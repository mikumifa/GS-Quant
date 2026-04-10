import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch.nn.utils.rnn import pad_sequence

from eval_llm import GenerationArguments

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Model identifier from the Hugging Face hub or local path."}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Configuration name if different from model_name_or_path. "
            "If not provided the configuration is loaded from model_name_or_path."
        },
    )
    tokens_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional JSON file containing a list of tokens to add to the tokenizer."
        },
    )
    target_modules: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={
            "help": "Comma separated list of module names where LoRA adapters are injected."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: float = field(
        default=16.0, metadata={"help": "LoRA alpha scaling factor."}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "LoRA dropout probability."}
    )
    lora_bias: str = field(
        default="none",
        metadata={
            "help": "LoRA bias type. Must be one of 'none', 'all', or 'lora_only'."
        },
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": "Torch dtype to load the base model with (e.g. 'float16', 'bfloat16', 'float32', 'auto')."
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device map passed to from_pretrained. Defaults to 'auto'. "
            "Set to None to disable device mapping."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Set True for architectures requiring remote code (e.g., Qwen/DeepSeek)."
        },
    )


@dataclass
class DataArguments:
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training data file (JSON/JSONL/CSV accepted)."},
    )
    text_column: str = field(
        default="instruction",
        metadata={"help": "Column containing the instruction or prompt."},
    )
    input_column: Optional[str] = field(
        default="input",
        metadata={
            "help": "Optional column with additional context appended to the instruction."
        },
    )
    response_column: str = field(
        default="output",
        metadata={"help": "Column containing the desired response/completion."},
    )
    source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length."},
    )
    target_max_len: int = field(
        default=64,
        metadata={"help": "Maximum target sequence length."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Truncate the number of training samples for debugging purposes."
        },
    )


DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class ScriptArguments:
    train_summary_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional JSON file to record training summary including metrics and artifact locations."
        },
    )


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def to_json_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, dict):
        return {k: to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_serializable(v) for v in value]
    return str(value)


def list_artifact_files(root: str) -> List[str]:
    collected: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            collected.append(os.path.abspath(os.path.join(dirpath, filename)))
    collected.sort()
    return collected


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)


def freeze_old_embeddings(model, new_token_ids: List[int]):
    embedding = model.get_input_embeddings()
    weight = embedding.weight
    requires_grad = torch.zeros_like(weight, dtype=torch.bool)
    requires_grad[new_token_ids] = True

    def hook(grad):
        grad_masked = grad.clone()
        grad_masked[~requires_grad] = 0
        return grad_masked

    weight.register_hook(hook)
    logger.info("Only new token embeddings will receive gradient updates.")


def mask_lm_head_output(model, new_token_ids: List[int]):
    lm_head = model.get_output_embeddings()
    with torch.no_grad():
        lm_head.weight[new_token_ids] = 0.0  # don't need to generate new token
    lm_head.weight.requires_grad = False
    logger.info("lm_head outputs for new tokens will stay 0 and frozen.")


def maybe_expand_tokenizer(
    tokenizer, model, tokens_file: Optional[str]
) -> Tuple[int, List[int]]:
    if not tokens_file:
        return 0, []
    with open(tokens_file, "r", encoding="utf-8") as handle:
        tokens = json.load(handle)

    existing_vocab = set(tokenizer.get_vocab())
    new_tokens = [tok for tok in tokens if tok not in existing_vocab]
    if not new_tokens:
        logger.info("No new tokens to add to tokenizer.")
        return 0, []
    old_vocab_size = len(tokenizer)
    added = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    new_token_ids = list(range(old_vocab_size, old_vocab_size + added))
    logger.info("Added %d tokens to tokenizer vocabulary.", added)
    return added, new_token_ids


def disable_new_token_generation(generation_config, new_token_ids: List[int]):
    bad_words_ids = [[i] for i in new_token_ids]
    generation_config.bad_words_ids = bad_words_ids
    logger.info("New tokens are disabled from generation.")


def parse_target_modules(target_modules: Optional[str]) -> Optional[List[str]]:
    if target_modules is None:
        return None
    modules = [module.strip() for module in target_modules.split(",") if module.strip()]
    return modules or None


def resolve_dtype(
    model_args: ModelArguments, training_args: TrainingArguments
) -> Optional[torch.dtype]:
    if training_args.bf16:
        return torch.bfloat16
    if training_args.fp16:
        return torch.float16
    if model_args.torch_dtype:
        key = model_args.torch_dtype.lower()
        if key == "auto":
            return None
        if key not in DTYPE_MAP:
            supported = ", ".join(sorted(DTYPE_MAP.keys()) + ["auto"])
            raise ValueError(
                f"Unsupported torch_dtype '{model_args.torch_dtype}'. Choose from: {supported}"
            )
        return DTYPE_MAP[key]
    return None


def infer_dataset_extension(path: str) -> str:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in {".jsonl", ".json"}:
        return "json"
    if ext == ".csv":
        return "csv"
    raise ValueError(f"Unsupported dataset file extension: {ext}")


class InstructionDataCollator:
    def __init__(self, tokenizer, data_args: DataArguments):
        self.tokenizer = tokenizer
        self.data_args = data_args

    def _build_source(
        self, instruction: Optional[str], input_text: Optional[str]
    ) -> str:
        prompt_parts: List[str] = []
        if instruction:
            prompt_parts.append(str(instruction).strip())
        if input_text:
            prompt_parts.append(str(input_text).strip())
        prompt = "\n\n".join(prompt_parts).strip()
        if self.tokenizer.bos_token:
            return f"{self.tokenizer.bos_token}{prompt}".strip()
        return prompt

    def _build_target(self, output: Optional[str]) -> str:
        target = str(output).strip() if output is not None else ""
        if self.tokenizer.eos_token:
            return f"{target}{self.tokenizer.eos_token}".strip()
        return target

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        if not features:
            return {
                "input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long),
            }

        sources: List[str] = []
        targets: List[str] = []
        for example in features:
            instruction = example[self.data_args.text_column]
            input_text = example[self.data_args.input_column]
            output = example.get(self.data_args.response_column)
            sources.append(self._build_source(instruction, input_text))
            targets.append(self._build_target(output))

        tokenized_sources = self.tokenizer(
            sources,
            max_length=self.data_args.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.data_args.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for src, tgt in zip(
            tokenized_sources["input_ids"], tokenized_targets["input_ids"]
        ):
            src_tensor = torch.tensor(src, dtype=torch.long)
            tgt_tensor = torch.tensor(tgt, dtype=torch.long)
            input_ids.append(torch.cat([src_tensor, tgt_tensor]))
            labels.append(
                torch.cat(
                    [
                        torch.full(
                            (src_tensor.size(0),),
                            IGNORE_INDEX,
                            dtype=torch.long,
                        ),
                        tgt_tensor,
                    ]
                )
            )

        padded_input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }


def prepare_datasets(data_args: DataArguments) -> Optional[Dataset]:
    data_files = data_args.train_file
    if data_files is None:
        raise ValueError("A training file must be provided via --train_file.")
    dataset_extension = infer_dataset_extension(data_files)
    raw_datasets: DatasetDict = load_dataset(
        dataset_extension,
        data_files={"train": data_files},
    )
    train_dataset = raw_datasets.get("train")
    limit = data_args.max_train_samples
    if limit is not None:
        capped = min(limit, len(train_dataset))
        train_dataset = train_dataset.select(range(capped))
        logger.info("Limiting training dataset to %d samples.", capped)
    logger.info("Loaded %d training examples.", len(train_dataset))
    return train_dataset


def main() -> None:
    parser = HfArgumentParser(
        [
            ModelArguments,  # type:ignore
            DataArguments,  # type:ignore
            TrainingArguments,  # type:ignore
            ScriptArguments,  # type:ignore
            GenerationArguments,  # type:ignore
        ]
    )
    model_args, data_args, training_args, script_args, generation_args = (
        parser.parse_args_into_dataclasses()
    )
    training_args.seed = 42
    training_args.data_seed = 42
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    training_args.remove_unused_columns = False

    setup_logging()

    set_seed(training_args.seed)

    torch_dtype = resolve_dtype(model_args, training_args)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code,
    )
    model_kwargs: Dict[str, object] = {
        "config": config,
    }

    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if model_args.device_map is not None and not training_args.deepspeed:
        model_kwargs["device_map"] = model_args.device_map

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added_tokens, new_token_ids = maybe_expand_tokenizer(
        tokenizer, model, model_args.tokens_file
    )
    trainable_token_indices = None
    if added_tokens:
        logger.info("Resized embeddings for %d newly added tokens.", added_tokens)
        # freeze_old_embeddings(model, new_token_ids)
        # mask_lm_head_output(model, new_token_ids)
        # disable_new_token_generation(training_args.generation_config, new_token_ids)
        trainable_token_indices = {"embed_tokens": new_token_ids}
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias=model_args.lora_bias,
        trainable_token_indices=trainable_token_indices,
        target_modules=parse_target_modules(model_args.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)  # type:ignore
    train_dataset = prepare_datasets(data_args)
    if train_dataset is None:
        raise ValueError("Failed to load training dataset from the provided file.")
    data_collator = InstructionDataCollator(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    if trainer.is_world_process_zero():
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Training/evaluation parameters %s", model_args)
        logger.info("Training/evaluation parameters %s", data_args)
        logger.info("Training/evaluation parameters %s", script_args)
        logger.info("Training/evaluation parameters %s", generation_args)
        model.print_trainable_parameters()  # type:ignore

    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        if script_args.train_summary_file:
            summary_payload: Dict[str, Any] = {
                "train_file": data_args.train_file,
                "output_dir": os.path.abspath(training_args.output_dir),
                "lora_adapter_path": os.path.abspath(training_args.output_dir),
                "num_train_examples": len(train_dataset),
                "metrics": to_json_serializable(metrics),
            }
            state = trainer.state
            summary_payload["global_step"] = to_json_serializable(
                getattr(state, "global_step", None)
            )
            if getattr(state, "epoch", None) is not None:
                summary_payload["completed_epochs"] = to_json_serializable(state.epoch)
            if getattr(state, "best_model_checkpoint", None):
                summary_payload["best_model_checkpoint"] = state.best_model_checkpoint
            if getattr(state, "best_metric", None) is not None:
                summary_payload["best_metric"] = to_json_serializable(state.best_metric)
            write_json(script_args.train_summary_file, summary_payload)
            logger.info(
                "Training summary written to %s", script_args.train_summary_file
            )


if __name__ == "__main__":
    main()
