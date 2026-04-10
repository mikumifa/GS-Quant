import csv
import json
import os
from pathlib import Path
import time
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    set_seed,
)


def load_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    normalized = name.lower()
    if normalized in {"auto", ""}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        supported = ", ".join(sorted(mapping) + ["auto"])
        raise ValueError(f"Unsupported torch_dtype '{name}'. Choose from: {supported}")
    return mapping[normalized]


@dataclass
class PiplineArguments:
    summary_config_path: str = field(
        default=None,
        metadata={"help": "Base model identifier from Hugging Face hub or local path."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Base model identifier from Hugging Face hub or local path."}
    )
    lora_adapter_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the LoRA adapter weights (directory containing adapter_config.json)."
        },
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the LoRA adapter weights (directory containing adapter_config.json)."
        },
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "torch dtype to load the base model with."},
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device map passed to from_pretrained. Use 'auto' for accelerate dispatch."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Set True for architectures requiring remote code (e.g., Qwen)."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="data", metadata={"help": "Root directory for data."}
    )
    process_path: str = field(
        default="processed_data",
        metadata={"help": "Root directory for processed data."},
    )
    dataset: str = field(default="FB15K-237", metadata={"help": "Dataset name."})
    file: str = field(
        default="test.jsonl",
        metadata={
            "help": "Path to the evaluation file (JSONL). Relative paths resolved from the current working directory."
        },
    )
    source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length."},
    )
    target_max_len: int = field(
        default=64,
        metadata={"help": "Maximum target sequence length."},
    )


@dataclass
class EvaluationArguments:
    batch_size: int = field(default=1, metadata={"help": "Batch size for generation."})
    topk: int = field(default=10, metadata={"help": "Reserved for compatibility."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to save predictions. "
            "If unset, defaults to processed_data/<dataset>/checkpoints/EVAL_LLM_<timestamp>."
        },
    )
    output_filename: str = field(
        default="predictions.jsonl",
        metadata={"help": "Filename for the prediction JSONL output."},
    )
    hits: List[int] = field(
        default_factory=lambda: [1, 3, 10],
        metadata={"help": "Hit@K thresholds to report."},
    )
    csv_filename: Optional[str] = field(
        default="rank_predictions.csv",
        metadata={
            "help": "Filename for detailed CSV output. Leave empty to skip writing."
        },
    )
    metrics_filename: Optional[str] = field(
        default="metrics.json",
        metadata={
            "help": "Filename for aggregated metrics JSON. Leave empty to skip writing."
        },
    )


@dataclass
class GenerationArguments:
    # control the length of the output
    max_new_tokens: Optional[int] = field(default=64)
    min_new_tokens: Optional[int] = field(default=1)

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

    num_return_sequences: Optional[int] = field(default=1)
    output_scores: Optional[bool] = field(default=False)
    return_dict_in_generate: Optional[bool] = field(default=True)


class Evaluator:
    def __init__(
        self,
        model,
        data_args: DataArguments,
        tokenizer,
        generation_config: GenerationConfig,
        batch_size: int,
    ) -> None:
        self.model = model
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.batch_size = batch_size

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> List[str]:
        encoded = self.tokenizer(
            prompts,
            max_length=self.data_args.source_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)
        generated_ids = self.model.generate(
            **encoded,
            generation_config=self.generation_config,
        )
        input_ids = encoded["input_ids"]
        prompt_lengths = [len(ids) for ids in input_ids]
        new_tokens_list = [
            generated_ids.sequences[i][prompt_lengths[i] :]
            for i in range(generated_ids.sequences.shape[0])
        ]
        outputs = self.tokenizer.batch_decode(
            new_tokens_list,
            max_length=self.data_args.source_max_len,
            skip_special_tokens=True,
            truncation=True,
        )
        return outputs


def prepare_model_and_tokenizer(
    model_args: ModelArguments, generation_config: GenerationConfig
):
    dtype = resolve_dtype(model_args.torch_dtype)
    model_kwargs = {}
    if model_args.device_map is not None:
        model_kwargs["device_map"] = model_args.device_map
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if model_args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    tokenizer_path = model_args.tokenizer_path or model_args.lora_adapter_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    if model_args.lora_adapter_path:
        model = PeftModel.from_pretrained(model, model_args.lora_adapter_path)
    model.eval()
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.bos_token_id = tokenizer.bos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("bos_token:", tokenizer.bos_token, tokenizer.bos_token_id)
    print(
        "generation_config:",
        generation_config.pad_token_id,
        generation_config.eos_token_id,
    )
    return model, tokenizer


def resolve_output_dir(data_args: DataArguments, eval_args: EvaluationArguments) -> str:
    if eval_args.output_dir:
        return os.path.abspath(eval_args.output_dir)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    default_dir = os.path.join(
        data_args.process_path,
        data_args.dataset,
        "checkpoints",
        f"EVAL_LLM_{timestamp}",
    )
    return os.path.abspath(default_dir)


def resolve_data_file(data_args: DataArguments) -> str:
    if os.path.isabs(data_args.file):
        return data_args.file
    return os.path.abspath(data_args.file)


def extract_primary_prediction(text: str) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def split_non_empty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_label(text: str) -> str:
    text = str(text)
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    return normalized


def build_prompt(sample: Dict[str, Any]) -> str:
    instruction = sample.get("instruction") or ""
    input_text = sample.get("input") or ""
    return f"{instruction}{input_text}"


def detect_direction(prompt: str) -> str:
    if "Here is a triplet with tail entity" in prompt:
        return "tail"
    if "Here is a triplet with head entity" in prompt:
        return "head"
    if "Predict the tail entity" in prompt:
        return "tail"
    if "Predict the head entity" in prompt:
        return "head"
    raise ValueError("Unable to detect prediction direction from prompt.")


def compute_predicted_rank(
    prediction: str, target: str, original_rank: int, topk_names: List[str]
) -> int:
    if prediction == target:
        return 1
    try:
        index = topk_names.index(prediction)
    except ValueError:
        return original_rank + 1
    if index >= original_rank:
        return original_rank + 1
    return original_rank


class MetricsAggregator:
    def __init__(self, hits: List[int]) -> None:
        if not hits:
            raise ValueError("At least one Hit@K threshold is required.")
        self.hits = sorted(set(int(k) for k in hits))
        self.ranks: Dict[str, List[int]] = {
            "head": [],
            "tail": [],
            "overall": [],
        }

    def add(self, direction: str, rank: int) -> None:
        if direction not in self.ranks:
            raise ValueError(f"Unsupported direction '{direction}'.")
        value = int(rank)
        self.ranks[direction].append(value)
        if direction != "overall":
            self.ranks["overall"].append(value)

    def metrics_for(self, direction: str) -> Dict[str, float]:
        ranks = self.ranks.get(direction, [])
        return compute_metrics(ranks, self.hits)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {direction: self.metrics_for(direction) for direction in self.ranks}

    def has_overall(self) -> bool:
        return bool(self.ranks["overall"])


def compute_metrics(
    ranks: List[int], hits: Optional[List[int]] = None
) -> Dict[str, float]:
    ks = sorted(set(int(k) for k in (hits or [1, 3, 10])))
    metrics: Dict[str, float] = {}
    if not ranks:
        metrics["count"] = 0
        for k in ks:
            metrics[f"hits{k}"] = 0.0
        metrics["mrr"] = 0.0
        return metrics
    arr = np.array(ranks, dtype=np.float32)
    for k in ks:
        metrics[f"hits{k}"] = float(np.mean(arr <= k))
    metrics["mrr"] = float(np.mean(1.0 / arr))
    metrics["count"] = int(len(ranks))
    return metrics


def main() -> None:
    hfparser = HfArgumentParser(
        [
            ModelArguments,
            DataArguments,
            EvaluationArguments,
            GenerationArguments,
            PiplineArguments,
        ]
    )
    (model_args, data_args, eval_args, generation_args, pipline_args) = (
        hfparser.parse_args_into_dataclasses()
    )

    set_seed(eval_args.seed)
    generation_config = GenerationConfig(**vars(generation_args))

    if pipline_args.summary_config_path:
        summary_path = os.path.abspath(pipline_args.summary_config_path)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        model_args.lora_adapter_path = summary["lora_adapter_path"]
        model_args.tokenizer_path = summary["lora_adapter_path"]
        data_args.dataset = Path(summary["train_file"]).parts[-3]
        test_candidate = Path(summary["train_file"]).with_name("test.jsonl")
        data_args.file = str(test_candidate)
        eval_args.output_dir = os.path.join(
            Path(summary["output_dir"]),
            f"EVAL_LLM_{time.strftime('%Y%m%d%H%M%S')}",
        )
        print(f"[Pipeline] Loaded training summary from {summary_path}")
        print(f"  ├─ LoRA Adapter Path: {model_args.lora_adapter_path}")
        print(f"  ├─ Tokenizer Path:    {model_args.tokenizer_path}")
        print(f"  ├─ Dataset:           {data_args.dataset}")
        print(f"  ├─ Data File:         {data_args.file}")
        print(f"  ├─ Eval Output Dir:   {eval_args.output_dir}")
        print("====================================================")

    output_dir = resolve_output_dir(data_args, eval_args)
    os.makedirs(output_dir, exist_ok=True)
    data_file = resolve_data_file(data_args)
    data = load_data(data_file)

    model, tokenizer = prepare_model_and_tokenizer(model_args, generation_config)
    evaluator = Evaluator(
        model, data_args, tokenizer, generation_config, eval_args.batch_size
    )
    aggregator = MetricsAggregator(eval_args.hits)
    csv_rows: List[Dict[str, Any]] = []
    output_path = os.path.join(output_dir, eval_args.output_filename)
    progress = tqdm(
        range(0, len(data), eval_args.batch_size),
        desc="Evaluating",
    )
    with open(output_path, "w", encoding="utf-8") as writer:
        for start in progress:
            batch = data[start : start + eval_args.batch_size]
            prompts = [build_prompt(sample) for sample in batch]
            predictions = evaluator.generate(prompts)
            for sample, prompt, prediction in zip(batch, prompts, predictions):
                record = {"prompt": prompt, "output": prediction}
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()

                target_lines = split_non_empty_lines(sample.get("output") or "")
                predicted_lines = split_non_empty_lines(prediction)
                normalized_target = [normalize_label(line) for line in target_lines]
                normalized_prediction = [
                    normalize_label(line) for line in predicted_lines
                ]
                normalized_topk = [
                    normalize_label(name) for name in (sample.get("topk_names") or [])
                ]
                base_rank = sample["rank"]
                direction = detect_direction(prompt)

                target_value = normalized_target[0] if normalized_target else ""
                predict_value = (
                    normalized_prediction[0] if normalized_prediction else ""
                )

                resolved_rank = compute_predicted_rank(
                    predict_value,
                    target_value,
                    base_rank,
                    normalized_topk,
                )
                aggregator.add(direction, resolved_rank)
                csv_rows.append(
                    {
                        "direction": direction,
                        "rank": resolved_rank,
                        "predicted_labels": " | ".join(predicted_lines),
                        "target_labels": " | ".join(target_lines),
                    }
                )

            if aggregator.has_overall():
                overall_metrics = aggregator.metrics_for("overall")
                postfix = {
                    "MRR": f"{overall_metrics['mrr']:.4f}",
                }
                for k in aggregator.hits:
                    postfix[f"H@{k}"] = f"{overall_metrics.get(f'hits{k}', 0.0):.4f}"
                postfix["N"] = str(int(overall_metrics.get("count", 0)))
                progress.set_postfix(postfix)

    metrics_summary = aggregator.summary()
    metrics_output = {
        "hits": aggregator.hits,
        "metrics": metrics_summary,
    }

    print("\nEvaluation Metrics:")
    for direction in ("tail", "head", "overall"):
        metrics = metrics_summary.get(direction)
        if not metrics:
            continue
        count = int(metrics["count"])
        print(f"{direction.upper()}:")
        print(f"  Samples: {count}")
        print(f"  MRR: {metrics['mrr']:.4f}")
        for k in aggregator.hits:
            value = metrics.get(f"hits{k}", 0.0)
            print(f"  Hits@{k}: {value:.4f}")
        print()

    metrics_path = None
    if eval_args.metrics_filename:
        metrics_path = os.path.join(output_dir, eval_args.metrics_filename)
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics_output, handle, ensure_ascii=False, indent=2)

    csv_path = None
    if eval_args.csv_filename and csv_rows:
        csv_path = os.path.join(output_dir, eval_args.csv_filename)
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "direction",
                    "rank",
                    "predicted_labels",
                    "target_labels",
                ],
            )
            csv_writer.writeheader()
            csv_writer.writerows(csv_rows)
        print(f"Saved detailed ranks to: {csv_path}")

    args_path = os.path.join(output_dir, "eval_args.json")
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_args": vars(model_args),
                "data_args": vars(data_args),
                "evaluation_args": vars(eval_args),
                "generation_args": vars(generation_args),
                "output_path": output_path,
                "data_file": data_file,
                "metrics_path": metrics_path,
                "csv_path": csv_path,
                "metrics": metrics_output,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
