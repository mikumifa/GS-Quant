import argparse
import json
import os
import re
import time
from typing import Dict, List, Sequence, Tuple

from utils import quantized_to_token

BEGIN_ENTITY_TOKEN = "<#begin_of_entity>"
END_ENTITY_TOKEN = "<#end_of_entity>"


def maybe_wrap_quant_tokens(token_str: str, enabled: bool) -> str:
    if not enabled:
        return token_str
    return f"{BEGIN_ENTITY_TOKEN}{token_str}{END_ENTITY_TOKEN}"


def ensure_special_tokens(tokens_path: str) -> None:
    if not os.path.exists(tokens_path):
        print(
            f"[Warn] tokens.json not found at {tokens_path}; skipping special token update."
        )
        return

    with open(tokens_path, "r", encoding="utf-8") as f:
        tokens = json.load(f)

    if not isinstance(tokens, list):
        print(f"[Warn] tokens.json at {tokens_path} is not a list; skipping update.")
        return

    updated = False
    for special in (BEGIN_ENTITY_TOKEN, END_ENTITY_TOKEN):
        if special not in tokens:
            tokens.append(special)
            updated = True

    if updated:
        with open(tokens_path, "w", encoding="utf-8") as f:
            json.dump(tokens, f, ensure_ascii=False, indent=2)
        print(f"[Info] Added special entity tokens to {tokens_path}")
    else:
        print(f"[Info] Special entity tokens already present in {tokens_path}")


def load_entity_mappings(entities_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    entity2id: Dict[str, int] = {}
    id2entity: Dict[int, str] = {}
    with open(entities_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, entity = line.split(None, 1)
            idx = int(idx_str)
            entity2id[entity] = idx
            id2entity[idx] = entity
    return entity2id, id2entity


def load_entity_info(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_entity_info_with_fallback(
    entity_info_path: str, kg_data_dir: str, kg_dataset: str
) -> Dict[str, Dict[str, str]]:
    if entity_info_path and os.path.isfile(entity_info_path):
        return load_entity_info(entity_info_path)

    # Fallback for datasets like WN18RR that ship entity2text.txt instead of entity.json
    fallback_path = os.path.join(kg_data_dir, kg_dataset, "entity2text.txt")
    if not os.path.isfile(fallback_path):
        raise FileNotFoundError(
            f"Entity info not found at {entity_info_path} and no fallback file at {fallback_path}"
        )

    entity_info: Dict[str, Dict[str, str]] = {}
    with open(fallback_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                continue
            ent_id, text = line.split("\t", 1)
            short_name = text.split(",", 1)[0].strip()
            entity_info[ent_id] = {
                "name": short_name,
                "text_label": short_name,
                "desc": text,
            }
    print(
        f"[Info] Loaded entity info from fallback: {fallback_path} (total {len(entity_info)})"
    )
    return entity_info


def build_candidate_entries(
    topk_entities: Sequence[str],
    entity2id: Dict[str, int],
    quantized_table: Sequence[Sequence[int]],
    entity_info: Dict[str, Dict[str, str]],
    token_num: int,
    wrap_token: bool,
) -> List[str]:
    entries: List[str] = []

    for idx, entity_key in enumerate(topk_entities):
        name = entity_info[entity_key]["name"]

        entity_idx = entity2id[entity_key]
        quantized = quantized_table[entity_idx]
        if token_num <= 0:
            entries.append(f"{name}")
        else:
            quant_token = quantized_to_token(quantized, token_num=token_num)
            quant_token = maybe_wrap_quant_tokens(quant_token, wrap_token)
            entries.append(f"{name} {quant_token}")

    return entries


def insert_known_entity_quant(text: str, quant: str) -> str:
    pattern = re.compile(r"(Following are some details about .+?:\n)")

    def repl(match: re.Match) -> str:
        return match.group(1) + f"Quantized representation: {quant}\n"

    new_text, count = pattern.subn(repl, text, count=1)
    if count == 0:
        return text + f"\nQuantized representation: {quant}\n"
    return new_text


def replace_candidate_block(text: str, new_block: str) -> str:
    pattern = re.compile(
        r"(Select one from the list:\s*\[)(.*)(\])(\s*\n\s*\[Answer\]:?)", re.DOTALL
    )

    def repl(match: re.Match) -> str:
        return match.group(1) + new_block + match.group(3) + match.group(4)

    new_text, count = pattern.subn(repl, text, count=1)
    if count == 0:
        appended = f"\nSelect one from the list: [{new_block}]\n\n[Answer]:"
        return text + appended
    return new_text


def clean_query_markers(text: str) -> str:
    text = re.sub(r"\s*\[QUERY\]", "", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\(\s+", "(", text)
    return text


def resolve_target_entity(
    head: str,
    tail: str,
    inverse: bool,
    topk_entities: Sequence[str],
    topk_names_orig: Sequence[str],
    entity_info: Dict[str, Dict[str, str]],
    output_text: str,
) -> str:
    matched = next(
        (
            topk_entities[idx]
            for idx, name in enumerate(topk_names_orig)
            if name == output_text and idx < len(topk_entities)
        ),
        None,
    )
    if matched:
        return matched

    matched = next(
        (
            entity_key
            for entity_key in topk_entities
            if entity_info[entity_key]["name"] == output_text
        ),
        None,
    )
    if matched:
        return matched

    matched = next(
        (
            entity_key
            for entity_key in (head, tail)
            if entity_info.get(entity_key, {}).get("text_label") == output_text
        ),
        None,
    )
    if matched:
        return matched

    return tail if not inverse else head


def adapt_sample(
    sample: Dict,
    entity2id: Dict[str, int],
    entity_info: Dict[str, Dict[str, str]],
    quantized_table: Sequence[Sequence[int]],
    token_num: int,
    wrap_token: bool,
) -> Dict:
    head, relation, tail = sample["triplet"]
    inverse = sample["inverse"]
    topk_entities = sample["topk_ents"]
    topk_names = sample["topk_names"]
    output_text = sample["output"]

    if inverse:
        predict_mode = "head"
    else:
        predict_mode = "tail"

    known_entity_key = head
    target_entity_key = tail
    known_entity_idx = entity2id[known_entity_key]

    candidate_entries = build_candidate_entries(
        topk_entities,
        entity2id,
        quantized_table,
        entity_info,
        token_num,
        wrap_token,
    )

    target_info = entity_info[target_entity_key]
    output_name = target_info["name"]
    assert output_text == output_name
    instruction = sample["input"]
    if token_num > 0:
        instruction = clean_query_markers(instruction)
        known_quant = quantized_to_token(
            quantized_table[known_entity_idx], token_num=token_num
        )
        known_quant = maybe_wrap_quant_tokens(known_quant, wrap_token)
        instruction = insert_known_entity_quant(instruction, known_quant)
        instruction = replace_candidate_block(
            instruction, "; ".join(candidate_entries)
        ).strip()

    return {
        "instruction": instruction,
        "input": "",
        "output": output_name,
        "rank": sample.get("rank"),
        "topk_names": topk_names,
        "metadata": {
            "triplet": [head, relation, tail],
            "predict": predict_mode,
            "inverse": inverse,
            "target_entity": target_entity_key,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt third-party KG fine-tuning data to project LoRA format."
    )
    parser.add_argument(
        "--kg_data_dir",
        type=str,
        default="data",
        help="Directory containing the knowledge graph data (entities.dict etc.).",
    )
    parser.add_argument(
        "--kg_dataset",
        type=str,
        default="FB15K-237",
        help="Knowledge graph dataset name inside kg_data_dir.",
    )
    parser.add_argument(
        "--dift_root",
        type=str,
        default="data/DIFT-dataset",
        help="Root directory containing external datasets.",
    )
    parser.add_argument(
        "--dift_dataset",
        type=str,
        default="FB15K237",
        help="Subdirectory name for the external dataset.",
    )
    parser.add_argument(
        "--dift_source",
        type=str,
        default="CoLE",
        help="Source/model folder name inside the external dataset directory.",
    )
    parser.add_argument(
        "--entity_info_path",
        type=str,
        default=None,
        help="Path to entity info JSON used for canonical names. Optional if entity2text.txt is available.",
    )
    parser.add_argument(
        "--quantized_path",
        type=str,
        required=True,
        help="Path to entity quantized codes JSON.",
    )
    parser.add_argument(
        "--token_num",
        type=int,
        default=4,
        help="Number of quantized tokens to keep per entity.",
    )
    parser.add_argument(
        "--wrap_token",
        action="store_true",
        help="Wrap quantized codes with special begin/end tokens and add them to tokens.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write adapted JSONL files. Defaults to processed_data/<kg_dataset>/adapter_<timestamp>.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to convert.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    entities_path = os.path.join(args.kg_data_dir, args.kg_dataset, "entities.dict")
    entity2id, _ = load_entity_mappings(entities_path)
    entity_info = load_entity_info_with_fallback(
        args.entity_info_path, args.kg_data_dir, args.kg_dataset
    )
    with open(args.quantized_path, "r", encoding="utf-8") as f:
        quantized_table = json.load(f)
    if args.wrap_token:
        tokens_path = os.path.join(os.path.dirname(args.quantized_path), "tokens.json")
        ensure_special_tokens(tokens_path)

    output_dir = (
        args.output_dir
        if args.output_dir
        else os.path.join(
            "processed_data",
            args.kg_dataset,
            f"adapter_{args.dift_source}_{time.strftime('%Y%m%d%H%M%S')}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    input_dir = os.path.join(
        args.dift_root, args.dift_dataset, args.dift_source, "data_KGELlama"
    )

    for split in args.splits:
        input_path = os.path.join(input_dir, f"{split}.json")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)

        output_path = os.path.join(output_dir, f"{split}.jsonl")
        converted: List[Dict] = []

        for sample in raw_samples:
            converted.append(
                adapt_sample(
                    sample,
                    entity2id=entity2id,
                    entity_info=entity_info,
                    quantized_table=quantized_table,
                    token_num=args.token_num,
                    wrap_token=args.wrap_token,
                )
            )

        with open(output_path, "w", encoding="utf-8") as fout:
            for record in converted:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[Done] Wrote {len(converted)} samples to {output_path}")


if __name__ == "__main__":
    main(parse_args())
