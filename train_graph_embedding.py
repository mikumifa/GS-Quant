import argparse
import json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from dataset.dataloader import (
    BidirectionalOneShotIterator,
    TrainDataset,
    build_data_args,
    read_data_from_args,
)
from graph_embedding.model import KGEmbeddingModel
from utils import (
    log_metrics,
    quantized_to_token,
    read_cluster_embeddings,
    read_entity_initial_embedding,
    read_triple,
    read_triples_info,
    seed_everything,
    set_logger,
)


def override_config(args):
    """
    Override model and data configuration
    """

    with open(os.path.join(args.init_checkpoint, "config.json"), "r") as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict["countries"]
    if args.data_path is None:
        args.data_path = argparse_dict["data_path"]
    args.hidden_dim = argparse_dict["hidden_dim"]
    args.test_batch_size = argparse_dict["test_batch_size"]


def construct_args():
    parser = argparse.ArgumentParser(description="KG-FIT")
    # Data paths
    build_data_args(parser)
    # Data paths
    # train, valid, test
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_valid", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument(
        "--evaluate_train", action="store_true", help="Evaluate on training data"
    )

    parser.add_argument(
        "--countries", action="store_true", help="Use Countries S1/S2/S3 datasets"
    )
    parser.add_argument(
        "--model", type=str, default="RotatE", help="Knowledge graph embedding model"
    )
    parser.add_argument(
        "--regions",
        type=int,
        nargs="+",
        default=None,
        help="Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET",
    )

    parser.add_argument("-n", "--negative_sample_size", default=128, type=int)
    parser.add_argument("-d", "--hidden_dim", default=500, type=int)
    parser.add_argument("-g", "--gamma", default=12.0, type=float)
    parser.add_argument("-a", "--adversarial_temperature", default=1.0, type=float)
    parser.add_argument("-b", "--batch_size", default=1024, type=int)
    parser.add_argument("-r", "--regularization", default=0.0, type=float)
    parser.add_argument(
        "--test_batch_size", default=4, type=int, help="valid/test batch size"
    )

    # Model hyperparameters
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="cosine",
        choices=["euclidean", "cosine", "complex", "pi", "rotate"],
        help="Distance metric for link prediction",
    )

    # Hyperparameters
    parser.add_argument(
        "--rho",
        type=float,
        default=0.6,
        help="Weight for the randomly initialized component",
    )
    parser.add_argument("--rerank", type=str, default="false", help="Use reranking")
    parser.add_argument(
        "--fuse_score", type=str, default="false", help="Use fused score"
    )
    # Training settings
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="Number of training epochs"
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=50000,
        help="Number of epochs for early stopping",
    )
    parser.add_argument("--cuda", action="store_true", help="Use GPU for training")
    parser.add_argument(
        "--uni_weight",
        action="store_true",
        help="Use uniform weight for positive and negative samples",
    )
    parser.add_argument("--save_hit_k", default=0, type=int, help="set 0 to disable")
    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float)
    parser.add_argument("-cpu", "--cpu_num", default=10, type=int)
    parser.add_argument("-init", "--init_checkpoint", default=None, type=str)
    parser.add_argument("-save", "--save_path", default=None, type=str)
    parser.add_argument("--max_steps", default=100000, type=int)
    parser.add_argument("--warm_up_steps", default=None, type=int)

    parser.add_argument("--save_checkpoint_steps", default=10000, type=int)
    parser.add_argument("--valid_steps", default=10000, type=int)
    parser.add_argument(
        "--log_steps", default=10, type=int, help="train log every xx steps"
    )
    parser.add_argument(
        "--test_log_steps", default=1000, type=int, help="valid/test log every xx steps"
    )

    parser.add_argument("--nentity", type=int, default=0, help="DO NOT MANUALLY SET")
    parser.add_argument("--nrelation", type=int, default=0, help="DO NOT MANUALLY SET")

    # Hit record & adapter export
    parser.add_argument(
        "--hit_topk",
        type=int,
        default=100,
        help="Top-K predictions to keep when saving hit records.",
    )
    parser.add_argument(
        "--entity_quantized_path",
        type=str,
        default=None,
        help="Optional path to entity_quantized.json for quantized tokens in adapter data.",
    )
    parser.add_argument(
        "--build_adapter_data",
        action="store_true",
        help="Generate adapter-style JSONL from hit records of the best model.",
    )
    parser.add_argument(
        "--adapter_output_dir",
        type=str,
        default=None,
        help="Where to write adapter JSONL files; defaults to <save_path>/adapter_data.",
    )
    parser.add_argument(
        "--adapter_token_num",
        type=int,
        default=4,
        help="Number of quantized tokens to keep per entity for adapter data.",
    )
    parser.add_argument(
        "--adapter_wrap_token",
        action="store_true",
        help="Wrap quantized tokens with begin/end markers for adapter data.",
    )

    args = parser.parse_args()
    args.data_path = f"{args.data_path}/{args.dataset}"
    args.save_path = f"{args.process_path}/{args.dataset}/checkpoints/{args.model}_{args.hierarchy_type}_batch_{args.batch_size}_hidden_{args.hidden_dim}_dist_{args.distance_metric}_{time.strftime('%Y%m%d%H%M%S')}"

    return args


args = construct_args()
seed_everything(42)


def load_entity_detail_map(args) -> Dict[str, Dict]:
    path = os.path.join(
        f"{args.process_path}/{args.dataset}",
        f"entity_info_{args.hierarchy_type}_hier.json",
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_entity_quantized(args) -> Optional[Sequence[Sequence[int]]]:
    if not args.entity_quantized_path:
        return None
    if not os.path.isfile(args.entity_quantized_path):
        logging.warning(
            "entity_quantized_path=%s not found, skip quantized tokens",
            args.entity_quantized_path,
        )
        return None
    with open(args.entity_quantized_path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_wrap_quant(token: str, wrap: bool) -> str:
    if not wrap:
        return token
    return f"<#begin_of_entity>{token}<#end_of_entity>"


def get_quant_token(
    entity_idx: int,
    quant_table: Optional[Sequence[Sequence[int]]],
    token_num: int,
    wrap: bool,
) -> Optional[str]:
    if quant_table is None or entity_idx >= len(quant_table):
        return None
    token = quantized_to_token(quant_table[entity_idx], token_num=token_num)
    return maybe_wrap_quant(token, wrap)


def format_candidate_entries(
    entity_ids: Sequence[int],
    id2entity: Dict[int, str],
    entity_detail: Dict[str, Dict],
    quant_table: Optional[Sequence[Sequence[int]]],
    token_num: int,
    wrap_token: bool,
) -> Tuple[List[str], List[str]]:
    names: List[str] = []
    entries: List[str] = []
    for ent_id in entity_ids:
        ent_key = id2entity.get(ent_id, str(ent_id))
        name = entity_detail.get(ent_key, {}).get("text_label", ent_key)
        names.append(name)
        quant = get_quant_token(ent_id, quant_table, token_num, wrap_token)
        if quant:
            entries.append(f"{name} {quant}")
        else:
            entries.append(name)
    return names, entries


def build_entity_triplet_index(
    triples: Iterable[Tuple[int, int, int]]
) -> Dict[int, List[Tuple[int, int, int]]]:
    idx: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for h, r, t in triples:
        idx[h].append((h, r, t))
        idx[t].append((h, r, t))
    return idx


def format_triplet(
    h: int,
    r: int,
    t: int,
    id2entity: Dict[int, str],
    id2relation: Dict[int, str],
) -> str:
    return f"({id2entity.get(h, h)}, {id2relation.get(r, r)}, {id2entity.get(t, t)})"


def make_instruction(
    known_name: str,
    known_desc: str,
    quant_token: Optional[str],
    relation_text: str,
    predict_mode: str,
    triplet_texts: Sequence[str],
    candidate_block: str,
) -> str:
    unknown_symbol = "h" if predict_mode == "head" else "t"
    if predict_mode == "head":
        triple_str = f"({unknown_symbol}, {relation_text}, {known_name})"
    else:
        triple_str = f"({known_name}, {relation_text}, {unknown_symbol})"

    desc_block = known_desc.strip() if known_desc else ""
    quant_line = f"Quantized representation: {quant_token}\n" if quant_token else ""
    triplets_joined = "; ".join(triplet_texts)

    instruction = (
        f"Here is a triplet with {predict_mode} entity {unknown_symbol} unknown: {triple_str}.\n\n"
        f"Following are some details about {known_name}:\n"
        f"{quant_line}{desc_block}\n\n"
        f"Following are some triplets about {known_name}:\n"
        f"[{triplets_joined}]\n\n"
        f"What is the entity name of {unknown_symbol}? Select one from the list: [{candidate_block}]\n\n[Answer]:"
    )
    return instruction


def build_adapter_records(
    hit_records: Sequence[Dict],
    id2entity: Dict[int, str],
    id2relation: Dict[int, str],
    entity_detail: Dict[str, Dict],
    quant_table: Optional[Sequence[Sequence[int]]],
    token_num: int,
    wrap_token: bool,
    all_true_triples: Sequence[Tuple[int, int, int]],
    topk: int,
) -> List[Dict]:
    entity_trip_idx = build_entity_triplet_index(all_true_triples)
    records: List[Dict] = []

    for rec in hit_records:
        h, r, t = rec["h"], rec["r"], rec["t"]
        for mode in ("tail", "head"):
            hits = rec.get("hits_tail" if mode == "tail" else "hits_head", [])
            if not hits:
                continue
            rank = rec.get("rank_tail" if mode == "tail" else "rank_head")
            target_id = t if mode == "tail" else h
            known_id = h if mode == "tail" else t
            known_key = id2entity.get(known_id, str(known_id))
            target_key = id2entity.get(target_id, str(target_id))

            known_detail = entity_detail.get(known_key, {})
            known_name = known_detail.get("text_label", known_key)
            known_desc = (
                known_detail.get("llm_description")
                or known_detail.get("original_description", "")
            )
            target_name = entity_detail.get(target_key, {}).get(
                "text_label", target_key
            )

            # candidates
            top_entities = list(hits[:topk])
            if target_id not in top_entities:
                top_entities.append(target_id)
            cand_names, cand_entries = format_candidate_entries(
                top_entities,
                id2entity=id2entity,
                entity_detail=entity_detail,
                quant_table=quant_table,
                token_num=token_num,
                wrap_token=wrap_token,
            )

            quant_token = get_quant_token(known_id, quant_table, token_num, wrap_token)
            relation_text = id2relation.get(r, str(r))

            # neighbor triplets
            neighbor_triples = entity_trip_idx.get(known_id, [])[:10]
            triplet_texts = [
                format_triplet(hh, rr, tt, id2entity, id2relation)
                for hh, rr, tt in neighbor_triples
            ]

            instruction = make_instruction(
                known_name,
                known_desc,
                quant_token,
                relation_text,
                predict_mode=mode,
                triplet_texts=triplet_texts,
                candidate_block="; ".join(cand_entries),
            )

            records.append(
                {
                    "instruction": instruction,
                    "input": "",
                    "output": target_name,
                    "rank": rank,
                    "topk_names": cand_names,
                    "metadata": {
                        "triplet": [
                            id2entity.get(h, h),
                            relation_text,
                            id2entity.get(t, t),
                        ],
                        "predict": mode,
                        "inverse": mode == "head",
                        "target_entity": target_key,
                    },
                }
            )
    return records


def save_hit_records(hit_records, args, split_name):
    """
    Save hit records for a specific split (valid/test/train) without interfering model checkpoint.
    """
    if hit_records is None or len(hit_records) == 0:
        logging.warning(f"[Skip Save] No hit records found for {split_name}")
        return

    K = int(getattr(args, "save_hit_k", 0))
    if K <= 0:
        logging.warning(
            f"[Skip Save] save_hit_k <= 0, not saving hit records for {split_name}"
        )
        return

    save_path = os.path.join(args.save_path, f"hit_{split_name}_top_{K}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(hit_records, f, indent=2, ensure_ascii=False)

    logging.info(f"[Hit Records Saved] → {save_path}")


def evaluate_and_save_hits(
    model: KGEmbeddingModel,
    triples: Sequence[Tuple[int, int, int]],
    all_true_triples: Sequence[Tuple[int, int, int]],
    entity_info_split: Sequence,
    args,
    split_name: str,
    save_dir: str,
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate with hit@K saving (using args.hit_topk) and persist both metrics and hit records.
    """
    original_k = getattr(args, "save_hit_k", 0)
    args.save_hit_k = max(args.hit_topk, original_k)
    metrics, hit_records = model.test_step(
        model,
        triples,
        all_true_triples,
        entity_info_split,
        args,
        save_hit_k=True,
    )
    args.save_hit_k = original_k

    os.makedirs(save_dir, exist_ok=True)
    hit_path = os.path.join(save_dir, f"hit_{split_name}_top_{args.hit_topk}.json")
    with open(hit_path, "w", encoding="utf-8") as f:
        json.dump(hit_records, f, ensure_ascii=False, indent=2)
    logging.info("Hit records for %s saved to %s", split_name, hit_path)
    logging.info(
        "[Best %s] overall metrics: %s", split_name, metrics.get("overall", {})
    )
    return metrics, hit_records


def load_checkpoint_into_model(model: KGEmbeddingModel, ckpt_dir: str, use_cuda: bool):
    checkpoint_path = os.path.join(ckpt_dir, "checkpoint")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cuda" if use_cuda else "cpu"
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    logging.info("Loaded best checkpoint from %s", checkpoint_path)
    return checkpoint


def main(args):
    if (
        (not args.do_train)
        and (not args.do_valid)
        and (not args.do_test)
        and (not args.evaluate_train)
    ):
        raise ValueError("one of train/val/test mode must be choosed.")
    if args.init_checkpoint:
        override_config(args)
    os.makedirs(args.save_path, exist_ok=True)
    set_logger(args)
    (
        entity2id,
        id2entity,
        relation2id,
    ) = read_data_from_args(args).values()
    id2relation = {v: k for k, v in relation2id.items()}
    entity_detail = load_entity_detail_map(args)
    quant_table = load_entity_quantized(args)
    train_triples = read_triple(
        os.path.join(args.data_path, "train.txt"), entity2id, relation2id
    )
    logging.info("#train: %d" % len(train_triples))
    valid_triples = read_triple(
        os.path.join(args.data_path, "valid.txt"), entity2id, relation2id
    )
    logging.info("#valid: %d" % len(valid_triples))
    test_triples = read_triple(
        os.path.join(args.data_path, "test.txt"), entity2id, relation2id
    )
    logging.info("#test: %d" % len(test_triples))
    entity_info_train = read_triples_info(
        os.path.join(
            f"{args.process_path}/{args.dataset}",
            f"entity_info_{args.hierarchy_type}_hier.json",
        ),
        train_triples,
        id2entity,
    )
    entity_info_valid = read_triples_info(
        os.path.join(
            f"{args.process_path}/{args.dataset}",
            f"entity_info_{args.hierarchy_type}_hier.json",
        ),
        valid_triples,
        id2entity,
    )
    entity_info_test = read_triples_info(
        os.path.join(
            f"{args.process_path}/{args.dataset}",
            f"entity_info_{args.hierarchy_type}_hier.json",
        ),
        test_triples,
        id2entity,
    )

    # All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    # Load the entity hierarchy and text embeddings
    entity_text_embeddings = read_entity_initial_embedding(args)
    # Load the cluster embeddings
    cluster_embeddings = read_cluster_embeddings(args)
    ###### KG-FIT Model ######
    kgfit_model = KGEmbeddingModel(
        model=args.model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        entity_text_embeddings=entity_text_embeddings,
        cluster_embeddings=cluster_embeddings,
        rho=args.rho,
        distance_metric=args.distance_metric,
    )

    logging.info("Model Parameter Configuration:")
    for name, param in kgfit_model.named_parameters():
        logging.info(
            "Parameter %s: %s, require_grad = %s"
            % (name, str(param.size()), str(param.requires_grad))
        )

    if args.cuda:
        kgfit_model = kgfit_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(
                train_triples,
                args.nentity,
                args.nrelation,
                args.negative_sample_size,
                entity_info_train,
                "head-batch",
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn,
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(
                train_triples,
                args.nentity,
                args.nrelation,
                args.negative_sample_size,
                entity_info_train,
                "tail-batch",
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn,
        )

        train_iterator = BidirectionalOneShotIterator(
            train_dataloader_head, train_dataloader_tail
        )

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kgfit_model.parameters()),
            lr=current_learning_rate,
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info("Loading checkpoint %s..." % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, "checkpoint"))
        init_step = checkpoint["step"]
        kgfit_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if args.do_train:
            current_learning_rate = checkpoint["current_learning_rate"]
            warm_up_steps = checkpoint["warm_up_steps"]
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    else:
        logging.info("Ramdomly Initializing ...")
        init_step = 0

    step = init_step

    logging.info("Start Training...")
    logging.info("init_step = %d" % init_step)
    logging.info("batch_size = %d" % args.batch_size)
    logging.info("hidden_dim = %d" % args.hidden_dim)
    logging.info("gamma = %f" % args.gamma)
    logging.info("adversarial_temperature = %f" % args.adversarial_temperature)

    # Track best validation checkpoint even when not training
    best_valid_mrr = float("-inf") if args.do_valid else None
    best_valid_step = -1
    best_save_path = os.path.join(args.save_path, "best")

    ###### Training ######
    if args.do_train:
        logging.info("learning_rate = %d" % current_learning_rate)

        training_logs = []
        patience_limit = (
            args.early_stop if args.do_valid and args.early_stop > 0 else None
        )
        patience_counter = 0
        early_stop_triggered = False

        # Training Loop
        for step in range(init_step, args.max_steps):
            log = kgfit_model.train_step(kgfit_model, optimizer, train_iterator, args)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info(
                    "Change learning_rate to %f at step %d"
                    % (current_learning_rate, step)
                )
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kgfit_model.parameters()),
                    lr=current_learning_rate,
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    "step": step,
                    "current_learning_rate": current_learning_rate,
                    "warm_up_steps": warm_up_steps,
                }
                kgfit_model.save_model(kgfit_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(
                        training_logs
                    )
                log_metrics("Training average", step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info("Evaluating on Valid Dataset...")
                metrics = kgfit_model.test_step(
                    kgfit_model,
                    valid_triples,
                    all_true_triples,
                    entity_info_valid,
                    args,
                )
                log_metrics("Valid", step, metrics)
                if best_valid_mrr is not None:
                    current_mrr = metrics.get("overall", {}).get("MRR")
                    if current_mrr is not None and current_mrr > best_valid_mrr:
                        best_valid_mrr = current_mrr
                        best_valid_step = step
                        patience_counter = 0
                        save_variable_list = {
                            "step": step,
                            "current_learning_rate": current_learning_rate,
                            "warm_up_steps": warm_up_steps,
                        }
                        kgfit_model.save_model(
                            kgfit_model,
                            optimizer,
                            save_variable_list,
                            args,
                            save_path=best_save_path,
                        )
                        logging.info(
                            "New best model saved at step %d with Valid MRR %.4f",
                            step,
                            current_mrr,
                        )
                    elif patience_limit is not None and current_mrr is not None:
                        patience_counter += 1
                        logging.info(
                            "No Valid MRR improvement for %d evals (limit %d)",
                            patience_counter,
                            patience_limit,
                        )
                        if patience_counter >= patience_limit:
                            logging.info(
                                "Early stopping triggered at step %d (patience %d)",
                                step,
                                patience_limit,
                            )
                            early_stop_triggered = True
                            break

        save_variable_list = {
            "step": step,
            "current_learning_rate": current_learning_rate,
            "warm_up_steps": warm_up_steps,
        }
        kgfit_model.save_model(kgfit_model, optimizer, save_variable_list, args)
        if args.do_valid and best_valid_step >= 0:
            logging.info(
                "Best validation MRR %.4f at step %d saved to %s",
                best_valid_mrr,
                best_valid_step,
                best_save_path,
            )
        if early_stop_triggered:
            logging.info(
                "Training stopped early at step %d after %d validation checks without improvement",
                step,
                patience_limit,
            )

    best_model_dir = args.save_path
    if args.do_valid and best_valid_step >= 0:
        best_model_dir = best_save_path
    elif not args.do_train and args.init_checkpoint:
        best_model_dir = args.init_checkpoint

    # Reload best checkpoint for downstream evaluations/exports
    if os.path.isdir(best_model_dir):
        ckpt = load_checkpoint_into_model(kgfit_model, best_model_dir, args.cuda)
        step = ckpt.get("step", step)
    else:
        logging.warning("Best model dir %s not found; skip reload", best_model_dir)

    export_splits = []
    if args.do_valid:
        export_splits.append(("valid", valid_triples, entity_info_valid))
    if args.do_test:
        export_splits.append(("test", test_triples, entity_info_test))
    if args.evaluate_train:
        export_splits.append(("train", train_triples, entity_info_train))

    adapter_output_dir = (
        args.adapter_output_dir
        if args.adapter_output_dir
        else os.path.join(best_model_dir, "adapter_data")
    )

    for split_name, split_triples, split_info in export_splits:
        logging.info("Evaluating (%s) with hit@%d saving ...", split_name, args.hit_topk)
        metrics, hit_records = evaluate_and_save_hits(
            kgfit_model,
            split_triples,
            all_true_triples,
            split_info,
            args,
            split_name,
            save_dir=best_model_dir,
        )
        if args.build_adapter_data:
            os.makedirs(adapter_output_dir, exist_ok=True)
            adapter_records = build_adapter_records(
                hit_records,
                id2entity=id2entity,
                id2relation=id2relation,
                entity_detail=entity_detail,
                quant_table=quant_table,
                token_num=args.adapter_token_num,
                wrap_token=args.adapter_wrap_token,
                all_true_triples=all_true_triples,
                topk=args.hit_topk,
            )
            adapter_path = os.path.join(
                adapter_output_dir, f"{split_name}_adapter.jsonl"
            )
            with open(adapter_path, "w", encoding="utf-8") as fout:
                for rec in adapter_records:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logging.info(
                "Adapter-style data for %s written to %s (records=%d)",
                split_name,
                adapter_path,
                len(adapter_records),
            )

    ######################


if __name__ == "__main__":
    args = construct_args()
    main(args)
