import argparse
import json
import logging
import os
import random
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from dataset.dataloader import (
    OneShotIterator,
    EmbDataset,
    build_data_args,
    read_data_from_args,
)
from utils import (
    log_metrics,
    quantized_to_token,
    read_entity_info,
    read_embedding_from_path,
    seed_everything,
    set_logger,
)
from codebook.rqvae import RQVAE


def construct_args():
    parser = argparse.ArgumentParser(description="CodeBook")
    # Data paths
    build_data_args(parser)
    # Data paths
    parser.add_argument(
        "-eep", "--entity_embeddings_path", type=str, help="entity data path."
    )
    parser.add_argument(
        "-cep", "--cluster_embeddings_path", type=str, help="cluster data path."
    )
    parser.add_argument(
        "-eip", "--entity_info_path", type=str, help="entity info path."
    )
    # Training settings
    parser.add_argument("--cuda", action="store_true", help="Use GPU for training")
    parser.add_argument("-d", "--hidden_dim", default=512, type=int)
    parser.add_argument("-cn", "--codebook_num", default=6, type=int)
    parser.add_argument("-cs", "--codebook_size", default=1024, type=int)
    parser.add_argument("--subspace_num", default=3, type=int)
    parser.add_argument("--subspace_size", default=1, type=int)

    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float)
    parser.add_argument("-cpu", "--cpu_num", default=10, type=int)
    parser.add_argument("-init", "--init_checkpoint", default=None, type=str)
    parser.add_argument("-save", "--save_path", default=None, type=str)
    parser.add_argument("--max_steps", default=500, type=int)
    parser.add_argument("--batch_size", default=16348, type=int)

    parser.add_argument("--warm_up_steps", default=None, type=int)
    parser.add_argument(
        "--log_steps", default=1, type=int, help="train log every xx steps"
    )

    parser.add_argument("--nentity", type=int, default=0, help="DO NOT MANUALLY SET")
    parser.add_argument("--nrelation", type=int, default=0, help="DO NOT MANUALLY SET")
    parser.add_argument(
        "--commit_loss_weight",
        type=float,
        default=0.25,
        help="Weight for commit_loss",
    )
    parser.add_argument(
        "--lambda_1",
        type=float,
        default=0.02,
        help="Weight for self_cluster_dist",
    )
    parser.add_argument(
        "--lambda_2",
        type=float,
        default=0.01,
        help="Secondary weighting term (kept for loss logging compatibility)",
    )
    parser.add_argument(
        "--self_recon_weight",
        type=float,
        default=1.0,
        help="Weight applied to self_recon_loss inside transformer recon term",
    )
    parser.add_argument(
        "--self_cluster_recon_weight",
        type=float,
        default=1.0,
        help="Weight applied to self_cluster_recon_loss",
    )
    parser.add_argument(
        "--parent_cluster_recon_weight",
        type=float,
        default=1.0,
        help="Weight applied to parent_cluster_recon_loss",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[512, 512, 512],
        help="hidden sizes of every layer",
    )
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")

    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument(
        "--reconstruction_layers",
        type=int,
        default=2,
        help="Number of LLaMA-style decoder layers for reconstruction",
    )
    parser.add_argument(
        "--reconstruction_heads",
        type=int,
        default=4,
        help="Number of attention heads in reconstruction decoder",
    )
    parser.add_argument(
        "--parent_recon_count",
        type=int,
        default=5,
        help="Number of parent cluster embeddings to reconstruct",
    )
    parser.add_argument(
        "--reconstruction_dropout",
        type=float,
        default=0.0,
        help="Dropout applied within reconstruction decoder",
    )
    parser.add_argument("--save_checkpoint_steps", default=10, type=int)
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of entities reserved for validation (set 0 to disable)",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=42,
        help="Random seed used when splitting train/validation entities",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1,
        help="Run validation every N steps when validation data is available",
    )
    parser.add_argument(
        "--run_name", type=str, default="Codebook", help="use bn or not"
    )
    parser.add_argument("--run_label", type=str, default="", help="use bn or not")
    parser.add_argument(
        "--entropy_sample_count",
        type=int,
        default=10,
        help="How many entropy points to keep for later analysis",
    )

    args = parser.parse_args()
    args.data_path = f"{args.data_path}/{args.dataset}"
    return args


def split_train_valid_indices(num_items, validation_split, seed):
    if num_items == 0:
        return [], []
    if validation_split <= 0.0:
        return list(range(num_items)), []
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be in the range (0, 1)")
    val_size = int(num_items * validation_split)
    if val_size == 0:
        val_size = 1
    if val_size >= num_items:
        val_size = num_items - 1
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(num_items, generator=generator).tolist()
    val_indices = sorted(permutation[:val_size])
    train_indices = sorted(permutation[val_size:])
    return train_indices, val_indices


def run_validation(model, dataloader, args):
    """
    Evaluate codebook usage and entropy across dataset.
    Uses model.get_indices(xs), which returns [B, num_layers].
    """
    model.eval()
    all_indices = []

    with torch.no_grad():
        for batch in dataloader:
            ids, cluster_ids, neighbor_ids, parent_ids = batch
            entity_embeddings = model.entity_embeddings[ids]
            indices = model.get_indices(entity_embeddings)
            all_indices.append(indices.cpu())

    all_indices = torch.cat(all_indices, dim=0)  # [N, num_layers]
    num_layers = all_indices.size(1)
    codebook_size = getattr(model, "codebook_size", args.codebook_size)

    stats = {}
    usage_list, entropy_list = [], []

    for layer in range(num_layers):
        idx = all_indices[:, layer]
        counts = torch.bincount(idx, minlength=codebook_size)
        freq = counts.float() / counts.sum()
        num_unique = idx.unique().numel()
        usage = (counts > 0).float().mean().item()
        nonzero_freq = freq[freq > 0]
        entropy = -(nonzero_freq * torch.log(nonzero_freq)).sum().item()
        usage_list.append(usage)
        entropy_list.append(entropy)
        logging.info(
            f"Layer {layer}: "
            f"tokens={num_unique}, "
            f"usage={usage * 100:.2f}%, "
            f"entropy={entropy:.4f}"
        )

    mean_usage = sum(usage_list) / len(usage_list)
    mean_entropy = sum(entropy_list) / len(entropy_list)
    stats["score"] = mean_entropy
    logging.info(
        f"[Codebook Summary] mean_usage={mean_usage * 100:.2f}%, "
        f"→ mean_entropy={mean_entropy:.4f}"
    )

    model.train()
    return stats


args = construct_args()
args.validation_seed = 42  # keep validation split deterministic
seed_everything(42)


def main(args):
    # if args.init_checkpoint:
    #     override_config(args)
    os.makedirs(args.save_path, exist_ok=True)
    set_logger(args)

    (
        entity2id,
        id2entity,
        relation2id,
    ) = read_data_from_args(args).values()

    entity_embeddings = read_embedding_from_path(args.entity_embeddings_path, args.cuda)
    cluster_embeddings = read_embedding_from_path(
        args.cluster_embeddings_path, args.cuda
    )
    entity_info = read_entity_info(args.entity_info_path, id2entity)
    train_indices, val_indices = split_train_valid_indices(
        len(entity_info), args.validation_split, args.validation_seed
    )
    if len(train_indices) == 0:
        raise ValueError("Training split is empty; decrease validation_split.")

    if len(val_indices) == 0:
        raise ValueError("Valid split is empty; increase validation_split.")
    model = RQVAE(
        entity_embeddings=entity_embeddings,
        cluster_embeddings=cluster_embeddings,
        codebook_size=args.codebook_size,
        codebook_num=args.codebook_num,
        subspace_num=args.subspace_num,
        subspace_size=args.subspace_size,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.layers,
        dropout_prob=args.dropout_prob,
        commit_loss_weight=args.commit_loss_weight,
        bn=args.bn,
        cuda=args.cuda,
        reconstruction_layers=args.reconstruction_layers,
        reconstruction_heads=args.reconstruction_heads,
        parent_recon_count=args.parent_recon_count,
        reconstruction_dropout=args.reconstruction_dropout,
        enable_self_cluster_loss=args.lambda_1 > 0,
        enable_neighbor_loss=args.lambda_2 > 0,
        enable_self_cluster_recon=args.self_cluster_recon_weight > 0,
        enable_parent_cluster_recon=args.parent_cluster_recon_weight > 0,
    )
    train_dataset = EmbDataset(entity_info, indices=train_indices)
    valid_dataset = EmbDataset(entity_info, indices=val_indices)
    logging.info("Model Parameter Configuration:")
    for name, param in model.named_parameters():
        logging.info(
            "Parameter %s: %s, require_grad = %s"
            % (name, str(param.size()), str(param.requires_grad))
        )
    logging.info("train_dataset len %s" % (len(train_dataset),))
    if valid_dataset is not None:
        logging.info("valid_dataset len %s" % (len(valid_dataset),))
    if args.cuda:
        model = model.cuda()

    # Set training dataloader iterator
    train_batch_size = args.batch_size
    if train_batch_size == 0:
        train_batch_size = len(train_dataset)
        if train_batch_size == 0:
            raise ValueError("Training dataset is empty.")
        logging.info(
            "batch_size set to 0; using full train_dataset size = %d", train_batch_size
        )
    args.max_steps = len(train_dataset) // train_batch_size * args.max_steps
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num),
        collate_fn=EmbDataset.collate_fn,
    )

    eval_batch_size = len(valid_dataset)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=max(1, args.cpu_num),
        collate_fn=EmbDataset.collate_fn,
    )

    # Set training configuration
    current_learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=current_learning_rate
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
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        current_learning_rate = checkpoint["current_learning_rate"]
        warm_up_steps = checkpoint["warm_up_steps"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    else:
        init_step = 0

    step = init_step
    logging.info("init_step = %d" % init_step)
    logging.info("hidden_dim = %d" % args.hidden_dim)
    logging.info("learning_rate = %lf" % current_learning_rate)

    training_logs = []
    train_iterator = OneShotIterator(train_dataloader)
    should_run_periodic_eval = valid_dataloader is not None and args.eval_steps > 0
    best_val_score = float("-inf")
    best_step = None
    best_epoch = None
    entropy_records = []
    entropy_history_path = os.path.join(args.save_path, "entropy_history.jsonl")
    entropy_snapshot_dir = os.path.join(args.save_path, "entropy_snapshots")
    if os.path.exists(entropy_history_path):
        os.remove(entropy_history_path)

    def save_entropy_snapshot(val_metrics, step_idx):
        """
        Persist quantized entities for this entropy so we can later sample diverse points.
        """
        entropy = val_metrics.get("score")
        if entropy is None:
            return
        os.makedirs(entropy_snapshot_dir, exist_ok=True)
        snapshot_dir = os.path.join(
            entropy_snapshot_dir, f"step_{step_idx}_entropy_{entropy:.4f}"
        )
        os.makedirs(snapshot_dir, exist_ok=True)
        entity_indices = model.get_indices(entity_embeddings)
        entity_indices_list = entity_indices.detach().cpu().tolist()
        entity_path = os.path.join(snapshot_dir, "entity_quantized.json")
        with open(entity_path, "w") as f:
            json.dump(entity_indices_list, f)
        unique_tokens = set()
        for entry in entity_indices_list:
            for code in entry:
                unique_tokens.add(quantized_to_token(int(code)))
        unique_tokens = sorted(unique_tokens)
        token_path = os.path.join(snapshot_dir, "tokens.json")
        with open(token_path, "w") as f:
            json.dump(unique_tokens, f, indent=2)
        record = {
            "step": step_idx,
            "entropy": entropy,
            "snapshot_dir": snapshot_dir,
            "entity_path": entity_path,
            "tokens_path": token_path,
        }
        entropy_records.append(record)
        with open(entropy_history_path, "a") as log_f:
            log_f.write(json.dumps(record) + "\n")
        logging.info(
            "Saved entropy snapshot at step %d (mean_entropy=%.4f) to %s",
            step_idx,
            entropy,
            snapshot_dir,
        )

    def persist_best(val_metrics, step_idx):
        nonlocal best_val_score, best_step, best_epoch
        if not val_metrics:
            return
        score = val_metrics.get("score")
        if score is None or score <= best_val_score:
            return
        best_val_score = score
        best_step = step_idx
        best_epoch = step_idx + 1
        best_payload = {
            "best_epoch": best_epoch,
            "best_step": best_step,
            "best_score": best_val_score,
            "metrics": val_metrics,
        }
        best_path = os.path.join(args.save_path, "best_validation.json")
        with open(best_path, "w") as best_file:
            json.dump(best_payload, best_file, indent=2)
        entity_indices = model.get_indices(entity_embeddings)
        entity_indices_list = entity_indices.detach().cpu().tolist()
        save_path = os.path.join(args.save_path, "entity_quantized.json")
        with open(save_path, "w") as f:
            json.dump(entity_indices_list, f)
        logging.info(f"save at {save_path}")
        unique_tokens = set()
        for entry in entity_indices_list:
            for code in entry:
                tok = quantized_to_token(int(code))
                unique_tokens.add(tok)
        unique_tokens = sorted(unique_tokens)
        token_save_path = os.path.join(args.save_path, "tokens.json")
        with open(token_save_path, "w") as f:
            json.dump(unique_tokens, f, indent=2)
        logging.info(f"Token vocab saved to {token_save_path}")
        logging.info(
            "New best validation mean_entropy %.4f at epoch %d",
            best_val_score,
            best_epoch,
        )

    # Training Loop
    for step in range(init_step, args.max_steps):
        step_start = time.perf_counter()
        log = model.train_step(optimizer, train_iterator, args)
        log["step_time"] = time.perf_counter() - step_start
        training_logs.append(log)
        if step >= warm_up_steps:
            current_learning_rate = current_learning_rate / 10
            logging.info(
                "Change learning_rate to %f at step %d" % (current_learning_rate, step)
            )
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=current_learning_rate,
            )
            warm_up_steps = warm_up_steps * 3

        if step % args.save_checkpoint_steps == 0:
            save_variable_list = {
                "step": step,
                "current_learning_rate": current_learning_rate,
                "warm_up_steps": warm_up_steps,
            }
            model.save_model(model, optimizer, save_variable_list, args)
        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs]) / len(
                    training_logs
                )
            log_metrics("Training average", step, metrics)
            training_logs = []
        if (
            step * 5 >= args.max_steps
            and should_run_periodic_eval
            and (step + 1) % args.eval_steps == 0
        ):
            val_metrics = run_validation(model, valid_dataloader, args)
            if val_metrics:
                log_metrics("Validation average", step, val_metrics)
                save_entropy_snapshot(val_metrics, step)
                persist_best(val_metrics, step)

    final_val_metrics = run_validation(model, valid_dataloader, args)
    if final_val_metrics:
        log_metrics("Validation final", step, final_val_metrics)
        save_entropy_snapshot(final_val_metrics, step)
        persist_best(final_val_metrics, step)

    def select_entropy_quantiles(records, sample_size):
        if not records or sample_size <= 0:
            return []
        unique_entropy_to_idx = {}
        for idx, rec in enumerate(records):
            entropy_val = rec["entropy"]
            if entropy_val not in unique_entropy_to_idx:
                unique_entropy_to_idx[entropy_val] = idx
        entropies = np.array(list(unique_entropy_to_idx.keys()))
        # Targets from min to max with large spacing
        targets = np.linspace(0.0, 1.0, num=min(sample_size, len(entropies)))
        selected = []
        used_entropy_values = set()
        for q in targets:
            target_entropy = float(np.quantile(entropies, q))
            best_entropy = None
            best_dist = None
            for entropy_val in entropies:
                if entropy_val in used_entropy_values:
                    continue
                dist = abs(entropy_val - target_entropy)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_entropy = entropy_val
            if best_entropy is not None:
                used_entropy_values.add(best_entropy)
                selected.append(records[unique_entropy_to_idx[best_entropy]])
        return selected

    selected_entropy_records = select_entropy_quantiles(
        entropy_records, args.entropy_sample_count
    )
    selection_path = os.path.join(args.save_path, "entropy_samples.json")
    with open(selection_path, "w") as f:
        payload = {
            "sample_count": len(selected_entropy_records),
            "requested": args.entropy_sample_count,
            "records": selected_entropy_records,
        }
        json.dump(payload, f, indent=2)
    logging.info(
        "Saved %d entropy samples (requested %d) to %s",
        len(selected_entropy_records),
        args.entropy_sample_count,
        selection_path,
    )

    if best_epoch is not None:
        logging.info(
            "Best validation mean_entropy %.4f achieved at epoch %d",
            best_val_score,
            best_epoch,
        )


if __name__ == "__main__":
    args = construct_args()
    main(args)
