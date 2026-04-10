import argparse
import json
import logging
import os
import time
import torch
from dataset.dataloader import (
    build_data_args,
)
from utils import quantized_to_token, read_embedding_from_path, set_logger
from codebook.rqvae import RQVAE


def construct_args():
    parser = argparse.ArgumentParser(description="CodeBook")
    # Data paths
    build_data_args(parser)
    # Data paths
    parser.add_argument("--init_checkpoint", required=True, help="Use GPU for training")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for training")

    args = parser.parse_args()
    args.data_path = f"{args.data_path}/{args.dataset}"
    args.save_path = f"{args.process_path}/{args.dataset}/checkpoints/CodeBookGenerate/CodeBookGenerate_batch_{time.strftime('%Y%m%d%H%M%S')}"

    return args


args = construct_args()


def main(args):
    with open(os.path.join(args.init_checkpoint, "config.json"), "r") as fjson:
        argparse_dict = json.load(fjson)
    argparse_dict["save_path"] = args.save_path
    argparse_dict["init_checkpoint"] = args.init_checkpoint
    argparse_dict["cuda"] = args.cuda
    args = argparse.Namespace(**argparse_dict)
    os.makedirs(args.save_path, exist_ok=True)
    set_logger(args)
    entity_embeddings = read_embedding_from_path(args.entity_embeddings_path, args.cuda)
    cluster_embeddings = read_embedding_from_path(
        args.cluster_embeddings_path, args.cuda
    )
    model = RQVAE(
        entity_embeddings=entity_embeddings,
        cluster_embeddings=cluster_embeddings,
        codebook_size=args.codebook_size,
        codebook_num=args.codebook_num,
        subspace_num=getattr(args, "subspace_num", 1),
        hidden_dim=args.hidden_dim,
        encoder_layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        cuda=args.cuda,
        reconstruction_layers=getattr(args, "reconstruction_layers", 2),
        reconstruction_heads=getattr(args, "reconstruction_heads", 8),
        neighbor_recon_count=getattr(args, "neighbor_recon_count", 3),
        parent_recon_count=getattr(args, "parent_recon_count", 5),
        reconstruction_dropout=getattr(args, "reconstruction_dropout", 0.0),
    )
    logging.info("Model Parameter Configuration:")
    for name, param in model.named_parameters():
        logging.info(
            "Parameter %s: %s, require_grad = %s"
            % (name, str(param.size()), str(param.requires_grad))
        )

    if args.cuda:
        model = model.cuda()
    # Restore model from checkpoint directory
    logging.info("Loading checkpoint %s..." % args.init_checkpoint)
    logging.info("batch_size = %d" % args.batch_size)
    checkpoint = torch.load(os.path.join(args.init_checkpoint, "checkpoint"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
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


if __name__ == "__main__":
    args = construct_args()
    main(args)
