import json
import random
import re
import string
from typing import Any, Dict
import numpy as np
import os
import torch
import logging
from tqdm import tqdm


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_hierarchy(args):
    """
    Read the hierarchy of the dataset from the json file.
    """
    file_path = (
        f"{args.process_path}/{args.dataset}/{args.hierarchy_type}_hierarchy.json"
    )
    hierarchy = json.load(open(file_path, "r"))

    return hierarchy


def read_entity_initial_embedding(args):
    """Read the initial entity embeddings from the json file."""
    file_path = f"{args.process_path}/{args.dataset}/entity_init_embeddings.npy"
    entity_init_embeddings = np.load(file_path)
    # convert to tensor
    entity_init_embeddings = torch.tensor(entity_init_embeddings, dtype=torch.float32)
    if args.cuda:
        entity_init_embeddings = entity_init_embeddings.cuda()
    return entity_init_embeddings


def read_cluster_embeddings(args):
    """Read the cluster embeddings from the json file."""
    file_path = f"{args.process_path}/{args.dataset}/clusters_embeddings_{args.hierarchy_type}.npy"
    cluster_embeddings = np.load(file_path)
    # convert to tensor
    cluster_embeddings = torch.tensor(cluster_embeddings, dtype=torch.float32)
    if args.cuda:
        cluster_embeddings = cluster_embeddings.cuda()
    return cluster_embeddings


def read_embedding_from_path(path, cuda=True):
    """Read the embeddings from the  file."""
    entity_init_embeddings = np.load(path)
    # convert to tensor
    entity_init_embeddings = torch.tensor(entity_init_embeddings, dtype=torch.float32)
    if cuda:
        entity_init_embeddings = entity_init_embeddings.cuda()
    return entity_init_embeddings


def read_triple(file_path, entity2id, relation2id):
    """
    Read triples and map them into ids.
    """
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split("\t")
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def read_triples_info(file_path, triples, id2entity):
    """
    Read triples information from the file.
    """
    entity_info = []
    with open(file_path, "r") as f:
        info = json.load(f)

    for triple in triples:
        head = id2entity[triple[0]]
        tail = id2entity[triple[2]]
        cluster_id_head = info[head]["cluster"]
        neighbor_clusters_ids_head = info[head]["nearest_clusters_lca"]
        parent_ids_head = info[head]["parent_path"]
        cluster_id_tail = info[tail]["cluster"]
        neighbor_clusters_ids_tail = info[tail]["nearest_clusters_lca"]
        parent_ids_tail = info[tail]["parent_path"]
        # link graph neighbors
        if "k_hop_neighbors" in info[head]:
            k_hop_neighbors_head = info[head]["k_hop_neighbors"]
        else:
            k_hop_neighbors_head = []
        if "k_hop_neighbors" in info[tail]:
            k_hop_neighbors_tail = info[tail]["k_hop_neighbors"]
        else:
            k_hop_neighbors_tail = []

        entity_info.append(
            (
                cluster_id_head,
                neighbor_clusters_ids_head,
                parent_ids_head,
                cluster_id_tail,
                neighbor_clusters_ids_tail,
                parent_ids_tail,
                k_hop_neighbors_head,
                k_hop_neighbors_tail,
            )
        )

    return entity_info


def log_metrics(mode, step, metrics):
    logging.info(f"{mode} step {step}: {metrics}")


def read_entity_info(file_path, entities):
    """
    Read entity information from the file.
    """
    entity_info = []
    with open(file_path, "r") as f:
        info = json.load(f)

    for id in entities.values():
        cluster_id = info[id]["cluster"]
        neighbor_clusters_ids = info[id]["nearest_clusters_lca"]
        parent_ids = info[id]["parent_path"]
        entity_info.append((cluster_id, neighbor_clusters_ids, parent_ids))

    return entity_info


def read_entity_info_dict(file_path, triples, id2entity):
    """
    Read entity information from the file.
    """
    entity_info = {}
    with open(file_path, "r") as f:
        info = json.load(f)

    for triple in tqdm(triples):
        head = id2entity[triple[0]]
        tail = id2entity[triple[2]]
        cluster_id_head = info[head]["cluster"]
        neighbor_clusters_ids_head = info[head]["nearest_clusters_lca"]
        parent_ids_head = info[head]["parent_path"]
        cluster_id_tail = info[tail]["cluster"]
        neighbor_clusters_ids_tail = info[tail]["nearest_clusters_lca"]
        parent_ids_tail = info[tail]["parent_path"]

        entity_info[triple[0]] = (
            cluster_id_head,
            neighbor_clusters_ids_head,
            parent_ids_head,
        )
        entity_info[triple[2]] = (
            cluster_id_tail,
            neighbor_clusters_ids_tail,
            parent_ids_tail,
        )

    return entity_info


def set_logger(args):
    """
    Write logs to checkpoint and console
    """

    if getattr(args, "do_train", True):
        log_file = os.path.join(args.save_path, "train.log")
    else:
        log_file = os.path.join(args.save_path, "test.log")

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


BASE_CHARS = string.ascii_lowercase
BASE = len(BASE_CHARS)


def int_to_str_code(x, base_chars=BASE_CHARS):
    if x < 0:
        raise ValueError("x must be non-negative")
    s = ""
    base = len(base_chars)
    if x == 0:
        return base_chars[0]
    while x > 0:
        x, rem = divmod(x, base)
        s = base_chars[rem] + s
    return s


def quantized_to_token(quantized, token_prefix="<#", token_suffix=">", token_num=32):
    if isinstance(quantized, (int, float)):
        quantized = int(quantized)
        return f"{token_prefix}{int_to_str_code(quantized)}{token_suffix}"
    elif isinstance(quantized, (list, tuple)):
        return "".join(
            [
                f"{token_prefix}{int_to_str_code(int(x))}{token_suffix}"
                for x in quantized[:token_num]
            ]
        )
    else:
        raise TypeError(f"Unsupported type: {type(quantized)}")


def str_code_to_int(s, base_chars=BASE_CHARS):
    base = len(base_chars)
    x = 0
    for ch in s:
        x = x * base + base_chars.index(ch)
    return x


def token_to_quantized(token_str, token_prefix="<#", token_suffix=">"):
    pattern = re.escape(token_prefix) + r"([A-Za-z0-9]+)" + re.escape(token_suffix)
    codes = re.findall(pattern, token_str)
    if not codes:
        raise ValueError(f"No valid tokens found in: {token_str}")
    values = [str_code_to_int(code) for code in codes]
    return values[0] if len(values) == 1 else values


def load_kg_hits(data_source: str) -> Dict[str, Any]:
    with open(data_source, "r", encoding="utf-8") as f:
        data = json.load(f)
    required_keys = ["h", "r", "t", "hits_head", "hits_tail"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key in JSON: {key}")
    return data


def load_relation_name(args) -> dict:
    """
    读取 relation.dict / relation2name.txt，返回 id → name 映射。

    relation.dict 示例:
        0   /organization/organization/headquarters./location/mailing_address/state_province_region
    relation2name.txt 示例:
        /time/event/locations   Connects events to their respective geographic locations where they occur.
    """
    dict_path = os.path.join(args.data_path, "relations.dict")
    id2rel = {}
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                rid, rtext = line.split("\t", 1)
            else:
                rid, rtext = line.split(" ", 1)
            id2rel[int(rid)] = rtext.strip()

    name_path = os.path.join(args.data_path, "relation2name.txt")
    rel2name = {}
    if os.path.exists(name_path):
        with open(name_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    rel, name = line.split("\t", 1)
                else:
                    rel, name = line.split(" ", 1)
                rel2name[rel.strip()] = name.strip()

    id2name = {}
    for rid, rel in id2rel.items():
        if rel in rel2name:
            id2name[rid] = rel2name[rel]
        else:
            id2name[rid] = rel.split("/")[-1].replace(".", "")
    return id2name


if __name__ == "__main__":
    x = [123, 456, 789]
    tokens = quantized_to_token(x)
    recovered = token_to_quantized(tokens)
    print(tokens)  # "<#...> <#...> <#...>"
    print(recovered)  # [123, 456, 789]
