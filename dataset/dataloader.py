#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os

import numpy as np
import torch

from torch.utils.data import Dataset
from abc import ABC


class TrainDataset(ABC):
    def __init__(
        self, triples, nentity, nrelation, negative_sample_size, entity_info, mode
    ):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.entity_info = entity_info

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        # (cluster_id_head, neighbor_clusters_ids_head, parent_ids_head, cluster_id_tail, neighbor_clusters_ids_tail, parent_ids_tail)
        entity_info = self.entity_info[idx]

        head, relation, tail = positive_sample
        (
            cluster_id_head,
            neighbor_clusters_ids_head,
            parent_ids_head,
            cluster_id_tail,
            neighbor_clusters_ids_tail,
            parent_ids_tail,
            _,
            _,
        ) = entity_info

        subsampling_weight = (
            self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        )
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(
                self.nentity, size=self.negative_sample_size * 2
            )
            if self.mode == "head-batch":
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True,
                )
            elif self.mode == "tail-batch":
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True,
                )
            else:
                raise ValueError("Training batch mode %s not supported" % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[
            : self.negative_sample_size
        ]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return (
            positive_sample,
            negative_sample,
            subsampling_weight,
            cluster_id_head,
            neighbor_clusters_ids_head,
            parent_ids_head,
            cluster_id_tail,
            neighbor_clusters_ids_tail,
            parent_ids_tail,
            self.mode,
        )

    @staticmethod
    def collate_fn(data):
        # size of positive sample: (batch_size, 3)
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        # size of negative sample: (batch_size, negative_sample_size)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        # size of subsample_weight: (batch_size, 1)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)

        mode = data[0][9]
        return (
            positive_sample,
            negative_sample,
            subsample_weight,
            # cluster_id_head,
            # cluster_id_tail,
            # padded_neighbor_clusters_ids_head,
            # padded_neighbor_clusters_ids_tail,
            # padded_parent_ids_head,
            # padded_parent_ids_tail,
            mode,
        )

    @staticmethod
    def count_frequency(triples, start=4):
        """
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        """
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        """
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        """

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(
                list(set(true_head[(relation, tail)]))
            )
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(
                list(set(true_tail[(head, relation)]))
            )

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(
        self,
        triples,
        all_true_triples,
        nentity,
        nrelation,
        entity_info,
        mode,
        rerank=False,
        alpha=0.5,
    ):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.entity_info = entity_info
        self.rerank = rerank
        self.alpha = alpha

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        (
            cluster_id_head,
            neighbor_clusters_ids_head,
            parent_ids_head,
            cluster_id_tail,
            neighbor_clusters_ids_tail,
            parent_ids_tail,
            k_hop_neighbors_head,
            k_hop_neighbors_tail,
        ) = self.entity_info[idx]

        if self.mode == "head-batch":
            tmp = [
                (0, rand_head)
                if (rand_head, relation, tail) not in self.triple_set
                else (-1, head)
                for rand_head in range(self.nentity)
            ]
            tmp[head] = (0, head)
            tmp[tail] = (-1, tail)
        elif self.mode == "tail-batch":
            tmp = [
                (0, rand_tail)
                if (head, relation, rand_tail) not in self.triple_set
                else (-1, tail)
                for rand_tail in range(self.nentity)
            ]
            tmp[tail] = (0, tail)
            tmp[head] = (-1, head)
        else:
            raise ValueError("negative batch mode %s not supported" % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()

        if self.rerank:
            if self.mode == "tail-batch":
                k_hop_neighbors_head = torch.LongTensor(k_hop_neighbors_head)
                filter_bias[k_hop_neighbors_head] += self.alpha
            elif self.mode == "head-batch":
                k_hop_neighbors_tail = torch.LongTensor(k_hop_neighbors_tail)
                filter_bias[k_hop_neighbors_tail] += self.alpha

        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return (
            positive_sample,
            negative_sample,
            filter_bias,
            self.mode,
        )

    @staticmethod
    def collate_fn(data):
        # size of positive sample: (batch_size, 3)
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        # size of negative sample: (batch_size, nentity)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        # size of filter_bias: (batch_size, nentity)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return (
            positive_sample,
            negative_sample,
            filter_bias,
            mode,
        )


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        """
        Transform a PyTorch Dataloader into python iterator
        """
        while True:
            for data in dataloader:
                yield data


def read_data_from_args(args):
    with open(os.path.join(args.data_path, "entities.dict")) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split("\t")
            entity2id[entity] = int(eid)
        id2entity = {v: k for k, v in entity2id.items()}

    with open(os.path.join(args.data_path, "relations.dict")) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split("\t")
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info("Data Path: %s" % args.data_path)
    logging.info("#entity: %d" % nentity)
    logging.info("#relation: %d" % nrelation)

    return {
        "entity2id": entity2id,
        "id2entity": id2entity,
        "relation2id": relation2id,
    }


def build_data_args(parser):
    parser.add_argument(
        "--data_path", type=str, default="data", help="Path to the dataset"
    )
    parser.add_argument(
        "--process_path",
        type=str,
        default="processed_data",
        help="Path to the entity hierarchy",
    )
    parser.add_argument("--dataset", type=str, default="FB15K-237", help="Dataset name")
    parser.add_argument(
        "--hierarchy_type",
        type=str,
        default="seed",
        choices=["seed", "llm"],
        help="Type of hierarchy to use",
    )


class EmbDataset(ABC):
    def __init__(self, entity_info, indices=None):
        self.entity_info = entity_info
        if indices is None:
            self.indices = list(range(len(entity_info)))
        else:
            self.indices = list(indices)
        self.nentity = len(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        entity_idx = self.indices[idx]
        cluster_id, neighbor_clusters_ids, parent_ids = self.entity_info[entity_idx]
        return entity_idx, cluster_id, neighbor_clusters_ids, parent_ids

    @staticmethod
    def collate_fn(data):
        """
        data: list of tuples returned by __getitem__,
              each tuple: (idx, cluster_id, neighbor_clusters_ids, parent_ids)
        return:
          cluster_id,
          padded_neighbor_clusters_ids,
          padded_parent_ids
        """
        # 0: idx
        # 1: cluster_id
        # 2: neighbor_clusters_ids
        # 3: parent_ids
        cluster_id = torch.tensor([sample[1] for sample in data], dtype=torch.long)
        id = torch.tensor([sample[0] for sample in data], dtype=torch.long)

        # neighbor clusters padding
        neighbor_clusters_ids = [torch.LongTensor(sample[2]) for sample in data]
        max_len_neighbor = max(
            (ids.numel() for ids in neighbor_clusters_ids), default=0
        )
        padded_neighbor_clusters_ids = (
            torch.stack(
                [
                    torch.nn.functional.pad(
                        ids, (0, max_len_neighbor - ids.numel()), value=-1
                    )
                    for ids in neighbor_clusters_ids
                ]
            )
            if len(neighbor_clusters_ids) > 0
            else torch.empty((0, max_len_neighbor), dtype=torch.long)
        )

        # parent ids padding
        parent_ids = [torch.LongTensor(sample[3]) for sample in data]
        max_len_parent = max((ids.numel() for ids in parent_ids), default=0)
        padded_parent_ids = (
            torch.stack(
                [
                    torch.nn.functional.pad(
                        ids, (0, max_len_parent - ids.numel()), value=-1
                    )
                    for ids in parent_ids
                ]
            )
            if len(parent_ids) > 0
            else torch.empty((0, max_len_parent), dtype=torch.long)
        )

        return id, cluster_id, padded_neighbor_clusters_ids, padded_parent_ids


class OneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)

    def __next__(self):
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        """
        Transform a PyTorch Dataloader into python iterator
        """
        while True:
            for data in dataloader:
                yield data
