import json
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.dataloader import TestDataset


class KGEmbeddingModel(nn.Module):
    def __init__(
        self,
        model,
        nentity,
        nrelation,
        hidden_dim,
        gamma,
        entity_text_embeddings=None,
        cluster_embeddings=None,
        rho=0.4,
        distance_metric="cosine",
    ):
        super(KGEmbeddingModel, self).__init__()
        self.model_name = model
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.distance_metric = distance_metric

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False,
        )
        self.entity_dim = hidden_dim
        if model == "RotatE" or model == "ComplEx":
            self.entity_dim = 2 * hidden_dim
        self.relation_dim = hidden_dim
        if model == "ComplEx":
            self.relation_dim = 2 * hidden_dim

        # Initialize relation embeddings (Equation 7)
        self.relation_embedding = nn.Parameter(
            torch.zeros(nrelation, self.relation_dim)
        )
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )
        print(f"Size of random relation_embedding: {self.relation_embedding.size()}")

        # Initialize randomly initialized component of entity embeddings
        self.entity_embedding_init = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding_init,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )
        print(
            f"Size of random entity_embedding_init: {self.entity_embedding_init.size()}"
        )

        ent_text_emb, ent_desc_emb = torch.chunk(entity_text_embeddings, 2, dim=1)
        clus_text_emb, clus_desc_emb = torch.chunk(cluster_embeddings, 2, dim=1)

        # concatenate ent_text_emb[:self.entity_dim/2] and ent_desc_emb[:self.entity_dim/2], size: (nentity, self.entity_dim)
        self.entity_text_embeddings = torch.cat(
            [
                ent_text_emb[:, : self.entity_dim // 2],
                ent_desc_emb[:, : self.entity_dim // 2],
            ],
            dim=1,
        )
        self.entity_text_embeddings.requires_grad = False
        print(f"Size of entity_text_embeddings: {self.entity_text_embeddings.size()}")
        # concatenate clus_text_emb[:self.entity_dim/2] and clus_desc_emb[:self.entity_dim/2], size: (nentity, self.entity_dim)
        self.cluster_embeddings = torch.cat(
            [
                clus_text_emb[:, : self.entity_dim // 2],
                clus_desc_emb[:, : self.entity_dim // 2],
            ],
            dim=1,
        )
        self.cluster_embeddings.requires_grad = False
        print(f"Size of cluster_embeddings: {self.cluster_embeddings.size()}")

        if model == "pRotatE":
            self.modulus = nn.Parameter(
                torch.Tensor([[0.5 * self.embedding_range.item()]])
            )

        if model not in ["TransE", "DistMult", "ComplEx", "RotatE", "pRotatE"]:
            raise ValueError("model %s not supported" % model)

        # Hyperparameters
        self.rho = rho  # Hyperparameter controlling the influence of the randomly initialized component in the embedding

    def score_func(self, head, relation, tail, mode="single"):
        """
        Compute the score for the given triple (head, relation, tail).
        """
        model_func = {
            "TransE": self.TransE,
            "DistMult": self.DistMult,
            "ComplEx": self.ComplEx,
            "RotatE": self.RotatE,
            "pRotatE": self.pRotatE,
            "HAKE": self.HAKE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError("model %s not supported" % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        """
        Compute the score using the TransE model.
        """
        if mode == "head-batch":
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        """
        Compute the score using the DistMult model.
        """
        if mode == "head-batch":
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        """
        Compute the score using the ComplEx model.
        """
        head_re, head_im = torch.chunk(head, 2, dim=2)
        relation_re, relation_im = torch.chunk(relation, 2, dim=2)
        tail_re, tail_im = torch.chunk(tail, 2, dim=2)

        if mode == "head-batch":
            re_score = relation_re * tail_re + relation_im * tail_im
            im_score = relation_re * tail_im - relation_im * tail_re
            score = head_re * re_score + head_im * im_score
        else:
            re_score = head_re * relation_re - head_im * relation_im
            im_score = head_re * relation_im + head_im * relation_re
            score = re_score * tail_re + im_score * tail_im

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        """
        Compute the score using the RotatE model.
        """
        pi = 3.14159265358979323846

        head_re, head_im = torch.chunk(head, 2, dim=2)
        tail_re, tail_im = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        relation_re = torch.cos(phase_relation)
        relation_im = torch.sin(phase_relation)

        if mode == "head-batch":
            re_score = relation_re * tail_re + relation_im * tail_im
            im_score = relation_re * tail_im - relation_im * tail_re
            re_score = re_score - head_re
            im_score = im_score - head_im
        else:
            re_score = head_re * relation_re - head_im * relation_im
            im_score = head_re * relation_im + head_im * relation_re
            re_score = re_score - tail_re
            im_score = im_score - tail_im

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        """
        Compute the score using the pRotatE model.
        """
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == "head-batch":
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    def HAKE(self, head, rel, tail, mode):
        """
        Compute the score using the HAKE model.
        """
        pi = 3.14159262358979323846

        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / pi)
        phase_relation = phase_relation / (self.embedding_range.item() / pi)
        phase_tail = phase_tail / (self.embedding_range.item() / pi)

        if mode == "head-batch":
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = bias_relation < -mod_relation
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (
            1 - bias_relation
        )

        phase_score = (
            torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        )
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)

    @staticmethod
    def get_masked_embeddings(indices, embeddings, dim_size):
        """
        Retrieves and applies a mask to embeddings based on provided indices.

        Args:
            indices (torch.Tensor): Tensor of indices with possible -1 indicating invalid entries.
            embeddings (torch.nn.Parameter): Embeddings from which to select.
            dim_size (tuple): The desired dimension sizes of the output tensor.

        Returns:
            torch.Tensor: Masked and selected embeddings based on valid indices.
        """
        valid_mask = indices != -1
        # Initialize tensor to hold the masked embeddings
        masked_embeddings = torch.zeros(
            *dim_size, dtype=embeddings.dtype, device=embeddings.device
        )
        # Apply mask to filter valid indices
        valid_indices = indices[valid_mask]
        selected_embeddings = torch.index_select(embeddings, dim=0, index=valid_indices)
        # Place selected embeddings back into the appropriate locations
        masked_embeddings.view(-1, embeddings.shape[1])[valid_mask.view(-1)] = (
            selected_embeddings
        )
        return masked_embeddings

    def get_entity_embedding(self):
        """
        Retrieve the embedding for the given entity ID.
        """
        return (
            self.rho * self.entity_embedding_init
            + (1 - self.rho) * self.entity_text_embeddings
        )

    def forward(self, sample, mode="single"):
        if mode == "single":
            relation = torch.index_select(
                self.relation_embedding, dim=0, index=sample[:, 1]
            ).unsqueeze(1)
            head_init = torch.index_select(
                self.entity_embedding_init, dim=0, index=sample[:, 0]
            ).unsqueeze(1)
            tail_init = torch.index_select(
                self.entity_embedding_init, dim=0, index=sample[:, 2]
            ).unsqueeze(1)
            head_text = torch.index_select(
                self.entity_text_embeddings, dim=0, index=sample[:, 0]
            ).unsqueeze(1)
            tail_text = torch.index_select(
                self.entity_text_embeddings, dim=0, index=sample[:, 2]
            ).unsqueeze(1)
            head_combined = self.rho * head_init + (1 - self.rho) * head_text
            tail_combined = self.rho * tail_init + (1 - self.rho) * tail_text
            link_pred_score = self.score_func(
                head_combined, relation, tail_combined, mode
            )
            return link_pred_score

        elif mode == "head-batch":
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            relation = torch.index_select(
                self.relation_embedding, dim=0, index=tail_part[:, 1]
            ).unsqueeze(1)
            tail_init = torch.index_select(
                self.entity_embedding_init, dim=0, index=tail_part[:, 2]
            ).unsqueeze(1)
            head_init = torch.index_select(
                self.entity_embedding_init, dim=0, index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            tail_text = torch.index_select(
                self.entity_text_embeddings, dim=0, index=tail_part[:, 2]
            ).unsqueeze(1)
            head_text = torch.index_select(
                self.entity_text_embeddings, dim=0, index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            tail_combined = self.rho * tail_init + (1 - self.rho) * tail_text
            head_combined = self.rho * head_init + (1 - self.rho) * head_text
            link_pred_score = self.score_func(
                head_combined, relation, tail_combined, mode
            )

        elif mode == "tail-batch":
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            relation = torch.index_select(
                self.relation_embedding, dim=0, index=head_part[:, 1]
            ).unsqueeze(1)
            head_init = torch.index_select(
                self.entity_embedding_init, dim=0, index=head_part[:, 0]
            ).unsqueeze(1)
            tail_init = torch.index_select(
                self.entity_embedding_init, dim=0, index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            head_text = torch.index_select(
                self.entity_text_embeddings, dim=0, index=head_part[:, 0]
            ).unsqueeze(1)
            tail_text = torch.index_select(
                self.entity_text_embeddings, dim=0, index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            head_combined = self.rho * head_init + (1 - self.rho) * head_text
            tail_combined = self.rho * tail_init + (1 - self.rho) * tail_text
            link_pred_score = self.score_func(
                head_combined, relation, tail_combined, mode
            )

        else:
            raise ValueError("mode %s not supported" % mode)

        return link_pred_score

    def rotate_distance(self, embeddings1, embeddings2):
        pi = 3.14159262358979323846

        phase1, mod1 = torch.chunk(embeddings1, 2, dim=-1)
        phase2, mod2 = torch.chunk(embeddings2, 2, dim=-1)

        phase1 = phase1 / (self.embedding_range.item() / pi)
        phase2 = phase2 / (self.embedding_range.item() / pi)

        phase_diff = torch.abs(torch.sin((phase1 - phase2) / 2))
        # return torch.mean(phase_diff, dim=-1)
        return torch.sum(phase_diff, dim=-1)

    def distance(self, embeddings1, embeddings2):
        """
        Compute the distance between two sets of embeddings.
        """
        if self.distance_metric == "euclidean":
            return torch.norm(embeddings1 - embeddings2, p=2, dim=-1)
        elif self.distance_metric == "manhattan":
            return torch.norm(embeddings1 - embeddings2, p=1, dim=-1)
        elif self.distance_metric == "cosine":
            embeddings1_norm = F.normalize(embeddings1, p=2, dim=-1)
            embeddings2_norm = F.normalize(embeddings2, p=2, dim=-1)
            cosine_similarity = torch.sum(embeddings1_norm * embeddings2_norm, dim=-1)
            cosine_distance = 1 - cosine_similarity
            return cosine_distance
        elif self.distance_metric == "rotate":
            return self.rotate_distance(embeddings1, embeddings2)
        elif self.distance_metric == "pi":
            pi = 3.14159262358979323846
            phase1 = embeddings1 / (self.embedding_range.item() / pi)
            phase2 = embeddings2 / (self.embedding_range.item() / pi)
            distance = torch.abs(torch.sin((phase1 - phase2) / 2))
            return 1 - torch.mean(distance, dim=-1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        """
        A single train step. Apply back-propation and return the loss
        """

        model.train()

        optimizer.zero_grad()

        (
            positive_sample,
            negative_sample,
            subsampling_weight,
            mode,
        ) = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model(
            (positive_sample, negative_sample),
            mode=mode,
        )

        # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        negative_score = (
            F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
            * F.logsigmoid(-negative_score)
        ).sum(dim=1)

        positive_score = model(
            positive_sample,
            mode="single",
        )

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
        positive_sample_loss = (
            -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        )
        negative_sample_loss = (
            -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        )

        ## Loss function
        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        log = {
            "positive_sample_loss": positive_sample_loss.item(),
            "negative_sample_loss": negative_sample_loss.item(),
            "loss": loss.item(),
        }
        loss_details = {}
        log.update(loss_details)
        return log

    @staticmethod
    def test_step(
        model, test_triples, all_true_triples, entity_info, args, save_hit_k=False
    ):
        model.eval()

        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                entity_info,
                "head-batch",
                rerank=True if args.rerank == "true" else False,
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                entity_info,
                "tail-batch",
                rerank=True if args.rerank == "true" else False,
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        logs_left, logs_right = [], []
        hit_records = {}

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        K = int(getattr(args, "save_hit_k", 0))
        save_hit_k &= K > 0

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for (
                    positive_sample,
                    negative_sample,
                    filter_bias,
                    mode,
                ) in tqdm(test_dataset):
                    # cuda setup
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    link_pred_score = model(
                        (positive_sample, negative_sample),
                        mode,
                    )

                    score = link_pred_score + filter_bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    positive_arg = (
                        positive_sample[:, 0]
                        if mode == "head-batch"
                        else positive_sample[:, 2]
                    )

                    if save_hit_k:
                        topk_entities = argsort[:, :K]

                    # === Evaluate ===
                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1
                        ranking = 1 + ranking.item()
                        record = {
                            "MRR": 1.0 / ranking,
                            "MR": float(ranking),
                            "HITS@1": 1.0 if ranking <= 1 else 0.0,
                            "HITS@3": 1.0 if ranking <= 3 else 0.0,
                            "HITS@5": 1.0 if ranking <= 5 else 0.0,
                            "HITS@10": 1.0 if ranking <= 10 else 0.0,
                            f"HITS@{K}": 1.0 if ranking <= K else 0.0,
                        }

                        if mode == "head-batch":
                            logs_left.append(record)
                        else:
                            logs_right.append(record)

                        if save_hit_k:
                            key = f"{int(positive_sample[i, 0].item())}-{int(positive_sample[i, 1].item())}-{int(positive_sample[i, 2].item())}"
                            if key not in hit_records:
                                hit_records[key] = {
                                    "h": int(positive_sample[i, 0].item()),
                                    "r": int(positive_sample[i, 1].item()),
                                    "t": int(positive_sample[i, 2].item()),
                                    "hits_head": [],
                                    "hits_tail": [],
                                }

                            if mode == "head-batch":
                                hit_records[key]["hits_head"] = [
                                    int(x.item()) for x in topk_entities[i]
                                ]
                                hit_records[key]["rank_head"] = ranking
                            else:
                                hit_records[key]["hits_tail"] = [
                                    int(x.item()) for x in topk_entities[i]
                                ]
                                hit_records[key]["rank_tail"] = ranking

                    if step % args.test_log_steps == 0:
                        logging.info(
                            "Evaluating the model... (%d/%d)" % (step, total_steps)
                        )
                    step += 1

            # === Compute metrics ===
            def compute_metrics(logs):
                metrics = {}
                if not logs:
                    return metrics
                for metric in logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
                return metrics

            metrics_left = compute_metrics(logs_left)
            metrics_right = compute_metrics(logs_right)
            metrics_total = compute_metrics(logs_left + logs_right)

        if save_hit_k:
            return {
                "left": metrics_left,
                "right": metrics_right,
                "overall": metrics_total,
            }, list(hit_records.values())

        return {
            "left": metrics_left,
            "right": metrics_right,
            "overall": metrics_total,
        }

    @staticmethod
    def save_model(
        model,
        optimizer,
        save_variable_list,
        args,
        save_path=None,
        checkpoint_name="checkpoint",
    ):
        """
        Save the parameters of the model and the optimizer,
        as well as some other variables such as step and learning_rate
        """

        save_dir = save_path if save_path else args.save_path
        os.makedirs(save_dir, exist_ok=True)
        argparse_dict = vars(args)
        with open(os.path.join(save_dir, "config.json"), "w") as fjson:
            json.dump(argparse_dict, fjson)

        torch.save(
            {
                **save_variable_list,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(save_dir, checkpoint_name),
        )

        entity_embedding = model.get_entity_embedding().detach().cpu().numpy()
        np.save(os.path.join(save_dir, "entity_embedding"), entity_embedding)

        relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(os.path.join(save_dir, "relation_embedding"), relation_embedding)
