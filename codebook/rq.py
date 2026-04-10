import torch
import torch.nn as nn
from torch.nn import functional as F

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_num,
        codebook_size,
        hidden_dim,
        mu,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, hidden_dim, mu)
                for _ in range(codebook_num)
            ]
        )

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def vq_initialization(self, x):
        residual = x
        for quantizer in self.vq_layers:
            x_res = quantizer.vq_initialization(residual)
            residual = residual - x_res

    def inner_triplet_loss(self, triplets, features, margin=0.2):
        triplets = torch.tensor(triplets, dtype=torch.long)
        anchors = features[triplets[:, 0]]
        positives = features[triplets[:, 1]]
        negatives = features[triplets[:, 2]]
        pos_distances = F.pairwise_distance(anchors, positives, p=2)
        neg_distances = F.pairwise_distance(anchors, negatives, p=2)
        losses = F.relu(pos_distances - neg_distances + margin)
        return losses.mean()

    def forward(self, x):
        all_losses = []
        all_indices = []
        x_qs = []
        residual = x

        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual)
            residual = residual - x_res
            x_qs.append(x_res)
            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        x_qs = torch.stack(x_qs, dim=1)
        return x_qs, mean_losses, all_indices
