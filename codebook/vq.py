import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e,
        e_dim,
        mu=0.25,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.mu = mu
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def get_codebook(self):
        return self.embedding.weight

    def vq_initialization(self, x):
        latent = x.view(-1, self.e_dim)
        d = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )

        indices = torch.argmin(d, dim=-1)
        x_q = self.embedding(indices).view(x.shape)
        return x_q

    def forward(self, x):
        latent = x.view(-1, self.e_dim)
        d = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )
        indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.mu * commitment_loss
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices
