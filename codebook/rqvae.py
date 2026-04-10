import json
import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


def exponential_decay_sum(x, base_lambda, reverse=False):
    iterator = reversed(x) if reverse else x
    total = None
    for i, v in enumerate(iterator, start=1):
        weight = base_lambda**i
        if total is None:
            total = weight * v
        else:
            total = total + weight * v

    return total / len(x)


def contrastive_cluster_loss(token_embed, cluster_emb, temperature=0.1):
    token_embed = F.normalize(token_embed, dim=-1)
    cluster_emb = F.normalize(cluster_emb, dim=-1)
    logits = token_embed @ cluster_emb.T  # [B, B]
    logits = logits / temperature

    labels = torch.arange(token_embed.size(0), device=token_embed.device)
    loss = F.cross_entropy(logits, labels)
    return loss


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(keepdim=True, dim=-1, p=2)
        rms = norm * norm / x.size(-1)
        return x * torch.rsqrt(rms + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, max_position_embeddings=4096):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, :, None, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, :, None, :], persistent=False
        )

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len_cached:
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()[None, :, None, :]
            sin = emb.sin()[None, :, None, :]
        else:
            cos = self.cos_cached[:, :seq_len, :, :]
            sin = self.sin_cached[:, :seq_len, :, :]
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, multiple_of=256, dropout=0.0):
        super().__init__()
        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = multiple_of * ((inner_size + multiple_of - 1) // multiple_of)
        self.gate_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.down_proj = nn.Linear(inner_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class LlamaAttention(nn.Module):
    def __init__(
        self, hidden_size, num_heads, dropout=0.0, max_position_embeddings=4096
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, max_position_embeddings=max_position_embeddings
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)

        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size, num_heads, dropout=dropout)
        self.mlp = LlamaMLP(hidden_size, dropout=dropout)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(x, attention_mask=attention_mask)

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x


class LlamaDecoder(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(hidden_size, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)

    def _build_causal_mask(self, seq_len, device, dtype):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        attn_mask = self._build_causal_mask(seq_len, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask)
        return self.norm(x)


class RQVAE(nn.Module):
    def __init__(
        self,
        entity_embeddings,
        cluster_embeddings,
        codebook_size=1024,
        codebook_num=6,
        subspace_num=3,  # zeta 2
        subspace_size=1,  # zeta 2
        hidden_dim=64,
        encoder_layers=None,
        dropout_prob=0.0,
        bn=False,
        cuda=False,
        commit_loss_weight=0.25,
        reconstruction_layers=2,
        reconstruction_heads=8,
        parent_recon_count=5,
        reconstruction_dropout=0.0,
        enable_self_cluster_loss=True,
        enable_neighbor_loss=True,
        enable_self_cluster_recon=True,
        enable_parent_cluster_recon=True,
    ):
        super(RQVAE, self).__init__()
        self.entity_embeddings = entity_embeddings
        self.in_dim = entity_embeddings.shape[-1]
        self.codebook_size = codebook_size
        self.codebook_num = codebook_num
        self.subspace_num = subspace_num
        if self.subspace_num <= 0:
            raise ValueError("subspace_num must be a positive integer")

        # self.codebooks_per_subspace = self.codebook_num // self.subspace_num
        self.subspace_size = subspace_size
        self.normspace_size = self.codebook_num - self.subspace_num * self.subspace_size
        if self.normspace_size <= 0:
            raise ValueError("normspace_size <=0 ")
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.usecuda = cuda

        if hidden_dim % reconstruction_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by reconstruction_heads ({reconstruction_heads})"
            )

        if self.in_dim % self.hidden_dim != 0:
            raise ValueError(
                f"entity_embeddings.shape[-1] ({self.in_dim}) must be divisible by hidden_dim ({hidden_dim})"
            )
        self.reconstruction_layers = reconstruction_layers
        self.reconstruction_heads = reconstruction_heads
        self.self_cluster_count = 1
        self.self_recon_token_count = int(self.in_dim / self.hidden_dim)
        self.parent_recon_count = parent_recon_count
        self.reconstruction_dropout = reconstruction_dropout
        self.enable_self_cluster_loss = enable_self_cluster_loss
        self.enable_neighbor_loss = enable_neighbor_loss
        self.enable_self_cluster_recon = enable_self_cluster_recon
        self.enable_parent_cluster_recon = enable_parent_cluster_recon
        self.total_recon_tokens = (
            self.self_cluster_count
            + self.self_recon_token_count
            + self.parent_recon_count
        )

        self.encode_layer_dims = [self.in_dim] + self.encoder_layers + [self.hidden_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims, dropout=self.dropout_prob, bn=self.bn
        )

        self.rq = ResidualVectorQuantizer(
            codebook_num,
            codebook_size,
            hidden_dim,
            commit_loss_weight,
        )

        # cluster embeddings
        clus_text_emb, clus_desc_emb = torch.chunk(cluster_embeddings, 2, dim=1)
        self.cluster_embeddings = torch.cat(
            [
                clus_text_emb[:, : self.hidden_dim // 2],
                clus_desc_emb[:, : self.hidden_dim // 2],
            ],
            dim=1,
        )
        self.cluster_embeddings.requires_grad = False
        print(f"Size of cluster_embeddings: {self.cluster_embeddings.size()}")

        # entity embeddings
        self.entity_embeddings = entity_embeddings
        self.entity_embeddings.requires_grad = False

        # reconstruction decoder (LLaMA-style)
        self.special_token1 = nn.Parameter(
            torch.randn(1, self.hidden_dim), requires_grad=False
        )
        self.special_token2 = nn.Parameter(torch.randn(1, self.hidden_dim))
        self.reconstruction_decoder = LlamaDecoder(
            hidden_size=self.hidden_dim,
            num_heads=self.reconstruction_heads,
            num_layers=self.reconstruction_layers,
            dropout=self.reconstruction_dropout,
        )

    def forward(self, ids, self_cluster_ids, neighbor_clusters_ids, parent_ids):
        if self.usecuda:
            ids = ids.cuda()
            self_cluster_ids = self_cluster_ids.cuda()
            neighbor_clusters_ids = neighbor_clusters_ids.cuda()
            parent_ids = parent_ids.cuda()

        x_raw_embeds = torch.index_select(self.entity_embeddings, dim=0, index=ids)
        x_embeds = self.encoder(x_raw_embeds)
        x_qs, commitment_loss, indices = self.rq(x_embeds)
        x_supspace_q_tokens = x_qs[:, self.normspace_size :]
        if self.enable_self_cluster_loss or self.enable_neighbor_loss:
            self_cluster_dist, neighbor_cluster_dist = self.cluster_forward(
                x_supspace_q_tokens,
                self_cluster_ids,
                neighbor_clusters_ids,
                compute_self=self.enable_self_cluster_loss,
                compute_neighbor=self.enable_neighbor_loss,
            )
        else:
            # Both cluster losses disabled
            self_cluster_dist, neighbor_cluster_dist = [], []
        (
            self_cluster_recon_loss,
            parent_cluster_recon_loss,
            self_recon_loss,
        ) = self.reconstruction_forward(
            x_qs,
            self_cluster_ids,
            parent_ids,
            x_raw_embeds,
            compute_self_cluster=self.enable_self_cluster_recon,
            compute_parent=self.enable_parent_cluster_recon,
        )
        return (
            self_cluster_dist,
            neighbor_cluster_dist,
            commitment_loss,
            self_cluster_recon_loss,
            parent_cluster_recon_loss,
            self_recon_loss,
        )

    def _truncate_or_pad_indices(self, indices, target_length):
        if target_length == 0:
            return indices[:, :0]
        current_length = indices.size(1)
        if current_length >= target_length:
            return indices[:, :target_length]
        pad_size = target_length - current_length
        padding = indices.new_full((indices.size(0), pad_size), -1)
        return torch.cat([indices, padding], dim=1)

    def _masked_mse(self, predictions, targets, mask):
        if mask.numel() == 0 or not mask.any():
            return predictions.new_tensor(0.0)
        mask = mask.float()
        per_token_loss = F.mse_loss(predictions, targets, reduction="none").mean(dim=-1)
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

    def reconstruction_forward(
        self,
        x_qs,
        self_cluster_ids,
        parent_ids,
        x_raw_embeds,
        compute_self_cluster=True,
        compute_parent=True,
    ):
        batch_size = x_qs.size(0)
        if batch_size == 0:
            zero = x_qs.new_tensor(0.0)
            return zero, zero, zero

        base_tokens = x_qs
        self_cluster_count = self.self_cluster_count if compute_self_cluster else 0
        parent_recon_count = self.parent_recon_count if compute_parent else 0
        decoder_inputs = [base_tokens]

        if self_cluster_count > 0:
            special_token1 = self.special_token1.unsqueeze(0).expand(
                batch_size, self_cluster_count, -1
            )
            decoder_inputs.append(special_token1)

        total_special2 = self.self_recon_token_count + parent_recon_count
        special_token2 = self.special_token2.unsqueeze(0).expand(
            batch_size, total_special2, -1
        )
        if total_special2 > 0:
            decoder_inputs.append(special_token2)

        decoder_input = torch.cat(decoder_inputs, dim=1)
        recon_tokens = self.reconstruction_decoder(decoder_input)

        offset = base_tokens.size(1)
        if self_cluster_count > 0:
            self_cluster_pred = recon_tokens[
                :, offset : offset + self_cluster_count, :
            ]
            offset += self_cluster_count
        else:
            self_cluster_pred = None

        self_pred_tokens = recon_tokens[
            :, offset : offset + self.self_recon_token_count, :
        ]
        offset += self.self_recon_token_count
        if parent_recon_count > 0:
            parent_pred = recon_tokens[:, offset : offset + parent_recon_count, :]
        else:
            parent_pred = None

        if self_cluster_count > 0:
            target_cluster = torch.index_select(
                self.cluster_embeddings, dim=0, index=self_cluster_ids
            ).unsqueeze(1)
            self_cluster_loss = F.mse_loss(
                self_cluster_pred, target_cluster, reduction="mean"
            )
        else:
            self_cluster_loss = x_qs.new_tensor(0.0)

        self_pred_flat = self_pred_tokens.reshape(batch_size, -1)
        target_raw = x_raw_embeds
        if self_pred_flat.size(1) > target_raw.size(1):
            self_pred_flat = self_pred_flat[:, : target_raw.size(1)]
        elif self_pred_flat.size(1) < target_raw.size(1):
            pad = self_pred_flat.new_zeros(
                batch_size, target_raw.size(1) - self_pred_flat.size(1)
            )
            self_pred_flat = torch.cat([self_pred_flat, pad], dim=1)
        self_recon_loss = F.mse_loss(self_pred_flat, target_raw, reduction="mean")

        if parent_recon_count > 0:
            parent_indices = self._truncate_or_pad_indices(parent_ids, parent_recon_count)
            parent_targets = self.get_masked_embeddings(
                parent_indices,
                self.cluster_embeddings,
                (
                    batch_size,
                    parent_recon_count,
                    self.hidden_dim,
                ),
            )
            parent_mask = parent_indices != -1
            parent_loss = self._masked_mse(parent_pred, parent_targets, parent_mask)
        else:
            parent_loss = x_qs.new_tensor(0.0)
        return (
            self_cluster_loss,
            parent_loss,
            self_recon_loss,
        )

    def cluster_forward(
        self,
        x_q_subspaces,
        self_cluster_ids,
        neighbor_clusters_ids,
        compute_self=True,
        compute_neighbor=True,
    ):
        if x_q_subspaces.dim() != 3:
            raise ValueError(
                "x_q_subspaces is expected to have shape (batch_size, subspace_num, hidden_dim)"
            )

        batch_size = x_q_subspaces.size(0)
        subspace_num = x_q_subspaces.size(1)
        neighbor_chunks = None
        if compute_neighbor:
            neighbor_chunks = self._reshape_indices(
                neighbor_clusters_ids, subspace_num
            )

        all_self_loss = []
        all_neighbor_loss = []

        for idx in range(subspace_num):
            token_embed = x_q_subspaces[:, idx, :]
            if compute_self:
                cluster_emb = torch.index_select(
                    self.cluster_embeddings, dim=0, index=self_cluster_ids
                )
                self_loss = contrastive_cluster_loss(token_embed, cluster_emb)
                all_self_loss.append(self_loss)

            if compute_neighbor and neighbor_chunks is not None:
                layer_losses = []
                neighbor_idx = neighbor_chunks[idx]
                neighbor_emb = self.get_masked_embeddings(
                    neighbor_idx,
                    self.cluster_embeddings,
                    (batch_size, neighbor_idx.size(1), self.hidden_dim),
                )
                for n in range(neighbor_emb.size(1)):
                    neighbor_n = neighbor_emb[:, n, :]  # [B, D]
                    loss_n = contrastive_cluster_loss(token_embed, neighbor_n)
                    layer_losses.append(loss_n)
                neighbor_loss = sum(layer_losses) / (len(layer_losses) + 1e6)
                all_neighbor_loss.append(neighbor_loss)

        return all_self_loss, all_neighbor_loss

    def _reshape_indices(self, indices, chunk_count):
        batch, length = indices.shape
        base = length // chunk_count
        remainder = length % chunk_count

        chunk_sizes = [base + 1 if i < remainder else base for i in range(chunk_count)]
        start_positions = [sum(chunk_sizes[:i]) % length for i in range(chunk_count)]
        end_positions = [s + chunk_sizes[i] for i, s in enumerate(start_positions)]

        chunks = [
            indices[:, s % length : e % length]
            if e <= length
            else torch.cat([indices[:, s:], indices[:, : e % length]], dim=1)
            for s, e in zip(start_positions, end_positions)
        ]
        return chunks  # list of [B, chunk_len_i]

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

    def vq_initialization(self, x):
        self.rq.vq_initialization(self.encoder(x))

    @torch.no_grad()
    def get_indices(self, xs):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e)
        return indices

    @torch.no_grad()
    def get_embs(self, xs):
        x_e = self.encoder(xs)
        return x_e

    @staticmethod
    def compute_loss_terms(outputs, args):
        (
            self_cluster_dist,
            neighbor_cluster_dist,
            commitment_loss,
            self_cluster_recon_loss,
            parent_cluster_recon_loss,
            self_recon_loss,
        ) = outputs

        zero = commitment_loss.new_tensor(0.0)
        if args.lambda_1 == 0 or len(self_cluster_dist) == 0:
            self_cluster_term = zero
        else:
            self_cluster_term = exponential_decay_sum(
                self_cluster_dist, args.lambda_1, reverse=False
            ).mean()

        if args.lambda_2 == 0 or len(neighbor_cluster_dist) == 0:
            neighbor_term = zero
        else:
            neighbor_term = exponential_decay_sum(
                neighbor_cluster_dist, args.lambda_2, reverse=True
            ).mean()

        if args.self_cluster_recon_weight == 0:
            self_cluster_recon_loss = zero
        if args.parent_cluster_recon_weight == 0:
            parent_cluster_recon_loss = zero

        reconstruction_term = (
            args.self_recon_weight * self_recon_loss
            + args.self_cluster_recon_weight * self_cluster_recon_loss
            + args.parent_cluster_recon_weight * parent_cluster_recon_loss
        )
        total_loss = (
            reconstruction_term + commitment_loss + self_cluster_term - neighbor_term
        )
        total_loss = total_loss.mean()
        loss_details = {
            "self_cluster_term": self_cluster_term.item(),
            "neighbor_term": float(neighbor_term),
            "commitment_loss": commitment_loss.item(),
            "self_cluster_recon_loss": self_cluster_recon_loss.item(),
            "parent_cluster_recon_loss": parent_cluster_recon_loss.item(),
            "self_recon_loss": self_recon_loss.item(),
            "weighted_recon_loss": reconstruction_term.item(),
        }
        return total_loss, loss_details

    def train_step(model, optimizer, train_iterator, args):
        """
        A single train step. Apply back-propation and return the loss
        """

        model.train()
        optimizer.zero_grad()

        (ids, self_cluster_ids, neighbor_cluster_ids, hier_ids) = next(train_iterator)
        outputs = model(ids, self_cluster_ids, neighbor_cluster_ids, hier_ids)
        loss, loss_details = model.compute_loss_terms(outputs, args)
        loss.backward()
        optimizer.step()

        log = {"loss": loss.item()}

        log.update(loss_details)
        return log

    @staticmethod
    def save_model(model, optimizer, save_variable_list, args):
        """
        Save the parameters of the model and the optimizer,
        as well as some other variables such as step and learning_rate
        """

        argparse_dict = vars(args)
        with open(os.path.join(args.save_path, "config.json"), "w") as fjson:
            json.dump(argparse_dict, fjson)

        torch.save(
            {
                **save_variable_list,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(args.save_path, "checkpoint"),
        )
