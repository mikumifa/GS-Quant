import json
import os
import time
import torch
from utils import quantized_to_token


def main():
    codes_path = "data/SSQR-embedding/FB15k-237_16_1024_notext.pt"
    emds_path = "data/SSQR-embedding/FB15k-237_16_1024_emd_notext.pt"
    save_dir = (
        "processed_data/FB15K-237/checkpoints/CodeBook/SSQR/FB15k-237_16_1024_notext"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    codes = torch.load(codes_path, map_location="cpu")  # [N, codebook_num]
    emds = torch.load(emds_path, map_location="cpu")  # [N, codebook_num, dim]

    entity_quantized = codes.tolist()
    with open(
        os.path.join(save_dir, "entity_quantized.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(entity_quantized, f, ensure_ascii=False)

    unique_codes = torch.unique(codes).tolist()
    tokens = sorted({quantized_to_token(int(x)) for x in unique_codes})
    with open(os.path.join(save_dir, "tokens.json"), "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

    metadata = {
        "dataset": "FB15K-237",
        "source_codes_path": codes_path,
        "source_embeddings_path": emds_path,
        "num_entities": int(codes.shape[0]),
        "codebook_num": int(codes.shape[1]),
        "codebook_size": int(codes.max().item() + 1),
        "embedding_dim": int(emds.shape[2]),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "Converted from SSQR embedding tensors to codebook-style outputs.",
    }
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to {save_dir}")


if __name__ == "__main__":
    main()
