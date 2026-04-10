#!/bin/bash

set -euo pipefail

# Define available GPU IDs to use
# GPU_IDS=(0 1 2 3 4 5 6 7)
# GPU_IDS=(0 4 5 6 7)
GPU_IDS=(0 1 2 3)

NUM_GPUS=${#GPU_IDS[@]}

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

PROCESS_PATH="processed_data"
DATA_PATH="data"
CODEBOOK_NUM=4
HIDDEN_DIM="${HIDDEN_DIM:-512}"
BATCH_SIZE="${BATCH_SIZE:-0}"

ADAPTER_TOKEN_NUM="${ADAPTER_TOKEN_NUM:-4}"
ADAPTER_SPLITS="${ADAPTER_SPLITS:-train valid test}"
ADAPTER_WRAP_TOKEN="${ADAPTER_WRAP_TOKEN:-0}"
KG_DATA_DIR="${KG_DATA_DIR:-data}"
DIFT_ROOT="${DIFT_ROOT:-data/DIFT-dataset}"
ADAPTER_DIFT_SOURCE="${ADAPTER_DIFT_SOURCE:-CoLE}"

declare -A ENTITY_EMBED_PATHS=(
  ["FB15K-237"]="${PROCESS_PATH}/FB15K-237/checkpoints/RotatE/RotatE_llm_batch_512_hidden_1024_dist_cosine_20251104174352/entity_embedding.npy"
  ["WN18RR"]="${PROCESS_PATH}/WN18RR/checkpoints/RotatE/RotatE_llm_batch_512_hidden_1024_dist_cosine_20251128172906/entity_embedding.npy"
)

declare -A CLUSTER_EMBED_PATHS=(
  ["FB15K-237"]="${PROCESS_PATH}/FB15K-237/clusters_embeddings_llm.npy"
  ["WN18RR"]="${PROCESS_PATH}/WN18RR/clusters_embeddings_llm.npy"
)

declare -A ENTITY_INFO_PATHS=(
  ["FB15K-237"]="${PROCESS_PATH}/FB15K-237/entity_info_llm_hier.json"
  ["WN18RR"]="${PROCESS_PATH}/WN18RR/entity_info_llm_hier.json"

)

declare -A DIFT_DATASETS=(
  ["FB15K-237"]="FB15K237"
  ["WN18RR"]="WN18RR"
)


format_tag() {
  local value="$1"
  local formatted
  formatted=$(printf "%g" "$value" 2>/dev/null || echo "$value")
  formatted=${formatted//-/"neg"}
  formatted=${formatted//./"p"}
  formatted=${formatted//e/"exp"}
  formatted=${formatted//E/"exp"}
  formatted=${formatted//+/}
  echo "$formatted"
}

# dataset|name|lambda1|lambda2|commit|self_recon|self_cluster_recon|parent_cluster_recon|batch_size|entity_embed_path
# Schema: dataset|name|lambda1|lambda2|commit|self_recon|self_cluster_recon|parent_cluster_recon|batch_size|entity_embed_path (last field optional; falls back to ENTITY_EMBED_PATHS)
# Baseline: WN18RR|v5.2|0.8|0.4|0.25|1.0|0.05|1|16348

declare -a LOSS_SWEEP=(
  # ==========================================
  # Group 1: Sensitivity for lambda1 (Base: 0.8)
  # Range: [0.2, 0.5, 0.8, 1.0, 1.2]
  # ==========================================
  # "FB15K-237|v5|0.8|0.4|0.25|1.0|1|1|0"
  "WN18RR|v6|1.0|0.4|0.25|1.0|0.05|1|16348"
  # "WN18RR|base_embedding_model|1.0|0.4|0.25|1.0|0.05|1|16348|processed_data/WN18RR/checkpoints/ComplEx_llm_batch_512_hidden_1024_dist_cosine_20251207025418/best/entity_embedding.npy"
  # "WN18RR|base_embedding_model|1.0|0.4|0.25|1.0|0.05|1|16348|processed_data/WN18RR/checkpoints/DistMult_llm_batch_512_hidden_1024_dist_cosine_20251207025407/best/entity_embedding.npy"
  # "WN18RR|base_embedding_model|1.0|0.4|0.25|1.0|0.05|1|16348|processed_data/WN18RR/checkpoints/pRotatE_llm_batch_512_hidden_1024_dist_cosine_20251207025513/best/entity_embedding.npy"
  # "WN18RR|base_embedding_model|1.0|0.4|0.25|1.0|0.05|1|16348|processed_data/WN18RR/checkpoints/TransE_llm_batch_512_hidden_1024_dist_cosine_20251207025357/best/entity_embedding.npy"

)
  

for i in "${!LOSS_SWEEP[@]}"; do
  while true; do
    gpu_index=$((i % NUM_GPUS))
    GPU_ID=${GPU_IDS[$gpu_index]}

    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | awk '{print $1}')
    gpu_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_ID" | awk '{print $1}')
    gpu_free=$((gpu_total - gpu_used))

    if (( gpu_free > 20000 )); then
      echo "✅ GPU ${GPU_ID} is free (${gpu_free} MB available). Starting job..."
      break
    else
      echo "⏳ GPU ${GPU_ID} busy (${gpu_free} MB free). Waiting..."
      sleep 15
    fi
  done

  IFS="|" read -r DATASET LABEL LAMBDA1 LAMBDA2 COMMIT_WEIGHT SELF_RECON_WEIGHT SELF_CLUSTER_RECON_WEIGHT PARENT_CLUSTER_RECON_WEIGHT BATCH_SIZE_ENTRY ENTITY_EMBED_OVERRIDE <<<"${LOSS_SWEEP[$i]}"

  ENTITY_EMBED_PATH=${ENTITY_EMBED_PATHS[$DATASET]:-}
  CLUSTER_EMBED_PATH=${CLUSTER_EMBED_PATHS[$DATASET]:-}
  ENTITY_INFO_PATH=${ENTITY_INFO_PATHS[$DATASET]:-}
  DIFT_DATASET=${DIFT_DATASETS[$DATASET]:-${DATASET//-/}}
  ADAPTER_WRAP_ARG=""

  if [[ -n "${ENTITY_EMBED_OVERRIDE:-}" ]]; then
    ENTITY_EMBED_PATH="$ENTITY_EMBED_OVERRIDE"
  fi

  if [[ -z "$ENTITY_EMBED_PATH" || -z "$CLUSTER_EMBED_PATH" || -z "$ENTITY_INFO_PATH" ]]; then
    echo "❌ Missing paths for dataset ${DATASET}. Please update ENTITY_*_PATHS maps."
    exit 1
  fi

  if [[ -z "$LABEL" ]]; then
    LABEL="cfg${i}"
  fi

  if [[ -n "${BATCH_SIZE_ENTRY:-}" ]]; then
    RUN_BATCH_SIZE="$BATCH_SIZE_ENTRY"
  else
    RUN_BATCH_SIZE="$BATCH_SIZE"
  fi

  tag_l1=$(format_tag "$LAMBDA1")
  tag_l2=$(format_tag "$LAMBDA2")
  tag_commit=$(format_tag "$COMMIT_WEIGHT")
  tag_sr=$(format_tag "$SELF_RECON_WEIGHT")
  tag_scr=$(format_tag "$SELF_CLUSTER_RECON_WEIGHT")
  tag_pcr=$(format_tag "$PARENT_CLUSTER_RECON_WEIGHT")
  if [[ "$RUN_BATCH_SIZE" -eq 0 ]]; then
    tag_bs="auto"
  else
    tag_bs=$(format_tag "$RUN_BATCH_SIZE")
  fi
  run_tag="${DATASET}_${LABEL}_l1_${tag_l1}-l2_${tag_l2}-clw_${tag_commit}-sr_${tag_sr}-scr_${tag_scr}-pcr_${tag_pcr}-bs_${tag_bs}"
  timestamp=$(date +%Y%m%d%H%M%S)
  base_save_dir="${PROCESS_PATH}/${DATASET}/checkpoints/CodeBook"
  if [[ -n "$LABEL" ]]; then
    SAVE_PATH="${base_save_dir}/${LABEL}/${run_tag}_${HIDDEN_DIM}_${timestamp}"
  else
    SAVE_PATH="${base_save_dir}/${run_tag}_${HIDDEN_DIM}_${timestamp}"
  fi
  if [[ "$ADAPTER_WRAP_TOKEN" == "1" ]]; then
    ADAPTER_WRAP_ARG="--wrap_token"
  fi
  log_file="$LOG_DIR/run_train_codebook_${run_tag}_gpu${GPU_ID}.log"

  echo "🚀 Launching on GPU ${GPU_ID}: ${LABEL} (dataset: ${DATASET})"
  nohup bash -c "
    set -euo pipefail
    CUDA_VISIBLE_DEVICES=${GPU_ID} uv run train_codebook.py \
      --entity_embeddings_path \"${ENTITY_EMBED_PATH}\" \
      --cluster_embeddings_path \"${CLUSTER_EMBED_PATH}\" \
      --entity_info_path \"${ENTITY_INFO_PATH}\" \
      --cuda \
      --codebook_num \"${CODEBOOK_NUM}\" \
      --run_label \"${LABEL}\" \
      --run_name \"${run_tag}\" \
      --data_path \"${DATA_PATH}\" \
      --dataset \"${DATASET}\" \
      --lambda_1 \"${LAMBDA1}\" \
      --lambda_2 \"${LAMBDA2}\" \
      --commit_loss_weight \"${COMMIT_WEIGHT}\" \
      --self_recon_weight \"${SELF_RECON_WEIGHT}\" \
      --self_cluster_recon_weight \"${SELF_CLUSTER_RECON_WEIGHT}\" \
      --parent_cluster_recon_weight \"${PARENT_CLUSTER_RECON_WEIGHT}\" \
      --save_path \"${SAVE_PATH}\" \
      --hidden_dim \"${HIDDEN_DIM}\" \
      --batch_size \"${RUN_BATCH_SIZE}\"

    if [[ -f \"${SAVE_PATH}/entity_quantized.json\" ]]; then
        echo \"✅ Training finished, starting adapter_lora_data.py for ${SAVE_PATH}\"
        uv run adapter_lora_data.py \
          --kg_data_dir \"${KG_DATA_DIR}\" \
          --kg_dataset \"${DATASET}\" \
          --dift_root \"${DIFT_ROOT}\" \
          --dift_dataset \"${DIFT_DATASET}\" \
          --dift_source \"${ADAPTER_DIFT_SOURCE}\" \
          --entity_info_path \"${ENTITY_INFO_PATH}\" \
          --quantized_path \"${SAVE_PATH}/entity_quantized.json\" \
          --token_num \"${ADAPTER_TOKEN_NUM}\" \
          --output_dir \"${SAVE_PATH}\" \
          ${ADAPTER_WRAP_ARG} \
          --splits ${ADAPTER_SPLITS}
    else
        echo \"⚠️ Skipping adapter_lora_data.py: ${SAVE_PATH}/entity_quantized.json not found\"
    fi
" >"$log_file" 2>&1 &
  sleep 10

done

wait
echo "✅ All jobs finished."
