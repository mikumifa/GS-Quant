#!/bin/bash
set -euo pipefail

# 用法:
#   bash scripts/batch_embedding.sh                     # 默认数据集 WN18RR，自动多卡调度
#   DATASET=WN18RR bash scripts/batch_embedding.sh
#   GPU_IDS="0 2 3" DATASET=WN18RR bash scripts/batch_embedding.sh

# GPU 列表（可用环境变量 GPU_IDS 覆盖）
IFS=" " read -r -a GPU_IDS <<<"${GPU_IDS:-0 1 2 3}"
NUM_GPUS=${#GPU_IDS[@]}

if (( NUM_GPUS == 0 )); then
  echo "❌ No GPU IDs provided. Set GPU_IDS env var."
  exit 1
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

DATASET="${DATASET:-WN18RR}"
# 覆盖默认超参时可用：HIDDEN_DIM、BATCH_SIZE、GAMMA、ADV_TEMP 等
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NEG_SIZE="${NEG_SIZE:-512}"
GAMMA="${GAMMA:-9.0}"
ADV_TEMP="${ADV_TEMP:-1.0}"
DISTANCE="${DISTANCE:-cosine}"
MAX_STEPS="${MAX_STEPS:-1600000}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"

MODELS=(TransE DistMult ComplEx pRotatE)

for i in "${!MODELS[@]}"; do
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

  MODEL=${MODELS[$i]}
  ts=$(date +%Y%m%d%H%M%S)
  LOG_FILE="$LOG_DIR/train_graph_embedding_${DATASET}_${MODEL}_${ts}_gpu${GPU_ID}.log"

  echo "🚀 Launching ${MODEL} on GPU ${GPU_ID} for ${DATASET}, log: ${LOG_FILE}"
  nohup bash -c "
    set -euo pipefail
    CUDA_VISIBLE_DEVICES=${GPU_ID} uv run train_graph_embedding.py \
      --do_train --do_valid --do_test --cuda \
      --data_path data \
      --dataset \"${DATASET}\" \
      --hierarchy_type llm \
      --model \"${MODEL}\" \
      --distance_metric \"${DISTANCE}\" \
      -n \"${NEG_SIZE}\" -b \"${BATCH_SIZE}\" -d \"${HIDDEN_DIM}\" \
      -g \"${GAMMA}\" -a \"${ADV_TEMP}\" \
      -lr 0.0001 --max_steps \"${MAX_STEPS}\" \
      --test_batch_size \"${TEST_BATCH_SIZE}\"
  " >"$LOG_FILE" 2>&1 &

  sleep 10
done

wait
echo "✅ All embedding jobs finished."
