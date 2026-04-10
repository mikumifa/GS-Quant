#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE'
Usage: scripts/batch_lora_codebook_first_gpu.sh DATASET CODEBOOK_DIR [CODEBOOK_DIR ...]

DATASET must be one of: WN18RR, FB15K-237
USAGE
  exit 1
fi

DATASET="$1"
case "$DATASET" in
  "WN18RR"|"FB15K-237") ;;
  *)
    echo "Invalid dataset: ${DATASET}. Use WN18RR or FB15K-237."
    exit 1
    ;;
esac
shift

export CUDA_VISIBLE_DEVICES="4,5"

# ==========================================
# prefix | Target_Modules | LoRA_R | LoRA_Alpha"
# ==========================================

declare -a LORA_CONFIGS=(
  "sens|q_proj,v_proj|64|32"
  # "v5|q_proj,v_proj|64|64"
  # "v5|q_proj,v_proj|128|64"
  # "v4|q_proj,v_proj|64|16"
  # "v4|q_proj,v_proj|128|64"
)

while [[ $# -gt 0 ]]; do
  CODEBOOK_DIR="${1%/}"
  shift
  if [[ ! -d "$CODEBOOK_DIR" ]]; then continue; fi
  TOKENS_FILE="${CODEBOOK_DIR}/tokens.json"
  TRAIN_FILE="${CODEBOOK_DIR}/train.jsonl"
  if [[ ! -f "$TOKENS_FILE" || ! -f "$TRAIN_FILE" ]]; then continue; fi
  CODEBOOK_NAME="$(basename "$CODEBOOK_DIR")"
  for CONFIG_STR in "${LORA_CONFIGS[@]}"; do
    IFS="|" read -r PREFIX T_MODULES L_R L_ALPHA <<< "$CONFIG_STR"
    TIMESTAMP=$(date +%Y%m%d_%H%M)
    RUN_TAG="${CODEBOOK_NAME}_${PREFIX}_${TIMESTAMP}"
    OUTPUT_DIR="processed_data/${DATASET}/checkpoints/LoRA_FT/${PREFIX}/${RUN_TAG}"
    mkdir -p "$OUTPUT_DIR"
    SUMMARY_FILE="${OUTPUT_DIR}/train_summary.json"
    TRAIN_LOG="${OUTPUT_DIR}/train.log"
    EVAL_LOG="${OUTPUT_DIR}/eval.log"
    MASTER_PORT=$((10000 + RANDOM % 20000))
    echo "------------------------------------------------------------"
    echo ">>> Task: ${RUN_TAG}"
    echo ">>> Config: Modules=[${T_MODULES}], R=${L_R}, Alpha=${L_ALPHA}"
    echo ">>> Output: ${OUTPUT_DIR}"
    echo "------------------------------------------------------------"
    nohup uv run torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train_lora.py \
      --model_name_or_path "/nvme1n1/LLM/Meta-Llama-3-8B-Instruct-8bit" \
      --tokens_file "$TOKENS_FILE" \
      --train_file "$TRAIN_FILE" \
      --text_column instruction \
      --output_dir "$OUTPUT_DIR" \
      --train_summary_file "$SUMMARY_FILE" \
      --overwrite_output_dir True \
      --per_device_train_batch_size 16 \
      --gradient_accumulation_steps 1 \
      --learning_rate 2e-4 \
      --source_max_len 2048 \
      --target_max_len 64 \
      --num_train_epochs 8.0 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type constant \
      --logging_steps 200 \
      --save_steps 200 \
      --do_sample True \
      --save_total_limit 1 \
      --logging_dir "$OUTPUT_DIR/logs" \
      --bf16 True \
      --lora_dropout 0.1 \
      --deepspeed configs/ds_config_zero3.json \
      --optim paged_adamw_32bit \
      --target_modules "$T_MODULES" \
      --lora_r "$L_R" \
      --lora_alpha "$L_ALPHA" \
      >"$TRAIN_LOG" 2>&1

    echo ">>> Training finished; starting evaluation..."
    uv run eval_llm.py \
      --summary_config_path "$SUMMARY_FILE" \
      --data_path data \
      --batch_size 64 \
      --max_new_tokens 64 \
      --min_new_tokens 1 \
      --source_max_len 2048 \
      --target_max_len 64 \
      --do_sample False \
      --num_beams 1 \
      --num_return_sequences 1 \
      --model_name_or_path "/nvme1n1/LLM/Meta-Llama-3-8B-Instruct-8bit" \
      >"$EVAL_LOG" 2>&1
    echo ">>> Evaluation completed ✅"
    echo ""
  done
done
echo ">>> Evaluation all completed ✅"
