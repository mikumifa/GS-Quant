#!/usr/bin/env bash

set -euo pipefail

# ==========================================
# 1. Check Input Arguments
# ==========================================
if [[ $# -lt 2 ]]; then
  cat <<'USAGE'
Usage: scripts/run_adapter_data.sh DATASET CODEBOOK_DIR [CODEBOOK_DIR ...]

Example:
  scripts/batch_adapter_lora_data.sh FB15K-237 processed_data/FB15K-237/checkpoints/CodeBook/CodeBook_*

Description:
  The script iterates through the provided directories, looks for 'entity_quantized.json',
  and calls 'adapter_lora_data.py' to generate training data for each.
USAGE
  exit 1
fi

# ==========================================
# 2. Dataset Arg + Environment & Defaults
# ==========================================
DATASET="$1"
case "$DATASET" in
  "WN18RR"|"FB15K-237") ;;
  *)
    echo "Invalid dataset: ${DATASET}. Use WN18RR or FB15K-237."
    exit 1
    ;;
esac
shift

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
# Append timestamp to log filename to avoid conflicts
LOG_FILE="${LOG_FILE:-${LOG_DIR}/$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S).log}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

KG_DATA_DIR="${KG_DATA_DIR:-data}"
KG_DATASET="${KG_DATASET:-$DATASET}"
DIFT_ROOT="${DIFT_ROOT:-data/DIFT-dataset}"
DIFT_DATASET_MAP_FALLBACK="${DIFT_DATASET:-}"
declare -A DIFT_MAP=(
  ["FB15K-237"]="FB15K237"
  ["WN18RR"]="WN18RR"
)
# Prefer env override; otherwise map from dataset
DIFT_DATASET="${DIFT_DATASET_MAP_FALLBACK:-${DIFT_MAP[$DATASET]}}"
DIFT_SOURCE="${DIFT_SOURCE:-CoLE}"

ENTITY_INFO_PATH="${ENTITY_INFO_PATH:-data/${KG_DATASET}/entity.json}"
TOKEN_NUM="${TOKEN_NUM:-4}"
SPLITS="${SPLITS:-train valid test}"
OVERWRITE="${OVERWRITE:-0}"

# Convert SPLITS string to array
read -r -a SPLIT_ARRAY <<< "${SPLITS}"

echo ">>> Task started. Logging to: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# ==========================================
# 3. Iterate through Input Directories
# ==========================================
while [[ $# -gt 0 ]]; do
  # Remove trailing slash if present
  CODEBOOK_DIR="${1%/}"
  shift

  # Check if directory exists
  if [[ ! -d "$CODEBOOK_DIR" ]]; then
    echo ">>> [SKIP] Invalid directory: ${CODEBOOK_DIR}" | tee -a "${LOG_FILE}"
    continue
  fi

  # Check for the core file 'entity_quantized.json'
  QUANTIZED_PATH="${CODEBOOK_DIR}/entity_quantized.json"
  if [[ ! -f "$QUANTIZED_PATH" ]]; then
    echo ">>> [SKIP] Missing entity_quantized.json in: ${CODEBOOK_DIR}" | tee -a "${LOG_FILE}"
    continue
  fi

  CODEBOOK_NAME="$(basename "${CODEBOOK_DIR}")"
  OUTPUT_DIR="${CODEBOOK_DIR}"

  echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
  echo "Processing ${CODEBOOK_NAME} (${CODEBOOK_DIR})" | tee -a "${LOG_FILE}"

  # Check if output exists (skip if OVERWRITE=0)
  if [[ "${OVERWRITE}" != "1" ]]; then
    ALL_SPLITS_PRESENT=1
    for split in "${SPLIT_ARRAY[@]}"; do
      if [[ ! -s "${OUTPUT_DIR}/${split}.jsonl" ]]; then
        ALL_SPLITS_PRESENT=0
        break
      fi
    done
    if [[ ${ALL_SPLITS_PRESENT} -eq 1 ]]; then
      echo "  -> [SKIP] Output files exist (train/valid/test.jsonl). Set OVERWRITE=1 to force run." | tee -a "${LOG_FILE}"
      continue
    fi
  fi

  # Construct Python command
  CMD=(
    "${PYTHON_BIN}"
    "adapter_lora_data.py"
    "--kg_data_dir" "${KG_DATA_DIR}"
    "--kg_dataset" "${KG_DATASET}"
    "--dift_root" "${DIFT_ROOT}"
    "--dift_dataset" "${DIFT_DATASET}"
    "--dift_source" "${DIFT_SOURCE}"
    "--entity_info_path" "${ENTITY_INFO_PATH}"
    "--quantized_path" "${QUANTIZED_PATH}"
    # "--wrap_token"
    "--token_num" "${TOKEN_NUM}"
    "--output_dir" "${OUTPUT_DIR}"
    "--splits"
  )

  # Append splits
  CMD+=("${SPLIT_ARRAY[@]}")
  echo "  -> Running Python script..." | tee -a "${LOG_FILE}"
  
  if ! "${CMD[@]}" >> "${LOG_FILE}" 2>&1; then
    echo "  -> [ERROR] Failed to process ${CODEBOOK_NAME}. Check logs for details." | tee -a "${LOG_FILE}"
  else
    echo "  -> [SUCCESS] Output saved to ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
  fi

done

echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
echo ">>> Batch adapter conversion completed." | tee -a "${LOG_FILE}"
