#!/usr/bin/env bash

set -euo pipefail

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/$(basename "$0" .sh).log}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

KG_DATA_DIR="${KG_DATA_DIR:-data}"
KG_DATASET="${KG_DATASET:-FB15K-237}"
DIFT_ROOT="${DIFT_ROOT:-data/DIFT-dataset}"
DIFT_DATASET="${DIFT_DATASET:-FB15K237}"
DIFT_SOURCE="${DIFT_SOURCE:-CoLE}"

ENTITY_INFO_PATH="${ENTITY_INFO_PATH:-data/${KG_DATASET}/entity.json}"
TOKEN_NUM="16"
SPLITS="${SPLITS:-train valid test}"

CODEBOOK_ROOT="processed_data/FB15K-237/checkpoints/CodeBook/SSQR"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -d "${CODEBOOK_ROOT}" ]]; then
  echo "CodeBook root directory not found: ${CODEBOOK_ROOT}" | tee -a "${LOG_FILE}"
  exit 1
fi

mapfile -t QUANTIZED_FILES < <(find "${CODEBOOK_ROOT}" -type f -name "entity_quantized.json" | sort)

if [[ ${#QUANTIZED_FILES[@]} -eq 0 ]]; then
  echo "No entity_quantized.json files found under ${CODEBOOK_ROOT}" | tee -a "${LOG_FILE}"
  exit 0
fi

read -r -a SPLIT_ARRAY <<< "${SPLITS}"

echo "Discovered ${#QUANTIZED_FILES[@]} quantized codebooks. Logging to ${LOG_FILE}" | tee -a "${LOG_FILE}"

for QUANTIZED_PATH in "${QUANTIZED_FILES[@]}"; do
  CODEBOOK_DIR="$(dirname "${QUANTIZED_PATH}")"
  CODEBOOK_NAME="$(basename "${CODEBOOK_DIR}")"
  OUTPUT_DIR="${CODEBOOK_DIR}"

  echo "Processing ${CODEBOOK_NAME} (${QUANTIZED_PATH})" | tee -a "${LOG_FILE}"

  if [[ "${OVERWRITE}" != "1" ]]; then
    ALL_SPLITS_PRESENT=1
    for split in "${SPLIT_ARRAY[@]}"; do
      if [[ ! -s "${OUTPUT_DIR}/${split}.jsonl" ]]; then
        ALL_SPLITS_PRESENT=0
        break
      fi
    done
    if [[ ${ALL_SPLITS_PRESENT} -eq 1 ]]; then
      echo "  -> All target files already exist, skipping (set OVERWRITE=1 to recompute)." | tee -a "${LOG_FILE}"
      continue
    fi
  fi

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

  CMD+=("${SPLIT_ARRAY[@]}")

  echo "  -> Running: ${CMD[*]}" | tee -a "${LOG_FILE}"
  if ! "${CMD[@]}" >> "${LOG_FILE}" 2>&1; then
    echo "  -> Failed while processing ${CODEBOOK_NAME}, see ${LOG_FILE} for details." | tee -a "${LOG_FILE}"
    exit 1
  fi

  echo "  -> Finished. Output stored in ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
done

echo "Batch adapter conversion complete." | tee -a "${LOG_FILE}"
