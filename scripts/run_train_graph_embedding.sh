export CUDA_VISIBLE_DEVICES=3

# First argument (optional) sets dataset name, defaults to FB15K-237
DATASET="${1:-FB15K-237}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "$0" .sh).log"

nohup uv run train_graph_embedding.py \
 --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data \
 --dataset "$DATASET" \
 --distance_metric cosine \
 --hierarchy_type llm \
 -n 512 -b 512 -d 1024 \
 -g 9.0 -a 1.0 \
 -lr 0.0001 --max_steps 1600000  \
 --hit_topk 100 \
 --build_adapter_data \
 --test_batch_size 8 \
 >"$LOG_FILE" 2>&1 
