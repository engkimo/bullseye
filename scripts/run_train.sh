#!/bin/bash
set -e

# Parse arguments
MODEL_TYPE=${1:-all}
RESUME=${2:-false}

echo "=== DocJA Training Script ==="
echo "Model type: $MODEL_TYPE"
echo "Resume: $RESUME"

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false

# Function to handle OOM
handle_oom() {
    echo "OOM detected, adjusting parameters..."
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    return 1
}

# Training functions
train_det() {
    echo "Training Text Detection models..."
    
    # DBNet++
    python -m src.train_det \
        --config configs/det_dbnet.yaml \
        --model dbnet \
        --resume $RESUME || handle_oom
    
    # YOLO
    python -m src.train_det \
        --config configs/det_yolo.yaml \
        --model yolo \
        --resume $RESUME || handle_oom
}

train_rec() {
    echo "Training Text Recognition models..."
    
    # ABINet
    python -m src.train_rec \
        --config configs/rec_abinet.yaml \
        --model abinet \
        --resume $RESUME || handle_oom
    
    # SATRN
    python -m src.train_rec \
        --config configs/rec_satrn.yaml \
        --model satrn \
        --resume $RESUME || handle_oom
}

train_layout() {
    echo "Training Layout Detection models..."
    
    python -m src.train_layout \
        --config configs/layout_yolo.yaml \
        --model yolo \
        --resume $RESUME || handle_oom
}

train_table() {
    echo "Training Table Recognition model..."
    
    python -m src.train_table \
        --config configs/table_tatr.yaml \
        --resume $RESUME || handle_oom
}

train_lora() {
    echo "Training LoRA adapter for gpt-oss-20B..."
    
    python -m src.train_doc_ja_lora \
        --config configs/training_lora.yaml \
        --resume $RESUME || handle_oom
}

# Main training logic
case $MODEL_TYPE in
    det|detection)
        train_det
        ;;
    rec|recognition)
        train_rec
        ;;
    layout)
        train_layout
        ;;
    table)
        train_table
        ;;
    lora|llm)
        train_lora
        ;;
    all)
        train_det
        train_rec
        train_layout
        train_table
        train_lora
        ;;
    *)
        echo "Unknown model type: $MODEL_TYPE"
        echo "Usage: $0 [det|rec|layout|table|lora|all] [true|false]"
        exit 1
        ;;
esac

# Sync to S3 if configured
if [ -n "$S3_BUCKET" ]; then
    echo "Syncing weights to S3..."
    aws s3 sync weights/ s3://$S3_BUCKET/weights/
fi

echo "=== Training completed ==="