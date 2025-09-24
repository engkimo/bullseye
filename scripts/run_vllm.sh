#!/bin/bash
set -e

echo "=== vLLM Server Startup Script ==="

# Parse arguments
MODEL_PATH=${1:-"gpt-oss-20B"}
LORA_PATH=${2:-"weights/lora/adapter"}
PORT=${3:-8000}

# Activate virtual environment
source venv/bin/activate

# Check GPU memory and adjust parameters
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU Memory: ${GPU_MEM} MB"

# Set parameters based on GPU memory
if [ $GPU_MEM -lt 16000 ]; then
    echo "Error: Insufficient GPU memory. Need at least 16GB"
    exit 1
elif [ $GPU_MEM -lt 24000 ]; then
    # For 16GB GPUs
    MAX_MODEL_LEN=4096
    GPU_MEMORY_UTIL=0.95
    QUANTIZATION="awq"
elif [ $GPU_MEM -lt 48000 ]; then
    # For 24GB GPUs (L4, RTX 4090)
    MAX_MODEL_LEN=8192
    GPU_MEMORY_UTIL=0.9
    QUANTIZATION="bitsandbytes"
else
    # For 40GB+ GPUs (A100, A6000)
    MAX_MODEL_LEN=16384
    GPU_MEMORY_UTIL=0.85
    QUANTIZATION="none"
fi

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  LoRA: $LORA_PATH"
echo "  Max length: $MAX_MODEL_LEN"
echo "  GPU utilization: $GPU_MEMORY_UTIL"
echo "  Quantization: $QUANTIZATION"

# Export environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export VLLM_USE_MODELSCOPE=False

# Create temporary config
cat > /tmp/vllm_config.yaml << EOF
model: $MODEL_PATH
tokenizer: $MODEL_PATH
enable_lora: true
lora_modules:
  - name: docja
    path: $LORA_PATH
    
max_model_len: $MAX_MODEL_LEN
gpu_memory_utilization: $GPU_MEMORY_UTIL
max_num_seqs: 256
max_num_batched_tokens: $MAX_MODEL_LEN

quantization: $QUANTIZATION
dtype: float16
trust_remote_code: true

# API settings
host: 0.0.0.0
port: $PORT
allow_credentials: false
allow_origins: ["*"]
allow_methods: ["*"]
allow_headers: ["*"]
EOF

# Kill any existing vLLM process
pkill -f "python -m vllm.entrypoints.openai.api_server" || true
sleep 2

# Start vLLM server
echo "Starting vLLM server on port $PORT..."

if [ -f "$LORA_PATH/adapter_config.json" ]; then
    # With LoRA adapter
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --tokenizer $MODEL_PATH \
        --enable-lora \
        --lora-modules docja=$LORA_PATH \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        --quantization $QUANTIZATION \
        --dtype float16 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port $PORT \
        2>&1 | tee logs/vllm_server.log &
else
    # Without LoRA (base model only)
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --tokenizer $MODEL_PATH \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        --quantization $QUANTIZATION \
        --dtype float16 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port $PORT \
        2>&1 | tee logs/vllm_server.log &
fi

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for server to start
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/v1/models > /dev/null; then
        echo "Server is ready!"
        break
    fi
    sleep 2
done

# Test the server
echo "Testing server..."
curl -X POST http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "prompt": "[文書] テストドキュメント [質問] これは何ですか？",
        "max_tokens": 50,
        "temperature": 0.1
    }' | jq .

# Create systemd service (optional)
if [ "$EUID" -eq 0 ]; then
    cat > /etc/systemd/system/vllm-docja.service << EOF
[Unit]
Description=vLLM DocJA Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/docja
Environment="PATH=/opt/docja/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/opt/docja/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --enable-lora \
    --lora-modules docja=$LORA_PATH \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --port $PORT
Restart=always

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    echo "Systemd service created: vllm-docja.service"
    echo "To enable: sudo systemctl enable vllm-docja"
    echo "To start: sudo systemctl start vllm-docja"
fi

echo "=== vLLM server is running ==="
echo "API endpoint: http://localhost:$PORT/v1"
echo "Logs: logs/vllm_server.log"
echo "PID: $VLLM_PID"

# Keep script running
wait $VLLM_PID