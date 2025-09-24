#!/bin/bash
set -e

echo "=== DocJA Bootstrap Script ==="

# Check if running on EC2
if [ ! -f /sys/hypervisor/uuid ] || [ $(head -c 3 /sys/hypervisor/uuid) != "ec2" ]; then
    echo "Warning: Not running on EC2 instance"
fi

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    git git-lfs curl wget unzip \
    libjpeg-dev libpng-dev libtiff-dev \
    tesseract-ocr tesseract-ocr-jpn \
    poppler-utils \
    tmux htop nvtop

# Install Docker (for potential containerized deployment)
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Setup Python environment
echo "Setting up Python environment..."
cd /opt/docja || cd ~/docja
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt || cat > requirements.txt << 'EOF'
# Core ML
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
opencv-python==4.8.1.78
pillow==10.1.0
matplotlib==3.8.2

# Deep Learning
transformers==4.36.2
accelerate==0.25.0
datasets==2.16.1
tokenizers==0.15.0
safetensors==0.4.1

# Document Processing
pypdf==3.17.4
pypdfium2==4.30.1
pikepdf==9.4.2
reportlab==4.1.0
pytesseract==0.3.10
python-doctr==0.7.0

# LLM Training
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
trl==0.7.10
peft==0.7.1
bitsandbytes==0.41.3

# Serving
vllm==0.2.7
fastapi==0.108.0
uvicorn==0.25.0
pydantic==2.5.3
requests==2.31.0

# Evaluation
rouge-score==0.1.2
sacrebleu==2.4.0
bert-score==0.3.13

# Utils
pyyaml==6.0.1
python-dotenv==1.0.0
tqdm==4.66.1
tensorboard==2.15.1
wandb==0.16.2
hydra-core==1.3.2
omegaconf==2.3.0

# Development
pytest==7.4.4
black==23.12.1
isort==5.13.2
flake8==7.0.0
mypy==1.8.0
EOF

pip install -r requirements.txt

# Optional: install evaluation extras (pycocotools, teds)
if [ "$DOCJA_INSTALL_EXTRAS" = "1" ]; then
    echo "Installing extras for evaluation (pycocotools, teds)..."
    pip install pycocotools teds || true
fi

# Setup CUDA environment
echo "Setting up CUDA environment..."
if [ -d "/usr/local/cuda" ]; then
    echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
    echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
fi

# Setup AWS CLI
echo "Configuring AWS CLI..."
if [ -n "$AWS_REGION" ]; then
    aws configure set region $AWS_REGION
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/{raw,processed,samples}
mkdir -p weights/{det,rec,layout,table,lora}
mkdir -p configs
mkdir -p logs
mkdir -p results
mkdir -p checkpoints

# Download sample data
echo "Preparing sample data..."
cat > data/samples/sample.txt << 'EOF'
これは日本語のサンプルテキストです。
縦書きにも対応しています。
表やレイアウトの解析も可能です。
EOF

# Setup S3 sync daemon
if [ -n "$S3_BUCKET" ]; then
    echo "Setting up S3 sync daemon..."
    cat > /tmp/s3_sync_daemon.sh << 'EOFS'
#!/bin/bash
while true; do
    aws s3 sync checkpoints/ s3://$S3_BUCKET/checkpoints/ --exclude "*.tmp"
    aws s3 sync weights/ s3://$S3_BUCKET/weights/ --exclude "*.tmp"
    sleep 60
done
EOFS
    chmod +x /tmp/s3_sync_daemon.sh
    nohup /tmp/s3_sync_daemon.sh > logs/s3_sync.log 2>&1 &
    echo "S3 sync daemon started (PID: $!)"
fi

# Verify GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"
    python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
else
    echo "Warning: No GPU detected"
fi

echo "=== Bootstrap completed successfully ==="
echo "Activate environment with: source venv/bin/activate"
