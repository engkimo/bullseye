#!/usr/bin/env bash
set -euo pipefail

echo "[uv_setup] Installing uv (if missing) and project deps"

# Detect uv
if ! command -v uv >/dev/null 2>&1; then
  echo "[uv_setup] uv not found; installing via install script"
  # Uses network; caller should ensure they allow it
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv (./.venv) if not exists
if [ ! -d ".venv" ]; then
  echo "[uv_setup] Creating .venv (Python 3.12)"
  # Prefer Python 3.12 for broad wheel availability (e.g., torch 2.2)
  uv venv --python 3.12 || uv venv
fi

source .venv/bin/activate

# Ensure requirements.txt exists (bootstrap writes a default if absent)
if [ ! -f requirements.txt ]; then
  echo "[uv_setup] requirements.txt not found; generating minimal requirements"
  cat > requirements.txt << 'EOF'
torch==2.2.0
torchvision==0.17.0
numpy<2
scikit-learn==1.3.2
pillow
opencv-python-headless
pypdfium2
reportlab
fastapi
uvicorn
pydantic
requests
huggingface_hub
datasets<3
tqdm
EOF
fi

echo "[uv_setup] Installing core dependencies via uv pip (sync)"
uv pip sync requirements.txt || uv pip install --upgrade --reinstall -r requirements.txt

if [ -f requirements-extras.txt ]; then
  echo "[uv_setup] Installing extras"
  uv pip install -r requirements-extras.txt || true
fi

# Ensure CPU-only PyTorch if CUDA libs are missing
echo "[uv_setup] Verifying torch import"
set +e
python - << 'PY'
try:
    import torch  # noqa: F401
    print('torch import OK:', torch.__version__)
except Exception as e:
    print('torch import failed:', e)
    raise SystemExit(1)
PY
if [ $? -ne 0 ]; then
  echo "[uv_setup] Reinstalling torch/torchvision with CPU-only wheels"
  uv pip uninstall -y torch torchvision || true
  uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 torchvision==0.17.0
fi
set -e

echo "[uv_setup] Done. Activate with: source .venv/bin/activate"
