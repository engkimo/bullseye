#!/usr/bin/env bash
set -euo pipefail

# P0 end-to-end pipeline (background-friendly) for Japanese Table strengthening
# Steps: system deps (optional) -> venv + pip -> synth gen -> train (table) -> eval -> demo export

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR" "$ROOT_DIR/results" "$ROOT_DIR/data" "$ROOT_DIR/weights/table"
LOG_FILE="$LOG_DIR/p0_pipeline.log"

echo "[P0] Start pipeline at $(date)" | tee -a "$LOG_FILE"
cd "$ROOT_DIR"

# 0) Optional system dependencies (Amazon Linux dnf / Ubuntu apt)
if [[ "${SKIP_SYSTEM_DEPS:-0}" != "1" ]]; then
  if command -v dnf >/dev/null 2>&1; then
    echo "[P0] Installing system deps via dnf" | tee -a "$LOG_FILE"
    sudo dnf install -y python3.11 python3.11-devel python3.11-venv \
      gcc gcc-c++ make tmux \
      mesa-libGL libXrender libXext libSM \
      freetype freetype-devel libpng libpng-devel libjpeg-turbo libjpeg-turbo-devel || true
  elif command -v apt-get >/dev/null 2>&1; then
    echo "[P0] Installing system deps via apt-get" | tee -a "$LOG_FILE"
    sudo apt-get update || true
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev \
      build-essential tmux \
      libgl1-mesa-glx libxrender1 libxext6 libsm6 \
      libfreetype6 libfreetype6-dev libpng-dev libjpeg-dev || true
  else
    echo "[P0] No known package manager found; skipping system deps" | tee -a "$LOG_FILE"
  fi
fi

# 1) Python venv
PYBIN=""
if command -v python3.11 >/dev/null 2>&1; then PYBIN=python3.11; 
elif command -v python3.10 >/dev/null 2>&1; then PYBIN=python3.10; 
else PYBIN=python3; fi

if [[ ! -d "$ROOT_DIR/venv" ]]; then
  echo "[P0] Creating venv with $PYBIN" | tee -a "$LOG_FILE"
  "$PYBIN" -m venv "$ROOT_DIR/venv"
fi
source "$ROOT_DIR/venv/bin/activate"
python -m pip install -U pip setuptools wheel >> "$LOG_FILE" 2>&1

# 2) Pip deps (GPU torch)
if python -c 'import torch; import sys; sys.exit(0)' 2>/dev/null; then
  echo "[P0] torch already installed" | tee -a "$LOG_FILE"
else
  echo "[P0] Installing torch/cu121" | tee -a "$LOG_FILE"
  pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121 >> "$LOG_FILE" 2>&1 || true
fi

pip install numpy opencv-python pillow pypdfium2 reportlab \
  fastapi uvicorn pydantic requests tqdm transformers datasets \
  pycocotools teds \
  >> "$LOG_FILE" 2>&1 || true

echo "[P0] GPU check:" | tee -a "$LOG_FILE"
nvidia-smi >> "$LOG_FILE" 2>&1 || true
python - <<'PY' 2>>"$LOG_FILE" | tee -a "$LOG_FILE"
import torch
print('CUDA available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())
PY

# 3) Synthesize datasets (small-to-mid scale; override with env vars)
TABLE_COUNT=${TABLE_COUNT:-2000}
LAYOUT_PAGES=${LAYOUT_PAGES:-500}
REC_TOTAL=${REC_TOTAL:-2000}
REC_JP_RATIO=${REC_JP_RATIO:-0.85}

echo "[P0] Generating tables: $TABLE_COUNT" | tee -a "$LOG_FILE"
python3 scripts/gen_synth_tables_ja.py --out data/table --count "$TABLE_COUNT" >> "$LOG_FILE" 2>&1

echo "[P0] Generating layout pages: $LAYOUT_PAGES" | tee -a "$LOG_FILE"
python3 scripts/gen_synth_layout_ja.py --out data/synth_layout_ja --pages "$LAYOUT_PAGES" >> "$LOG_FILE" 2>&1

echo "[P0] Generating recognition lines: $REC_TOTAL (JP ratio=$REC_JP_RATIO)" | tee -a "$LOG_FILE"
python3 scripts/gen_synth_rec_lines.py --out data/rec --total "$REC_TOTAL" --jp-ratio "$REC_JP_RATIO" >> "$LOG_FILE" 2>&1 || true

# 4) Train (Table / TATR; shortened epochs configured)
echo "[P0] Training TATR (table)" | tee -a "$LOG_FILE"
python -m src.train_table --config configs/table_tatr.yaml >> "$LOG_FILE" 2>&1

# 5) Evaluate (table)
echo "[P0] Evaluating table" | tee -a "$LOG_FILE"
python -m src.eval_table --config configs/table_tatr.yaml --output results/table_metrics.json >> "$LOG_FILE" 2>&1 || true

# 6) Demo export (bullseye_demo)
echo "[P0] Demo export (md/html/pdf + vis)" | tee -a "$LOG_FILE"
python -m src.cli data/samples/sample.pdf --layout --table --reading-order --lite --vis -f md -o results/bullseye_demo >> "$LOG_FILE" 2>&1 || true
python -m src.cli data/samples/sample.pdf --layout --table --lite -f html -o results/bullseye_demo >> "$LOG_FILE" 2>&1 || true
python -m src.cli data/samples/sample.pdf --layout --table --lite -f pdf -o results/bullseye_demo >> "$LOG_FILE" 2>&1 || true

echo "[P0] Done at $(date)" | tee -a "$LOG_FILE"
echo "[P0] Logs: $LOG_FILE" | tee -a "$LOG_FILE"
