# BullsEye - Japanese Document AI Integration System

[English] | [日本語](README_JA.md)

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB.svg)

An integrated AI system for production‑grade Japanese document processing: OCR, layout analysis, table structure recognition, reading order, and LLM integration.

<img width="3024" height="1702" alt="image" src="https://github.com/user-attachments/assets/2ee70864-ceaf-4969-bdb9-0888a76788c5" />

Independent implementation with compatibility to upstream model families (bullseye integration).

## Features

- **Text Detection**: High‑accuracy region detection with DBNet++ / YOLO
- **Text Recognition**: ABINet / SATRN with 7,000+ JP chars (vertical/handwriting)
- **Layout Detection**: DocLayNet‑compatible classes (YOLO/DETR switchable)
- **Table Recognition**: TATR‑like structure recognition and cell OCR
- **Reading Order**: Graph‑based ordering with rule post‑processing
- **LLM Integration**: gpt‑oss‑20B for QA/summary/extraction
- **Export**: HTML/Markdown/JSON/CSV/Searchable‑PDF

## Requirements

- Python 3.12 (3.10+ compatible)
- CUDA 12.1+ for GPU inference/training
- AWS CLI configured (for cloud deployment)
- Hardware (minimum → recommended)
  - CPU: 8 cores → 16 cores
  - RAM: 32 GB → 64 GB
  - GPU: 16 GB → L4 24 GB (target p95 ≤ 2s/page)
  - Disk: free 100 GB → 300 GB

## Quick Start

```bash
# Setup (uv recommended)
make setup-uv
source .venv/bin/activate

# Run API (FastAPI)
uvicorn src.api:app --host 0.0.0.0 --port 8001

# Minimal analyze (Unified Doc JSON)
curl -X POST http://localhost:8001/v1/di/analyze \
  -F file=@data/samples/sample.pdf \
  -F 'options={"output_format":"json","detect_layout":true,"detect_tables":true,"extract_reading_order":true,"lite":true}' | jq '.'

# CLI example
python -m src.cli data/samples/sample.pdf -f md -o results \
  --layout --table --reading-order --lite --vis

# Train models
make train-all

# Evaluate
make eval-all
```

## Backend/Frontend: Run

Minimal environment variables and commands. See Advanced ENV in docs for tuning knobs.

### Backend (FastAPI)

```bash
# Minimal ENV (redacted and de-duplicated)
export DOCJA_API_KEY=dev-123                # Use a strong random value in prod
export DOCJA_LLM_LANG=ja                    # internal reasoning=en, responses=ja
export DOCJA_READING_ORDER_SIMPLE=1         # optional stability for large pages
export DOCJA_VIS_PROFILE=clean              # overlays: clean|debug|raw (clean recommended)

# Bullseye (upstream) providers and repos
export DOCJA_PROVIDER_ALIAS_LABEL=bullseye
export DOCJA_DET_PROVIDER=bullseye
export DOCJA_REC_PROVIDER=bullseye
export DOCJA_LAYOUT_PROVIDER=bullseye
export DOCJA_TABLE_PROVIDER=bullseye

# Default upstream repos (HF) for bullseye
export DOCJA_BULLSEYE_LOCAL_DIR="$PWD/bullseye/src"
export DOCJA_BULLSEYE_DET_REPO=Ryousukee/bullseye-dbnet
export DOCJA_BULLSEYE_REC_REPO=Ryousukee/bullseye-recparseq
export DOCJA_BULLSEYE_LAYOUT_REPO=Ryousukee/bullseye-layoutrtdetrv
export DOCJA_BULLSEYE_TABLE_REPO=Ryousukee/bullseye-tablertdetrv

# Choose ONE provider block below (do not set both)

## Provider: Ollama (default)
export DOCJA_LLM_PROVIDER=ollama
export DOCJA_OLLAMA_ENDPOINT=http://localhost:11434
export DOCJA_OLLAMA_MODEL=gptoss-20b
export DOCJA_LLM_TIMEOUT=45

## Provider: Gemma 3 (OpenAI-compatible)
# export DOCJA_LLM_PROVIDER=gemma3
# export DOCJA_LLM_ENDPOINT=http://localhost:8000/v1
# export DOCJA_LLM_MODEL=google/gemma-3-12b-it
# export DOCJA_LLM_TIMEOUT=30
# export DOCJA_LLM_USE_IMAGE=1

# Optional (private): set externally, do not hardcode in README
# export HUGGING_FACE_HUB_TOKEN=***
# export OPENAI_API_KEY=***

# (uv env)
make setup-uv && source .venv/bin/activate

# Run API
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

### Frontend (Vite dev server)

```bash
cd front
npm install
npm run dev -- --host 0.0.0.0
# Example: open http://<HOST>:5173 in your browser
```



## Installation

### Local Setup

```bash
git clone https://github.com/engkimo/bullseye.git
cd docja
make setup-uv
source .venv/bin/activate
```

### AWS Deployment

```bash
# Configure AWS credentials
export AWS_PROFILE=your-profile
export KEY_NAME=your-key

# Deploy infrastructure
make infra-plan
make infra-apply

# Connect and setup
make ssh
bash scripts/bootstrap.sh
```

## Usage
## Refactoring & Stability

- We trialed a large directory refactor (core/pipeline/io) for Clean Code alignment, but rolled it back immediately because it broke upstream adapters and analysis paths. Stability takes priority.
- The refactor plan is now incremental and safe. No breaking changes to public APIs/modules will be merged without shims and verifiable smoke tests.
- See `docs/15_bullseye_refactor.md` for the staged plan, guardrails, and rollback notes.

### CLI

```bash
# Basic OCR
docja input.pdf -o output/

# With options
docja input.pdf \
  -f md \              # Output format (md/html/json/csv/pdf)
  -o results/ \        # Output directory
  --layout \           # Enable layout detection
  --table \            # Enable table extraction
  --reading-order \    # Enable reading order
  --llm \              # Enable LLM analysis
  -v                   # Verbose logging

# Providers (bullseye)
# You can set the bullseye provider via env vars or CLI
DOCJA_PROVIDER_ALIAS_LABEL=bullseye \
python -m src.cli input.pdf -f json -o out \
  --rec-provider bullseye \
  --det-provider bullseye \
  --layout-provider bullseye \
  --table-provider bullseye
```

### Python API

```python
from src.hf_compat import DocJaProcessor

# Load processor
processor = DocJaProcessor.from_pretrained("weights/docja-v1")

# Process document
result = processor(
    "input.pdf",
    detect_layout=True,
    detect_tables=True,
    extract_reading_order=True
)

# Access results
for page in result.pages:
    print(f"Page {page.page_num}")
    for block in page.text_blocks:
        print(f"  {block.text}")
```

## Model Architecture

### Text Detection
- **DBNet++**: ResNet backbone + FPN + Differentiable Binarization
- **YOLO**: YOLOv8n/s for word/line detection

### Text Recognition  
- **ABINet**: Vision-Language fusion with 7,000+ character support
- **SATRN**: Shallow CNN + Transformer decoder

### Layout Detection
- **YOLO**: Fast inference with DocLayNet labels
- **DETR**: Transformer-based detection

### Table Recognition
- **TATR**: Table structure + cell detection + header recognition

## Training

### Data Preparation

```bash
# Register dataset paths
export DATASET_SYNTHDOG_JA_PATH=/path/to/synthdog
export DATASET_DOCLAYNET_PATH=/path/to/doclaynet

# Or use Python API
from datasets.registry import registry
registry.register_path('synthdog_ja', '/path/to/data')
```

### Model Training

```bash
# Train specific models
make train-det    # Text detection
make train-rec    # Text recognition
make train-layout # Layout detection
make train-table  # Table recognition

# Train LoRA adapter
make train-lora
```

### Evaluation

```bash
# Run all evaluations
make eval-all

# Specific evaluations
python -m src.eval_jsquad
python -m src.eval_docqa
python -m src.eval_jsonextract
```

## LLM Integration

### Harmony Format

```python
# Low reasoning
prompt = "[文書] 請求書... [質問] 金額は？"
response = "1,234,567円"

# Medium reasoning  
prompt = "[文書] 契約書... [質問] 契約期間を説明して"
response = "契約期間は2025年4月1日から2026年3月31日までの1年間です。"

# High reasoning
prompt = "[文書] 財務諸表... [質問] 財務状況を分析して"
response = "売上高は前年比15%増加し..."
```

### vLLM Deployment

```bash
# Start vLLM server
make serve

# Test API
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-docja",
    "prompt": "[文書] ... [質問] ...",
    "max_tokens": 512
  }'
```

### Ollama (local) – optional

```bash
ollama serve &
ollama pull gptoss-20b
export DOCJA_LLM_PROVIDER=ollama
export DOCJA_OLLAMA_ENDPOINT=http://localhost:11434
export DOCJA_OLLAMA_MODEL=gptoss-20b
```

## Bullseye Integration

This project ships an upstream‑compat wrapper called "bullseye". It can load local upstream sources (`bullseye/src`) or HF‑hosted weights and normalizes `metadata.providers` to `bullseye-*`.

- Required/recommended environment variables (examples)
  - `export DOCJA_BULLSEYE_LOCAL_DIR=$PWD/bullseye/src`
  - `export DOCJA_PROVIDER_ALIAS_LABEL=bullseye`
  - `export DOCJA_DET_PROVIDER=bullseye`
  - `export DOCJA_REC_PROVIDER=bullseye`
  - `export DOCJA_LAYOUT_PROVIDER=bullseye`
  - `export DOCJA_TABLE_PROVIDER=bullseye`
  - HF weights (for evaluation): `export HF_TOKEN=…`
    - `export DOCJA_BULLSEYE_DET_REPO=Ryousukee/bullseye-dbnet`
    - `export DOCJA_BULLSEYE_REC_REPO=Ryousukee/bullseye-recparseq`
    - `export DOCJA_BULLSEYE_LAYOUT_REPO=Ryousukee/bullseye-layoutrtdetrv`
    - `export DOCJA_BULLSEYE_TABLE_REPO=Ryousukee/bullseye-tablertdetrv`
    - Explicit recognizer name: `export DOCJA_BULLSEYE_REC_NAME=parseq|parseqv2`

- Logging labels are unified as `bullseye.*` (duplicate handlers removed).

- Note on table structure:
  - Bullseye's table module expects layout `table_boxes`. This pipeline extracts and forwards them automatically; no extra settings are required.

## DI API v1 (Unified Doc JSON / Flow / Gantt)

Endpoints (excerpt)
- `POST /v1/di/analyze`: Run OCR/layout/table/reading‑order → Unified Doc JSON (or return artifacts)
- `POST /v1/di/reprocess`: Re‑run failed/low‑confidence pages (future extension)
- `GET /v1/di/jobs/{id}`: Page‑level status/metrics/overlays (future extension)

Request example
```bash
curl -X POST http://localhost:8001/v1/di/analyze \
  -H 'x-api-key: $DOCJA_API_KEY' \
  -F file=@data/samples/sample.pdf \
  -F 'options={
    "output_format":"json",
    "detect_layout":true,
    "detect_tables":true,
    "extract_reading_order":true,
    "with_llm":false,
    "lite":true,
    "vis":false
  }'
```

Response (JSON output_format)
```json
{
  "filename": "sample.pdf",
  "pages": [ {"page_num":1, "text_blocks":[], "layout_elements":[], "tables":[]} ],
  "metadata": { "providers": {"detector":"bullseye-dbnet:..."} }
}
```

## Configuration

Key configuration files:
- `configs/det_dbnet.yaml`: Text detection settings
- `configs/rec_abinet.yaml`: Text recognition settings  
- `configs/layout_yolo.yaml`: Layout detection settings
- `configs/table_tatr.yaml`: Table recognition settings
- `configs/training_lora.yaml`: LoRA training settings
- `configs/vllm.yaml`: vLLM serving settings

### Environment variables (compact)

- Core
  - `DOCJA_API_KEY` (required in production)
  - `DOCJA_LLM_PROVIDER`=`ollama|gemma3|gptoss`
  - Per provider: `DOCJA_OLLAMA_ENDPOINT/MODEL` or `DOCJA_LLM_ENDPOINT/MODEL`
  - `DOCJA_LLM_LANG`=`ja|en` (internal prompts in English, responses in Japanese recommended)
- Runtime flags (optional)
  - `DOCJA_READING_ORDER_SIMPLE=1` (stability for large pages)
  - `DOCJA_VIS_PROFILE=clean|debug|raw` (UI: clean recommended)
  - `DOCJA_NO_INTERNAL_FALLBACK=1`, `DOCJA_NO_HF=1`
- Cache/Temp (optional, storage management)
  - `UV_CACHE_DIR=/mnt/uv-cache`
  - `XDG_CACHE_HOME=/mnt/hf-cache`
  - `TMPDIR=/mnt/tmp`

See `.env.sample` for visualization knobs, Flow/Gantt tuning, and upstream bullseye settings.

## License

Licensed under the Apache License, Version 2.0. See the LICENSE file for details.

## Citation

```bibtex
@software{docja2025,
  title={DocJA: Japanese Document AI System},
  author={Your Organization},
  year={2025},
  url={https://github.com/engkimo/bullseye.git}
}
```
