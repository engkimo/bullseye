# BullsEye - 日本語 Document AI Integration System

実務レベルの日本語文書処理を実現する統合AIシステム。OCR、レイアウト解析、表構造認識、読み順推定、LLM連携を提供。

<img width="3024" height="1702" alt="image" src="https://github.com/user-attachments/assets/2ee70864-ceaf-4969-bdb9-0888a76788c5" />

**上流モデル群と互換性のある独自実装システム（bullseye統合）**

## Features

- **Text Detection**: DBNet++ / YOLO による高精度文字領域検出
- **Text Recognition**: ABINet / SATRN による7,000字以上対応の文字認識（縦書き・手書き対応）
- **Layout Detection**: DocLayNet互換のレイアウト解析（YOLO/DETR切替可能）
- **Table Recognition**: TATR互換の表構造認識とセル抽出
- **Reading Order**: グラフベースの読み順推定
- **LLM Integration**: gpt-oss-20B による文書理解・QA・情報抽出
- **Export**: HTML/Markdown/JSON/CSV/サーチャブルPDF出力

## Requirements

- Python 3.10+
- CUDA 11.8+ (GPU training/inference)
- AWS CLI configured (for cloud deployment)
- 24GB+ GPU Memory (recommended: NVIDIA L4)

## Quick Start

```bash
# Setup
make setup-local
source venv/bin/activate

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

## Backend/Frontend: Run (ENV presets)

以下はローカル実行のための環境変数プリセットと起動手順です（Ollama と vLLM いずれかが利用可能）。必要に応じて値を調整してください。

### Backend (FastAPI)

```bash
# 言語・LLM関連
export DOCJA_LLM_LANG=ja
export DOCJA_LLM_TIMEOUT=45
export DOCJA_LLM_ENDPOINT=http://localhost:8000/v1/completions
export DOCJA_LLM_MODEL=openai/gpt-oss-20b

# Ollama プロバイダ
export DOCJA_LLM_PROVIDER=ollama
export DOCJA_OLLAMA_ENDPOINT=http://localhost:11434
export DOCJA_OLLAMA_MODEL=gpt-oss:20b
export DOCJA_PROVIDER_ALIAS_LABEL=bullseye

# 読み順（安定化）
export DOCJA_READING_ORDER_SIMPLE=1

# PATH
export PATH="$HOME/.local/bin:$PATH"

# Hugging Face（必要に応じて設定）
export HF_TOKEN=<your_hf_token>
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Gantt 解析
export DOCJA_GANTT_COL_THR=0.12
export DOCJA_GANTT_HEADER_UP=120
export DOCJA_GANTT_HEADER_DN=60
export DOCJA_GANTT_MIN_COLS=14
export DOCJA_GANTT_HMIN=5
export DOCJA_GANTT_HMAX=30
export DOCJA_GANTT_SMIN=60
export DOCJA_GANTT_VMIN=120
export DOCJA_GANTT_FORCE_CELL=1
export DOCJA_GANTT_VK=11
export DOCJA_GANTT_HK=9
export DOCJA_GANTT_CELL_OCC_THR=12
export DOCJA_GANTT_CELL_OCC_RATE=0.03
export DOCJA_GANTT_DRAW_CELL_LINKS=0
export DOCJA_GANTT_CELL_SMIN=40
export DOCJA_GANTT_BAR_HMIN=5
export DOCJA_GANTT_BAR_HMAX=45
export DOCJA_GANTT_PALETTE_DIST2=1400

# bullseye（上流ラッパを利用する場合）
export DOCJA_DET_PROVIDER=bullseye
export DOCJA_REC_PROVIDER=bullseye
export DOCJA_LAYOUT_PROVIDER=bullseye
export DOCJA_TABLE_PROVIDER=bullseye
export DOCJA_BULLSEYE_LOCAL_DIR=$PWD/bullseye/src
export DOCJA_BULLSEYE_DET_REPO=Ryousukee/bullseye-dbnet
export DOCJA_BULLSEYE_REC_REPO=Ryousukee/bullseye-recparseq
export DOCJA_BULLSEYE_LAYOUT_REPO=Ryousukee/bullseye-layoutrtdetrv
export DOCJA_BULLSEYE_TABLE_REPO=Ryousukee/bullseye-tablertdetrv
export DOCJA_VIS_PROFILE=clean

# 仮想環境と依存
conda deactivate || true
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt --upgrade

# API 起動
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

### Frontend (Vite dev server)

```bash
cd front
npm install
npm run dev -- --host 0.0.0.0
# 例: http://<HOST>:5173 をブラウザで開く
```

### Batch Scripts (copy/paste)

バックエンド/フロントの起動をワンコマンド化したスクリプト例です。必要に応じてパスやトークンを編集してから実行してください。

```bash
# Save as: scripts/dev_backend.sh
#!/usr/bin/env bash
set -euo pipefail

# ===== ENV: LLM =====
export DOCJA_LLM_LANG="ja"
export DOCJA_LLM_TIMEOUT="45"
export DOCJA_LLM_ENDPOINT="http://localhost:8000/v1/completions"
export DOCJA_LLM_MODEL="openai/gpt-oss-20b"

# ===== ENV: Ollama =====
export DOCJA_LLM_PROVIDER="ollama"
export DOCJA_OLLAMA_ENDPOINT="http://localhost:11434"
export DOCJA_OLLAMA_MODEL="gpt-oss:20b"
export DOCJA_PROVIDER_ALIAS_LABEL="bullseye"

# ===== ENV: Reading Order / PATH =====
export DOCJA_READING_ORDER_SIMPLE="1"
export PATH="$HOME/.local/bin:$PATH"

# ===== ENV: HF (optional) =====
export HF_TOKEN="<YOUR_HF_TOKEN>"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ===== ENV: Gantt =====
export DOCJA_GANTT_COL_THR=0.12
export DOCJA_GANTT_HEADER_UP=120
export DOCJA_GANTT_HEADER_DN=60
export DOCJA_GANTT_MIN_COLS=14
export DOCJA_GANTT_HMIN=5
export DOCJA_GANTT_HMAX=30
export DOCJA_GANTT_SMIN=60
export DOCJA_GANTT_VMIN=120
export DOCJA_GANTT_FORCE_CELL=1
export DOCJA_GANTT_VK=11
export DOCJA_GANTT_HK=9
export DOCJA_GANTT_CELL_OCC_THR=12
export DOCJA_GANTT_CELL_OCC_RATE=0.03
export DOCJA_GANTT_DRAW_CELL_LINKS=0
export DOCJA_GANTT_CELL_SMIN=40
export DOCJA_GANTT_BAR_HMIN=5
export DOCJA_GANTT_BAR_HMAX=45
export DOCJA_GANTT_PALETTE_DIST2=1400

# ===== ENV: bullseye upstream (optional) =====
export DOCJA_DET_PROVIDER="bullseye"
export DOCJA_REC_PROVIDER="bullseye"
export DOCJA_LAYOUT_PROVIDER="bullseye"
export DOCJA_TABLE_PROVIDER="bullseye"
export DOCJA_BULLSEYE_LOCAL_DIR="$PWD/bullseye/src"
export DOCJA_BULLSEYE_DET_REPO="Ryousukee/bullseye-dbnet"
export DOCJA_BULLSEYE_REC_REPO="Ryousukee/bullseye-recparseq"
export DOCJA_BULLSEYE_LAYOUT_REPO="Ryousukee/bullseye-layoutrtdetrv"
export DOCJA_BULLSEYE_TABLE_REPO="Ryousukee/bullseye-tablertdetrv"
export DOCJA_VIS_PROFILE="clean"

# ===== Venv & deps =====
conda deactivate || true
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -r requirements.txt --upgrade
else
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip && pip install -r requirements.txt
fi

exec uvicorn src.api:app --host 0.0.0.0 --port 8001
```

```bash
# Save as: scripts/dev_frontend.sh
#!/usr/bin/env bash
set -euo pipefail
cd front
npm install
exec npm run dev -- --host 0.0.0.0
```

実行例:

```bash
bash scripts/dev_backend.sh
# 別ターミナル
bash scripts/dev_frontend.sh
```

## Installation

### Local Setup

```bash
git clone https://github.com/engkimo/bullseye.git
cd docja
make setup-local
source venv/bin/activate
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
# 環境変数またはCLIで bullseye プロバイダを指定可能
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

本プロジェクトは、上流互換ラッパを bullseye として同梱しています。ローカル上流（bullseye/src）またはHFのリポジトリから重みを取得し、`metadata.providers` を `bullseye-*` に正規化します。

- 必須/推奨環境変数（例）
  - `export DOCJA_BULLSEYE_LOCAL_DIR=$PWD/bullseye/src`
  - `export DOCJA_PROVIDER_ALIAS_LABEL=bullseye`
  - `export DOCJA_DET_PROVIDER=bullseye`
  - `export DOCJA_REC_PROVIDER=bullseye`
  - `export DOCJA_LAYOUT_PROVIDER=bullseye`
  - `export DOCJA_TABLE_PROVIDER=bullseye`
  - HF重み（評価用途）: `export HF_TOKEN=…`
    - `export DOCJA_BULLSEYE_DET_REPO=<user>/bullseye-dbnet`
    - `export DOCJA_BULLSEYE_REC_REPO=<user>/bullseye-recparseq`
    - `export DOCJA_BULLSEYE_LAYOUT_REPO=<user>/bullseye-layoutrtdetrv`
    - `export DOCJA_BULLSEYE_TABLE_REPO=<user>/bullseye-tablertdetrv`
    - 認識器名の明示: `export DOCJA_BULLSEYE_REC_NAME=parseq|parseqv2`

- ログ表記は `bullseye.*` で統一（重複ハンドラは除去）

- 表構造認識の注意:
  - bullseye表構造はレイアウトの table 領域（`table_boxes`）を前提とします。本パイプラインはレイアウト出力から自動抽出して渡すため、追加設定不要です。

## DI API v1（Unified Doc JSON / Flow / Gantt）

エンドポイント（抜粋）
- `POST /v1/di/analyze`: OCR/レイアウト/表/読み順 → Unified Doc JSON（またはアーティファクト返却）
- `POST /v1/di/reprocess`: 失敗/低信頼ページの再処理（将来拡張）
- `GET /v1/di/jobs/{id}`: ページ粒度の状態/メトリクス/オーバレイ参照（将来拡張）

リクエスト例
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

レスポンス（JSON指定時）
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

### Environment variables (抜粋)
- `DOCJA_API_KEY`: REST認証（必須推奨）
- `DOCJA_LLM_PROVIDER={ollama|gemma3|gptoss}` / `DOCJA_OLLAMA_*` / `DOCJA_LLM_ENDPOINT` など
- `DOCJA_READING_ORDER_SIMPLE=1`（大規模ページの安定化）
- `DOCJA_PROVIDER_ALIAS_LABEL=bullseye`（メタ表記統一）

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
