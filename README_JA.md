## BullsEye - 日本語 Document AI 統合システム

[日本語版] | [English](README.md)

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB.svg)

実務レベルの日本語文書処理を実現する統合AIシステムです。OCR、レイアウト解析、表構造認識、読み順推定、LLM連携を提供します。

独自実装で上流モデル群との互換性を確保（bullseye 統合）。

## 特長

- 文字検出: 高精度な文字領域検出（DBNet++ / YOLO）
- 文字認識: ABINet / SATRN（7,000+日本語、縦書き/手書き対応）
- レイアウト解析: DocLayNet互換（YOLO/DETR切替）
- 表構造認識: TATR系（セルOCR連携）
- 読み順推定: グラフ+規則ハイブリッド
- LLM連携: gpt‑oss‑20B によるQA/要約/抽出
- エクスポート: HTML/Markdown/JSON/CSV/検索可能PDF

## 必要要件

- Python 3.12（3.10+互換）
- CUDA 12.1+（GPU推論/学習）
- AWS CLI（クラウド運用時）
- ハードウェア（最小 → 推奨）
  - CPU: 8コア → 16コア
  - メモリ: 32GB → 64GB
  - GPU: 16GB → L4 24GB（p95 ≤ 2秒/頁 目標）
  - ディスク: 空き100GB → 300GB

## クイックスタート

```bash
# セットアップ（uv推奨）
make setup-uv
source .venv/bin/activate

# API 起動
uvicorn src.api:app --host 0.0.0.0 --port 8001

# 最小実行（Unified Doc JSON）
curl -X POST http://localhost:8001/v1/di/analyze \
  -F file=@data/samples/sample.pdf \
  -F 'options={"output_format":"json","detect_layout":true,"detect_tables":true,"extract_reading_order":true,"lite":true}' | jq '.'

# CLI 例
python -m src.cli data/samples/sample.pdf -f md -o results \
  --layout --table --reading-order --lite --vis

# 学習
make train-all

# 評価
make eval-all
```

## Backend/Frontend: 起動

最小の環境変数とコマンドのみを記載。詳細なチューニングは docs の Advanced ENV を参照してください。

### Backend (FastAPI)

```bash
# 最小ENV
export DOCJA_API_KEY=dev-123                # 本番は強ランダムを使用
export DOCJA_LLM_PROVIDER=ollama            # または gemma3|gptoss
export DOCJA_OLLAMA_ENDPOINT=http://localhost:11434
export DOCJA_OLLAMA_MODEL=gptoss-20b        # 例
export DOCJA_LLM_LANG=ja                    # 内部推論=en, 応答=ja
export DOCJA_LLM_TIMEOUT=45

# bullseye（上流）プロバイダとリポジトリ
export DOCJA_PROVIDER_ALIAS_LABEL=bullseye
export DOCJA_DET_PROVIDER=bullseye
export DOCJA_REC_PROVIDER=bullseye
export DOCJA_LAYOUT_PROVIDER=bullseye
export DOCJA_TABLE_PROVIDER=bullseye

# bullseye の既定HFリポジトリ
export DOCJA_BULLSEYE_DET_REPO=Ryousukee/bullseye-dbnet
export DOCJA_BULLSEYE_REC_REPO=Ryousukee/bullseye-recparseq
export DOCJA_BULLSEYE_LAYOUT_REPO=Ryousukee/bullseye-layoutrtdetrv
export DOCJA_BULLSEYE_TABLE_REPO=Ryousukee/bullseye-tablertdetrv
export DOCJA_BULLSEYE_LOCAL_DIR="$PWD/bullseye/src"

# (uv env)
make setup-uv && source .venv/bin/activate

# API 起動
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

### Frontend (Vite dev server)

```bash
cd front
npm install
npm run dev -- --host 0.0.0.0
# 例: http://<HOST>:5173
```

## インストール

### ローカルセットアップ

```bash
git clone https://github.com/engkimo/bullseye.git
cd docja
make setup-uv
source .venv/bin/activate
```

### AWS デプロイ

```bash
# AWS 資格情報
export AWS_PROFILE=your-profile
export KEY_NAME=your-key

# インフラ適用
make infra-plan
make infra-apply

# 接続とセットアップ
make ssh
bash scripts/bootstrap.sh
```

## Usage
## リファクタリングと安定性

- 大規模なディレクトリリファクタを試行しましたが、上流アダプタや解析パスを壊すため即時ロールバック。安定性を優先します。
- 以後は段階的かつ安全なリファクタ方針。公開API/モジュールに破壊的変更は、シムとスモークテストを伴わない限り行いません。

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

## モデルアーキテクチャ

### Text Detection
- DBNet++: ResNet + FPN + Differentiable Binarization
- YOLO: 単語/行検出（高速）

### Text Recognition
- ABINet: 7,000+文字対応、縦書き/手書き
- SATRN: 軽量CNN + Transformer

### Layout Detection
- YOLO: 高速、DocLayNetラベル
- DETR: Transformerベース検出

### Table Recognition
- TATR: 構造復元 + セル/ヘッダ + HTML/MD

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

## LLM 統合

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

### vLLM デプロイ

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

### Ollama（ローカル・任意）

```bash
ollama serve &
ollama pull gptoss-20b
export DOCJA_LLM_PROVIDER=ollama
export DOCJA_OLLAMA_ENDPOINT=http://localhost:11434
export DOCJA_OLLAMA_MODEL=gptoss-20b
```

## Bullseye 統合

本プロジェクトは「bullseye」ラッパを同梱します。ローカル上流（`bullseye/src`）やHFの重みを読み込み、`metadata.providers` を `bullseye-*` に正規化します。

- 必須/推奨環境変数（例）
  - `export DOCJA_BULLSEYE_LOCAL_DIR=$PWD/bullseye/src`
  - `export DOCJA_PROVIDER_ALIAS_LABEL=bullseye`
  - `export DOCJA_DET_PROVIDER=bullseye`
  - `export DOCJA_REC_PROVIDER=bullseye`
  - `export DOCJA_LAYOUT_PROVIDER=bullseye`
  - `export DOCJA_TABLE_PROVIDER=bullseye`
  - HF重み（評価用途）: `export HF_TOKEN=…`
    - `export DOCJA_BULLSEYE_DET_REPO=Ryousukee/bullseye-dbnet`
    - `export DOCJA_BULLSEYE_REC_REPO=Ryousukee/bullseye-recparseq`
    - `export DOCJA_BULLSEYE_LAYOUT_REPO=Ryousukee/bullseye-layoutrtdetrv`
    - `export DOCJA_BULLSEYE_TABLE_REPO=Ryousukee/bullseye-tablertdetrv`
    - 認識器名の明示: `export DOCJA_BULLSEYE_REC_NAME=parseq|parseqv2`

- ログ表記は `bullseye.*` に統一（重複ハンドラ除去）

- 表構造の注意:
  - bullseye表モジュールはレイアウトの `table_boxes` を前提とします。本パイプラインが自動抽出して渡すため追加設定は不要です。

## DI API v1（Unified Doc JSON / Flow / Gantt）

エンドポイント（抜粋）
- `POST /v1/di/analyze`: OCR/レイアウト/表/読み順 → Unified Doc JSON（またはアーティファクト返却）
- `POST /v1/di/reprocess`: 失敗/低信頼ページの再処理（将来拡張）
- `GET /v1/di/jobs/{id}`: ページ粒度の状態/メトリクス/オーバレイ（将来拡張）

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

## 設定

主要設定ファイル:
- `configs/det_dbnet.yaml`: Text detection
- `configs/rec_abinet.yaml`: Text recognition
- `configs/layout_yolo.yaml`: Layout detection
- `configs/table_tatr.yaml`: Table recognition
- `configs/training_lora.yaml`: LoRA training
- `configs/vllm.yaml`: vLLM serving

### 環境変数（コンパクト版）

- Core
  - `DOCJA_API_KEY`（本番必須）
  - `DOCJA_LLM_PROVIDER`=`ollama|gemma3|gptoss`
  - プロバイダ別: `DOCJA_OLLAMA_ENDPOINT/MODEL` または `DOCJA_LLM_ENDPOINT/MODEL`
  - `DOCJA_LLM_LANG`=`ja|en`（内部プロンプト=英語、応答=日本語 推奨）
- Runtime flags（任意）
  - `DOCJA_READING_ORDER_SIMPLE=1`（大規模ページの安定化）
  - `DOCJA_FORCE_YOMITOKU=1`, `DOCJA_NO_INTERNAL_FALLBACK=1`, `DOCJA_NO_HF=1`
- Cache/Temp（容量対策・任意）
  - `UV_CACHE_DIR=/mnt/uv-cache`
  - `XDG_CACHE_HOME=/mnt/hf-cache`
  - `TMPDIR=/mnt/tmp`

詳細な可視化・Flow/Gantt 調整や bullseye 上流設定は `docs/requirements_definition/14_gemma3_yomitoku_integration.md` と `.env.sample` を参照してください。

## ライセンス

Apache-2.0。詳細は LICENSE を参照してください。

## 引用

```bibtex
@software{docja2025,
  title={DocJA: Japanese Document AI System},
  author={Your Organization},
  year={2025},
  url={https://github.com/engkimo/bullseye.git}
}
```
