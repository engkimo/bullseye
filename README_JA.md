## BullsEye - 日本語 Document AI 統合システム

[日本語版] | [English](README.md)

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)

実務レベルの日本語文書処理を実現する統合AIシステムです。OCR、レイアウト解析、表構造認識、読み順推定、LLM連携を提供します。

**上流モデル群と互換性のある独自実装（bullseye統合）**

### 特長
- 文字検出: DBNet++/YOLO による高精度検出
- 文字認識: ABINet/SATRN（7,000+字、縦書き/手書き対応）
- レイアウト解析: DocLayNet互換（YOLO/DETR切替）
- 表構造認識: TATR系（セルOCR連携、HTML/MD出力）
- 読み順推定: グラフ+規則ハイブリッド
- LLM連携: gpt-oss-20B によるQA/要約/抽出
- エクスポート: HTML/Markdown/JSON/CSV/検索可能PDF

### 必要要件
- Python 3.12（3.10+互換）
- CUDA 12.1+（GPU推論/学習）
- ハードウェア（最小 → 推奨）
  - CPU: 8コア → 16コア
  - メモリ: 32GB → 64GB
  - GPU: 16GB → L4 24GB（p95 ≤ 2秒/頁の目標）
  - ディスク: 空き100GB → 300GB

### クイックスタート
```bash
# セットアップ（uv推奨）
make setup-uv
source .venv/bin/activate

# API起動
uvicorn src.api:app --host 0.0.0.0 --port 8001

# 最小実行（Unified Doc JSON）
curl -X POST http://localhost:8001/v1/di/analyze \
  -F file=@data/samples/sample.pdf \
  -F 'options={"output_format":"json","detect_layout":true,"detect_tables":true,"extract_reading_order":true,"lite":true}' | jq '.'
```

### Backend/Frontend 起動

#### Backend (FastAPI)
```bash
# 最小ENV
export DOCJA_API_KEY=dev-123
export DOCJA_LLM_PROVIDER=ollama            # gemma3|gptoss も可
export DOCJA_OLLAMA_ENDPOINT=http://localhost:11434
export DOCJA_OLLAMA_MODEL=gptoss-20b
export DOCJA_LLM_LANG=ja                    # 推論は英語/応答は日本語
export DOCJA_LLM_TIMEOUT=45

make setup-uv && source .venv/bin/activate
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

#### Frontend (Vite)
```bash
cd front
npm install
npm run dev -- --host 0.0.0.0
# 例: http://<HOST>:5173
```

### 環境変数（コンパクト）
- Core: `DOCJA_API_KEY`, `DOCJA_LLM_PROVIDER`, `DOCJA_OLLAMA_ENDPOINT/MODEL` または `DOCJA_LLM_ENDPOINT/MODEL`, `DOCJA_LLM_LANG`
- Flags: `DOCJA_READING_ORDER_SIMPLE=1`, `DOCJA_FORCE_YOMITOKU=1`, `DOCJA_NO_INTERNAL_FALLBACK=1`, `DOCJA_NO_HF=1`
- Cache: `UV_CACHE_DIR`, `XDG_CACHE_HOME`, `TMPDIR`

詳細は `docs/requirements_definition/14_gemma3_yomitoku_integration.md` と `.env.sample` を参照してください。

### ライセンス
Apache-2.0。詳細は LICENSE を参照してください。

