# DocJA Front (React)

- 開発サーバ: Vite + React（/front）
- バックエンド: DocJA API (`/v1/di/analyze`, `/v1/di/chat`)、Ollama (`/api/chat`)

## 使い方

1) 依存導入
```
cd front
npm install
```

2) 開発サーバ起動（Vite）
```
npm run dev
```
- ブラウザ: http://localhost:5173
- プロキシ:
  - `/docja/*` → `http://localhost:8001/*`（`vite.config.js`）
  - `/ollama/*` → `http://localhost:11434/*`

3) DocJA 側（別ターミナル）
```
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

4) Ollama 側（ローカル）
```
ollama serve &
# あなたのモデルタグに合わせる
export DOCJA_OLLAMA_MODEL=gpt-oss:20b
```

5) 画面フロー
- ファイルをアップロード → 解析する（/v1/di/analyze with vis=true）
- visサムネ表示（`vis_previews`）
- チャット欄で Provider=ollama を選択し会話開始

## 環境
- `vite.config.js` のプロキシターゲットは環境変数 `VITE_DOCJA_BASE` / `VITE_OLLAMA_BASE` で上書き可

