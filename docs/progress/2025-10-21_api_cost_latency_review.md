# 主要マルチモーダルAPIの料金・レイテンシ検証（2025-10-21）

## 1. 前提とシナリオ
- 対象: GPT-5、Claude Opus 4.1、Gemini 2.5 Pro、Llama 4 Maverick、Qwen3-Max、およびBullseyeローカル重み。
- 目的: 1,000ページバッチ処理を想定したコスト仮定とレイテンシ想定が2025年10月時点の公開情報と整合しているかを確認する。
- 為替レート: $1 = ¥150.8（2025-10-20時点のミッドレート）。citeturn7search0
- 入力トークン仮定: 1ページあたり1,800トークン（レイアウト指示+OCRテキスト）。
- 出力トークン仮定: 1ページあたり900トークン（構造化JSON+推論ログ）。
- 画像課金: Gemini 2.5 Proのみ1ページ1画像（$0.005/枚）を加算。citeturn0search0
- Llama 4 Maverickは300dpi PDFベースのバッチをBlackwell B200 8GPUノード（$3.75/GPU-hrのオンデマンドレンタル）で処理し、NVIDIAが公表した1,000 tokens/s帯の推論性能を参照する。citeturn1search1turn5search3turn9search1

## 2. 料金スナップショット

| モデル | 入力単価 (USD/Mtok) | 出力単価 (USD/Mtok) | 追加課金 | 1,000頁トータルUSD | 1,000頁トータルJPY |
|--------|----------------------|----------------------|-----------|--------------------|--------------------|
| Bullseye Local | 電力+減価償却で約¥580想定 | - | - | $3.84相当 | **¥580** |
| GPT-5 | $1.25 | $10.00 | - | $11.25 | **¥1,700** |
| Claude Opus 4.1 | $15.00 | $75.00 | - | $94.50 | **¥14,300** |
| Gemini 2.5 Pro | $1.25 | $10.00 | 画像 $0.005/頁 | $16.25 | **¥2,460** |
| Llama 4 Maverick | GPUレンタル換算（0.5時間×$30/h） | - | - | $15.00 | **¥2,300** |
| Qwen3-Max | $1.20 | $6.00 | - | $7.56 | **¥1,140** |

- GPT-5単価: OpenAI公式発表。citeturn0search3
- Claude Opus 4.1単価: Anthropic公式ドキュメント。citeturn0search2
- Gemini 2.5 Pro単価: Google発表（≤200Kトークン帯）+画像課金。citeturn0search0turn0search1
- Llama 4 Maverick: ModelBooth/Together価格では$0.20〜$0.60/MTokの事例もあるが、企業用途の安定SLAを想定しBlackwell B200 8GPUノードを0.5時間利用した場合で算出（Genesis Cloud on-demand $3.75/GPU-hr）。citeturn3search5turn9search1turn5search3
- Qwen3-Max単価: Alibaba Model Studioの階段制料金。citeturn4search1

> メモ: Bullseyeの¥580はL4 24GBを0.3kW・p95 2.2秒/頁で稼働、電力単価¥30/kWh＋年間保守費を1,000頁あたり¥500で按分した値。

## 3. レイテンシ指標

| モデル | 公開メトリクス | 補足 |
|--------|----------------|------|
| GPT-5 | p50 450ms / p90 1.2s / p99 3.5s、初回トークン<100ms。citeturn1search1 | Priority Tier SLA: 99%で50tok/s以上。citeturn1search0 |
| Claude Opus 4.1 | 初回トークン約1.8s、出力速度45〜65 tok/s。citeturn6search2turn6search8 | - |
| Gemini 2.5 Pro | フォーラム報告で平均35s、p99 8分の遅延事例。citeturn1search2 | 大規模プロンプト（100K〜500K）では10分超の事例。citeturn1search4 |
| Llama 4 Maverick | Groqベンチ: 145 tok/s、TTFT 0.6ms。citeturn5search0 | NVIDIA Blackwell最適化で1,000 tok/s超の報告。citeturn5search3 |
| Qwen3-Max | Vals.ai平均レイテンシ223s、AI Primerで2,804sの長尾報告。citeturn8search0turn8search1 |

## 4. スコアボードへの反映
- 旧コスト値（2025-10-13版）: GPT-4o miniベースの想定に依存しており、最新料金とは乖離。
- 上記計算に合わせ、`docs/progress/2025-10-13_pseudo_eval.md`のコスト列と`results/pseudo_eval`のモックログを更新。
- Bullseye優位性: Claude/Gemini/GPT-5に対しては依然として低コスト・低レイテンシ。Llama 4 Maverickは自前推論の固定費次第で逆転可能、Qwen3-Maxは料金優位だが長尾レイテンシが課題。

## 5. リスクと示唆
1. **Claude Opus 4.1のコスト増大**: 出力トークン課金が支配的（¥14k/千頁）。ハイブリッド構成時は回答をClaudeに限定し、構造化はBullseye側で完結させる設計が合理的。
2. **Gemini 2.5 Proの尾部遅延**: QA用途の再試行・キュー制御を組み込む必要。成果物の納期SLOが厳しい場合はPriorityサポート枠の契約が前提。
3. **Llama 4 MaverickのTCO**: トークン課金モデルでは廉価だが、Blackwell級GPUを確保する場合は¥2,000〜¥5,000/千頁レンジ。推論時間短縮と稼働率向上でペイするかを要検証。
4. **Qwen3-MaxのTail Latency**: 200秒超の中央値はプロダクション投入時のSLO違反リスク。分散キュー＋キャンセル制御が必須。
5. **ドキュメント特化モデルの進展**: 最新研究では高解像度ページを少数トークンへ圧縮しながらレイテンシを抑制する手法が登場しており、Bullseyeの構造化改善方針を継続的に見直す必要がある。
