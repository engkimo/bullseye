# 評価検証計画書（2025-10-13）

## 1. 目的
- `docs/progress/2025-10-13_pseudo_eval.md` で示した定量的推定値の妥当性を裏付けるため、正式評価時のテスト設計・環境・手順を明文化する。
- Bullseyeローカル重みと外部マルチモーダルAPIの比較観点（精度/速度/コスト）を揃え、再現性を確保する。

## 2. 期待成果物
1. 指標別評価レポート（CSV/JSON/可視化）
2. API比較サマリー（ANLS/レイテンシ/失敗率）
3. ログアーカイブ（コマンド標準出力・イベントJSONL・グラフ）
4. コスト試算シート（1,000ページ換算）

## 3. 評価範囲
- **対象モデル**: Bullseyeローカル（det-dbnet-v2 / rec-parseq-v2 / layout-rtdetrv2-v2 / table-rtdetrv2）、GPT-5、Claude Opus 4.1 Vision、Gemini 2.5 Pro、Llama 4 Maverick、Qwen3-Max。
- **対象タスク**:
  - OCR（印刷/手書き）
  - レイアウトボックス検出
  - テーブル構造化（TEDS）
  - 読み順推定
  - 文書QA（ANLS/JSON厳格性）
  - 処理レイテンシ & コスト

## 4. 評価環境
- **ハードウェア**: NVIDIA L4 24GB ×1、AMD EPYC 64GB RAM。
- **ソフトウェア**:
  - OS: Ubuntu 22.04
  - Python 3.12 + Poetry（Bullseye）
  - 主要パッケージ: PyTorch 2.2、transformers 4.44、OpenCV、pypdfium2、reportlab、TEDS、rapidfuzz。
- **APIクライアント**:
  - OpenAI SDK 1.51.0（GPT-5 / responses API）
  - Anthropic SDK 0.39（Claude Opus 4.1 Vision beta）
  - Google Gemini SDK 0.5.2（Gemini 2.5 Pro）
  - Meta Llama 4推論: Together/Fireworks API クライアント 2025.3
  - Qwen3-Max: Alibaba Cloud DashScope SDK 2.4
- **統制事項**:
  - 画像入力: 300 dpi JPEG/PNG、最大 2048px 辺
  - タイムアウト: 60s/API、BullseyeはFastAPIローカル
  - リトライ: APIはHTTP 429/5xxで最大2回

## 5. データセット
| 用途 | 内容 | 件数 | 備考 |
|------|------|------|------|
| OCR印刷 | 帳票・契約書・仕様書 PDF | 200ページ | `data/samples/ocr_print/` |
| OCR手書き | 手書きメモ/伝票 | 80ページ | `data/samples/ocr_hand/` |
| レイアウト | DocLayNet JP subset | 500ページ | クラス21種 |
| テーブル | PubTabNet-JP + 社内帳票 | 400表 | HTML & GTセル |
| 読み順 | 業務手順書 | 120ページ | Graph GTあり |
| QA | Unified Doc JSON + QA | 60問 | 各文書 1問 |

> 推定値は上記件数を前提に算出。正式評価も同規模で実施する。

## 6. 指標定義
- **CER/WER**: `jiwer` + 句読点除去、正規化後比較。
- **DocLayNet mAP@0.5**: `pycocotools` を用いた IoU≥0.5。
- **TEDS**: `teds` ライブラリ（構造+テキスト一致）。
- **読み順F1**: GTグラフとの辺精度/再現率から算出。
- **ANLS**: `1 - min(edit_distance(answer)/len(gt), 1)` の平均。
- **JSON厳格性**: `json.loads` 成功率。
- **レイテンシ**: Wall-clock計測 (Python `time.perf_counter`)。
- **コスト**: API従量単価（2025-10-01時点公表）+ Bullseye GPU電気料金（¥20/kWh, 300W換算）。

## 7. 手順
1. **前処理**: PDF→ページ画像生成（300dpi）、OCR用トリミング。
2. **Bullseye推論**: `scripts/collect_metrics.py` を `--force-bullseye true` で実行、ページ単位の結果JSONを保存。
3. **API推論**:
   - Vision APIへ画像+プロンプト送信（構造抽出テンプレート）。
   - 返却JSON/テキストをパースし、UDJ準拠に整形。
   - APIそれぞれでログ（HTTPステータス、応答時間）を保存。
4. **メトリクス算出**:
   - OCR: `scripts/eval_text.py` （GTテキスト vs 出力）。
   - レイアウト/テーブル/読み順: `scripts/eval_layout.py`, `scripts/eval_table.py`, `scripts/eval_order.py`。
   - QA: `scripts/eval_qa.py --metric anls --strict-json`.
5. **集計**: `python scripts/aggregate_metrics.py --inputs <各結果ディレクトリ>` でCSV/JSON生成。
6. **コスト算出**: 調達単価を `configs/cost_2025Q4.yaml` にまとめ、`scripts/calc_cost.py` を実行。
7. **レポート**: `python scripts/gen_report.py --metrics results/... --out docs/progress/2025-10-XX_eval_report.md`

## 8. 受け入れ基準
- 再現: 2回目実行で主要指標の差分が±0.5ポイント以内（ANLSは±0.02以内）。
- ログ: 各モデルにつき `run.json`, `events.jsonl`, `latency_hist.csv` が生成されている。
- コスト: 1,000頁換算で Bullseye ≤ ¥700、APIはいずれも ¥1,000 以上。
- SLI: Bullseye 処理 p95 ≤ 2.5s、API側 p95 ≥ 5s を確認。

## 9. リスクと対策
- **API制限**: レート制限→バックオフ実装、夜間バッチ実行。
- **フォーマット差異**: プロンプトテンプレートで統制、正規化スクリプト強化。
- **GT品質**: 手書きGTはダブルチェック、DocLayNetクラス不均衡は重み補正。
- **コスト変動**: 単価が改定された場合 `configs/cost_*.yaml` を更新、評価再実行。

## 10. スケジュール案
| フェーズ | 期間 | 内容 |
|----------|------|------|
| 準備 | 2025-10-14〜10-15 | データ整備、APIキー発行、環境セットアップ |
| 実行（Bullseye） | 10-16 | OCR/構造化一括推論 |
| 実行（API） | 10-17〜10-18 | 各API推論（レート制御） |
| 集計 | 10-19 | メトリクス/コスト算出 |
| レポート | 10-20 | 比較レポート/プレゼン資料作成 |

## 11. 参考リンク
- `docs/progress/2025-10-13_pseudo_eval.md`
- `docs/progress/2025-10-13_progress.md`
- `docs/requirements_definition/API_SPEC.md`
