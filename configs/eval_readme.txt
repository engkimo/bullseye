Usage examples for local evaluation paths (actual files present in this repo):

1) Layout mAP (COCO-style) using synthetic JA layout data
- Config: configs/eval_layout_synth.json
- Run:    python3 -m src.eval_layout --config configs/eval_layout_synth.json --output results/layout_metrics.json
- Notes:  Uses data/synth_layout_ja/images + annotations.json (DocLayNet-like classes).

2) Doc QA (ANLS/EM/F1) on local samples
- Place QA file next to each input as <stem>.qa.jsonl
  Example already provided: data/samples/sample.qa.jsonl
- Run: python3 scripts/collect_metrics.py --data-root data/samples --out results/metrics --cli bullseye --format json --with-llm true --eval-qa true

3) Tables TEDS (optional)
- Provide GT HTML next to each input with the same stem, e.g., data/table/images/table_000001.gt.html
- Enable TEDS: add --teds true to scripts/collect_metrics.py
- You can also provide <stem>.tables.json as [{"html": "<table>...</table>"}, ...].

