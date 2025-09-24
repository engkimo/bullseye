#!/bin/bash
set -e

echo "=== DocJA Evaluation Script ==="

# Activate virtual environment
source venv/bin/activate

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/eval_${TIMESTAMP}"
mkdir -p $RESULT_DIR

# Function to run evaluation and save results
run_eval() {
    local eval_name=$1
    local eval_cmd=$2
    
    echo "Running $eval_name evaluation..."
    eval $eval_cmd > "$RESULT_DIR/${eval_name}.json" 2> "$RESULT_DIR/${eval_name}.log"
    
    # Extract key metrics
    python -c "
import json
import sys
try:
    with open('$RESULT_DIR/${eval_name}.json') as f:
        data = json.load(f)
    print(f'$eval_name Results:')
    for k, v in data.items():
        if isinstance(v, (int, float)):
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
except Exception as e:
    print(f'Error reading results: {e}')
"
}

# Detection evaluation
echo "=== Text Detection Evaluation ==="
python -m src.eval_detection \
    --config configs/det_eval.yaml \
    --output "$RESULT_DIR/detection_metrics.json"

# Recognition evaluation  
echo "=== Text Recognition Evaluation ==="
python -m src.eval_recognition \
    --config configs/rec_eval.yaml \
    --output "$RESULT_DIR/recognition_metrics.json"

# Layout evaluation
echo "=== Layout Detection Evaluation ==="
python -m src.eval_layout \
    --config configs/layout_eval.yaml \
    --output "$RESULT_DIR/layout_metrics.json"

# Table evaluation
echo "=== Table Recognition Evaluation ==="
python -m src.eval_table \
    --config configs/table_eval.yaml \
    --output "$RESULT_DIR/table_metrics.json"

# LLM evaluations
echo "=== LLM Evaluations ==="

# JSQuAD
run_eval "jsquad" "python -m src.eval_jsquad --model weights/lora/adapter --base gpt-oss-20B"

# JaQuAD  
run_eval "jaquad" "python -m src.eval_jaquad --model weights/lora/adapter --base gpt-oss-20B"

# JDocQA
run_eval "jdocqa" "python -m src.eval_docqa --model weights/lora/adapter --base gpt-oss-20B"

# JSON Extraction
run_eval "json_extract" "python -m src.eval_jsonextract --model weights/lora/adapter --base gpt-oss-20B"

# Summary
run_eval "summary" "python -m src.eval_summary --model weights/lora/adapter --base gpt-oss-20B"

# Generate comparison report
echo "=== Generating Comparison Report ==="
python -c "
import json
import os
from pathlib import Path

result_dir = Path('$RESULT_DIR')
report = {}

# Load all evaluation results
for json_file in result_dir.glob('*.json'):
    try:
        with open(json_file) as f:
            data = json.load(f)
            report[json_file.stem] = data
    except:
        pass

# Save consolidated report
with open(result_dir / 'consolidated_report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# Generate markdown report
with open(result_dir / 'report.md', 'w') as f:
    f.write('# DocJA Evaluation Report\\n\\n')
    f.write(f'Generated: {Path(result_dir).name}\\n\\n')
    
    for eval_name, metrics in report.items():
        f.write(f'## {eval_name}\\n\\n')
        f.write('| Metric | Value |\\n')
        f.write('|--------|-------|\\n')
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                f.write(f'| {k} | {v:.4f}' if isinstance(v, float) else f'| {k} | {v}')
                f.write(' |\\n')
        f.write('\\n')

print(f'\\nEvaluation report saved to: {result_dir}/report.md')
"

# Compare with baseline if exists
if [ -d "results/baseline" ]; then
    echo "=== Comparing with Baseline ==="
    python -m src.compare_metrics \
        --baseline results/baseline \
        --current $RESULT_DIR \
        --output "$RESULT_DIR/comparison.json"
fi

# Gate thresholds (optional enforcement)
echo "=== Evaluation Gate Check ==="
CER_MAX=${CER_MAX:-0.90}
DET_HMEAN_MIN=${DET_HMEAN_MIN:-0.86}
LAY_MAP_5095_MIN=${LAY_MAP_5095_MIN:-0.72}
TAB_TEDS_MIN=${TAB_TEDS_MIN:-0.88}
ENFORCE=${GATE_ENFORCE:-0}

python -m scripts.gate_eval \
  --dir "$RESULT_DIR" \
  --cer-max "$CER_MAX" \
  --det-hmean-min "$DET_HMEAN_MIN" \
  --layout-map-5095-min "$LAY_MAP_5095_MIN" \
  --table-teds-min "$TAB_TEDS_MIN" \
  $( [ "$ENFORCE" = "1" ] && echo --enforce ) || true

echo "=== Evaluation completed ==="
echo "Results saved to: $RESULT_DIR"
