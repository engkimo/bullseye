#!/usr/bin/env bash
set -euo pipefail

# Quick A/B trial for SATRN recognizer (short run)
# Usage: bash scripts/ab_rec_satrn_quick.sh [max_steps]

MAX_STEPS=${1:-1200}

source venv/bin/activate || true

OUTDIR=${REC_LOGDIR:-results/ab_satrn_quick}
mkdir -p "$OUTDIR"

CFG=$(realpath configs/rec_abinet.yaml)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}

echo "[SATRN] Training short trial: steps=$MAX_STEPS"
python -m src.train_rec --config "$CFG" --model satrn --max-steps "$MAX_STEPS" 2>&1 | tee "$OUTDIR/train_satrn.log"

# Extract best CER from rec_eval.jsonl if present
EVAL_LOG=$(rg -l "rec_eval.jsonl" -n runs results logs || true | head -n1)
if [ -z "$EVAL_LOG" ]; then
  EVAL_LOG="logs/rec_eval.jsonl"
fi

python - "$OUTDIR" "$EVAL_LOG" << 'PY'
import json,sys,os
outdir, logpath = sys.argv[1], sys.argv[2]
best=1e9
best_row=None
try:
    with open(logpath,'r',encoding='utf-8') as f:
        for line in f:
            try:
                row=json.loads(line)
                cer=float(row.get('cer', row.get('CER', 1e9)))
                if cer<best:
                    best=cer
                    best_row=row
            except: pass
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir,'best_eval.json'),'w',encoding='utf-8') as f:
        json.dump({'best_cer':best,'row':best_row},f,ensure_ascii=False,indent=2)
    print(f"[SATRN] best CER: {best:.6f} (saved to {outdir}/best_eval.json)")
except Exception as e:
    print(f"[SATRN] eval parsing failed: {e}")
PY

echo "[SATRN] Done. See: $OUTDIR"

