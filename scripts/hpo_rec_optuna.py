#!/usr/bin/env python3
"""
Hyper-parameter optimization for Text Recognition (ABINet) using Optuna (if available),
with a builtin random-search fallback.

Each trial runs a short training (limited by --max-steps) in an isolated working dir,
then reads the last CER from logs/rec_eval.jsonl as the objective.

Usage examples:
  # Optuna (if installed) or fallback random search (10 trials):
  python -m scripts.hpo_rec_optuna \
    --base-config configs/rec_abinet.yaml \
    --trials 10 \
    --max-steps 1200 \
    --out results/hpo_runs

Notes:
  - No external services (e.g., W&B) are used.
  - Each trial writes weights/logs under its own directory.
  - Best trial's params and config override are saved under --out.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def merge_cfg(base: dict, override: dict) -> dict:
    out = json.loads(json.dumps(base))  # deep copy
    def _merge(d, o):
        for k, v in o.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                _merge(d[k], v)
            else:
                d[k] = v
    _merge(out, override)
    return out


def write_yaml(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding='utf-8')


def run_trial(trial_id: int, base_cfg_path: Path, out_root: Path, max_steps: int, params: dict) -> float:
    work = out_root / f"trial_{trial_id:03d}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding='utf-8'))
    # Adjust eval frequency for short runs
    eval_steps = max(100, int(max_steps // 3))
    override = {
        'training': {
            'epochs': 20,
            'optimizer': {
                'lr': float(params['lr'])
            },
            'label_smoothing': float(params['label_smoothing']),
            'eval_steps': int(eval_steps),
            # Enable AMP to reduce GPU memory pressure on short trials
            'use_amp': True,
        },
        'data': {
            'target_width': int(params['target_width']),
            # Robust loader settings for diverse environments
            'batch_size': 32,   # safer default for HPO to avoid OOM
            'num_workers': 0,   # avoid SemLock/Permission issues
            'prefetch_factor': 2,
            'timeout': 0,
            'augment': True,
            'aug_blur_prob': float(params['aug_blur_prob']),
            'aug_noise_prob': float(params['aug_noise_prob']),
            'aug_erase_prob': float(params['aug_erase_prob']),
        },
        'model': {
            'max_len': int(params['max_len'])
        }
    }
    cfg = merge_cfg(base_cfg, override)
    cfg_path = work / 'config.yaml'
    write_yaml(cfg_path, cfg)

    log_path = work / 'train.log'
    env = os.environ.copy()
    # Isolated logs/weights/TB per trial
    env['TB_LOGDIR'] = str(work / 'runs')
    env['REC_LOGDIR'] = str(work / 'logs')
    env['REC_WEIGHTSDIR'] = str(work / 'weights')
    # Ensure repo root is importable as module path for `-m src.train_rec`
    repo_root = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = repo_root + (os.pathsep + env['PYTHONPATH'] if 'PYTHONPATH' in env and env['PYTHONPATH'] else '')
    # CUDA memory fragmentation mitigation (optional, harmless on CPU)
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')
    # Optional: allow users to force CPU for debugging by setting HPO_FORCE_CPU=1
    if env.get('HPO_FORCE_CPU', '0') in ('1', 'true', 'True'):
        env['CUDA_VISIBLE_DEVICES'] = ''
    cfg_abs = str(cfg_path.resolve())
    cmd = [sys.executable, '-m', 'src.train_rec', '--config', cfg_abs, '--model', 'abinet', '--max-steps', str(max_steps)]
    with open(log_path, 'w') as lf:
        rc = None
        try:
            proc = subprocess.run(cmd, cwd=repo_root, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
            rc = proc.returncode
        except KeyboardInterrupt:
            rc = -2
        except Exception:
            rc = -1
        # Persist return code for diagnostics
        try:
            (work / 'status.json').write_text(json.dumps({'returncode': rc}, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

    # Read CER
    cer = 1.0
    eval_log = work / 'logs' / 'rec_eval.jsonl'
    if eval_log.exists():
        try:
            lines = [json.loads(l) for l in eval_log.read_text(encoding='utf-8').strip().splitlines() if l.strip()]
            if lines:
                # Use the BEST (minimum) CER observed during the trial, not the last line
                best_entry = min(lines, key=lambda x: float(x.get('cer', 1.0)))
                cer = float(best_entry.get('cer', 1.0))
                # Persist best_eval for reference
                (work / 'best_eval.json').write_text(json.dumps(best_entry, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass
    else:
        # If eval log is missing, annotate the trial directory for quick triage
        try:
            (work / 'NO_EVAL_LOG').write_text('Evaluation log not found; training likely exited early.\n', encoding='utf-8')
        except Exception:
            pass
    return cer


def sample_params(trial=None):
    def U(a, b):
        import random
        return a + (b - a) * random.random()
    params = {
        'lr': (trial.suggest_float('lr', 1e-4, 8e-4, log=True) if trial else U(1e-4, 8e-4)),
        'label_smoothing': (trial.suggest_categorical('label_smoothing', [0.0, 0.05, 0.1]) if trial else [0.0, 0.05, 0.1][int(U(0,2.999))]),
        'target_width': (trial.suggest_categorical('target_width', [320, 480]) if trial else (320 if U(0,1)<0.5 else 480)),
        'aug_blur_prob': (trial.suggest_float('aug_blur_prob', 0.0, 0.2) if trial else U(0.0, 0.2)),
        'aug_noise_prob': (trial.suggest_float('aug_noise_prob', 0.0, 0.2) if trial else U(0.0, 0.2)),
        'aug_erase_prob': (trial.suggest_float('aug_erase_prob', 0.0, 0.1) if trial else U(0.0, 0.1)),
        'max_len': (trial.suggest_categorical('max_len', [50, 64, 80]) if trial else ([50,64,80][int(U(0,2.999))])),
    }
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-config', default='configs/rec_abinet.yaml')
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--max-steps', type=int, default=1200)
    ap.add_argument('--out', default='results/hpo')
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    base_cfg_path = Path(args.base_config)

    best = {'cer': 1.0, 'params': None}

    try:
        import optuna
        def objective(trial):
            params = sample_params(trial)
            cer = run_trial(trial.number, base_cfg_path, out_root, args.max_steps, params)
            trial.set_user_attr('params', params)
            return cer

        study = optuna.create_study(direction='minimize', study_name=f"rec_hpo_{int(time.time())}",
                                     sampler=optuna.samplers.TPESampler(),
                                     pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
        best_trial = study.best_trial
        best['cer'] = best_trial.value
        best['params'] = best_trial.user_attrs.get('params')
    except Exception as e:
        print(f"[HPO] Optuna not available ({e}); falling back to random search")
        for t in range(args.trials):
            params = sample_params(None)
            cer = run_trial(t, base_cfg_path, out_root, args.max_steps, params)
            if cer < best['cer']:
                best = {'cer': cer, 'params': params}

    # Save best
    (out_root / 'best.json').write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8')
    print("Best CER:", best['cer'])
    print("Best params:", json.dumps(best['params'], ensure_ascii=False))

    # Emit merged config override for final training
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding='utf-8'))
    final_cfg = merge_cfg(base_cfg, {
        'training': {
            'optimizer': {'lr': float(best['params']['lr'])},
            'label_smoothing': float(best['params']['label_smoothing'])
        },
        'data': {
            'target_width': int(best['params']['target_width']),
            'augment': True,
            'aug_blur_prob': float(best['params']['aug_blur_prob']),
            'aug_noise_prob': float(best['params']['aug_noise_prob']),
            'aug_erase_prob': float(best['params']['aug_erase_prob'])
        },
        'model': {'max_len': int(best['params']['max_len'])}
    })
    write_yaml(out_root / 'best_config.yaml', final_cfg)


if __name__ == '__main__':
    main()
