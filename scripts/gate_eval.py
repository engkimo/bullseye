#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

def load_json(p: Path):
    try:
        with p.open() as f:
            return json.load(f)
    except Exception:
        return {}

def pick(d: dict, keys):
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    # nested common pattern
    if 'metrics' in d and isinstance(d['metrics'], dict):
        return pick(d['metrics'], keys)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='Result directory containing *_metrics.json')
    ap.add_argument('--cer-max', type=float, default=0.90)
    ap.add_argument('--det-hmean-min', type=float, default=0.86)
    ap.add_argument('--layout-map-5095-min', type=float, default=0.72)
    ap.add_argument('--table-teds-min', type=float, default=0.88)
    ap.add_argument('--enforce', action='store_true')
    args = ap.parse_args()

    rd = Path(args.dir)
    out = {
        'thresholds': {
            'cer_max': args.cer_max,
            'det_hmean_min': args.det_hmean_min,
            'layout_map_50_95_min': args.layout_map_5095_min,
            'table_teds_min': args.table_teds_min,
        },
        'metrics': {},
        'results': {},
        'passed': True,
        'errors': []
    }

    # Recognition
    rec = load_json(rd / 'recognition_metrics.json')
    cer = pick(rec, ['cer','CER','character_error_rate'])
    if cer is not None:
        out['metrics']['cer'] = cer
        if cer > args.cer_max:
            out['results']['recognition'] = 'fail'
            out['errors'].append(f'CER {cer:.4f} > {args.cer_max}')
            out['passed'] = False
        else:
            out['results']['recognition'] = 'pass'
    else:
        out['results']['recognition'] = 'skip'

    # Detection
    det = load_json(rd / 'detection_metrics.json')
    hmean = pick(det, ['hmean','HMean','f1','F1'])
    if hmean is not None:
        out['metrics']['detection_hmean'] = hmean
        if hmean < args.det_hmean_min:
            out['results']['detection'] = 'fail'
            out['errors'].append(f'HMean {hmean:.4f} < {args.det_hmean_min}')
            out['passed'] = False
        else:
            out['results']['detection'] = 'pass'
    else:
        out['results']['detection'] = 'skip'

    # Layout
    lay = load_json(rd / 'layout_metrics.json')
    map_5095 = pick(lay, ['map_50_95','mAP@0.5:0.95','map@[.50:.95]','map'])
    if map_5095 is not None:
        out['metrics']['layout_map_50_95'] = map_5095
        if map_5095 < args.layout_map_5095_min:
            out['results']['layout'] = 'fail'
            out['errors'].append(f'mAP@0.5:0.95 {map_5095:.4f} < {args.layout_map_5095_min}')
            out['passed'] = False
        else:
            out['results']['layout'] = 'pass'
    else:
        out['results']['layout'] = 'skip'

    # Table
    tab = load_json(rd / 'table_metrics.json')
    teds = pick(tab, ['teds','TEDS'])
    if teds is not None:
        out['metrics']['table_teds'] = teds
        if teds < args.table_teds_min:
            out['results']['table'] = 'fail'
            out['errors'].append(f'TEDS {teds:.4f} < {args.table_teds_min}')
            out['passed'] = False
        else:
            out['results']['table'] = 'pass'
    else:
        out['results']['table'] = 'skip'

    # Save
    with (rd / 'gate_summary.json').open('w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Human-readable summary
    status = 'PASS' if out['passed'] else 'FAIL'
    print(f"Gate: {status} | Metrics: " + \
          ", ".join([f"{k}={v:.4f}" for k,v in out.get('metrics',{}).items()]))
    if out['errors']:
        print("Reasons: " + "; ".join(out['errors']))

    if args.enforce and not out['passed']:
        sys.exit(1)

if __name__ == '__main__':
    main()

