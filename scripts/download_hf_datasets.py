#!/usr/bin/env python3
"""
Download HF datasets used by Visionalizer-AI into local data/ directory via datasets.load_dataset.

Datasets:
- SynthDoG-ja: naver-clova-ix/synthdog-ja
- DocLayNet:  ds4sd/DocLayNet
- PubTabNet:  ajimeno/PubTabNet (mirror)

Writes to: data/hf_cache/<name>/<split>

CLI options (or env vars):
- --only {all,doclaynet,pubtabnet,synthdog,publaynet,docbank} (HF_DL_ONLY)
- --limit-synthdog N  (HF_DL_LIMIT_SYNTHDOG)
- --limit-doclaynet N (HF_DL_LIMIT_DOCLAYNET)
- --limit-pubtabnet N (HF_DL_LIMIT_PUBTABNET)

Defaults: --only all, no limits (full download). Use limits for quick smoke.
"""
import os
from pathlib import Path
from typing import Optional

import os
import argparse
from datasets import load_dataset
from huggingface_hub import snapshot_download


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_index(ds, out_dir: Path, id_col: Optional[str] = None):
    idx_path = out_dir / "index.txt"
    with idx_path.open("w", encoding="utf-8") as f:
        for i in range(len(ds)):
            if id_col and id_col in ds.features:
                f.write(str(ds[i][id_col]))
            else:
                f.write(str(i))
            f.write("\n")


def _apply_limit(ds, limit: int):
    if limit and limit > 0:
        n = min(limit, len(ds)) if hasattr(ds, "__len__") else limit
        try:
            return ds.select(range(n))
        except Exception:
            return ds
    return ds


def download_synthdog(root: Path, limit: int = 0):
    name = "synthdog_ja"
    base = root / "hf_cache" / name
    ensure_dir(base)
    ds = load_dataset("naver-clova-ix/synthdog-ja")
    for split in ds.keys():
        out_dir = base / split
        ensure_dir(out_dir)
        subset = _apply_limit(ds[split], limit)
        subset.save_to_disk(str(out_dir))
        save_index(subset, out_dir)
    print(f"[download] Saved SynthDoG-ja to {base}")


def download_doclaynet(root: Path, limit: int = 0):
    name = "doclaynet"
    base = root / "hf_cache" / name
    ensure_dir(base)
    try:
        ds = load_dataset("ds4sd/DocLayNet")
        for split in ds.keys():
            out_dir = base / split
            ensure_dir(out_dir)
            subset = _apply_limit(ds[split], limit)
            subset.save_to_disk(str(out_dir))
            save_index(subset, out_dir)
        print(f"[download] Saved DocLayNet to {base}")
        return
    except Exception as e:
        print(f"[info] load_dataset failed for DocLayNet, falling back to snapshot_download: {e}")
    # Fallback: snapshot one of candidate repos
    candidates_env = os.getenv("HF_DOCLAYNET_REPOS")
    candidates: List[str] = []
    if candidates_env:
        candidates = [x.strip() for x in candidates_env.split(',') if x.strip()]
    if not candidates:
        candidates = [
            "ds4sd/DocLayNet-v1.2",
            "pierreguillou/DocLayNet-base",
        ]
    token = os.getenv("HF_TOKEN")
    last_err = None
    for rid in candidates:
        try:
            dest = base / (rid.replace('/', '__'))
            ensure_dir(dest)
            local_path = snapshot_download(
                repo_id=rid,
                repo_type="dataset",
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                token=token,
            )
            print(f"[download] DocLayNet snapshot saved to {local_path}")
            return
        except Exception as e:
            print(f"[warn] snapshot {rid} failed: {e}")
            last_err = e
    if last_err:
        raise last_err

def download_synthchartnet(root: Path, limit: int = 0):
    name = "synthchartnet"
    base = root / "hf_cache" / name
    ensure_dir(base)
    token = os.getenv("HF_TOKEN")
    dest = base / "raw_repo"
    ensure_dir(dest)
    local_path = snapshot_download(
        repo_id=os.getenv("HF_SYNTHCHARTNET_REPO", "ds4sd/SynthChartNet"),
        repo_type="dataset",
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"[download] SynthChartNet snapshot saved to {local_path}")


def download_pubtabnet(root: Path, limit: int = 0):
    name = "pubtabnet"
    base = root / "hf_cache" / name
    ensure_dir(base)
    ds = load_dataset("ajimeno/PubTabNet")
    for split in ds.keys():
        out_dir = base / split
        ensure_dir(out_dir)
        subset = _apply_limit(ds[split], limit)
        subset.save_to_disk(str(out_dir))
        save_index(subset, out_dir)
    print(f"[download] Saved PubTabNet to {base}")


def download_publaynet(root: Path, limit: int = 0):
    """Download PubLayNet via snapshot (requires repo id)."""
    name = "publaynet"
    base = root / "hf_cache" / name
    ensure_dir(base)
    repo_id = os.getenv("HF_PUBLAYNET_REPO")
    if not repo_id:
        print("[warn] PubLayNet requires HF_PUBLAYNET_REPO env (e.g., 'ibm/PublayNet')")
        return
    dest = base / "raw_repo"
    ensure_dir(dest)
    token = os.getenv("HF_TOKEN")
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"[download] PubLayNet snapshot saved to {local_path}")


def download_docbank(root: Path, limit: int = 0):
    """Download DocBank via snapshot (requires repo id)."""
    name = "docbank"
    base = root / "hf_cache" / name
    ensure_dir(base)
    repo_id = os.getenv("HF_DOCBANK_REPO")
    if not repo_id:
        print("[warn] DocBank requires HF_DOCBANK_REPO env (e.g., 'doc-analysis/DocBank_500K')")
        return
    dest = base / "raw_repo"
    ensure_dir(dest)
    token = os.getenv("HF_TOKEN")
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"[download] DocBank snapshot saved to {local_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "doclaynet", "pubtabnet", "synthdog", "publaynet", "docbank", "synthchartnet"], default=os.getenv("HF_DL_ONLY", "all"))
    parser.add_argument("--limit-synthdog", type=int, default=int(os.getenv("HF_DL_LIMIT_SYNTHDOG", "0")))
    parser.add_argument("--limit-doclaynet", type=int, default=int(os.getenv("HF_DL_LIMIT_DOCLAYNET", "0")))
    parser.add_argument("--limit-pubtabnet", type=int, default=int(os.getenv("HF_DL_LIMIT_PUBTABNET", "0")))
    args = parser.parse_args()
    root = Path(os.getcwd()) / "data"
    ensure_dir(root)
    # Order: doclaynet -> pubtabnet -> synthdog (heavy)
    targets = ["doclaynet", "pubtabnet", "publaynet", "docbank", "synthchartnet", "synthdog"] if args.only == "all" else [args.only]
    for t in targets:
        try:
            if t == "doclaynet":
                download_doclaynet(root, limit=args.limit_doclaynet)
            elif t == "pubtabnet":
                download_pubtabnet(root, limit=args.limit_pubtabnet)
            elif t == "synthdog":
                download_synthdog(root, limit=args.limit_synthdog)
            elif t == "publaynet":
                download_publaynet(root, limit=0 if args.only == "publaynet" else args.limit_doclaynet)
            elif t == "docbank":
                download_docbank(root, limit=0 if args.only == "docbank" else args.limit_doclaynet)
            elif t == "synthchartnet":
                download_synthchartnet(root, limit=0)
        except Exception as e:
            print(f"[warn] {t} failed: {e}")


if __name__ == "__main__":
    main()
