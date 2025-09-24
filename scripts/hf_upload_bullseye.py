#!/usr/bin/env python3
"""
Upload local bullseye model folders to Hugging Face Hub.

Usage:
  python scripts/hf_upload_bullseye.py --user <hf_username> \
    --root models/bullseye \
    --private  # optional

Requires:
  pip install huggingface_hub
  Export HF_TOKEN (or login via huggingface-cli login)

  It will create/upload the following repos if the folders exist:
  <user>/bullseye-dbnet
  <user>/bullseye-recparseq
  <user>/bullseye-layoutrtdetrv
  <user>/bullseye-tablertdetrv
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple


def ensure_hf():
    try:
        import huggingface_hub  # noqa
    except Exception as e:
        raise SystemExit("huggingface_hub is required. Install: pip install huggingface_hub") from e


def upload_folder(api, repo_id: str, folder: Path, private: bool):
    from huggingface_hub import HfApi
    assert folder.exists() and folder.is_dir(), f"Folder not found: {folder}"
    # Create repo if missing
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    except Exception as e:
        print(f"[WARN] create_repo failed (may already exist): {e}")
    # Upload (large files supported via LFS)
    print(f"[INFO] Uploading {folder} -> {repo_id}")
    api.upload_folder(repo_id=repo_id, folder_path=str(folder), path_in_repo=".")


def main():
    ensure_hf()
    from huggingface_hub import HfApi

    p = argparse.ArgumentParser()
    p.add_argument('--user', required=True, help='HF username/organization (namespace)')
    p.add_argument('--root', default='models/bullseye', help='Root folder for bullseye models')
    p.add_argument('--private', action='store_true', help='Create private repos')
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    api = HfApi()
    plan: List[Tuple[str, Path]] = []
    mapping = [
        (f"{args.user}/bullseye-dbnet", root / 'det-dbnet-v2'),
        (f"{args.user}/bullseye-recparseq", root / 'rec-parseq-v2'),
        (f"{args.user}/bullseye-layoutrtdetrv", root / 'layout-rtdetrv2-v2'),
        (f"{args.user}/bullseye-tablertdetrv", root / 'table-rtdetrv2'),
    ]
    for repo_id, folder in mapping:
        if folder.exists():
            plan.append((repo_id, folder))

    if not plan:
        raise SystemExit("No model folders found to upload.")

    for repo_id, folder in plan:
        upload_folder(api, repo_id, folder, args.private)

    print("[DONE] Upload completed")


if __name__ == '__main__':
    main()
