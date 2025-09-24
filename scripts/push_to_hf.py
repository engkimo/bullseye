#!/usr/bin/env python3
"""
Push local weights/config to Hugging Face Hub.
Requires environment variables:
  - HF_TOKEN: token with write permission
  - HF_REPO:  repo name like "username/docja-weights" (created if missing)

Uploads folders under weights/ as subpaths:
  weights/det -> repo/det
  weights/rec -> repo/rec
  weights/layout -> repo/layout
  weights/table -> repo/table
  weights/lora -> repo/lora
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    token = os.getenv("HF_TOKEN")
    repo = os.getenv("HF_REPO")
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    if not repo:
        raise SystemExit("HF_REPO is not set (e.g., username/docja-weights)")

    api = HfApi(token=token)
    # Create repo if not exists
    try:
        create_repo(repo_id=repo, token=token, private=True, exist_ok=True)
    except Exception as e:
        print(f"[push_to_hf] create_repo warning: {e}")

    root = Path("weights")
    if not root.exists():
        raise SystemExit("weights/ directory not found")

    # Map local subdirs to hub paths
    subdirs = ["det", "rec", "layout", "table", "lora"]
    for sd in subdirs:
        p = root / sd
        if p.exists():
            print(f"[push_to_hf] Uploading {p} -> {repo}/{sd}")
            upload_folder(
                repo_id=repo,
                folder_path=str(p),
                path_in_repo=sd,
                token=token,
                ignore_patterns=["*.tmp", "*.log", "*.png", "*.jpg"],
            )
        else:
            print(f"[push_to_hf] Skipping missing: {p}")

    print("[push_to_hf] Completed.")


if __name__ == "__main__":
    main()

