#!/usr/bin/env python3
"""Offline preprocess for recognition line images.

Reads images from --in-dir, applies the same crop/rotate/CLAHE pipeline
as training, and writes 48x320 grayscale PNGs to --out-dir. This shifts
CPU-heavy preprocessing out of the training loop to increase GPU usage.

Usage:
  python scripts/precache_rec_lines.py --in-dir data/rec/lines \
    --out-dir data/rec/lines_pre --max-workers 8
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_one(src: Path, dst: Path):
    try:
        img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Crop via Otsu
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(bin_inv > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            mh = int(0.05 * (y2 - y1 + 1))
            mw = int(0.05 * (x2 - x1 + 1))
            y1 = max(0, y1 - mh)
            y2 = min(img.shape[0] - 1, y2 + mh)
            x1 = max(0, x1 - mw)
            x2 = min(img.shape[1] - 1, x2 + mw)
            img = img[y1:y2 + 1, x1:x2 + 1]
        # Rotate if tall
        h, w = img.shape[:2]
        if h > w * 1.5:
            img = np.rot90(img, k=-1)
            h, w = img.shape[:2]
        # CLAHE
        gray2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray2 = clahe.apply(gray2)
        # Resize to 48x320 with letterbox
        target_h, target_w = 48, 320
        ratio = min(target_w / w, target_h / h)
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        resized = cv2.resize(gray2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        out = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255,))
        # Save
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='Process only first N files (0=all)')
    ap.add_argument('--height', type=int, default=48)
    ap.add_argument('--width', type=int, default=320)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')])
    if args.limit and len(files) > args.limit:
        files = files[: args.limit]
    if not files:
        print('No images found')
        return
    ok = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        # partial binding of target size
        def submit(p):
            return ex.submit(process_one_sz, p, out_dir / p.name, args.height, args.width)
        futs = {submit(p): p for p in files}
        for i, f in enumerate(as_completed(futs), 1):
            if f.result():
                ok += 1
            if i % 1000 == 0:
                print(f"Processed {i}/{len(files)} ...")
    print(f"Done. Success {ok}/{len(files)} -> {out_dir}")


def process_one_sz(src: Path, dst: Path, target_h: int, target_w: int) -> bool:
    try:
        img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Crop via Otsu
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(bin_inv > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            mh = int(0.05 * (y2 - y1 + 1))
            mw = int(0.05 * (x2 - x1 + 1))
            y1 = max(0, y1 - mh)
            y2 = min(img.shape[0] - 1, y2 + mh)
            x1 = max(0, x1 - mw)
            x2 = min(img.shape[1] - 1, x2 + mw)
            img = img[y1:y2 + 1, x1:x2 + 1]
        # Rotate if tall
        h, w = img.shape[:2]
        if h > w * 1.5:
            img = np.rot90(img, k=-1)
            h, w = img.shape[:2]
        # CLAHE
        gray2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray2 = clahe.apply(gray2)
        # Resize to target_h x target_w with letterbox
        ratio = min(target_w / w, target_h / h)
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        resized = cv2.resize(gray2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        out = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255,))
        # Save
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out)
        return True
    except Exception:
        return False


if __name__ == '__main__':
    main()
