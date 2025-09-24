from __future__ import annotations

import os
from typing import List, Tuple, Optional

import numpy as np
import cv2


def _smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k) | 1)
    return cv2.GaussianBlur(x.reshape(1, -1), (k, 1), 0).flatten()


def _find_local_maxima(y: np.ndarray, min_dist: int, thr_ratio: float) -> List[int]:
    if y.size == 0:
        return []
    thr = float(np.max(y)) * float(thr_ratio)
    peaks: List[int] = []
    last = -10_000
    for i in range(1, len(y) - 1):
        if y[i] >= thr and y[i] > y[i - 1] and y[i] >= y[i + 1]:
            if i - last >= min_dist:
                peaks.append(i)
                last = i
    return peaks


def _autocorr_period(y: np.ndarray, min_period: int, max_period: int) -> Optional[float]:
    if y.size == 0 or max_period <= min_period:
        return None
    # normalize
    z = (y - np.mean(y))
    z = z / (np.std(z) + 1e-6)
    # compute autocorrelation via FFT convolution
    n = int(2 ** np.ceil(np.log2(len(z) * 2)))
    f = np.fft.rfft(z, n)
    ac = np.fft.irfft(f * np.conj(f))[: len(z)]
    # search between min and max
    s = int(min_period)
    e = int(min(max_period, len(ac) - 1))
    if e <= s:
        return None
    k = s + int(np.argmax(ac[s : e + 1]))
    return float(k)


def estimate_columns(
    image_rgb: np.ndarray,
    roi: Tuple[int, int, int, int],  # (x_left, y_top, x_right, y_bottom)
    band_y: Optional[Tuple[int, int]],
    bar_boxes: List[Tuple[int, int, int, int]],
    tick_xs: List[float],
) -> List[int]:
    """Estimate Gantt vertical grid columns robustly.

    - Build gradient profile in the orange header band if available.
    - Fuse 3 cues: profile peaks, OCR tick x-positions, and bar endpoints.
    - Estimate grid period via autocorrelation; search offset for minimal cost.
    """

    H, W = image_rgb.shape[:2]
    xL, yT, xR, yB = roi
    xL = max(0, min(xL, W - 1))
    xR = max(xL + 1, min(xR, W - 1))
    yT = max(0, min(yT, H - 1))
    yB = max(yT + 1, min(yB, H - 1))

    # choose band for profile
    if band_y is not None:
        yh0, yh1 = band_y
        yh0 = max(0, yh0 - 1)
        yh1 = min(H - 1, yh1 + 1)
        band = image_rgb[yh0:yh1, xL:xR]
    else:
        # fallback to top 10% of ROI
        band = image_rgb[yT : yT + max(5, int(0.12 * (yB - yT))), xL:xR]

    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY) if band.size else np.zeros((10, max(1, xR - xL)), np.uint8)
    # Scharr is more sensitive for fine edges
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    g = np.mean(np.abs(gx), axis=0)
    g = _smooth_1d(g, k=max(5, int((xR - xL) / 100)))

    # peak candidates from profile
    thr_ratio = float(os.getenv('DOCJA_GANTT_COL_THR', '0.18'))
    min_sep = max(6, int((xR - xL) / 64))
    pk_idx = _find_local_maxima(g, min_dist=min_sep, thr_ratio=thr_ratio)
    pk_x = [int(xL + i) for i in pk_idx]

    # bar endpoints cue
    end_xs: List[int] = []
    for (bx1, by1, bx2, by2) in bar_boxes:
        end_xs.append(int(bx1))
        end_xs.append(int(bx2))
    end_xs.sort()

    # OCR ticks
    lbl_xs = sorted(int(x) for x in tick_xs if xL <= x <= xR)

    # initial period estimation
    width = float(max(1, xR - xL))
    min_cols = int(os.getenv('DOCJA_GANTT_MIN_COLS', '0') or 0)
    # plausible period range
    min_period = max(6, int(width / 64))
    max_period = max(min_period + 1, int(width / max(6, min_cols if min_cols > 0 else 12)))

    # try from peaks autocorrelation
    period = _autocorr_period(g, min_period, max_period)
    # refine from label distances if we have at least 2 labels
    if lbl_xs and len(lbl_xs) >= 2:
        diffs = np.diff(lbl_xs)
        guess = float(np.median(diffs))
        if period is None or (abs(guess - period) / max(period, 1.0)) < 0.5:
            period = guess
    # refine from bar endpoint diffs
    if (period is None or period < min_period or period > 2 * max_period) and len(end_xs) >= 4:
        d = np.diff(end_xs)
        d = d[(d >= min_period) & (d <= max(2 * max_period, min_period + 1))]
        if d.size:
            period = float(np.median(d))
    # final fallback
    if period is None or period < 3:
        period = max(8.0, width / float(max(min_cols, 14) if min_cols > 0 else 16))

    # candidate offsets: from peaks/labels/endpoints, normalized into [0, period)
    supports = pk_x + lbl_xs + end_xs
    if not supports:
        # place a uniform grid
        n = int(np.clip(width / period, 4, 64))
        return [int(round(xL + i * period)) for i in range(n)]

    def _cost_for_offset(off: float) -> float:
        # build grid positions across ROI
        x0 = xL + ((off - (xL % period)) % period)
        grid: List[float] = []
        x = x0
        while x <= xR:
            grid.append(float(x))
            x += period
        # weights
        w_peak = float(os.getenv('DOCJA_GANTT_W_PEAK', '0.7'))
        w_lbl = float(os.getenv('DOCJA_GANTT_W_LABEL', '1.0'))
        w_end = float(os.getenv('DOCJA_GANTT_W_END', '0.5'))
        # pre-sort supports
        pks = np.array(pk_x, dtype=np.float32) if pk_x else None
        lbs = np.array(lbl_xs, dtype=np.float32) if lbl_xs else None
        eds = np.array(end_xs, dtype=np.float32) if end_xs else None
        cost = 0.0
        for gx in grid:
            if pks is not None and pks.size:
                cost += w_peak * float(np.min(np.abs(pks - gx)))
            if lbs is not None and lbs.size:
                cost += w_lbl * float(np.min(np.abs(lbs - gx)))
            if eds is not None and eds.size:
                cost += w_end * float(np.min(np.abs(eds - gx)))
        # penalty for too few columns
        min_cols_local = int(os.getenv('DOCJA_GANTT_MIN_COLS', '0') or 0)
        if min_cols_local > 0 and len(grid) < min_cols_local:
            cost += (min_cols_local - len(grid)) * period
        return cost / max(1, len(grid))

    # search offset in [0, period)
    steps = max(16, int(period / 2))
    offs = np.linspace(0.0, float(period), num=steps, endpoint=False)
    # seed offsets from supports modulo period (avoid re-evaluating many duplicates)
    offs_seed = []
    for s in supports[:64]:
        offs_seed.append(float(s % period))
    offs = np.unique(np.concatenate([offs, np.array(offs_seed, dtype=np.float32)]))

    best_off = 0.0
    best_cost = 1e18
    for off in offs:
        c = _cost_for_offset(off)
        if c < best_cost:
            best_cost, best_off = c, float(off)

    # compose final grid
    x0 = xL + ((best_off - (xL % period)) % period)
    xs: List[int] = []
    x = x0
    while x <= xR:
        xs.append(int(round(x)))
        x += period

    # enforce minimal columns if requested by uniform extension
    min_cols_local = int(os.getenv('DOCJA_GANTT_MIN_COLS', '0') or 0)
    if min_cols_local > 0 and len(xs) < min_cols_local and len(xs) >= 2:
        step = float(np.median(np.diff(xs)))
        while len(xs) < min_cols_local:
            xs.insert(0, int(round(xs[0] - step)))
            if len(xs) >= min_cols_local:
                break
            xs.append(int(round(xs[-1] + step)))

    return xs

