"""
GanttParser (v0.2): テキストブロックのみから軸スケール（px/day）と行ラベル、
日付レンジを可能な限り復元するヒューリスティック実装。

- 軸: 上部/下部に配置された日付テキストから tick を抽出し px/day を推定
- 行: 左余白にある短文テキストをタスク名候補として行クラスタを作成
- 期間: 同じ行帯域にある「M/D〜M/D」「YYYY-MM-DD〜...」のようなレンジ表記を抽出
  → px/day と原点の tick から start/end_px を復元（可能な場合）

将来的に色付きバー検出/凡例色対応は画像処理で補完予定。
"""
from typing import Dict, Any, List, Tuple, Optional
import os
import re
from datetime import datetime
import numpy as np
import cv2
from .gantt_columnizer import estimate_columns


DATE_PATTERNS = [
    re.compile(r"(?P<y>\d{4})[./-](?P<m>\d{1,2})[./-](?P<d>\d{1,2})"),
    re.compile(r"(?P<m>\d{1,2})[./-](?P<d>\d{1,2})"),
    re.compile(r"(?P<m>\d{1,2})月(?P<d>\d{1,2})日"),
]
RANGE_SEP = r"\s*(?:~|〜|―|ｰ|-|–|to|→)\s*"
DOW = set(list("月火水木金土日"))


def _parse_date(s: str, default_year: int) -> Optional[datetime]:
    for pat in DATE_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        gd = m.groupdict()
        y = int(gd.get('y') or default_year)
        mth = int(gd['m'])
        day = int(gd['d'])
        try:
            return datetime(y, mth, day)
        except Exception:
            return None
    return None


def _center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def _cluster_rows(text_blocks: List[Dict[str, Any]], W: float, H: float) -> List[Dict[str, Any]]:
    """左側25%にある短文を行ラベル候補として抽出し、Y座標で安直にクラスタ。"""
    margin = 0.25 * W
    cands = []
    for tb in text_blocks:
        t = (tb.get('text') or '').strip()
        if not t:
            continue
        x1, y1, x2, y2 = tb.get('bbox', [0, 0, 0, 0])
        cx, cy = _center([x1, y1, x2, y2])
        if cx <= margin and 1 <= len(t) <= 32:
            cands.append({'name': t.splitlines()[0], 'y': cy, 'bbox': [x1, y1, x2, y2]})
    cands.sort(key=lambda v: v['y'])
    rows = []
    for i, c in enumerate(cands):
        rows.append({'index': i, 'label': c['name'], 'y': c['y'], 'bbox': c['bbox']})
    return rows


def _estimate_grid_roi(image_rgb: np.ndarray, rows: List[Dict[str, Any]], bars_hint: List[Tuple[int,int,int,int]]) -> Tuple[int,int,int,int]:
    """Estimate grid ROI (x_left, y_top, x_right, y_bottom)."""
    H, W = image_rgb.shape[:2]
    # Vertical/horizontal lines emphasis
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    # Lines concentration map
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, H//80)))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, W//80), 1))
    ver = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ver_kernel, iterations=1)
    hor = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, hor_kernel, iterations=1)
    lines = cv2.bitwise_or(ver, hor)
    # Use rows/bars hints to constrain
    if rows:
        y_top = max(0, int(min(r['bbox'][1] for r in rows) - 40))
        y_bot = min(H-1, int(max(r['bbox'][3] for r in rows) + 40))
    else:
        y_top, y_bot = int(0.25*H), int(0.95*H)
    if bars_hint:
        x_left = max(0, min(b[0] for b in bars_hint) - 20)
        x_right = min(W-1, max(b[2] for b in bars_hint) + 20)
    else:
        x_left, x_right = int(0.35*W), int(0.98*W)
    # Refine by projecting line density
    sub = lines[y_top:y_bot, x_left:x_right]
    if sub.size > 0:
        col_sum = np.sum(sub>0, axis=0)
        row_sum = np.sum(sub>0, axis=1)
        # trim margins without lines
        def _trim(arr, thresh_ratio=0.05):
            thresh = max(1, int(thresh_ratio * np.max(arr)))
            xs = np.where(arr >= thresh)[0]
            if xs.size:
                return int(xs[0]), int(xs[-1])
            return 0, arr.shape[0]-1
        l_idx, r_idx = _trim(col_sum)
        t_idx, b_idx = _trim(row_sum)
        x_left = x_left + l_idx
        x_right = x_left + r_idx
        y_top = y_top + t_idx
        y_bot = y_top + b_idx
    return x_left, y_top, x_right, y_bot


def parse_gantt_from_page(page_dict: Dict[str, Any], image_rgb: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    text_blocks = page_dict.get('text_blocks', [])
    if not text_blocks:
        return []
    W = float(page_dict.get('width') or page_dict.get('size', {}).get('width') or 0)
    H = float(page_dict.get('height') or page_dict.get('size', {}).get('height') or 0)
    if not W or not H:
        return []
    # safe defaults for grid ROI (avoid UnboundLocalError)
    xL = 0
    yT = 0
    xR = int(W)
    yB = int(H)

    # 1) 軸tick抽出（上端/下端近傍）
    ticks = []  # [{'x': cx, 'date': datetime}]
    y_top_th = 0.15 * H
    y_bot_th = 0.85 * H
    now_year = datetime.now().year
    for tb in text_blocks:
        t = (tb.get('text') or '').strip()
        if not t:
            continue
        x1, y1, x2, y2 = tb.get('bbox', [0, 0, 0, 0])
        cx, cy = _center([x1, y1, x2, y2])
        if cy <= y_top_th or cy >= y_bot_th:
            dt = _parse_date(t, now_year)
            if dt:
                ticks.append({'x': cx, 'date': dt})
    ticks.sort(key=lambda v: v['x'])

    # px/dayの推定
    px_per_day = None
    x_origin_px = None
    if len(ticks) >= 2:
        t0, t1 = ticks[0], ticks[-1]
        days = (t1['date'] - t0['date']).days
        if days != 0:
            px_per_day = (t1['x'] - t0['x']) / float(days)
            x_origin_px = float(t0['x'])

    # 2) 行ラベル抽出
    rows = _cluster_rows(text_blocks, W, H)
    # 表グリッドのY範囲を行ラベルから推定（列線クリップ用）
    rows_y_min = None
    rows_y_max = None
    try:
        if rows:
            ys1 = [r['bbox'][1] for r in rows]
            ys2 = [r['bbox'][3] for r in rows]
            rows_y_min = float(min(ys1))
            rows_y_max = float(max(ys2))
    except Exception:
        rows_y_min = None
        rows_y_max = None

    # 3) 各行帯域にある日付レンジ表記から期間を復元
    tasks = []
    for row in rows:
        y = row['y']
        band_h = 0.03 * H
        # 同帯域のテキストを収集
        band_texts = []
        for tb in text_blocks:
            x1, y1, x2, y2 = tb.get('bbox', [0, 0, 0, 0])
            cx, cy = _center([x1, y1, x2, y2])
            if abs(cy - y) <= band_h:
                band_texts.append((tb, (tb.get('text') or '').strip()))
        # レンジ表記検出
        start_dt = end_dt = None
        for _tb, txt in band_texts:
            # 例: 7/1-7/15, 2025-07-01 ~ 2025-07-15, 7月1日〜7月15日
            parts = re.split(RANGE_SEP, txt)
            if len(parts) >= 2:
                sdt = _parse_date(parts[0], now_year)
                edt = _parse_date(parts[1], now_year)
                if sdt and edt:
                    start_dt, end_dt = sdt, edt
                    break
        # px復元
        start_px = end_px = None
        if start_dt and end_dt and px_per_day is not None and x_origin_px is not None and ticks:
            # ticks[0]のdateを原点とする
            base_date = ticks[0]['date']
            start_px = x_origin_px + (start_dt - base_date).days * px_per_day
            end_px = x_origin_px + (end_dt - base_date).days * px_per_day

        tasks.append({
            'id': f'T{row["index"]+1}',
            'name': row['label'],
            'row_index': row['index'],
            'start_px': float(start_px) if start_px is not None else None,
            'end_px': float(end_px) if end_px is not None else None,
            'start_date': start_dt.strftime('%Y-%m-%d') if start_dt else None,
            'end_date': end_dt.strftime('%Y-%m-%d') if end_dt else None,
            'color_rgba': None,
            'dependency_ids': []
        })

    # 4) カラーバー検出（画像がある場合）→ タスクの start/end_px を補完
    legend = []
    bars: List[Tuple[int,int,int,int]] = []
    if image_rgb is not None:
        try:
            Himg, Wimg = image_rgb.shape[:2]
            # Labのクロマで淡色も拾う
            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            a = lab[:, :, 1].astype(np.float32) - 128.0
            bch = lab[:, :, 2].astype(np.float32) - 128.0
            chroma = np.sqrt(a*a + bch*bch)
            # 背景（白/グレー）を除外するためにクロマ閾値を低めに設定
            sat_mask = (chroma > 8).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
            mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # レジェンド候補（上部右寄りの小さめ矩形）とバー分離
            leg_patches = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 10 or h < 4:
                    continue
                ar = w / max(1.0, h)
                # 右側グリッドっぽい領域（横長）をバー候補に
                if ar >= 3 and x > 0.35 * Wimg:
                    bars.append((x, y, x + w, y + h))
                # 上部の小パッチをレジェンド候補に
                elif y < 0.25 * Himg and w * h < 20000:
                    leg_patches.append((x, y, x + w, y + h))
            # レジェンドの色とラベル（近傍テキスト）
            def _mean_color(b):
                x1, y1, x2, y2 = b
                roi = image_rgb[y1:y2, x1:x2]
                if roi.size == 0:
                    return (0, 0, 0)
                return tuple(np.mean(roi.reshape(-1, 3), axis=0).tolist())
            for b in leg_patches:
                color = _mean_color(b)
                # 近傍テキスト（左/右隣）から凡例ラベルを推定
                label = None
                for tb, txt in [(tb, (tb.get('text') or '').strip()) for tb in text_blocks]:
                    bx1, by1, bx2, by2 = tb.get('bbox', [0, 0, 0, 0])
                    if abs((by1 + by2) / 2 - (b[1] + b[3]) / 2) < 20 and abs(bx1 - b[2]) < 120:
                        label = txt.splitlines()[0][:30]
                        break
                legend.append({'label': label or 'series', 'color_rgb': [int(c) for c in color]})
            # グリッドROIを見直してからバーを行に割当て（細かな色パッチを同一行でマージ）
            xL, yT, xR, yB = _estimate_grid_roi(image_rgb, rows, bars)
            # 行ごとに水平結合
            merged_bars: List[Tuple[int,int,int,int]] = []
            bars_sorted = sorted([b for b in bars if (b[0]>=xL and b[2]<=xR and b[1]>=yT and b[3]<=yB)], key=lambda b: (int((b[1]+b[3])/2), b[0]))
            current = None
            for (x1, y1, x2, y2) in bars_sorted:
                if current is None:
                    current = [x1, y1, x2, y2]
                else:
                    cy = (current[1]+current[3])/2
                    ny = (y1+y2)/2
                    if abs(ny - cy) <= max(6, int(0.008*Himg)) and (x1 - current[2]) <= max(6, int(0.004*Wimg)):
                        current[2] = max(current[2], x2)
                        current[1] = min(current[1], y1)
                        current[3] = max(current[3], y2)
                    else:
                        merged_bars.append(tuple(int(v) for v in current))
                        current = [x1, y1, x2, y2]
            if current is not None:
                merged_bars.append(tuple(int(v) for v in current))
            # マージ後バーを行に割当て
            for (x1, y1, x2, y2) in merged_bars:
                cy = (y1 + y2) / 2
                # 最近傍の行を探索
                best = None
                bestd = 1e12
                for row in rows:
                    d = abs(row['y'] - cy)
                    if d < bestd:
                        best, bestd = row, d
                if best is not None:
                    # 既存タスクがあるなら区間を追記、ないなら新規
                    tasks.append({
                        'id': f'TBAR_{best["index"]}_{int(x1)}',
                        'name': best['label'],
                        'row_index': best['index'],
                        'start_px': float(x1),
                        'end_px': float(x2),
                        'start_date': None,
                        'end_date': None,
                        'color_rgba': None,
                        'dependency_ids': []
                    })
        except Exception:
            pass

    # 5) 縦グリッド列の推定と列ラベルの割当（曜日など）
    columns = []
    if image_rgb is not None:
        try:
            Himg, Wimg = image_rgb.shape[:2]
            if rows:
                y_top = max(0, int(min(r['bbox'][1] for r in rows) - 40))
                y_bot = min(Himg - 1, int(max(r['bbox'][3] for r in rows) + 40))
            else:
                y_top, y_bot = int(0.25 * Himg), int(0.95 * Himg)
            if bars:
                x_left = max(0, min(b[0] for b in bars) - 20)
                x_right = min(Wimg - 1, max(b[2] for b in bars) + 20)
            else:
                x_left, x_right = int(0.35 * Wimg), int(0.98 * Wimg)
            # Set default grid ROI for later references
            xL, yT, xR, yB = int(x_left), int(y_top), int(x_right), int(y_bot)
            # まずヘッダー帯（オレンジ）を検出して y 範囲を補正
            def _find_orange_band(img: np.ndarray, x0: int, x1: int) -> Optional[Tuple[int,int]]:
                try:
                    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                    Hmin = int(os.getenv('DOCJA_GANTT_HMIN', '5'))
                    Hmax = int(os.getenv('DOCJA_GANTT_HMAX', '30'))
                    Smin = int(os.getenv('DOCJA_GANTT_SMIN', '60'))
                    Vmin = int(os.getenv('DOCJA_GANTT_VMIN', '120'))
                    # マスク作成（x範囲で切り出し）
                    sub = hsv[:, x0:x1, :]
                    H = sub[:, :, 0]
                    S = sub[:, :, 1]
                    V = sub[:, :, 2]
                    m = ((H >= Hmin) & (H <= Hmax) & (S >= Smin) & (V >= Vmin)).astype(np.uint8) * 255
                    # 横方向に連結
                    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(31, int(0.06*(x1-x0))), 3))
                    m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
                    cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # 幅の大きい水平帯を優先
                    best = None
                    bestw = 0
                    for c in cnts:
                        x, y, w, h = cv2.boundingRect(c)
                        if w >= 0.5*(x1-x0) and 8 <= h <= 200:
                            if w > bestw:
                                best, bestw = (int(y), int(y+h)), w
                    if best is not None:
                        return best
                except Exception:
                    return None
                return None

            band_y = _find_orange_band(image_rgb, x_left, x_right)
            if band_y is not None:
                # ヘッダー帯の上下に少し拡張してROI確定
                pad_up = int(os.getenv('DOCJA_GANTT_HEADER_UP', '30'))
                pad_dn = int(os.getenv('DOCJA_GANTT_HEADER_DN', '20'))
                y_top = max(0, band_y[0] - pad_up)
                y_bot = min(Himg - 1, band_y[1] + pad_dn)

            roi = image_rgb[y_top:y_bot, x_left:x_right]
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            # 1) Try Hough-based vertical line detection (strong lines)
            edges = cv2.Canny(gray, 70, 150)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=max(60, int(0.15 * (y_bot - y_top))),
                minLineLength=max(20, int(0.20 * (y_bot - y_top))),
                maxLineGap=8
            )
            xs: List[int] = []
            if lines is not None:
                for l in lines.reshape(-1, 4):
                    x1, y1, x2, y2 = l
                    if abs(x1 - x2) <= 2:
                        xs.append(int((x1 + x2) / 2) + x_left)
            xs.sort()
            merged: List[int] = []
            for x in xs:
                if not merged or abs(x - merged[-1]) > 8:
                    merged.append(x)
            # フォールバック: 縦投影ピーク（オレンジ帯を強調）
            if not merged:
                # まずヘッダー帯（検出済み）でピークを取り、無ければ y_top 近傍
                if band_y is not None:
                    yh0, yh1 = max(0, band_y[0]-2), min(Himg-1, band_y[1]+2)
                else:
                    header_margin_up = int(float(os.getenv('DOCJA_GANTT_HEADER_UP', '30')))
                    header_margin_dn = int(float(os.getenv('DOCJA_GANTT_HEADER_DN', '20')))
                    yh0 = max(0, y_top - header_margin_up)
                    yh1 = min(Himg - 1, y_top + header_margin_dn)
                roi_hdr = image_rgb[yh0:yh1, x_left:x_right]
                gray_hdr = cv2.cvtColor(roi_hdr, cv2.COLOR_RGB2GRAY) if roi_hdr.size else gray

                def _vertical_peaks(gray_img: np.ndarray, x_off: int) -> List[int]:
                    # Sobelベースの縦エッジ投影 + 平滑化 + 閾値走査
                    sobelx = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=3)
                    sobelx = np.abs(sobelx).astype(np.float32)
                    proj = np.sum(sobelx, axis=0)
                    # 平滑化窓は画像幅に応じて調整
                    k = max(5, int((proj.shape[0] / 80))) | 1
                    proj_sm = cv2.GaussianBlur(proj.reshape(1, -1), (k, 1), 0).flatten()
                    # 環境変数で閾値調整（デフォルト0.18倍）
                    thr_ratio = float(os.getenv('DOCJA_GANTT_COL_THR', '0.18'))
                    thr = max(5.0, thr_ratio * float(np.max(proj_sm) + 1e-6))
                    # ローカルピーク抽出（最小距離は8px）
                    peaks_idx: List[int] = []
                    last = -999
                    for i, v in enumerate(proj_sm):
                        if v >= thr:
                            if i - last > 8:
                                peaks_idx.append(i)
                                last = i
                    return [int(i) + x_off for i in peaks_idx]

                merged = _vertical_peaks(gray_hdr, x_left)
                # さらに、ROI本体でも試し、両者を統合（重複除去）
                merged_all = set(merged)
                for x in _vertical_peaks(gray, x_left):
                    if not merged_all or min(abs(x - m) for m in merged_all) > 6:
                        merged_all.add(x)
                merged = sorted(list(merged_all))

            # Columnizer（ピーク+ラベル+バー端点の統合）でも列候補を推定し、精度が上がるなら置換
            try:
                col_xs = estimate_columns(
                    image_rgb=image_rgb,
                    roi=(x_left, y_top, x_right, y_bot),
                    band_y=band_y,
                    bar_boxes=bars,
                    tick_xs=[t['x'] for t in ticks] if ticks else [],
                )
                if col_xs and (len(col_xs) > len(merged) or not merged):
                    merged = sorted(list({int(x) for x in col_xs}))
            except Exception:
                pass
            # ラベルの割当（曜日1文字など）
            label_points = []
            for tb in text_blocks:
                ttxt = (tb.get('text') or '').strip()
                if len(ttxt) == 1 and ttxt in DOW:
                    x1, y1, x2, y2 = tb.get('bbox', [0, 0, 0, 0])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    # ヘッダー帯の近傍（やや下も許容）にある曜日ラベルを採用
                    if cy < (y_top + max(8, int(0.01 * Himg))):
                        label_points.append((cx, ttxt))
            # スケジュール領域の左端候補（曜日ラベルの最小x）
            schedule_x_min = None
            if label_points:
                try:
                    schedule_x_min = float(min(p[0] for p in label_points)) - 8.0
                except Exception:
                    schedule_x_min = None
            if merged:
                for x in merged:
                    lab = None
                    bestd = 1e9
                    for cx, ttxt in label_points:
                        d = abs(cx - x)
                        if d < bestd and d < 40:
                            bestd = d
                            lab = ttxt
                columns.append({'x_px': float(x), 'label': lab})
            elif label_points:
                # ラベルのみで列を定義（文字の中心xをカラムとする）
                for cx, ttxt in sorted(label_points, key=lambda v: v[0]):
                    columns.append({'x_px': float(cx), 'label': ttxt})
            # さらに列が見つからない場合は、バーのx端点から列候補を生成
            if not columns and bars:
                xs_raw: List[int] = []
                for (x1, y1, x2, y2) in bars:
                    xs_raw.extend([x1, x2])
                xs_raw.sort()
                xs_merged: List[int] = []
                for x in xs_raw:
                    if not xs_merged or abs(x - xs_merged[-1]) > 10:
                        xs_merged.append(x)
                for x in xs_merged:
                    columns.append({'x_px': float(x), 'label': None})

            # 最低列数の下限（不足時は一様補間で補完）
            try:
                min_cols = int(os.getenv('DOCJA_GANTT_MIN_COLS', '0'))
            except Exception:
                min_cols = 0
            if columns and min_cols > 0 and len(columns) < min_cols:
                cols_sorted = sorted(columns, key=lambda c: c['x_px'])
                if len(cols_sorted) >= 2:
                    # 均等間隔で補完（端の2点から平均間隔推定）
                    xs = [c['x_px'] for c in cols_sorted]
                    mean_step = float(np.median(np.diff(xs))) if len(xs) >= 3 else float((xs[-1]-xs[0]) / max(1, len(xs)-1))
                    # 端から補う
                    while len(cols_sorted) < min_cols:
                        xs = [c['x_px'] for c in cols_sorted]
                        # 左側に追加
                        x_new_left = xs[0] - mean_step
                        cols_sorted.insert(0, {'x_px': float(x_new_left), 'label': None})
                        if len(cols_sorted) >= min_cols:
                            break
                        # 右側に追加
                        x_new_right = xs[-1] + mean_step
                        cols_sorted.append({'x_px': float(x_new_right), 'label': None})
                    columns = cols_sorted[:min_cols]

            # --- FORCE: dotted grid cell occupancy detection ---
            force_cell = (os.getenv('DOCJA_GANTT_FORCE_CELL', '0') == '1')
            # Disable automatic fallback; run only when explicitly requested
            if force_cell:
                try:
                    # derive vertical and horizontal dotted grids via adaptive threshold and morphology
                    # Use the broader grid ROI (xL,yT,xR,yB) but clamp to schedule_x_min if available
                    xL_eff = int(xL) if 'xL' in locals() else int(x_left)
                    if schedule_x_min is not None:
                        xL_eff = max(xL_eff, int(schedule_x_min))
                    yT_eff = int(yT) if 'yT' in locals() else int(y_top)
                    xR_eff = int(xR) if 'xR' in locals() else int(x_right)
                    yB_eff = int(yB) if 'yB' in locals() else int(y_bot)
                    roi_gray = cv2.cvtColor(image_rgb[yT_eff:yB_eff, xL_eff:xR_eff], cv2.COLOR_RGB2GRAY)
                    bin_inv = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                     cv2.THRESH_BINARY_INV, 31, 10)
                    vk = max(3, int(os.getenv('DOCJA_GANTT_VK', '7')))
                    hk = max(3, int(os.getenv('DOCJA_GANTT_HK', '7')))
                    vmask = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)), iterations=1)
                    hmask = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)), iterations=1)
                    # project to get candidate lines
                    vx = np.sum(vmask > 0, axis=0)
                    vy = np.sum(hmask > 0, axis=1)
                    # cluster peaks
                    def _peaks(arr, min_sep=8, thr_ratio=0.25):
                        thr = max(1.0, float(np.max(arr)) * thr_ratio)
                        p = []
                        last = -10_000
                        for i in range(1, len(arr) - 1):
                            if arr[i] >= thr and arr[i] >= arr[i-1] and arr[i] >= arr[i+1]:
                                if i - last >= min_sep:
                                    p.append(i)
                                    last = i
                        return p
                    c_idx = _peaks(vx, min_sep=max(6, int(0.01*(xR_eff-xL_eff))), thr_ratio=0.2)
                    r_idx = _peaks(vy, min_sep=max(6, int(0.01*(yB_eff-yT_eff))), thr_ratio=0.2)
                    # map to absolute coords
                    col_abs = [int(xL_eff + i) for i in c_idx]
                    row_abs = [int(yT_eff + i) for i in r_idx]
                    if len(col_abs) >= 3 and len(row_abs) >= 2:
                        # detect colored occupancy inside each cell, restricted to task-like colors
                        hsv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
                        # legend palette (RGB) for color proximity filter
                        legend_palette = []
                        try:
                            for lg in (legend or []):
                                col = lg.get('color_rgb') if isinstance(lg, dict) else None
                                if isinstance(col, (list, tuple)) and len(col) == 3:
                                    legend_palette.append(tuple(int(v) for v in col))
                        except Exception:
                            legend_palette = []
                        bar_hmin = int(os.getenv('DOCJA_GANTT_BAR_HMIN', '5'))
                        bar_hmax = int(os.getenv('DOCJA_GANTT_BAR_HMAX', '45'))
                        bar_smin = int(os.getenv('DOCJA_GANTT_CELL_SMIN', '40'))
                        # cells_active: list of {row_index, col_index, bbox}
                        cells_active: List[Dict[str, Any]] = []
                        # also merge contiguous actives into task segments per row
                        for ri in range(len(row_abs)-1):
                            y1 = row_abs[ri]
                            y2 = row_abs[ri+1]
                            row_hits: List[int] = []
                            for ci in range(len(col_abs)-1):
                                x1 = col_abs[ci]
                                x2 = col_abs[ci+1]
                                roi_hsv = hsv_img[y1:y2, x1:x2]
                                if roi_hsv.size == 0:
                                    continue
                                H = roi_hsv[:, :, 0].astype(np.float32)
                                S = roi_hsv[:, :, 1].astype(np.float32)
                                # primary rule: orange-ish high saturation
                                mask_bar = ((H >= bar_hmin) & (H <= bar_hmax) & (S >= bar_smin))
                                score = float(np.mean(mask_bar.astype(np.float32)))
                                # secondary: near legend colors (euclidean distance in RGB)
                                near_legend = False
                                if not legend_palette:
                                    near_legend = False
                                else:
                                    roi_rgb = image_rgb[y1:y2, x1:x2]
                                    mean_rgb = tuple(np.mean(roi_rgb.reshape(-1, 3), axis=0).tolist())
                                    for pal in legend_palette:
                                        dr = mean_rgb[0] - pal[0]
                                        dg = mean_rgb[1] - pal[1]
                                        db = mean_rgb[2] - pal[2]
                                        if (dr*dr + dg*dg + db*db) <= (int(os.getenv('DOCJA_GANTT_PALETTE_DIST2', '1400'))):
                                            near_legend = True
                                            break
                                if score >= float(os.getenv('DOCJA_GANTT_CELL_RATE', '0.03')) or near_legend:
                                    row_hits.append(ci)
                                    cells_active.append({'row_index': ri, 'col_index': ci, 'bbox': [x1, y1, x2, y2]})
                            # merge hits into segments and add as tasks
                            if row_hits:
                                start = row_hits[0]
                                for j in range(1, len(row_hits)+1):
                                    if j == len(row_hits) or row_hits[j] != row_hits[j-1] + 1:
                                        end = row_hits[j-1]
                                        sx = col_abs[start]
                                        ex = col_abs[end+1]
                                        y1_band = y1
                                        y2_band = y2
                                        tasks.append({
                                            'id': f'TCELL_{ri}_{start}_{end}',
                                            'name': rows[ri]['label'] if ri < len(rows) else '',
                                            'row_index': int(ri),
                                            'start_px': float(sx),
                                            'end_px': float(ex),
                                            'start_date': None,
                                            'end_date': None,
                                            'color_rgba': None,
                                            'dependency_ids': [],
                                            'start_col_index': int(start),
                                            'end_col_index': int(end),
                                            'bbox': [float(sx), float(y1_band), float(ex), float(y2_band)],
                                        })
                                        start = row_hits[j] if j < len(row_hits) else start
                        # adopt forced columns if we had no columns or they are very few
                        if not columns or len(columns) <= 2 or force_cell:
                            # If schedule_x_min is known, drop columns to its left
                            if schedule_x_min is not None:
                                col_abs = [x for x in col_abs if x >= schedule_x_min]
                            columns = [{'x_px': float(x), 'label': None} for x in col_abs]
                        # stash for overlay and grid bbox from detected grid
                        chart_cells_active = [{'row_index': c['row_index'], 'col_index': c['col_index'], 'bbox': c['bbox']} for c in cells_active]
                        # attach to this function scope via columns list marker (hacky: add to columns meta later below)
                        locals()['chart_cells_active'] = chart_cells_active
                        if col_abs and row_abs:
                            locals()['chart_grid_bbox'] = {
                                'x_left': int(min(col_abs)),
                                'x_right': int(max(col_abs)),
                                'y_top': int(min(row_abs)),
                                'y_bottom': int(max(row_abs))
                            }
                except Exception:
                    pass
        except Exception:
            pass

    # 列情報からタスクの開始/終了カラムを推定
    if columns:
        xs = [c['x_px'] for c in columns]
        def _nearest_col(x):
            if x is None or not xs:
                return None, None, None
            dists = [abs(x - xi) for xi in xs]
            idx = int(np.argmin(dists))
            return idx, columns[idx].get('label'), xs[idx]
        for t in tasks:
            if t.get('start_px') is not None:
                idx, lbl, snapped = _nearest_col(t.get('start_px'))
                t['start_col_index'] = idx
                t['start_col_label'] = lbl
                if snapped is not None:
                    t['start_px_snapped'] = float(snapped)
            if t.get('end_px') is not None:
                idx, lbl, snapped = _nearest_col(t.get('end_px'))
                t['end_col_index'] = idx
                t['end_col_label'] = lbl
                if snapped is not None:
                    t['end_px_snapped'] = float(snapped)

    chart = {
        'type': 'gantt',
        'calendar': {
            'x_origin_px': float(x_origin_px) if x_origin_px is not None else 0.0,
            'px_per_day': float(px_per_day) if px_per_day is not None else None,
            'timezone': 'UTC'
        },
        'axis_ticks': [
            {'x_px': float(t['x']), 'date_iso': t['date'].strftime('%Y-%m-%d')} for t in ticks
        ],
        'legend': legend,
        'columns': columns if columns else None,
        'tasks': tasks,
        'dependencies': []
    }
    # grid bbox for overlay range (limit thin gray lines to inside table grid)
    try:
        if 'chart_grid_bbox' in locals():
            chart['grid_bbox'] = locals()['chart_grid_bbox']
        else:
            x_left_grid = int(min([c['x_px'] for c in columns])) if columns else int(xL)
            x_right_grid = int(max([c['x_px'] for c in columns])) if columns else int(xR)
            # y範囲は行ラベルから推定（なければROI）
            y_top_fb = int(rows_y_min) if rows_y_min is not None else int(yT)
            y_bot_fb = int(rows_y_max) if rows_y_max is not None else int(yB)
            chart['grid_bbox'] = {
                'x_left': max(0, x_left_grid - 1),
                'x_right': min(int(W), x_right_grid + 1),
                'y_top': max(0, y_top_fb - 2),
                'y_bottom': min(int(H), y_bot_fb + 2)
            }
    except Exception:
        chart['grid_bbox'] = {'x_left': int(xL), 'x_right': int(xR), 'y_top': int(yT), 'y_bottom': int(yB)}
    # Attach forced active cells if present
    if 'chart_cells_active' in locals():
        chart['cells_active'] = locals()['chart_cells_active']
    return [chart]
