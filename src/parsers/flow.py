"""
FlowParser (v0.2): ヒューリスティックによりテキストブロックから
簡易的な業務フロー（有向グラフ）を復元します。

- ノード: 読み順（reading_order）に沿ってテキストブロックを順列化し、
  各ブロックを process ノードとして登録
- エッジ: 隣接ノードを直線で連結（順序のみ）
- レーン: 左端近傍の「部/課/室/チーム/工程」等を含む見出しっぽいテキストを
  横帯レーンとして推定し、ノード中心Yでレーンに割り当て

将来的に線分/矢印・レーン境界の画像処理で精度を上げます。
"""
from typing import Dict, Any, List, Tuple, Optional
import os
import numpy as np
import cv2


LANE_HINTS = ('部', '課', '室', 'チーム', '工程', 'レーン', '担当', '部署')


def _center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def _infer_swimlanes(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """左側20%の幅にある見出し風テキストから簡易レーンを推定。"""
    W = float(page.get('width') or page.get('size', {}).get('width', 0) or 0)
    text_blocks = page.get('text_blocks', [])
    lanes: List[Dict[str, Any]] = []
    if not W or not text_blocks:
        return lanes
    margin = 0.2 * W
    # 候補抽出
    cand = []
    for tb in text_blocks:
        bbox = tb.get('bbox') or [0, 0, 0, 0]
        cx, cy = _center(bbox)
        if cx <= margin:
            t = (tb.get('text') or '').strip()
            if any(h in t for h in LANE_HINTS) and len(t) <= 20:
                cand.append({'y': cy, 'label': t, 'bbox': bbox})
    # 重複をY座標でクラスタ
    cand.sort(key=lambda x: x['y'])
    lanes = []
    for i, c in enumerate(cand):
        lanes.append({'id': f'L{i+1}', 'label': c['label'], 'bbox': c['bbox']})
    return lanes


def _angle(a: Tuple[float,float], b: Tuple[float,float], c: Tuple[float,float]) -> float:
    """Return angle ABC (in degrees)."""
    import math
    bax, bay = a[0]-b[0], a[1]-b[1]
    bcx, bcy = c[0]-b[0], c[1]-b[1]
    dot = bax*bcx + bay*bcy
    na = math.hypot(bax, bay) + 1e-6
    nb = math.hypot(bcx, bcy) + 1e-6
    cosv = max(-1.0, min(1.0, dot/(na*nb)))
    return abs(math.degrees(math.acos(cosv)))


def _detect_arrow_tips(image_rgb: np.ndarray) -> List[Tuple[int,int]]:
    """Detect potential arrow tips (apex of small triangles) via contour approx.

    Returns list of (x,y) integer coordinates for apex candidates.
    """
    tips: List[Tuple[int,int]] = []
    try:
        H, W = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Emphasize dark shapes
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 80, 160)
        # Close small gaps
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_min = max(60, int(0.00001 * W * H))
        area_max = max(800, int(0.002 * W * H))
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a < area_min or a > area_max:
                continue
            eps = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) == 3:  # triangle
                pts = [tuple(int(v) for v in p[0]) for p in approx]
                # compute angles and choose sharpest as apex
                angs = [
                    _angle(pts[(i-1)%3], pts[i], pts[(i+1)%3]) for i in range(3)
                ]
                i_apex = int(np.argmin(angs))
                if angs[i_apex] <= 50.0:  # sharp enough
                    tips.append((pts[i_apex][0], pts[i_apex][1]))
    except Exception:
        return []
    return tips


def parse_flow_from_page(page_dict: Dict[str, Any], image_rgb: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    text_blocks = page_dict.get('text_blocks', [])
    if len(text_blocks) < 2:
        return []

    order = page_dict.get('reading_order') or list(range(len(text_blocks)))
    nodes = []
    edges = []

    # 推定レーン
    lanes = _infer_swimlanes(page_dict)

    # ノード生成 + レーン割当（テキストブロック）
    for i, idx in enumerate(order):
        if not (0 <= idx < len(text_blocks)):
            continue
        tb = text_blocks[idx]
        bbox = tb.get('bbox') or [0, 0, 0, 0]
        cx, cy = _center(bbox)
        lane_id = None
        # ノード中心Yがレーンbbox内にあるものを選択
        for ln in lanes:
            x1, y1, x2, y2 = ln.get('bbox', [0, 0, 0, 0])
            if y1 <= cy <= y2:
                lane_id = ln['id']
                break
        nid = f"N{i+1}"
        nodes.append({'id': nid, 'bbox': bbox, 'shape': 'process', 'text': tb.get('text', ''), 'lane': lane_id})
        if i > 0:
            edges.append({'id': f'E{i}', 'source': f'N{i}', 'target': nid, 'kind': 'arrow'})

    # レイアウト要素のfigureもノードとして追加（アイコン等の見落とし補完）
    for el in page_dict.get('layout_elements', []) or []:
        lab = (el.get('label') or el.get('type') or '').lower()
        if lab in ('figure', 'icon') and 'bbox' in el:
            nodes.append({'id': f'F{len(nodes)+1}', 'bbox': el['bbox'], 'shape': 'figure', 'text': '', 'lane': None})

    # 画像がある場合、線分検出で補助（Hough）して近接ノード間を補強 + 形状ノードの追加
    arrow_tips: List[Tuple[int,int]] = []
    if image_rgb is not None and len(nodes) >= 2:
        try:
            img = image_rgb.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges_img = cv2.Canny(gray, 80, 160)
            lines = cv2.HoughLinesP(edges_img, 1, np.pi/180, threshold=80, minLineLength=40, maxLineGap=10)
            if lines is not None:
                # ノード中心に最も近い線の端点ペアから推定エッジを追加
                centers = [(n['id'], _center(n['bbox'])) for n in nodes]
                def _nearest_node(pt):
                    x, y = pt
                    best = None
                    bestd = 1e12
                    for nid, (cx, cy) in centers:
                        d = (cx-x)**2 + (cy-y)**2
                        if d < bestd:
                            bestd, best = d, nid
                    return best
                used_pairs = set((e['source'], e['target']) for e in edges)
                cnt = 0
                for l in lines.reshape(-1, 4):
                    x1, y1, x2, y2 = map(float, l)
                    a = _nearest_node((x1, y1))
                    b = _nearest_node((x2, y2))
                    if a and b and a != b and (a, b) not in used_pairs:
                        cnt += 1
                        eid = f'H{cnt}'
                        edges.append({'id': eid, 'source': a, 'target': b, 'kind': 'connector'})
                        used_pairs.add((a, b))
            # 形状ノード（大きめの塊）
            contours, _ = cv2.findContours(cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            Himg, Wimg = img.shape[:2]
            def _iou(b1, b2):
                x11,y11,x12,y12 = b1
                x21,y21,x22,y22 = b2
                xi1, yi1 = max(x11,x21), max(y11,y21)
                xi2, yi2 = min(x12,x22), min(y12,y22)
                w, h = max(0, xi2-xi1), max(0, yi2-yi1)
                inter = w*h
                a1 = max(0,(x12-x11)*(y12-y11))
                a2 = max(0,(x22-x21)*(y22-y21))
                uni = a1 + a2 - inter + 1e-6
                return inter/uni
            existing = [n['bbox'] for n in nodes]
            addc = 0
            for cntc in contours:
                x, y, w, h = cv2.boundingRect(cntc)
                if w*h < 800 or w*h > 200000:
                    continue
                bbox = [x, y, x+w, y+h]
                # 既存と重複大ならスキップ
                if any(_iou(bbox, eb) > 0.3 for eb in existing):
                    continue
                addc += 1
                nodes.append({'id': f'C{addc}', 'bbox': bbox, 'shape': 'icon', 'text': '', 'lane': None})

            # 矢印先端テンプレでエッジ方向を補正
            arrow_tips = _detect_arrow_tips(img)
        except Exception:
            pass

    # 方向補正（arrow_tipsに基づく）
    if arrow_tips and nodes and edges:
        try:
            # 検索用にノード中心辞書
            centers = {n['id']: _center(n['bbox']) for n in nodes if 'bbox' in n}
            # 閾値（px）: 画像短辺の比率で設定 + ENV上書き
            Htmp, Wtmp = (image_rgb.shape[:2] if image_rgb is not None else (1000, 1000))
            base_thr = 0.05 * float(min(Htmp, Wtmp))
            try:
                base_thr = float(os.getenv('DOCJA_FLOW_ARROW_MAX_DIST', str(base_thr)))
            except Exception:
                pass
            new_edges = []
            for e in edges:
                a = centers.get(e.get('source'))
                b = centers.get(e.get('target'))
                if not a or not b:
                    new_edges.append(e)
                    continue
                # 近い先端がどちら側にあるか
                import math
                def _min_dist(pt, pts):
                    return min(math.hypot(pt[0]-q[0], pt[1]-q[1]) for q in pts) if pts else 1e9
                da = _min_dist(a, arrow_tips)
                db = _min_dist(b, arrow_tips)
                corrected = False
                if db < base_thr and da >= db * 0.9:
                    # 矢印がtarget側 → 方向そのまま、kindをarrowに昇格
                    e2 = dict(e)
                    e2['kind'] = 'arrow'
                    e2['direction_corrected'] = True
                    new_edges.append(e2)
                    corrected = True
                elif da < base_thr and db > da * 1.1:
                    # 矢印がsource側 → 方向逆転
                    e2 = dict(e)
                    src, tgt = e2.get('source'), e2.get('target')
                    e2['source'], e2['target'] = tgt, src
                    e2['kind'] = 'arrow'
                    e2['direction_corrected'] = True
                    new_edges.append(e2)
                    corrected = True
                if not corrected:
                    new_edges.append(e)
            edges = new_edges
        except Exception:
            pass

    return [{
        'type': 'flow',
        'swimlanes': lanes,
        'nodes': nodes,
        'edges': edges,
    }]
