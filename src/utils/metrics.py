from typing import List, Tuple, Dict, Any
import numpy as np


def bbox_iou(box1: List[float], box2: List[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0
    inter = (xi_max - xi_min) * (yi_max - yi_min)
    area1 = max(0.0, (x1_max - x1_min)) * max(0.0, (y1_max - y1_min))
    area2 = max(0.0, (x2_max - x2_min)) * max(0.0, (y2_max - y2_min))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds: List[List[float]], gts: List[List[float]], iou_thresh: float = 0.5) -> Tuple[int, int, int]:
    matched_gt = set()
    tp = 0
    for pb in preds:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gts):
            if j in matched_gt:
                continue
            iou = bbox_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def detection_metrics(pred_boxes: List[List[float]], gt_boxes: List[List[float]], iou_thresh: float = 0.5) -> Dict[str, float]:
    tp, fp, fn = greedy_match(pred_boxes, gt_boxes, iou_thresh)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # H-mean is equivalent to F1 for two classes; keep separate for convention
    hmean = f1
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hmean': hmean,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def levenshtein(a: str, b: str) -> int:
    # iterative DP
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / max(1, len(ref))


def normalized_edit_similarity(a: str, b: str) -> float:
    # proxy for TEDS: 1 - normalized edit distance
    if not a and not b:
        return 1.0
    dist = levenshtein(a, b)
    denom = max(len(a), len(b), 1)
    return 1.0 - (dist / denom)


def _table_signature(html: str) -> str:
    """Build a simple structural signature from HTML table.
    Counts cells per row and header flags; ignores attributes.
    """
    s = html.lower()
    # split by rows
    parts = s.split('<tr')
    sig = []
    for part in parts[1:]:
        row_html = part.split('</tr>')[0]
        th = row_html.count('<th')
        td = row_html.count('<td')
        sig.append(f"{th}h{td}d")
    return '|'.join(sig)


def _strip_tags(html: str) -> str:
    out = []
    inside = False
    for ch in html:
        if ch == '<':
            inside = True
        elif ch == '>':
            inside = False
            out.append(' ')
        elif not inside:
            out.append(ch)
    return ''.join(out).strip()


def teds_like_score(gt_html: str, pred_html: str, structure_weight: float = 0.5) -> float:
    """Approximate TEDS by combining structural signature similarity and text similarity.
    This is a placeholder until full TEDS is wired in.
    """
    sig_gt = _table_signature(gt_html)
    sig_pr = _table_signature(pred_html)
    struct_sim = normalized_edit_similarity(sig_gt, sig_pr)

    txt_gt = _strip_tags(gt_html)
    txt_pr = _strip_tags(pred_html)
    text_sim = normalized_edit_similarity(txt_gt, txt_pr)

    w = max(0.0, min(1.0, structure_weight))
    return w * struct_sim + (1 - w) * text_sim
