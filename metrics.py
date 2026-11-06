import numpy as np
import torch
from common import NC, box_iou


def average_precision(rec, prec):
    """VOC all-point interpolated AP: area under the monotone precision envelope."""
    r = np.concatenate([[0], rec, [1]])
    p = np.concatenate([[0], prec, [0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    i = np.where(r[1:] != r[:-1])[0]
    return float(((r[i + 1] - r[i]) * p[i + 1]).sum())


def evaluate(preds, gts, masks, iou_thr=0.5):
    """preds: list of (K,6) tensors (x1,y1,x2,y2,score,cls); gts: list of (N,5) tensors (cls,x1,y1,x2,y2);
    masks: list of annotated-class lists. A class is evaluated only on images whose dataset annotates it.
    Returns per class dict with AP, precision/recall curve, best-F1 point and the number of GT boxes."""
    out = {}
    for c in range(NC):
        dets, npos = [], 0
        for i, (p, g, m) in enumerate(zip(preds, gts, masks)):
            if c not in m:
                continue
            gc = g[g[:, 0] == c, 1:]
            pc = p[p[:, 5] == c]
            npos += len(gc)
            pc = pc[pc[:, 4].argsort(descending=True)]
            used = torch.zeros(len(gc), dtype=torch.bool)
            iou = box_iou(pc[:, :4], gc) if len(gc) and len(pc) else None
            for k in range(len(pc)):
                tp = 0
                if iou is not None:
                    v, j = (iou[k] * ~used).max(0)
                    if v >= iou_thr:
                        used[j] = True
                        tp = 1
                dets.append((pc[k, 4].item(), tp))
        if not npos:
            continue
        dets.sort(key=lambda d: -d[0])
        tp = np.cumsum([d[1] for d in dets]) if dets else np.zeros(0)
        fp = np.cumsum([1 - d[1] for d in dets]) if dets else np.zeros(0)
        rec = tp / npos
        prec = tp / np.maximum(tp + fp, 1e-9)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        b = int(f1.argmax()) if len(f1) else 0
        out[c] = {"ap": average_precision(rec, prec), "rec": rec, "prec": prec, "n": npos,
                  "scores": np.array([d[0] for d in dets]),
                  "f1": float(f1[b]) if len(f1) else 0.0, "p": float(prec[b]) if len(f1) else 0.0,
                  "r": float(rec[b]) if len(f1) else 0.0, "thr": dets[b][0] if dets else 1.0}
    return out
