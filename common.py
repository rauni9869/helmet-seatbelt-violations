import json
import os
import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
RUNS = os.path.join(ROOT, "runs")
FIGS = os.path.join(ROOT, "figs")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "outputs")
for d in (PROC, RUNS, FIGS, RESULTS, OUT):
    os.makedirs(d, exist_ok=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"

# unified detector classes; 1 and 4 are the violations
CLASSES = ["rider_helmet", "rider_nohelmet", "car", "seatbelt", "no_seatbelt", "plate"]
NC = len(CLASSES)
COLORS = [(0, 200, 0), (0, 0, 255), (255, 160, 0), (0, 200, 200), (255, 0, 255), (0, 255, 255)]
IMG = 640

# characters of the plate OCR (the character dataset has no letter O; O is read as 0)
CHARS = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
CH, CW = 40, 30


def imread(p, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, np.uint8), flag)


def imwrite(p, im, q=95):
    cv2.imencode(os.path.splitext(p)[1], im, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tofile(p)


def load_split(split):
    with open(os.path.join(PROC, f"{split}.json")) as f:
        return json.load(f)


def box_iou(a, b):
    """IoU between every box of a (N,4) and b (M,4), xyxy."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def nms(boxes, scores, thr=0.5):
    """Greedy non-maximum suppression (Lec 17): keep the most confident box, drop the ones overlapping it."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        iou = box_iou(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][iou <= thr]
    return torch.stack(keep) if keep else torch.zeros(0, dtype=torch.long, device=boxes.device)


def letterbox(im, size=IMG):
    """Resize keeping aspect ratio and pad to a size x size square. Returns image, scale, (padx, pady)."""
    h, w = im.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    out = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out, s, (px, py)
