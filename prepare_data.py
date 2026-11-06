"""Builds the unified detection dataset, the anchor boxes and the two OCR datasets from data/raw."""
import glob
import hashlib
import json
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import cv2
import numpy as np
from common import RAW, PROC, FIGS, RESULTS, CLASSES, NC, COLORS, IMG, CHARS, CH, CW, imread, imwrite

random.seed(0)
np.random.seed(0)
EXT = (".jpg", ".jpeg", ".png")


def yolo_labels(img_path):
    lab = re.sub(r"[\\/]images[\\/]", "/labels/", img_path)
    lab = os.path.splitext(lab)[0] + ".txt"
    if not os.path.exists(lab):
        return []
    out = []
    for line in open(lab).read().split("\n"):
        p = line.split()
        if len(p) == 5:
            c, x, y, w, h = int(p[0]), *map(float, p[1:])
            out.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return out


def images(pattern):
    return sorted(f for f in glob.glob(pattern, recursive=True) if f.lower().endswith(EXT))


def rf_source(path):
    """Roboflow exports name augmented copies '<source>_jpg.rf.<hash>.jpg'."""
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", os.path.basename(path))[0]


def inside(head, box):
    cx, cy = (head[1] + head[3]) / 2, (head[2] + head[4]) / 2
    return box[1] <= cx <= box[3] and box[2] <= cy <= box[4]


# ---------------- per-dataset readers: (path, [(cls, x1, y1, x2, y2) normalised], annotated classes, group) -------------
def read_helmet_violations():
    recs = []
    for f in images(f"{RAW}/helmet/HelmetViolationsV2/*/images/*"):
        b = [({0: 5, 1: 0, 2: 1}[c], *xy) for c, *xy in yolo_labels(f)]
        has_rider = any(c in (0, 1) for c, *_ in b)
        # images with only a plate annotated often contain unlabelled riders -> do not penalise rider classes there
        recs.append((f, b, [0, 1, 5] if has_rider else [5], rf_source(f)))
    return recs, "group"


def read_rider_helmet_plate():
    recs = []
    for f in images(f"{RAW}/rider-with-helmet-without-helmet-number-plate/*/images/*"):
        lab = yolo_labels(f)
        heads = [l for l in lab if l[0] in (0, 1)]
        b = [(5, *l[1:]) for l in lab if l[0] == 3]
        for r in (l for l in lab if l[0] == 2):
            hs = [h for h in heads if inside(h, r)]
            if hs:  # rider unit is a violator if any head inside it is without a helmet
                b.append((1 if any(h[0] == 1 for h in hs) else 0, *r[1:]))
        recs.append((f, b, [0, 1, 5], os.path.basename(f)))
    return recs, "group"


def read_seatbelt_anpr():
    recs = []
    for f in images(f"{RAW}/seatbelt-detection/*/*/images/*"):
        b = [({0: 4, 1: 3, 2: 2}[c], *xy) for c, *xy in yolo_labels(f)]
        frame = int(re.search(r"mp4-(\d+)_jpg", f).group(1))
        recs.append((f, b, [2, 3, 4], frame))
    return recs, "time"


def read_seatbelt_akaike():
    recs = []
    for f in images(f"{RAW}/seatbelt-detection-v1i/*/images/*"):
        b = [({1: 3, 2: 4, 3: 2}[c], *xy) for c, *xy in yolo_labels(f) if c != 0]
        recs.append((f, b, [2, 3, 4], rf_source(f)))
    return recs, "time"


def read_indian_plates():
    recs = []
    for f in images(f"{RAW}/indian-license-plates-with-labels/images/*"):
        recs.append((f, [(5, *xy) for _, *xy in yolo_labels(f)], [5], os.path.basename(f)))
    return recs, "group"


READERS = {"helmetviol": read_helmet_violations, "riderhp": read_rider_helmet_plate,
           "seatbelt_anpr": read_seatbelt_anpr, "seatbelt_akaike": read_seatbelt_akaike,
           "inplates": read_indian_plates}


def split_groups(keys, mode):
    """70/15/15 split of group keys. 'time' keeps consecutive video frames together (no near-duplicate leakage)."""
    keys = sorted(set(keys))
    if mode == "group":
        random.shuffle(keys)
    n = len(keys)
    a, b = int(0.7 * n), int(0.85 * n)
    return {**{k: "train" for k in keys[:a]}, **{k: "val" for k in keys[a:b]}, **{k: "test" for k in keys[b:]}}


def build_detection():
    out = {"train": [], "val": [], "test": []}
    os.makedirs(f"{PROC}/images", exist_ok=True)
    seen = set()
    for name, reader in READERS.items():
        recs, mode = reader()
        where = split_groups([r[3] for r in recs], mode)
        used = set()
        for i, (f, boxes, mask, g) in enumerate(recs):
            split = where[g]
            if split != "train" and g in used:  # augmented copies of one source: keep one for val/test
                continue
            raw = open(f, "rb").read()
            h = hashlib.md5(raw).hexdigest()
            im = imread(f)
            if h in seen or im is None:
                print("skipped", "duplicate" if h in seen else "unreadable", f)
                continue
            seen.add(h)
            used.add(g)
            H, W = im.shape[:2]
            s = IMG / max(H, W)
            im = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            h2, w2 = im.shape[:2]
            bx = []
            for c, x1, y1, x2, y2 in boxes:
                x1, x2 = np.clip([x1 * w2, x2 * w2], 0, w2)
                y1, y2 = np.clip([y1 * h2, y2 * h2], 0, h2)
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    bx.append([c, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
            fn = f"images/{name}_{i:05d}.jpg"
            imwrite(f"{PROC}/{fn}", im)
            out[split].append({"file": fn, "src": os.path.relpath(f, RAW), "ds": name, "w": w2, "h": h2,
                               "boxes": bx, "mask": mask})
    for s, v in out.items():
        with open(f"{PROC}/{s}.json", "w") as fp:
            json.dump(v, fp)
    return out


def kmeans_anchors(data, k=9, iters=300):
    """YOLOv2-style anchors: K-means (Lec 13) on box (w, h) with distance 1 - IoU."""
    wh = np.array([[b[3] - b[1], b[4] - b[2]] for r in data for b in r["boxes"]])
    cen = wh[np.random.choice(len(wh), k, replace=False)]
    for _ in range(iters):
        inter = np.minimum(wh[:, None], cen[None]).prod(2)
        iou = inter / (wh.prod(1)[:, None] + cen.prod(1)[None] - inter)
        a = iou.argmax(1)
        new = np.array([np.median(wh[a == j], 0) if (a == j).any() else cen[j] for j in range(k)])
        if np.allclose(new, cen):
            break
        cen = new
    cen = cen[np.argsort(cen.prod(1))]
    inter = np.minimum(wh[:, None], cen[None]).prod(2)
    mean_iou = (inter / (wh.prod(1)[:, None] + cen.prod(1)[None] - inter)).max(1).mean()
    return cen.round(1).tolist(), float(mean_iou)


def build_chars():
    """Character classification set (35 classes) -> 80/10/10 split, images resized to CH x CW."""
    X, y = [], []
    for j, c in enumerate(CHARS):
        for f in sorted(glob.glob(f"{RAW}/license-plate-digits-classification-dataset/CNN letter Dataset/{c}/*")):
            X.append(cv2.resize(imread(f, cv2.IMREAD_GRAYSCALE), (CW, CH), interpolation=cv2.INTER_AREA))
            y.append(j)
    X, y = np.stack(X), np.array(y)
    p = np.random.permutation(len(y))
    a, b = int(0.8 * len(y)), int(0.9 * len(y))
    np.savez_compressed(f"{PROC}/chars.npz", Xtr=X[p[:a]], ytr=y[p[:a]], Xva=X[p[a:b]], yva=y[p[a:b]],
                        Xte=X[p[b:]], yte=y[p[b:]])
    return len(y)


def plate_rows(boxes):
    """Groups character boxes into 1 or 2 text rows using the largest gap in their y-centres."""
    cy = np.array([(b[1] + b[3]) / 2 for b in boxes])
    hgt = np.median([b[3] - b[1] for b in boxes])
    o = np.argsort(cy)
    gaps = np.diff(cy[o])
    if len(gaps) and gaps.max() > 0.5 * hgt:
        cut = gaps.argmax() + 1
        rows = [o[:cut], o[cut:]]
    else:
        rows = [o]
    return [sorted(r, key=lambda i: boxes[i][0]) for r in rows]


def build_plates():
    """Indian plate crops with per-character boxes: ground truth for segmentation and full-plate OCR."""
    recs = []
    for xf in sorted(glob.glob(f"{RAW}/license-plate-characters-detection-ocr/LP-characters/annotations/*.xml")):
        r = ET.parse(xf).getroot()
        img = glob.glob(f"{RAW}/license-plate-characters-detection-ocr/LP-characters/images/{os.path.splitext(r.find('filename').text)[0]}.*")
        if not img:
            continue
        objs = [(o.find("name").text.upper().replace("O", "0"),
                 *[int(float(o.find("bndbox").find(t).text)) for t in ("xmin", "ymin", "xmax", "ymax")])
                for o in r.findall("object")]
        boxes = [o[1:] for o in objs]
        rows = plate_rows(boxes)
        text = "".join(objs[i][0] for row in rows for i in row)
        recs.append({"file": os.path.relpath(img[0], RAW), "text": text, "rows": len(rows),
                     "chars": [[objs[i][0], *boxes[i]] for row in rows for i in row]})
    random.shuffle(recs)
    n = len(recs) // 3
    with open(f"{PROC}/plates.json", "w") as fp:
        json.dump({"dev": recs[:n], "test": recs[n:]}, fp)
    return len(recs[:n]), len(recs[n:]), sum(r["rows"] == 2 for r in recs)


def sample_figure(data):
    tiles = []
    for ds in READERS:
        r = next(r for r in data["train"] if r["ds"] == ds and len(r["boxes"]) >= 2)
        im = imread(f"{PROC}/{r['file']}")
        for c, *b in r["boxes"]:
            p = list(map(int, b))
            cv2.rectangle(im, p[:2], p[2:], COLORS[c], 2)
            cv2.putText(im, CLASSES[c], (p[0], max(12, p[1] - 4)), 0, 0.5, COLORS[c], 2)
        tiles.append(cv2.resize(im, (400, 400)))
    imwrite(f"{FIGS}/dataset_samples.jpg", np.hstack(tiles), 90)


if __name__ == "__main__":
    data = build_detection()
    anchors, miou = kmeans_anchors(data["train"])
    with open(f"{PROC}/anchors.json", "w") as fp:
        json.dump(anchors, fp)
    nchars = build_chars()
    ndev, ntest, two = build_plates()
    sample_figure(data)

    lines = ["Detection dataset (images / boxes per class)"]
    lines.append(f"{'dataset':16s}" + "".join(f"{s:>8s}" for s in data))
    for ds in READERS:
        lines.append(f"{ds:16s}" + "".join(f"{sum(r['ds'] == ds for r in data[s]):8d}" for s in data))
    lines.append(f"{'total':16s}" + "".join(f"{len(data[s]):8d}" for s in data))
    lines.append("")
    lines.append(f"{'class':16s}" + "".join(f"{s:>8s}" for s in data))
    for c in range(NC):
        lines.append(f"{CLASSES[c]:16s}" + "".join(f"{sum(b[0] == c for r in data[s] for b in r['boxes']):8d}" for s in data))
    lines.append("")
    lines.append(f"K-means anchors (w,h at {IMG}px): {anchors}")
    lines.append(f"Mean best-anchor IoU of training boxes: {miou:.3f}")
    lines.append(f"Character images: {nchars}")
    lines.append(f"Plate OCR set: {ndev} dev / {ntest} test plates ({two} two-row plates)")
    print("\n".join(lines))
    with open(f"{RESULTS}/data_stats.txt", "w") as fp:
        fp.write("\n".join(lines) + "\n")
