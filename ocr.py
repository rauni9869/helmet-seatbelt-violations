"""Number-plate reading with classical image processing + a CNN character classifier.
Steps: CLAHE -> deskew with Canny edges + Hough transform (Lec 11-12) -> Otsu binarisation -> connected components
-> row grouping by K-means (Lec 13) -> CNN per character -> Indian plate-format constrained decoding."""
import itertools
import cv2
import numpy as np
import torch
from common import CHARS, CH, CW, dev
from charnet import norm

PH = 96  # working height of a plate crop
TAU = 3.0  # max log-probability cost of the plate-format correction (chosen on the dev plates)
DIGITS = [CHARS.index(c) for c in "0123456789"]
LETTERS = [i for i in range(len(CHARS)) if i not in DIGITS]


def skew_angle(gray):
    """Dominant near-horizontal line angle (degrees) from the Hough transform of the Canny edge map."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(0.35 * gray.shape[1]))
    if lines is None:
        return 0.0, edges, []
    th = np.degrees(lines[:, 0, 1]) - 90  # theta = 90 deg is a horizontal line
    keep = np.abs(th) < 20
    if not keep.any():
        return 0.0, edges, []
    return float(np.median(th[keep][:10])), edges, lines[keep][:10, 0]


def rotate(im, ang, border):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border), M


def components(binary):
    """Character-like connected components of a binary image (characters = 255)."""
    H, W = binary.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cand = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (max(8, 0.12 * H) < h < 0.95 * H and w < 0.3 * W and 0.08 < w / h < 1.3 and a > 0.12 * w * h):
            continue
        cand.append([x, y, x + w, y + h])
    if not cand:
        return []
    cand = np.array(cand)
    hs = (cand[:, 3] - cand[:, 1]).astype(float)
    # characters share (roughly) one height: every component votes for its height, the best supported one wins
    votes = [(np.abs(hs - h) < 0.35 * h).sum() + h / 1000 for h in hs]
    href = hs[int(np.argmax(votes))]
    cand = cand[np.abs(hs - href) < 0.35 * href]
    out = []
    for x1, y1, x2, y2 in cand:  # split blobs of touching characters
        k = int(round((x2 - x1) / (0.62 * (y2 - y1))))
        if k >= 2:
            xs = np.linspace(x1, x2, k + 1)
            out += [[int(xs[j]), y1, int(xs[j + 1]), y2] for j in range(k)]
        else:
            out.append([x1, y1, x2, y2])
    return out


def kmeans_rows(boxes, h):
    """Splits characters into 1 or 2 rows: 1-D K-means (K = 2) on the y-centres; 2 rows only if well separated."""
    cy = np.array([(b[1] + b[3]) / 2 for b in boxes], float)
    if len(cy) < 4:
        return [sorted(range(len(boxes)), key=lambda i: boxes[i][0])]
    c = np.array([cy.min(), cy.max()])
    for _ in range(20):
        a = np.abs(cy[:, None] - c[None]).argmin(1)
        c = np.array([cy[a == k].mean() if (a == k).any() else c[k] for k in range(2)])
    if abs(c[1] - c[0]) < 0.6 * h or min((a == 0).sum(), (a == 1).sum()) < 2:
        return [sorted(range(len(boxes)), key=lambda i: boxes[i][0])]
    order = np.argsort(c)
    return [sorted(np.where(a == k)[0].tolist(), key=lambda i: boxes[i][0]) for k in order]


def fit_char(c):
    """Pads a character crop to the CH:CW aspect ratio with its background level and resizes it."""
    h, w = c.shape
    tw = int(round(h * CW / CH))
    if w < tw:
        bg = int(np.median(np.r_[c[:, 0], c[:, -1]]))
        c = cv2.copyMakeBorder(c, 0, 0, (tw - w) // 2, tw - w - (tw - w) // 2, cv2.BORDER_CONSTANT, value=bg)
    return cv2.resize(c, (CW, CH), interpolation=cv2.INTER_AREA)


def char_crops(gray, boxes, dark_text):
    """Crops characters (8% margin) and makes them dark-on-light, CH x CW."""
    out = []
    for x1, y1, x2, y2 in boxes:
        p = max(1, int(0.08 * (y2 - y1)))
        c = gray[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p]
        out.append(fit_char(c if dark_text else 255 - c))
    return np.stack(out) if out else np.zeros((0, CH, CW), np.uint8)


@torch.no_grad()
def classify(model, crops):
    if not len(crops):
        return np.zeros((0, len(CHARS)))
    return torch.softmax(model(norm(torch.from_numpy(crops).to(dev))), 1).cpu().numpy()


def indian_patterns(n):
    """Indian registration format: state (2 letters), district (1-2 digits), series (0-3 letters), number (1-4 digits)."""
    for d, s, k in itertools.product((1, 2), range(4), range(1, 5)):
        if 2 + d + s + k == n:
            yield "L" * 2 + "D" * d + "L" * s + "D" * k


def decode(probs, grammar=True, tau=TAU):
    """Arg-max reading, replaced by the best Indian-format reading if that costs less than tau in log-probability
    (a larger cost means the segmentation is probably wrong and forcing the format would only add errors)."""
    raw = "".join(CHARS[i] for i in probs.argmax(1))
    if not grammar:
        return raw
    lp = np.log(probs + 1e-9)
    best, text = lp.max(1).sum() - tau, raw
    for pat in indian_patterns(len(probs)):
        idx = [LETTERS if t == "L" else DIGITS for t in pat]
        sc = sum(lp[i, ix].max() for i, ix in enumerate(idx))
        if sc > best:
            best, text = sc, "".join(CHARS[ix[int(lp[i, ix].argmax())]] for i, ix in enumerate(idx))
    return text


def segment(bgr, deskew=True, model=None):
    """Returns the working gray image, character boxes (working coords, row-ordered), polarity, class
    probabilities (if a model is given) and debug images.
    Candidate binarisations: global Otsu and two local-mean thresholds, each for dark-on-light and light-on-dark
    text. With a model, the candidate whose characters the CNN recognises most confidently is kept
    (recognition-driven selection); without one, the candidate with most character-like components."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    s = PH / gray.shape[0]
    gray = cv2.resize(gray, (max(8, int(round(gray.shape[1] * s))), PH), interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(2.0, (4, 4)).apply(gray)
    ang, edges, lines = skew_angle(gray) if deskew else (0.0, None, [])
    M = cv2.getRotationMatrix2D((gray.shape[1] / 2, PH / 2), 0, 1.0)
    if abs(ang) > 0.5:
        gray, M = rotate(gray, ang, int(np.median(gray)))
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bins = [otsu] + [cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, 5) for bs in (33, 17)]
    best, score = None, -np.inf
    for b in bins:
        for dark, bb in ((True, 255 - b), (False, b)):
            bx = components(cv2.morphologyEx(bb, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)))
            if bx:
                rows = kmeans_rows(bx, np.median([q[3] - q[1] for q in bx]))
                bx = [bx[i] for r in rows for i in r]
            probs = classify(model, char_crops(gray, bx, dark)) if model is not None else None
            if model is not None:
                sc = probs.max(1).sum() - 2 * max(0, len(bx) - 10) if len(bx) else -1.0
            else:
                sc = len(bx) if len(bx) <= 11 else 22 - len(bx)
            if sc > score:
                best, score = (dark, bx, bb, probs, len(rows) if bx else 0), sc
    dark, boxes, binary, probs, nrows = best
    return gray, boxes, dark, probs, {"edges": edges, "lines": lines, "angle": ang, "binary": binary, "M": M,
                                      "scale": s, "rows": nrows}


def read_plate(bgr, model, deskew=True, grammar=True):
    gray, boxes, dark, probs, dbg = segment(bgr, deskew, model)
    text = decode(probs, grammar) if len(boxes) else ""
    conf = float(probs.max(1).mean()) if len(boxes) else 0.0
    return text, conf, boxes, dbg


def to_original(boxes, dbg):
    """Maps working-image boxes back to the coordinates of the input crop."""
    Mi = cv2.invertAffineTransform(dbg["M"])
    out = []
    for x1, y1, x2, y2 in boxes:
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], float) @ Mi.T / dbg["scale"]
        out.append([*pts.min(0), *pts.max(0)])
    return out
