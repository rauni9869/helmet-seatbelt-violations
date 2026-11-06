"""Evaluates plate OCR on the annotated Indian plate crops.  python eval_ocr.py [dev|test]"""
import json
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from common import RAW, PROC, RESULTS, FIGS, CHARS, imread, box_iou
from charnet import load_model
from ocr import read_plate, segment, char_crops, classify, to_original, PH


def lev(a, b):
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return int(d[-1])


def main(split):
    recs = json.load(open(f"{PROC}/plates.json"))[split]
    model = load_model()
    ims = [imread(f"{RAW}/{r['file']}") for r in recs]
    lines = [f"Plate OCR on the {split} split ({len(recs)} plates)"]

    # 1) character segmentation vs ground-truth character boxes
    tp = npred = ngt = 0
    for im, r in zip(ims, recs):
        _, boxes, _, _, dbg = segment(im, model=model)
        pb = torch.tensor(to_original(boxes, dbg), dtype=torch.float32).reshape(-1, 4)
        gb = torch.tensor([c[1:] for c in r["chars"]], dtype=torch.float32)
        npred, ngt = npred + len(pb), ngt + len(gb)
        if len(pb):
            iou = box_iou(pb, gb)
            for _ in range(min(len(pb), len(gb))):  # greedy one-to-one matching
                v = iou.max()
                if v < 0.5:
                    break
                i, j = divmod(int(iou.argmax()), iou.shape[1])
                tp += 1
                iou[i, :] = 0
                iou[:, j] = 0
    lines.append(f"Character segmentation (IoU>=0.5): precision {tp / max(npred, 1):.4f} recall {tp / ngt:.4f}")

    # 2) classifier alone on the ground-truth character boxes
    ok = tot = 0
    for im, r in zip(ims, recs):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        s = PH / gray.shape[0]
        gray = cv2.createCLAHE(2.0, (4, 4)).apply(cv2.resize(gray, (int(round(gray.shape[1] * s)), PH), interpolation=cv2.INTER_CUBIC))
        dark = segment(im, model=model)[2]
        boxes = [[int(v * s) for v in c[1:]] for c in r["chars"]]
        p = classify(model, char_crops(gray, boxes, dark)).argmax(1)
        ok += sum(CHARS[k] == c[0] for k, c in zip(p, r["chars"]))
        tot += len(p)
    lines.append(f"Character classification on ground-truth boxes: accuracy {ok / tot:.4f} ({tot} characters)")

    # 3) end-to-end plate reading with ablations
    shown = []
    for name, kw in (("full pipeline", {}), ("without format-constrained decoding", {"grammar": False}),
                     ("without Hough deskew", {"deskew": False})):
        exact, err, nch = 0, 0, 0
        for im, r in zip(ims, recs):
            text = read_plate(im, model, **kw)[0]
            exact += text == r["text"]
            err += lev(text, r["text"])
            nch += len(r["text"])
            if not kw:
                shown.append(text)
        lines.append(f"End-to-end, {name}: exact plate match {exact / len(recs):.4f}, character error rate {err / nch:.4f}")
    print("\n".join(lines))
    with open(f"{RESULTS}/ocr_{split}.txt", "w") as fp:
        fp.write("\n".join(lines) + "\n")

    if split == "test":
        figure_steps(ims, recs, model)
        fig, ax = plt.subplots(4, 4, figsize=(11, 5))
        for a, im, r, t in zip(ax.flat, ims, recs, shown):
            a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            a.set_title(f"gt {r['text']}\npred {t}", fontsize=8, color="g" if t == r["text"] else "r")
            a.axis("off")
        plt.tight_layout()
        plt.savefig(f"{FIGS}/ocr_results.png", dpi=150)


def figure_steps(ims, recs, model):
    """Intermediate stages of the OCR pipeline on the most skewed correctly read test plate."""
    angs = [abs(segment(im)[4]["angle"]) if read_plate(im, model)[0] == r["text"] else 0 for im, r in zip(ims, recs)]
    k = int(np.argmax(angs))
    im = ims[k]
    gray0 = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (int(round(im.shape[1] * PH / im.shape[0])), PH))
    gray, boxes, dark, _, dbg = segment(im, model=model)
    text = read_plate(im, model)[0]
    hough = cv2.cvtColor(dbg["edges"], cv2.COLOR_GRAY2RGB)
    for rho, th in dbg["lines"]:
        a, b = np.cos(th), np.sin(th)
        p = np.array([a * rho, b * rho])
        cv2.line(hough, tuple((p + 2000 * np.array([-b, a])).astype(int)), tuple((p - 2000 * np.array([-b, a])).astype(int)), (255, 0, 0), 1)
    seg = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(seg, (x1, y1), (x2, y2), (0, 200, 0), 1)
    panels = [(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), "input crop"), (gray0, "grayscale"),
              (hough, f"Canny + Hough lines (skew {dbg['angle']:.1f} deg)"), (gray, "CLAHE + deskewed"),
              (dbg["binary"], "Otsu threshold"), (seg, f"components, {dbg['rows']} row(s) -> {text}")]
    fig, ax = plt.subplots(2, 3, figsize=(11, 3.6))
    for a, (p, t) in zip(ax.flat, panels):
        a.imshow(p, cmap="gray")
        a.set_title(t, fontsize=9)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIGS}/ocr_steps.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
