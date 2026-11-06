"""Test-set evaluation of a trained detector.  python eval_detector.py [--name main]"""
import argparse
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from common import PROC, RUNS, RESULTS, FIGS, CLASSES, COLORS, dev, imread, load_split, letterbox
from detector import Detector
from metrics import evaluate
from train_detector import run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
    args = ap.parse_args()
    model = Detector(pretrained=False).to(dev)
    model.load_state_dict(torch.load(f"{RUNS}/det_{args.name}.pt", map_location=dev))
    model.eval()

    va, te = load_split("val"), load_split("test")
    res_va, _, _ = run_eval(model, va)
    res, preds, gts = run_eval(model, te)
    masks = [r["mask"] for r in te]
    coco = {c: np.mean([evaluate(preds, gts, masks, t)[c]["ap"] for t in np.arange(0.5, 0.96, 0.05)]) for c in res}

    lines = [f"Detector '{args.name}' on the test split ({len(te)} images)",
             f"{'class':16s}{'#gt':>6s}{'AP@.5':>8s}{'AP@.5:.95':>11s}{'P':>7s}{'R':>7s}{'F1':>7s}{'thr*':>7s}"]
    thr = {}
    for c, v in res.items():
        thr[CLASSES[c]] = round(res_va[c]["thr"], 3)  # operating point = best F1 on the validation split
        k = int((v["scores"] >= thr[CLASSES[c]]).sum())
        p, r = (v["prec"][k - 1], v["rec"][k - 1]) if k else (0.0, 0.0)
        f1 = 2 * p * r / max(p + r, 1e-9)
        lines.append(f"{CLASSES[c]:16s}{v['n']:6d}{v['ap']:8.3f}{coco[c]:11.3f}{p:7.3f}{r:7.3f}{f1:7.3f}{thr[CLASSES[c]]:7.3f}")
    lines.append(f"{'mean':16s}{'':6s}{np.mean([v['ap'] for v in res.values()]):8.3f}{np.mean(list(coco.values())):11.3f}")
    lines.append("P/R/F1 on test at thr* = the best-F1 score threshold chosen on the validation split")
    print("\n".join(lines))
    with open(f"{RESULTS}/det_{args.name}.txt", "w") as fp:
        fp.write("\n".join(lines) + "\n")
    if args.name != "main":
        return
    with open(f"{RUNS}/thresholds.json", "w") as fp:
        json.dump(thr, fp)

    fig, ax = plt.subplots(figsize=(5, 4))
    for c, v in res.items():
        ax.plot(v["rec"], v["prec"], label=f"{CLASSES[c]} (AP {v['ap']:.2f})", color=np.array(COLORS[c][::-1]) / 255)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGS}/pr_curves.png", dpi=150)

    picks = []
    for ds in ("helmetviol", "riderhp", "seatbelt_anpr", "seatbelt_akaike", "inplates"):
        picks += [i for i, r in enumerate(te) if r["ds"] == ds and len(r["boxes"]) >= 2][:2]
    tiles = []
    for i in picks[:10]:
        im = imread(f"{PROC}/{te[i]['file']}")
        s = 640 / max(im.shape[:2])
        px, py = (640 - round(im.shape[1] * s)) // 2, (640 - round(im.shape[0] * s)) // 2
        for x1, y1, x2, y2, sc, c in preds[i].tolist():
            if sc < thr[CLASSES[int(c)]]:
                continue
            p1, p2 = (int((x1 - px) / s), int((y1 - py) / s)), (int((x2 - px) / s), int((y2 - py) / s))
            cv2.rectangle(im, p1, p2, COLORS[int(c)], 2)
            cv2.putText(im, f"{CLASSES[int(c)]} {sc:.2f}", (p1[0], max(12, p1[1] - 4)), 0, 0.5, COLORS[int(c)], 2)
        tiles.append(letterbox(im, 400)[0])
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % 5)
    grid = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
    cv2.imwrite(f"{FIGS}/detections.jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
