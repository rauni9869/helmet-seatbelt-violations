"""End-to-end system: detect helmet / seat-belt violators, link each to its vehicle's number plate and read it.
    python pipeline.py --eval                  test-set metrics + annotated outputs
    python pipeline.py --image a.jpg [b.jpg]   annotate images
    python pipeline.py --video in.mp4          annotate a video"""
import argparse
import csv
import json
import os
import cv2
import numpy as np
import torch
from common import RAW, PROC, RUNS, RESULTS, FIGS, OUT, CLASSES, IMG, dev, imread, imwrite, letterbox, load_split, box_iou
from detector import Detector, predict
from charnet import load_model
from train_detector import MEAN, STD
from ocr import read_plate
import helmetnet

RIDER_OK, RIDER_BAD, CAR, BELT_OK, BELT_BAD, PLATE = range(6)


def center_in(box, outer, pad=0.05):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = outer[2] - outer[0], outer[3] - outer[1]
    return outer[0] - pad * w <= cx <= outer[2] + pad * w and outer[1] - pad * h <= cy <= outer[3] + pad * h


def overlap(inner, outer):
    """Fraction of `inner` covered by `outer`."""
    iw = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    ih = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return iw * ih / max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-9)


class ViolationSystem:
    def __init__(self, name="main"):
        self.det = Detector(pretrained=False).to(dev)
        self.det.load_state_dict(torch.load(f"{RUNS}/det_{name}.pt", map_location=dev))
        self.det.eval()
        t = json.load(open(f"{RUNS}/thresholds.json"))
        self.thr = np.array([t[c] for c in CLASSES])
        self.chars = load_model()
        # second-stage helmet classifier (train_helmet.py); without it the rider rules below are used alone
        self.helmet = helmetnet.load_model() if os.path.exists(f"{RUNS}/helmetnet.pt") else None
        self.not_rider = 0.5  # 'not a rider' probability above which a rider box is dropped
        self.car_rules = self.helmet is None  # the classifier makes the car rules unnecessary (val: F1 0.913 vs 0.864)

    def detect(self, bgr):
        im, s, (px, py) = letterbox(bgr)
        x = torch.from_numpy(((im[..., ::-1] - MEAN) / STD).transpose(2, 0, 1).copy())[None].to(dev)
        with torch.autocast("cuda", torch.float16, enabled=dev == "cuda"):
            p = predict(self.det, x, conf=float(self.thr.min()), all_scores=True)[0].cpu().numpy()
        p = p[p[:, 4] >= self.thr[p[:, 5].astype(int)]]
        p[:, [0, 2]] = ((p[:, [0, 2]] - px) / s).clip(0, bgr.shape[1])
        p[:, [1, 3]] = ((p[:, [1, 3]] - py) / s).clip(0, bgr.shape[0])
        return [{"cls": int(r[5]), "score": float(r[4]), "box": [float(v) for v in r[:4]], "scores": r[6:].tolist()}
                for r in p]

    def arbitrate(self, dets, bgr=None):
        """Makes the rider detections of one image consistent before linking (settings chosen on the val split).
        1. with the helmet classifier, every rider box is re-judged on its head-and-shoulders crop: dropped if it
           is not a rider (a car, a parked bike), otherwise relabelled helmet / no helmet.
        2. one rider gets one answer: of overlapping rider boxes only the most confident one is kept.
        3. the partial-label loss never trains the rider classes against cars, so cars also get rider boxes. Such a
           box is dropped when it contains a confident driver (seat-belt / no-seat-belt) box, or when the image has
           drivers and the car score at the box's anchor exceeds its rider score."""
        key = lambda d: d["score"] / self.thr[d["cls"]]
        iou = lambda a, b: float(box_iou(torch.tensor([a["box"]]), torch.tensor([b["box"]]))[0, 0])
        riders = [d for d in dets if d["cls"] in (RIDER_OK, RIDER_BAD)]
        drop = set()
        judged = self.helmet is not None and bgr is not None
        if judged:
            for d, p in zip(riders, helmetnet.classify(self.helmet, bgr, [d["box"] for d in riders])):
                d["helmet_p"] = p.tolist()
                if p[2] > self.not_rider:
                    drop.add(id(d))
                d["cls"] = RIDER_OK if p[0] >= p[1] else RIDER_BAD
        riders.sort(key=lambda d: -key(d))
        for i, d in enumerate(riders):
            if id(d) not in drop:
                drop |= {id(e) for e in riders[i + 1:] if (judged or e["cls"] != d["cls"]) and iou(d, e) > 0.5}
        drivers = [d for d in dets if d["cls"] in (BELT_OK, BELT_BAD) and d["score"] >= 0.35]
        for d in riders if self.car_rules else []:
            car_margin = d["scores"][CAR] - max(d["scores"][RIDER_OK], d["scores"][RIDER_BAD])
            if any(center_in(b["box"], d["box"], 0) for b in drivers) or (drivers and car_margin > 0) or car_margin > 0.3:
                drop.add(id(d))
        return [d for d in dets if id(d) not in drop]

    @staticmethod
    def link(dets):
        """Associates every violator with a vehicle (seat belt case) and a number plate."""
        plates = [d for d in dets if d["cls"] == PLATE]
        cars = [d for d in dets if d["cls"] == CAR]
        out = []
        for d in dets:
            if d["cls"] == RIDER_BAD:  # rider box covers the whole two-wheeler: plate must lie inside it
                vehicle = d
            elif d["cls"] == BELT_BAD:  # driver box lies inside the car box; the plate lies inside the car box
                cov = [overlap(d["box"], c["box"]) for c in cars]
                vehicle = cars[int(np.argmax(cov))] if cov and max(cov) > 0.6 else None
            else:
                continue
            cand = [p for p in plates if vehicle is not None and center_in(p["box"], vehicle["box"])]
            if d["cls"] == BELT_BAD:  # plate is on the lower part of the car (front / rear bumper)
                cand = [p for p in cand if (p["box"][1] + p["box"][3]) / 2 > (vehicle["box"][1] + vehicle["box"][3]) / 2]
            out.append({"type": "no helmet" if d["cls"] == RIDER_BAD else "no seat belt", "det": d,
                        "vehicle": vehicle, "plate": max(cand, key=lambda p: p["score"]) if cand else None})
        return out

    def read(self, bgr, plate):
        x1, y1, x2, y2 = plate["box"]
        mx, my = 0.06 * (x2 - x1), 0.12 * (y2 - y1)
        crop = bgr[int(max(0, y1 - my)):int(y2 + my), int(max(0, x1 - mx)):int(x2 + mx)]
        if crop.shape[0] < 8 or crop.shape[1] < 16:
            return "", 0.0
        text, conf, _, _ = read_plate(crop, self.chars)
        return text, conf

    def __call__(self, bgr):
        dets = self.arbitrate(self.detect(bgr), bgr)
        viols = self.link(dets)
        for v in viols:
            v["text"], v["ocr_conf"] = self.read(bgr, v["plate"]) if v["plate"] else ("", 0.0)
        return dets, viols


def draw(bgr, dets, viols):
    im = bgr.copy()
    lw = max(2, round(max(im.shape[:2]) / 400))
    fs = max(0.5, max(im.shape[:2]) / 1100)
    for d in dets:
        if d["cls"] in (RIDER_OK, BELT_OK, CAR):
            b = list(map(int, d["box"]))
            cv2.rectangle(im, b[:2], b[2:], (0, 180, 0) if d["cls"] != CAR else (200, 160, 0), max(1, lw // 2))
    for v in viols:
        b = list(map(int, v["det"]["box"]))
        cv2.rectangle(im, b[:2], b[2:], (0, 0, 255), lw)
        label = f"{v['type'].upper()} {v['det']['score']:.2f}"
        if v["plate"]:
            p = list(map(int, v["plate"]["box"]))
            cv2.rectangle(im, p[:2], p[2:], (0, 255, 255), lw)
            label += f" | {v['text'] or '?'}"
        else:
            label += " | plate not visible"
        (tw, th), _ = cv2.getTextSize(label, 0, fs, lw)
        y = max(th + 4, b[1])
        cv2.rectangle(im, (b[0], y - th - 6), (b[0] + tw + 4, y), (0, 0, 255), -1)
        cv2.putText(im, label, (b[0] + 2, y - 4), 0, fs, (255, 255, 255), lw)
    return im


def evaluate_test(sys_):
    """Violation-level metrics on the test split, using the ground-truth boxes."""
    te = [r for r in load_split("test") if r["ds"] != "inplates"]
    stats = {k: dict(tp=0, fp=0, fn=0, link_ok=0, link_n=0) for k in (RIDER_BAD, BELT_BAD)}
    rows, examples = [], []
    on_cars = 0  # "no helmet" on seat-belt images: these contain cars only, but riders are not labelled there
    os.makedirs(f"{OUT}/test", exist_ok=True)
    for r in te:
        proc = imread(f"{PROC}/{r['file']}")
        dets, viols = sys_(proc)
        g = np.array(r["boxes"]).reshape(-1, 5)
        if r["ds"].startswith("seatbelt"):
            on_cars += sum(v["det"]["cls"] == RIDER_BAD for v in viols)
        for k in stats:
            if k not in r["mask"]:
                continue
            gv = torch.tensor(g[g[:, 0] == k, 1:], dtype=torch.float32)
            pv = [v for v in viols if v["det"]["cls"] == k]
            used = set()
            for v in sorted(pv, key=lambda v: -v["det"]["score"]):
                iou = box_iou(torch.tensor([v["det"]["box"]]), gv)[0] if len(gv) else torch.zeros(0)
                iou[list(used)] = 0
                if len(iou) and iou.max() >= 0.5:
                    j = int(iou.argmax())
                    used.add(j)
                    stats[k]["tp"] += 1
                    # linking: compare to the ground-truth plate (helmet) / car (seat belt) of this violator
                    if k == RIDER_BAD and PLATE in r["mask"]:
                        gp = [b[1:] for b in g if b[0] == PLATE and center_in(b[1:], gv[j].tolist())]
                        if gp:
                            stats[k]["link_n"] += 1
                            ok = v["plate"] is not None and box_iou(torch.tensor([v["plate"]["box"]]), torch.tensor(gp, dtype=torch.float32)).max() >= 0.5
                            stats[k]["link_ok"] += int(ok)
                    if k == BELT_BAD:
                        gc = [b[1:] for b in g if b[0] == CAR and overlap(gv[j].tolist(), b[1:]) > 0.6]
                        if gc:
                            stats[k]["link_n"] += 1
                            ok = v["vehicle"] is not None and box_iou(torch.tensor([v["vehicle"]["box"]]), torch.tensor(gc, dtype=torch.float32)).max() >= 0.5
                            stats[k]["link_ok"] += int(ok)
                else:
                    stats[k]["fp"] += 1
            stats[k]["fn"] += len(gv) - len(used)

        # annotated output at full resolution (plates are read from the original image)
        src = imread(os.path.join(RAW, *r["src"].split("\\")))
        dets, viols = sys_(src)
        imwrite(f"{OUT}/test/{os.path.basename(r['file'])}", draw(src, dets, viols), 85)
        for v in viols:
            rows.append([r["src"], v["type"], f"{v['det']['score']:.3f}", [round(x) for x in v["det"]["box"]],
                         [round(x) for x in v["plate"]["box"]] if v["plate"] else "", v["text"], f"{v['ocr_conf']:.3f}"])
        if viols:
            examples.append((r, src, dets, viols))

    with open(f"{OUT}/violations_test.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["image", "violation", "score", "violator_box", "plate_box", "plate_text", "ocr_conf"])
        w.writerows(rows)
    lines = [f"Violation-level evaluation on the test split ({len(te)} images, IoU >= 0.5)"]
    for k, s in stats.items():
        p = s["tp"] / max(s["tp"] + s["fp"], 1)
        rc = s["tp"] / max(s["tp"] + s["fn"], 1)
        lines.append(f"{CLASSES[k]:15s} TP {s['tp']:3d} FP {s['fp']:3d} FN {s['fn']:3d}  precision {p:.3f} recall {rc:.3f} F1 {2 * p * rc / max(p + rc, 1e-9):.3f}")
        what = "plate" if k == RIDER_BAD else "vehicle"
        lines.append(f"{'':15s} correct {what} linked for {s['link_ok']}/{s['link_n']} detected violators with a labelled {what}")
    lines.append(f"{'':15s} plus {on_cars} 'no helmet' detections on the seat-belt (car-only) images, not counted above")
    n_pl = sum(v["plate"] is not None for *_, vs in examples for v in vs)
    n_v = sum(len(vs) for *_, vs in examples)
    lines.append(f"Predicted violations: {n_v}; with a linked plate: {n_pl}; with a non-empty plate reading: "
                 f"{sum(bool(v['text']) for *_, vs in examples for v in vs)}")
    print("\n".join(lines))
    with open(f"{RESULTS}/pipeline.txt", "w") as fp:
        fp.write("\n".join(lines) + "\n")
    return examples


def example_figure(examples):
    pick = []
    for ds in ("helmetviol", "riderhp", "seatbelt_anpr", "seatbelt_akaike"):
        want = "no seat belt" if ds.startswith("seatbelt") else "no helmet"
        cand = [e for e in examples if e[0]["ds"] == ds and any(v["type"] == want for v in e[3])]
        cand.sort(key=lambda e: -sum(v["plate"] is not None and v["type"] == want for v in e[3]))
        pick += cand[:2]
    # fill the last row with further helmet violators whose plate was read
    more = [e for e in examples if e[0]["ds"] in ("helmetviol", "riderhp") and all(e is not p for p in pick)
            and any(v["type"] == "no helmet" and v["text"] for v in e[3])]
    more.sort(key=lambda e: -max(v["ocr_conf"] for v in e[3]))
    pick += more[:-len(pick) % 4]
    tiles = []
    for r, src, dets, viols in pick:
        im = draw(src, dets, viols)
        if r["ds"] == "seatbelt_akaike":  # zoom on the violating car of the 1920x1080 frames
            v = next(v for v in viols if v["type"] == "no seat belt")
            b = v["vehicle"]["box"] if v["vehicle"] else v["det"]["box"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            S = max(1.6 * max(b[2] - b[0], b[3] - b[1]), 700)
            im = im[int(max(0, cy - S / 2)):int(cy + S / 2), int(max(0, cx - S / 2)):int(cx + S / 2)]
        tiles.append(letterbox(im, 480)[0])
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % 4) if tiles else []
    grid = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    if grid:
        imwrite(f"{FIGS}/pipeline_examples.jpg", np.vstack(grid), 85)


def run_video(sys_, path, out):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        dets, viols = sys_(f)
        vw.write(draw(f, dets, viols))
        n += 1
    vw.release()
    print(f"wrote {out} ({n} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--image", nargs="*")
    ap.add_argument("--video")
    a = ap.parse_args()
    S = ViolationSystem()
    if a.eval:
        example_figure(evaluate_test(S))
    for p in a.image or []:
        im = imread(p)
        dets, viols = S(im)
        imwrite(f"{OUT}/{os.path.splitext(os.path.basename(p))[0]}_out.jpg", draw(im, dets, viols))
        for v in viols:
            print(p, v["type"], f"{v['det']['score']:.2f}", "plate:", v["text"] or ("unreadable" if v["plate"] else "not visible"))
    if a.video:
        run_video(S, a.video, f"{OUT}/{os.path.splitext(os.path.basename(a.video))[0]}_out.mp4")
