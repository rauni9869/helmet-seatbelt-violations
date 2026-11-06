"""Trains the violation / plate detector.  python train_detector.py [--nomask] [--name main]"""
import argparse
import math
import os
import time
from collections import Counter
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from common import PROC, RUNS, RESULTS, CLASSES, NC, IMG, dev, imread, letterbox, load_split
from detector import Detector, detection_loss, predict
from metrics import evaluate

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


class DetSet(Dataset):
    def __init__(self, recs, train):
        self.recs, self.train = recs, train

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        im, s, (px, py) = letterbox(imread(f"{PROC}/{r['file']}"))
        b = np.array([x[1:] for x in r["boxes"]], np.float32).reshape(-1, 4) * s + [px, py, px, py]
        c = np.array([x[0] for x in r["boxes"]], np.int64)
        if self.train:
            im, b, c = augment(im, b, c)
        mask = np.zeros(NC, np.float32)
        mask[r["mask"]] = 1
        x = torch.from_numpy(((im[..., ::-1] - MEAN) / STD).transpose(2, 0, 1).copy())
        return x, torch.from_numpy(np.c_[c, b].astype(np.float32)), torch.from_numpy(mask)


def augment(im, b, c):
    """Random scale/translate (affine), horizontal flip, HSV jitter and random grayscale."""
    sc = np.random.uniform(0.5, 1.5)
    tx, ty = np.random.uniform(-0.2, 0.2, 2) * IMG
    M = np.array([[sc, 0, (1 - sc) * IMG / 2 + tx], [0, sc, (1 - sc) * IMG / 2 + ty]], np.float32)
    im = cv2.warpAffine(im, M, (IMG, IMG), borderValue=(114, 114, 114))
    if len(b):
        area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) * sc * sc
        b = b * sc + [M[0, 2], M[1, 2], M[0, 2], M[1, 2]]
        b = b.clip(0, IMG)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = (area > 0.35 * area0) & (b[:, 2] - b[:, 0] > 3) & (b[:, 3] - b[:, 1] > 3)
        b, c = b[keep], c[keep]
    if np.random.rand() < 0.5:
        im = im[:, ::-1]
        b = b.copy()
        b[:, [0, 2]] = IMG - b[:, [2, 0]]
    hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv *= np.random.uniform([0.97, 0.5, 0.6], [1.03, 1.5, 1.4])
    im = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if np.random.rand() < 0.2:
        im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return im, b, c


def collate(batch):
    xs, ts, ms = zip(*batch)
    t = torch.cat([torch.cat([torch.full((len(t), 1), i), t], 1) for i, t in enumerate(ts)])
    return torch.stack(xs), t, torch.stack(ms)


def run_eval(model, recs, bs=16, workers=2):
    model.eval()
    loader = DataLoader(DetSet(recs, False), bs, num_workers=workers, collate_fn=collate)
    preds, gts = [], []
    for x, t, _ in loader:
        with torch.autocast("cuda", torch.float16):
            p = predict(model, x.to(dev), conf=0.01, iou=0.5)
        preds += [q.cpu() for q in p]
        gts += [t[t[:, 0] == i, 1:] for i in range(len(x))]
    res = evaluate(preds, gts, [r["mask"] for r in recs])
    return res, preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nomask", action="store_true", help="ablation: penalise all classes on every image")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    tr, va = load_split("train"), load_split("val")
    if args.nomask:
        tr = [{**r, "mask": list(range(NC))} for r in tr]
    n = Counter(r["ds"] for r in tr)
    share = {d: math.sqrt(k) for d, k in n.items()}  # sample datasets ~ sqrt(size) so the big plate set does not dominate
    w = [share[r["ds"]] / n[r["ds"]] for r in tr]
    loader = DataLoader(DetSet(tr, True), args.bs, sampler=WeightedRandomSampler(w, len(tr)), num_workers=args.workers,
                        collate_fn=collate, drop_last=True, persistent_workers=True)

    model = Detector().to(dev)
    backbone = [p for m in (model.stem, model.c3, model.c4, model.c5) for p in m.parameters()]
    ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr * 0.2}, {"params": rest}], args.lr, weight_decay=5e-4)
    total, warm = args.epochs * len(loader), 2 * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: (k + 1) / warm if k < warm else 0.5 * (1 + math.cos(math.pi * (k - warm) / (total - warm))) * 0.95 + 0.05)
    scaler = torch.amp.GradScaler()

    os.makedirs(RUNS, exist_ok=True)
    log = open(f"{RESULTS}/train_{args.name}.txt", "w")
    best = -1
    for ep in range(args.epochs):
        model.train()
        t0, lc, lb = time.time(), 0.0, 0.0
        for x, t, m in loader:
            x, t, m = x.to(dev, non_blocking=True), t.to(dev), m.to(dev)
            with torch.autocast("cuda", torch.float16):
                outs = model(x)
            loss, a, b = detection_loss(outs, model.anchors, t, m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            lc, lb = lc + a, lb + b
        msg = f"epoch {ep + 1}: cls {lc / len(loader):.4f} box {lb / len(loader):.4f} ({time.time() - t0:.0f}s)"
        if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
            res, _, _ = run_eval(model, va)
            mAP = np.mean([v["ap"] for v in res.values()])
            msg += f" | val mAP@0.5 {mAP:.4f} " + " ".join(f"{CLASSES[c]} {v['ap']:.3f}" for c, v in res.items())
            if mAP > best:
                best = mAP
                torch.save(model.state_dict(), f"{RUNS}/det_{args.name}.pt")
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
    print(f"best val mAP {best:.4f}")
    log.write(f"best val mAP@0.5 {best:.4f}\n")


if __name__ == "__main__":
    main()
