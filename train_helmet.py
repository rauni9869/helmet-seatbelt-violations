"""Trains the second-stage helmet classifier (helmetnet.py) on rider boxes of the detection splits.
Samples: ground-truth rider boxes (with box jitter), and the detector's own rider boxes on the same images,
labelled by matching them to the ground truth. Detector rider boxes on the seat-belt images (cars only) and
unmatched rider boxes on fully labelled two-wheeler images (parked bikes, background) are 'not a rider'.
The test split is only used for the final numbers."""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from common import PROC, RESULTS, RUNS, imread, load_split, box_iou
from helmetnet import HelmetNet, LABELS, crop, to_tensor, load_model

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DS = ("helmetviol", "riderhp", "seatbelt_anpr", "seatbelt_akaike")


def detector_boxes(split):
    """Rider boxes predicted by the detector on a split (cached, the detector is slow on a CPU)."""
    path = f"{PROC}/rider_dets_{split}.json"
    if os.path.exists(path):
        return json.load(open(path))
    from pipeline import ViolationSystem
    det = ViolationSystem()
    out = {r["file"]: [d["box"] for d in det.detect(imread(f"{PROC}/{r['file']}")) if d["cls"] in (0, 1)]
           for r in load_split(split) if r["ds"] in DS}
    json.dump(out, open(path, "w"))
    return out


def samples(split, n_jitter, seed):
    rng = np.random.default_rng(seed)
    pred = detector_boxes(split)
    X, y = [], []
    for r in load_split(split):
        if r["ds"] not in DS:
            continue
        im = imread(f"{PROC}/{r['file']}")
        g = np.array(r["boxes"]).reshape(-1, 5)
        gr = g[np.isin(g[:, 0], (0, 1))]
        labelled = 0 in r["mask"]
        for c, *b in gr:
            X += [crop(im, b)] + [crop(im, b, rng) for _ in range(n_jitter)]
            y += [int(c)] * (1 + n_jitter)
        for b in pred[r["file"]]:
            if r["ds"].startswith("seatbelt"):
                lab = 2
            elif not labelled:  # plate-only images: riders may be present but unlabelled
                continue
            else:
                iou = box_iou(torch.tensor([b]), torch.tensor(gr[:, 1:], dtype=torch.float32))[0] if len(gr) else torch.zeros(1)
                lab = int(gr[int(iou.argmax()), 0]) if iou.max() >= 0.5 else 2 if iou.max() < 0.2 else None
            if lab is not None:
                X.append(crop(im, b))
                y.append(lab)
    return np.array(X), np.array(y)


def augment(x):
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    gain = 1 + 0.25 * (torch.rand(len(x), 1, 1, 1, device=x.device) * 2 - 1)
    bias = 0.25 * (torch.rand(len(x), 1, 1, 1, device=x.device) * 2 - 1)
    return x * gain + bias


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.cat([model(to_tensor(X[i:i + 256]).to(device)).argmax(1).cpu() for i in range(0, len(X), 256)]).numpy()


def report(name, p, y):
    acc = (p == y).mean()
    per = " ".join(f"{LABELS[k]} {(p[y == k] == k).mean():.3f} (n={(y == k).sum()})" for k in range(len(LABELS)))
    conf = np.zeros((3, 3), int)
    np.add.at(conf, (y, p), 1)
    return f"{name}: accuracy {acc:.4f} | recall per class: {per}\n  confusion (rows true, cols predicted {LABELS}):\n  {conf.tolist()}"


if __name__ == "__main__":
    torch.manual_seed(0)
    Xtr, ytr = samples("train", 5, 0)
    Xva, yva = samples("val", 0, 1)
    print("train", np.bincount(ytr), "val", np.bincount(yva))
    model = HelmetNet().to(device)
    EPOCHS = 15
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    wt = torch.tensor(len(ytr) / (3 * np.bincount(ytr)), dtype=torch.float32, device=device)  # class balance
    log = open(f"{RESULTS}/helmet.txt", "w")
    log.write(f"train samples per class {dict(zip(LABELS, np.bincount(ytr).tolist()))}, "
              f"val {dict(zip(LABELS, np.bincount(yva).tolist()))}\n")
    best = 0
    for ep in range(EPOCHS):
        model.train()
        for b in torch.randperm(len(ytr)).split(64):
            x = augment(to_tensor(Xtr[b.numpy()]).to(device))
            loss = F.cross_entropy(model(x), torch.from_numpy(ytr[b.numpy()]).to(device), weight=wt, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        p = predict(model, Xva)
        bal = np.mean([(p[yva == k] == k).mean() for k in range(3)])  # balanced accuracy selects the epoch
        if bal >= best:
            best = bal
            torch.save(model.state_dict(), f"{RUNS}/helmetnet.pt")
        msg = f"epoch {ep + 1}: loss {loss.item():.4f} val accuracy {(p == yva).mean():.4f} balanced {bal:.4f}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
    model = load_model(device)
    Xte, yte = samples("test", 0, 2)
    for name, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        msg = report(name, predict(model, X), y)
        print(msg)
        log.write(msg + "\n")
