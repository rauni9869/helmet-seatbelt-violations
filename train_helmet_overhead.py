"""Overhead helmet classifier for fixed traffic cameras (seat-belt dataset camera). Starts from the street-level
helmet classifier (runs/helmetnet.pt) and is fine-tuned on two-wheeler crops found by background subtraction
(overhead_crops.py) and labelled by eye (data/processed/overhead/labels.txt; H / N / X). Splits follow the
detector's leak-free time-range splits. Saves runs/helmetnet_overhead.pt and results/helmet_overhead.txt."""
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from common import PROC, RUNS, RESULTS
from helmetnet import LABELS, HelmetNet, crop, to_tensor, load_model
from train_helmet import augment, report

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MAP = {"H": 0, "N": 1, "X": 2}
EPOCHS = 25


def load():
    C = json.load(open(f"{PROC}/overhead/crops.json"))
    L = "".join(l.split("#")[0].replace(" ", "").strip() for l in open(f"{PROC}/overhead/labels.txt"))
    assert len(C) == len(L)
    cache, data = {}, {"train": [], "val": [], "test": []}
    for c, l in zip(C, L):
        im = cache.setdefault(c["file"], cv2.imread(f"{PROC}/{c['file']}"))
        data[c["split"]].append((im, c["box"], MAP[l]))
    return data


def crops(items, rng=None, reps=1):
    X, y = [], []
    for _ in range(reps):
        for im, box, l in items:
            X.append(crop(im, box, rng)); y.append(l)
    return np.stack(X), np.array(y)


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.cat([F.softmax(model(to_tensor(X[i:i + 128]).to(device)), 1).cpu() for i in range(0, len(X), 128)]).numpy()


if __name__ == "__main__":
    torch.manual_seed(0); rng = np.random.default_rng(0)
    data = load()
    Xva, yva = crops(data["val"]); Xte, yte = crops(data["test"])
    base = load_model(device)
    lines = [report("street-level classifier (before) on overhead val", predict(base, Xva).argmax(1), yva),
             report("street-level classifier (before) on overhead test", predict(base, Xte).argmax(1), yte)]
    model = HelmetNet(pretrained=False).to(device)
    model.load_state_dict(base.state_dict())
    opt = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    ytr_all = np.array([l for *_, l in data["train"]])
    wt = torch.tensor(len(ytr_all) / (3 * np.bincount(ytr_all, minlength=3)), dtype=torch.float32, device=device)
    best = (-1, None)
    for ep in range(EPOCHS):
        Xtr, ytr = crops(data["train"], rng, reps=2)  # fresh box jitter every epoch
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            b = perm[i:i + 32].numpy()
            x = augment(to_tensor(Xtr[b]).to(device))
            loss = F.cross_entropy(model(x), torch.from_numpy(ytr[b]).to(device), weight=wt, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        p = predict(model, Xva).argmax(1)
        bal = np.mean([(p[yva == k] == k).mean() for k in range(3)])
        print(f"epoch {ep + 1}: loss {loss.item():.3f} val acc {(p == yva).mean():.3f} balanced {bal:.3f}", flush=True)
        if bal > best[0]:
            best = (bal, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    torch.save(model.state_dict(), f"{RUNS}/helmetnet_overhead.pt")
    lines += [f"selected epoch {best[2]} (balanced val accuracy {best[0]:.3f})",
              report("overhead classifier (after) on overhead val", predict(model, Xva).argmax(1), yva),
              report("overhead classifier (after) on overhead test", predict(model, Xte).argmax(1), yte)]
    open(f"{RESULTS}/helmet_overhead.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
