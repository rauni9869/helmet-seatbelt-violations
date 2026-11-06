"""Character classifier for plate OCR, trained from scratch (Lec 14-15).
Training data: the Kaggle character set + synthetic characters rendered with 20 fonts (closes the font gap to
Indian plates) + characters cropped from the dev plates. The test plates are never used."""
import glob
import json
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from common import RAW, PROC, RUNS, RESULTS, FIGS, CHARS, CH, CW, dev, imread
from charnet import CharNet, norm, load_model
from ocr import fit_char, char_crops, segment, PH

FONTS = ["arial", "arialbd", "ARIALN", "ARIALNB", "ariblk", "bahnschrift", "calibri", "calibrib", "consola",
         "consolab", "courbd", "impact", "segoeui", "segoeuib", "seguibl", "tahomabd", "verdanab", "trebucbd",
         "framd", "lucon"]


def render(ch, rng, fonts):
    """One synthetic character: random font, weight, condensing, rotation, neighbours, blur, noise, resolution."""
    font = fonts[rng.integers(len(fonts))]
    fg, bg = int(rng.integers(0, 90)), int(rng.integers(140, 256))
    img = Image.new("L", (300, 140), bg)
    d = ImageDraw.Draw(img)
    sw = int(rng.integers(0, 3))
    x0, y0 = 110, 20
    box = d.textbbox((x0, y0), ch, font=font, stroke_width=sw)
    d.text((x0, y0), ch, fill=fg, font=font, stroke_width=sw, stroke_fill=fg)
    gap = rng.uniform(0.05, 0.3) * (box[3] - box[1])
    if rng.random() < 0.6:  # neighbouring characters that may leak into the crop margin
        d.text((box[2] + gap, y0), CHARS[rng.integers(len(CHARS))], fill=fg, font=font, stroke_width=sw, stroke_fill=fg)
        lw = d.textbbox((0, 0), "8", font=font, stroke_width=sw)[2]
        d.text((box[0] - gap - lw, y0), CHARS[rng.integers(len(CHARS))], fill=fg, font=font, stroke_width=sw, stroke_fill=fg)
    a = np.array(img)
    sx = rng.uniform(0.6, 1.1)  # condensed fonts
    ang = rng.uniform(-6, 6)
    M = cv2.getRotationMatrix2D(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), ang, 1.0)
    M[0] *= sx
    a = cv2.warpAffine(a, M, (int(300 * sx) + 1, 140), borderValue=bg)
    pts = np.array([[box[0], box[1], 1], [box[2], box[1], 1], [box[0], box[3], 1], [box[2], box[3], 1]], float) @ M.T
    x1, y1 = pts.min(0).astype(int)
    x2, y2 = pts.max(0).astype(int)
    p = max(1, int(0.08 * (y2 - y1)))
    c = a[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p]
    h = int(rng.integers(14, 48))  # low resolution of far-away plates
    c = cv2.resize(c, (max(4, int(c.shape[1] * h / c.shape[0])), h), interpolation=cv2.INTER_AREA)
    c = cv2.GaussianBlur(c, (3, 3), rng.uniform(0.1, 1.2))
    c = (c + rng.normal(0, rng.uniform(0, 12), c.shape)).clip(0, 255).astype(np.uint8)
    return fit_char(c)


def synth(n_per_class, seed):
    rng = np.random.default_rng(seed)
    fonts = [ImageFont.truetype(f"C:/Windows/Fonts/{f}.ttf", 80) for f in FONTS
             if os.path.exists(f"C:/Windows/Fonts/{f}.ttf")]
    if not fonts:
        print("no fonts found in C:/Windows/Fonts: training without synthetic characters")
        return np.zeros((0, CH, CW), np.uint8), np.zeros(0, np.int64)
    X = [render(ch, rng, fonts) for ch in CHARS for _ in range(n_per_class)]
    return np.stack(X), np.repeat(np.arange(len(CHARS)), n_per_class)


def plate_chars(split):
    """Characters cropped from the annotated plates of a split, processed exactly like in the OCR pipeline."""
    X, y = [], []
    for r in json.load(open(f"{PROC}/plates.json"))[split]:
        im = imread(f"{RAW}/{r['file']}")
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        s = PH / gray.shape[0]
        gray = cv2.createCLAHE(2.0, (4, 4)).apply(cv2.resize(gray, (int(round(gray.shape[1] * s)), PH), interpolation=cv2.INTER_CUBIC))
        dark = segment(im)[2]
        X.append(char_crops(gray, [[int(v * s) for v in c[1:]] for c in r["chars"]], dark))
        y += [CHARS.index(c[0]) for c in r["chars"]]
    return np.concatenate(X), np.array(y)


def augment(x):
    """Random affine (rotation, scale, shear, shift), blur and noise, done on the GPU."""
    n = len(x)
    a = torch.empty(n, device=dev).uniform_(-0.15, 0.15)
    s = torch.empty(n, device=dev).uniform_(0.9, 1.1)
    sh = torch.empty(n, device=dev).uniform_(-0.15, 0.15)
    t = torch.empty(n, 2, device=dev).uniform_(-0.08, 0.08)
    th = torch.stack([torch.stack([a.cos() / s, -a.sin() / s + sh, t[:, 0]], 1),
                      torch.stack([a.sin() / s, a.cos() / s, t[:, 1]], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    if torch.rand(1).item() < 0.5:
        x = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)
    return x + 0.1 * torch.randn_like(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    d = np.load(f"{PROC}/chars.npz")
    Xs, ys = synth(800, 0)
    Xsv, ysv = synth(60, 1)
    Xd, yd = plate_chars("dev")
    Xt, yt = plate_chars("test")
    if len(ys):
        cv2.imwrite(f"{FIGS}/chars_synth.png", np.vstack([np.hstack(Xs[k * 800 + np.arange(0, 800, 100)]) for k in range(0, 35, 5)]))
    Xtr = np.concatenate([d["Xtr"], Xs] + [Xd] * 10)
    ytr = np.concatenate([d["ytr"], ys] + [yd] * 10)
    T = lambda a: torch.from_numpy(a).to(dev)
    Xtr, ytr = T(Xtr), T(ytr)
    sets = {"kaggle val": (T(d["Xva"]), T(d["yva"]))}
    if len(ysv):
        sets["synthetic val"] = (T(Xsv), T(ysv))

    model = CharNet().to(dev)
    EPOCHS = 12
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    @torch.no_grad()
    def predict(X):
        return torch.cat([model(norm(X[i:i + 2000])).argmax(1) for i in range(0, len(X), 2000)])

    log = open(f"{RESULTS}/chars.txt", "w")
    log.write(f"train: {len(d['ytr'])} Kaggle + {len(ys)} synthetic + {len(yd)} dev-plate characters (x10)\n")
    best = 0
    for ep in range(EPOCHS):
        model.train()
        for b in torch.randperm(len(Xtr), device=dev).split(128):
            loss = F.cross_entropy(model(augment(norm(Xtr[b]))), ytr[b], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        acc = {k: (predict(X) == y).float().mean().item() for k, (X, y) in sets.items()}
        if np.mean(list(acc.values())) >= best:
            best = np.mean(list(acc.values()))
            torch.save(model.state_dict(), f"{RUNS}/charnet.pt")
        msg = f"epoch {ep + 1}: loss {loss.item():.4f} " + " ".join(f"{k} acc {v:.4f}" for k, v in acc.items())
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    model = load_model()
    te = {"Kaggle test": (T(d["Xte"]), T(d["yte"])), "test-plate characters (GT boxes)": (T(Xt), T(yt))}
    for k, (X, y) in te.items():
        pred = predict(X)
        msg = f"{k}: accuracy {(pred == y).float().mean().item():.4f} ({len(y)} images)"
        print(msg)
        log.write(msg + "\n")
    pred, y = predict(T(Xt)).cpu(), torch.from_numpy(yt)
    conf = {}
    for a, b in zip(y.tolist(), pred.tolist()):
        if a != b:
            conf[(CHARS[a], CHARS[b])] = conf.get((CHARS[a], CHARS[b]), 0) + 1
    msg = "Most frequent confusions on test plates (true, pred): count " + str(sorted(conf.items(), key=lambda kv: -kv[1])[:8])
    print(msg)
    log.write(msg + "\n")

    wrong = (pred != y).nonzero().flatten()[:8].tolist()
    fig, ax = plt.subplots(2, 8, figsize=(10, 3.4))
    for a, j in zip(ax.flat, list(range(0, 80, 10)) + wrong):
        a.imshow(Xt[j], cmap="gray")
        a.set_title(f"gt {CHARS[y[j]]} / pred {CHARS[pred[j]]}", fontsize=8, color="k" if pred[j] == y[j] else "r")
        a.axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIGS}/chars_results.png", dpi=150)
