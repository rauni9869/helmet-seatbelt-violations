"""Collects two-wheeler boxes from the fixed ANPR traffic camera (seat-belt dataset) by background subtraction,
per split, for training an overhead helmet classifier. Output: data/processed/overhead/crops.json + sheets."""
import json, os, re, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app, helmetnet
from common import PROC

recs = []
for split in ("train", "val", "test"):
    for r in json.load(open(f"{PROC}/{split}.json")):
        if "ANPR" in r["src"]:
            recs.append((split, int(re.search(r"mp4-(\d+)_jpg", r["src"]).group(1)), r["file"]))
recs.sort(key=lambda t: t[1])
train_frames = [cv2.imread(f"{PROC}/{f}") for s, _, f in recs if s == "train"]
bg = np.median(np.stack(train_frames[::max(1, len(train_frames) // 80)]), 0).astype(np.uint8)  # train frames only
cv2.imwrite(f"{PROC}/overhead/background.jpg", bg)
out = []
for split, frame, f in recs:
    im = cv2.imread(f"{PROC}/{f}")
    dets, _, _ = app.run(im, app.OURS)
    for b in app.two_wheelers(im, bg, dets):
        out.append({"split": split, "frame": frame, "file": f, "box": [int(v) for v in b]})
json.dump(out, open(f"{PROC}/overhead/crops.json", "w"))
print({s: sum(o["split"] == s for o in out) for s in ("train", "val", "test")}, "frames", len(recs))
# contact sheets, 10 x 6 per sheet, index printed on each crop
tiles = []
for i, o in enumerate(out):
    x1, y1, x2, y2 = o["box"]
    c = cv2.resize(im := cv2.imread(f"{PROC}/{o['file']}")[y1:y1 + int(0.6 * (y2 - y1)), x1:x2], (150, 150))
    c = cv2.copyMakeBorder(c, 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(c, str(i), (4, 168), 0, 0.6, (0, 0, 0), 2)
    tiles.append(c)
for k in range(0, len(tiles), 60):
    t = tiles[k:k + 60]; t += [np.full_like(tiles[0], 255)] * (60 - len(t))
    cv2.imwrite(f"{PROC}/overhead/sheet_{k // 60:02d}.jpg", np.vstack([np.hstack(t[r:r + 10]) for r in range(0, 60, 10)]))
