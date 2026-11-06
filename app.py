"""Web demo of the violation system (Gradio).
    python app.py          then open http://127.0.0.1:7860
Upload a traffic image or video; the system marks helmet / seat-belt violators, links each to its vehicle's plate
and reads the plate. The 'detector only' mode switches the helmet classifier off, to show what it fixes."""
import os
import time
from difflib import SequenceMatcher
import cv2
import gradio as gr
import numpy as np
from common import OUT
from pipeline import ViolationSystem, draw, RIDER_OK, RIDER_BAD, CAR, BELT_OK, BELT_BAD, PLATE

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_examples")
OURS, DET_ONLY = "Full system (with helmet classifier)", "Detector only (no helmet classifier)"
S = ViolationSystem()
HELMET = S.helmet
OVERHEAD = None  # helmet classifier fine-tuned on top-view two-wheelers of a fixed traffic camera
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "helmetnet_overhead.pt")):
    import torch
    import helmetnet
    OVERHEAD = helmetnet.HelmetNet(pretrained=False).to(helmetnet.dev)
    OVERHEAD.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                                                     "helmetnet_overhead.pt"), map_location=helmetnet.dev))
    OVERHEAD.eval()
SURE = 0.6  # top-view helmet decisions below this probability are left as "two-wheeler"


def run(bgr, mode):
    """Runs the system in the chosen mode; the classifier is switched off for the 'detector only' comparison."""
    S.helmet = HELMET if mode == OURS else None
    S.car_rules = False  # compare the plain detector with the full system, without the fallback rules
    t = time.time()
    dets, viols = S(bgr)
    return dets, viols, time.time() - t


def table(viols):
    rows = []
    for i, v in enumerate(viols, 1):
        if v["plate"] is None:
            plate, conf = "plate not visible", ""
        else:
            plate, conf = v["text"] or "unreadable", f"{v['ocr_conf']:.2f}"
        rows.append([i, v["type"], f"{v['det']['score']:.2f}", plate, conf])
    return rows or [["-", "no violation found", "", "", ""]]


def summary(dets, viols, sec, mode):
    n = lambda c: sum(d["cls"] == c for d in dets)
    helmet = sum(v["type"] == "no helmet" for v in viols)
    belt = sum(v["type"] == "no seat belt" for v in viols)
    read = sum(bool(v.get("text")) for v in viols)
    return (f"**{len(viols)} violation(s)**: {helmet} without helmet, {belt} without seat belt; "
            f"{read} plate(s) read.  \n"
            f"Detected: {n(RIDER_OK) + n(RIDER_BAD)} rider(s), {n(CAR)} car(s), {n(BELT_OK) + n(BELT_BAD)} driver(s), "
            f"{n(PLATE)} plate(s). Mode: {mode.split(' (')[0].lower()}, {sec:.1f} s.")


def on_image(rgb, mode):
    if rgb is None:
        return None, [], "Upload an image or pick an example below."
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, -1)
    bgr = np.ascontiguousarray(rgb[..., 2::-1])  # drops an alpha channel if present
    try:
        dets, viols, sec = run(bgr, mode)
    except Exception as e:  # show the problem in the page instead of an empty table
        return None, [["-", "error", "", str(e)[:80], ""]], f"**Error while processing:** {e}"
    return draw(bgr, dets, viols)[..., ::-1], table(viols), summary(dets, viols, sec, mode)


def on_compare(rgb):
    """Both modes side by side on one image."""
    if rgb is None:
        return None, None, "Upload an image or pick an example below."
    bgr = np.ascontiguousarray((np.stack([rgb] * 3, -1) if rgb.ndim == 2 else rgb)[..., 2::-1])
    outs, notes = [], []
    for mode in (DET_ONLY, OURS):
        dets, viols, _ = run(bgr, mode)
        outs.append(draw(bgr, dets, viols)[..., ::-1])
        notes.append(f"**{mode.split(' (')[0]}:** {sum(v['type'] == 'no helmet' for v in viols)} helmet "
                     f"violation(s), {sum(v['type'] == 'no seat belt' for v in viols)} seat-belt violation(s)")
    return outs[0], outs[1], "  \n".join(notes)


def on_video(path, mode, step, progress=gr.Progress()):
    """Annotates every `step`-th frame (the detector runs at ~1 frame/s on a CPU) and lists each violation once."""
    if path is None:
        return None, [], "Upload a video."
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w, h = int(cap.get(3)), int(cap.get(4))
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"demo_{int(time.time())}.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), max(1.0, fps / step), (w, h))
    if not vw.isOpened():  # OpenCV builds without H.264
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / step), (w, h))
    rows, seen, k = [], {}, 0
    bg, bikes = background(path), 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % step == 0:
            progress(k / total, desc=f"frame {k} / {total}")
            dets, viols, _ = run(f, mode)
            img = draw(f, dets, viols)
            if bg is not None and not any(d["cls"] in (RIDER_OK, RIDER_BAD) for d in dets):
                # only where the detector sees no rider: a street-level frame already has its riders judged
                judged = judge_bikes(f, two_wheelers(f, bg, dets), dets)
                bikes += len(judged)
                img = draw_bikes(img, judged)
                for *_, v in judged:
                    if v is not None:
                        add(seen, v, k / fps)
            vw.write(img)
            for v in viols:
                add(seen, v, k / fps)
        k += 1
    vw.release()
    rows = [[f"{r['t']:.1f}", r["type"], f"{r['score']:.2f}", r["text"], f"{r['conf']:.2f}" if r["conf"] else "",
             r["n"]] for r in sorted(seen.values(), key=lambda r: r["t"])]
    note = (f"  \nFixed camera: {bikes} two-wheeler box(es) found by background subtraction, judged by the "
            "top-view helmet classifier (orange = not sure)." if bikes else "")
    return out, rows, (f"**{len(rows)} distinct violation(s)** in {k} frames ({len(range(0, k, step))} analysed)."
                       + note)


def background(path, n=60):
    """Median of up to n frames spread over the video: the empty road for a fixed traffic camera. Returns None for
    a moving camera (most pixels then differ from the median in most frames)."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    for i in np.linspace(0, max(total - 1, 0), min(n, max(total, 1))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    if len(frames) < 10:
        return None
    bg = np.median(np.stack(frames), 0).astype(np.uint8)
    moving = [np.mean(cv2.cvtColor(cv2.absdiff(f, bg), cv2.COLOR_BGR2GRAY) > 35) for f in frames]
    return bg if np.median(moving) < 0.15 else None


def two_wheelers(f, bg, dets):
    """Background subtraction for a fixed camera: moving blobs that are not inside a detected car or rider box and
    have a two-wheeler's size and upright shape. The helmet is not judged (the classifier never saw top views)."""
    diff = cv2.cvtColor(cv2.absdiff(f, bg), cv2.COLOR_BGR2GRAY)
    m = (cv2.GaussianBlur(diff, (5, 5), 0) > 35).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    for d in dets:
        if d["cls"] in (CAR, RIDER_OK, RIDER_BAD):
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(m, (x1, y1), (x2, y2), 0, -1)
    H, W = m.shape
    s = (H * W / 640 ** 2)
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for x, y, w, h, a in st[1:]:
        if 600 * s < a < 12000 * s and 1.2 < h / w < 5 and w < 90 * s ** 0.5 and x > 2 and x + w < W - 2:
            out.append((x, y, x + w, y + h))
    return out


def judge_bikes(f, boxes, dets):
    """Top-view helmet decision for each two-wheeler box (overhead classifier); a confident 'no helmet' becomes a
    violation linked to the plate whose centre lies in the box. Returns [(box, label, prob, violation or None)]."""
    if OVERHEAD is None or not boxes:
        return [(b, "two-wheeler", 0.0, None) for b in boxes]
    out = []
    plates = [d for d in dets if d["cls"] == PLATE]
    for b, p in zip(boxes, helmetnet.classify(OVERHEAD, f, [list(map(float, b)) for b in boxes])):
        if p[1] >= SURE:
            x1, y1, x2, y2 = b
            inside = [d for d in plates if x1 <= (d["box"][0] + d["box"][2]) / 2 <= x2
                      and y1 <= (d["box"][1] + d["box"][3]) / 2 <= y2]
            plate = max(inside, key=lambda d: d["score"]) if inside else None
            text, conf = S.read(f, plate) if plate else ("", 0.0)
            v = {"type": "no helmet (top view)", "det": {"score": float(p[1]), "box": list(map(float, b))},
                 "plate": plate, "text": text, "ocr_conf": conf}
            out.append((b, "no helmet", float(p[1]), v))
        elif p[0] >= SURE:
            out.append((b, "helmet", float(p[0]), None))
        else:
            out.append((b, "two-wheeler", 0.0, None))
    return out


def draw_bikes(img, judged):
    for (x1, y1, x2, y2), label, prob, v in judged:
        col = {"no helmet": (0, 0, 255), "helmet": (0, 200, 0)}.get(label, (0, 140, 255))
        text = "TWO-WHEELER" if label == "two-wheeler" else f"{label.upper()} {prob:.2f} (top view)"
        if v is not None and v["text"]:
            text += f" | {v['text']}"
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return img


def add(seen, v, t):
    """Merges the readings of one plate across frames (OCR differs slightly from frame to frame) and keeps the
    most confident one; violations without a readable plate are kept per type."""
    text = v["text"] or ("unreadable" if v["plate"] else "plate not visible")
    key = next((k for k in seen if v["text"] and seen[k]["type"] == v["type"] and seen[k]["conf"] and
                SequenceMatcher(None, k, v["text"]).ratio() >= 0.6), None) if v["text"] else f"{v['type']}: {text}"
    key = key or v["text"]
    r = seen.setdefault(key, {"t": t, "type": v["type"], "score": 0, "text": text, "conf": 0, "n": 0})
    r["n"] += 1
    if v["ocr_conf"] > r["conf"] or not r["conf"]:
        r.update(score=v["det"]["score"], text=text, conf=v["ocr_conf"])


def examples(kind):
    if not os.path.isdir(EXAMPLES):
        return []
    ext = (".jpg", ".jpeg", ".png") if kind == "image" else (".mp4",)
    return sorted(os.path.join(EXAMPLES, f) for f in os.listdir(EXAMPLES) if f.lower().endswith(ext))


LEGEND = ("Red: violator (with the plate text) · Yellow: plate linked to the violator · Green: rider with helmet / "
          "driver with seat belt · Blue: car")

with gr.Blocks(title="Helmet and Seat-belt Violation Detector") as demo:
    gr.Markdown("# Helmet and Seat-belt Violation Detector\n"
                "Finds two-wheeler riders without a helmet and drivers without a seat belt, links each violator to "
                "the number plate of their own vehicle and reads it.  \n"
                "EE604 course project, IIT Kanpur · "
                "[code](https://github.com/rauni9869/helmet-seatbelt-violations)")
    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="Traffic image", type="numpy")
                mode = gr.Radio([OURS, DET_ONLY], value=OURS, label="Mode")
                go = gr.Button("Detect violations", variant="primary")
            with gr.Column(scale=1):
                img_out = gr.Image(label="Result")
                info = gr.Markdown()
        viol = gr.Dataframe(headers=["#", "violation", "score", "plate text", "OCR confidence"],
                            label="Violations", interactive=False)
        gr.Markdown(LEGEND)
        gr.Examples([[p] for p in examples("image")], inputs=[img_in])
        go.click(on_image, [img_in, mode], [img_out, viol, info])
    with gr.Tab("Compare: detector only vs. full system"):
        gr.Markdown("The detector alone is trained on five datasets that each label only some classes, so it "
                    "never learns that a car is not a rider. The helmet classifier, trained on the detector's own "
                    "mistakes, removes these false violations and gives each rider one helmet decision.")
        cmp_in = gr.Image(label="Traffic image", type="numpy")
        cmp_go = gr.Button("Compare", variant="primary")
        with gr.Row():
            cmp_a = gr.Image(label="Detector only")
            cmp_b = gr.Image(label="Full system (ours)")
        cmp_info = gr.Markdown()
        gr.Examples([[p] for p in examples("image")], inputs=[cmp_in])
        cmp_go.click(on_compare, [cmp_in], [cmp_a, cmp_b, cmp_info])
    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column(scale=1):
                vid_in = gr.Video(label="Traffic video")
                vmode = gr.Radio([OURS, DET_ONLY], value=OURS, label="Mode")
                step = gr.Slider(1, 30, value=10, step=1, label="Analyse every n-th frame (higher is faster)")
                vgo = gr.Button("Process video", variant="primary")
            with gr.Column(scale=1):
                vid_out = gr.Video(label="Annotated video")
                vinfo = gr.Markdown()
        vtab = gr.Dataframe(headers=["first seen (s)", "violation", "score", "plate text (best frame)",
                                     "OCR confidence", "frames"],
                            label="Violations found", interactive=False)
        gr.Examples([[p] for p in examples("video")], inputs=[vid_in])
        vgo.click(on_video, [vid_in, vmode, step], [vid_out, vtab, vinfo])

if __name__ == "__main__":
    demo.launch()
