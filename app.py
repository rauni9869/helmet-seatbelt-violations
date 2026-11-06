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
    return rows


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
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    dets, viols, sec = run(bgr, mode)
    return draw(bgr, dets, viols)[..., ::-1], table(viols), summary(dets, viols, sec, mode)


def on_compare(rgb):
    """Both modes side by side on one image."""
    if rgb is None:
        return None, None, "Upload an image or pick an example below."
    bgr = np.ascontiguousarray(rgb[..., ::-1])
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
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % step == 0:
            progress(k / total, desc=f"frame {k} / {total}")
            dets, viols, _ = run(f, mode)
            vw.write(draw(f, dets, viols))
            for v in viols:
                add(seen, v, k / fps)
        k += 1
    vw.release()
    rows = [[f"{r['t']:.1f}", r["type"], f"{r['score']:.2f}", r["text"], f"{r['conf']:.2f}" if r["conf"] else "",
             r["n"]] for r in sorted(seen.values(), key=lambda r: r["t"])]
    return out, rows, f"**{len(rows)} distinct violation(s)** in {k} frames ({len(range(0, k, step))} analysed)."


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
