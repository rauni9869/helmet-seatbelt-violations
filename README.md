# Detection of Helmet and Seat-belt Violators with their Vehicle Number Plates

EE604 (Image Processing) course project by Sanskaar Srivastava, Raunak Raj and Prashant Shekhar. The system finds two-wheeler riders without a helmet and car drivers
without a seat belt, links each violator to the number plate of their vehicle, and reads the plate.

```
image ──► YOLO-style detector (ResNet-34 + U-Net neck, 3 scales × 3 K-means anchors, NMS)
              │  rider_helmet · rider_nohelmet · car · seatbelt · no_seatbelt · plate
              ▼
          helmet classifier on every rider box (ResNet-18 on the head-and-shoulders crop):
              helmet · no helmet · not a rider (drops rider boxes on cars / parked bikes)
              ▼
          violator ↔ vehicle ↔ plate association (geometric containment)
              ▼
          plate OCR: CLAHE → Canny + Hough deskew → Otsu / local thresholds → connected components
                     → K-means row grouping → CNN character classifier → Indian-format decoding
```

## Datasets (Kaggle, downloaded to `data/raw/`)

| Kaggle dataset | Used for | Licence |
|---|---|---|
| `pkdarabi/helmet` (HelmetViolations v2) | riders with / without helmet, plates | CC BY 4.0 |
| `aneesarom/rider-with-helmet-without-helmet-number-plate` | riders, helmet / no-helmet heads, plates | Apache 2.0 |
| `uzairahmad1434/seatbelt-detection` | traffic-camera frames: vehicle, seat belt, no seat belt | Apache 2.0 (CC BY 4.0 on Roboflow) |
| `mariamabdelghanie/seatbelt-detection-v1i` | vehicle, seat belt, no seat belt | Apache 2.0 (CC BY 4.0 on Roboflow) |
| `kedarsai/indian-license-plates-with-labels` | Indian number plates (boxes) | CC0 |
| `aladdinss/license-plate-digits-classification-dataset` | character images 0-9, A-Z | unknown |
| `francescopettini/license-plate-characters-detection-ocr` | Indian plate crops with per-character boxes (OCR ground truth) | CC0 |

Download with the Kaggle CLI, e.g.
`kaggle datasets download pkdarabi/helmet -p data/raw/helmet --unzip` (folder names as used in `prepare_data.py`:
`helmet`, `rider-with-helmet-without-helmet-number-plate`, `seatbelt-detection`, `seatbelt-detection-v1i`,
`indian-license-plates-with-labels`, `license-plate-digits-classification-dataset`,
`license-plate-characters-detection-ocr`).

## Getting started

1. `pip install -r requirements.txt` (install PyTorch with CUDA from pytorch.org; on a Mac the default build uses
   the Apple GPU through MPS).
2. **Trained weights** are not stored in git (the detector is 96 MB). Download `det_main.pt`, `charnet.pt` and
   `helmetnet.pt` from the [Releases page](https://github.com/rauni9869/helmet-seatbelt-violations/releases) into `runs/`.
3. **Datasets:** download the 7 Kaggle datasets above into `data/raw/` and run `python prepare_data.py`. The
   split files (`data/processed/train.json`, `val.json`, `test.json`), the anchors and the OCR sets are in git,
   and `prepare_data.py` rebuilds the same split (fixed seeds).
4. Run the system on your own image: `python pipeline.py --image some.jpg` (writes `outputs/some_out.jpg`).

| Folder | Content |
|---|---|
| `data/processed/` | split files, anchors, plate OCR sets, cached detector rider boxes (images and `chars.npz` are rebuilt by `prepare_data.py`) |
| `runs/` | `thresholds.json` and training logs (weights: see Releases) |
| `results/` | all numeric results (`det_main.txt`, `ocr_test.txt`, `pipeline.txt`, `helmet.txt`, `chars.txt`, `data_stats.txt`) |
| `figs/` | result figures (PR curves, example outputs, plate reading) |

Paths are relative to this folder, so it can be cloned anywhere.

Current results (test split): detector mAP@0.5 0.746 (no-helmet AP 0.87, no-seat-belt AP 0.55, plate AP 0.96);
complete system with the helmet classifier: no-helmet violators precision 0.895 / recall 0.850 / F1 0.872
(detector alone 0.643 / 0.900 / 0.750), no "no helmet" boxes on cars (detector alone: 81); no-seat-belt F1 0.500;
plate OCR exact match 0.48, character error rate 0.22. Notes for improving the project:

- The detector was stopped at epoch 45 of a 60-epoch schedule (laptop on battery); training to 60 should help.
- The ablation without the partial-label mask (`python train_detector.py --nomask --name nomask`) was not run.
- Known weakness of the detector: rider and car classes confuse each other across datasets, and helmet vs
  no helmet is often a near tie (see the write-up's discussion). The helmet classifier (`helmetnet.py`,
  `train_helmet.py`) corrects this in `ViolationSystem.arbitrate`; without `runs/helmetnet.pt` the pipeline
  falls back to rules (driver box inside a rider box -> car). Pseudo-labelling and retraining the detector
  would fix the cause.
- The helmet classifier still calls some plate close-ups without a visible rider "no helmet". Suggested fix in
  `train_helmet.py`: on plate-only images, use detector rider boxes in which the plate covers > 10% of the box
  as "not a rider" (labelled riders: at most 5.7%).
- If the folder is on an iCloud-synced Desktop with "Optimize Mac Storage", macOS can evict files during long
  runs and training then hangs; keep the project in a non-synced folder for training.
- `train_chars.py` renders synthetic characters from fonts in `C:/Windows/Fonts`; on Linux/Mac change the
  `FONTS` path, otherwise training runs without the synthetic characters.
- Training needs ~3.5 GB GPU memory and enough RAM for 6 data-loader workers; use `--workers 3` on 8-16 GB RAM.

## Running

```
python prepare_data.py            # unified dataset, splits, K-means anchors, OCR sets  -> data/processed
python train_detector.py          # detector with partial-label (masked) loss            -> runs/det_main.pt
python train_detector.py --nomask --name nomask    # ablation
python eval_detector.py           # test mAP, PR curves, thresholds                      -> results/, figs/
python eval_detector.py --name nomask
python train_chars.py             # character CNN                                        -> runs/charnet.pt
python train_helmet.py            # helmet classifier on rider crops                      -> runs/helmetnet.pt
python eval_ocr.py test           # plate OCR evaluation
python pipeline.py --eval         # violation-level evaluation + annotated outputs       -> outputs/
python pipeline.py --image some.jpg
python pipeline.py --video some.mp4
```

## Files

| File | Content |
|---|---|
| `common.py` | paths, class list, image I/O, IoU, NMS, letterbox |
| `prepare_data.py` | dataset unification, leakage-free splits, de-duplication, K-means anchors, OCR data |
| `detector.py` | model, target assignment, masked focal-BCE + MSE + GIoU loss, decoding |
| `train_detector.py`, `eval_detector.py`, `metrics.py` | training, evaluation, VOC AP |
| `charnet.py`, `train_chars.py` | character CNN and its training (real + synthetic characters) |
| `ocr.py`, `eval_ocr.py` | plate reading pipeline and its evaluation |
| `helmetnet.py`, `train_helmet.py` | second-stage helmet classifier (helmet / no helmet / not a rider) and its training |
| `pipeline.py` | complete violation system |
