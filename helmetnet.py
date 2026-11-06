"""Second-stage helmet classifier: looks at the upper part of a detected rider box (head and shoulders) and
decides helmet / no helmet / not a rider. The detector sees the helmet only as a small part of a box that also
covers the two-wheeler, and is never trained to tell riders from cars (partial labels), so it is re-judged here."""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from common import RUNS, dev, letterbox

LABELS = ["helmet", "no helmet", "not a rider"]
S = 160  # crop size
TOP = 0.6  # fraction of the rider box (from the top) that is classified
MEAN = np.array([0.485, 0.456, 0.406]) * 255
STD = np.array([0.229, 0.224, 0.225]) * 255


def crop(bgr, box, jitter=None):
    """Upper part of a rider box, letterboxed to S x S. jitter = rng for training-time box noise."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if jitter is not None:  # detector boxes are off by ~10% of their size
        x1, x2 = x1 + jitter.normal(0, 0.06) * w, x2 + jitter.normal(0, 0.06) * w
        y1, y2 = y1 + jitter.normal(0, 0.06) * h, y2 + jitter.normal(0, 0.06) * h
        h = y2 - y1
    y2 = y1 + TOP * h
    H, W = bgr.shape[:2]
    c = bgr[int(np.clip(y1, 0, H - 1)):int(np.clip(y2, 1, H)), int(np.clip(x1, 0, W - 1)):int(np.clip(x2, 1, W))]
    if c.size == 0:
        c = np.full((S, S, 3), 114, np.uint8)
    return letterbox(c, S)[0]


def to_tensor(crops):
    """uint8 (N, S, S, 3) BGR -> normalised float (N, 3, S, S) RGB."""
    x = (np.asarray(crops)[..., ::-1] - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).astype(np.float32))


class HelmetNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(LABELS)))

    def forward(self, x):
        return self.net(x)


def load_model(device=dev):
    m = HelmetNet(pretrained=False).to(device)
    m.load_state_dict(torch.load(f"{RUNS}/helmetnet.pt", map_location=device))
    return m.eval()


@torch.no_grad()
def classify(model, bgr, boxes):
    """Class probabilities (N, 3) for rider boxes of one image."""
    if not len(boxes):
        return np.zeros((0, len(LABELS)))
    x = to_tensor([crop(bgr, b) for b in boxes]).to(next(model.parameters()).device)
    return model(x).softmax(1).cpu().numpy()
