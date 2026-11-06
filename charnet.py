import torch
import torch.nn as nn
from common import RUNS, CHARS, dev


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class CharNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(1, 32), nn.MaxPool2d(2), block(32, 64), nn.MaxPool2d(2),
                                      block(64, 128), nn.MaxPool2d(2), nn.Flatten())  # 40x30 -> 5x3
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 * 5 * 3, 256), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(256, len(CHARS)))

    def forward(self, x):
        return self.fc(self.features(x))


def norm(x):
    """uint8 (N,H,W) dark text on light background -> zero-mean unit-variance per image."""
    x = x.float().unsqueeze(1) / 255.0
    return (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-3)


def load_model():
    m = CharNet().to(dev)
    m.load_state_dict(torch.load(f"{RUNS}/charnet.pt", map_location=dev))
    return m.eval()
