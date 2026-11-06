"""YOLO-style single-stage detector (Lec 17) on a pre-trained ResNet-34 backbone (Lec 15) with a
U-Net/FCN-style top-down neck that re-uses encoder activations through skip connections (Lec 16)."""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from common import PROC, NC, box_iou, nms

STRIDES = (8, 16, 32)
NA = 3  # anchors per grid cell at each scale


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Detector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.c3, self.c4, self.c5 = r.layer2, r.layer3, r.layer4  # strides 8, 16, 32
        self.l5 = nn.Conv2d(512, 256, 1)
        self.d4 = block(256 + 256, 256)  # upsampled P5 concatenated with C4
        self.d3 = block(256 + 128, 128)  # upsampled P4 concatenated with C3
        self.heads = nn.ModuleList(nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
                                                 nn.Conv2d(c, NA * (4 + NC), 1)) for c in (128, 256, 256))
        self.register_buffer("anchors", torch.tensor(json.load(open(f"{PROC}/anchors.json"))).view(3, NA, 2))
        for h in self.heads:  # class prior of 1% so the focal loss starts stable (RetinaNet init)
            nn.init.constant_(h[-1].bias.view(NA, 4 + NC)[:, 4:], -4.6)

    def forward(self, x):
        c3 = self.c3(self.stem(x))
        c4 = self.c4(c3)
        p5 = self.l5(self.c5(c4))
        p4 = self.d4(torch.cat([F.interpolate(p5, scale_factor=2), c4], 1))
        p3 = self.d3(torch.cat([F.interpolate(p4, scale_factor=2), c3], 1))
        outs = []
        for p, h in zip((p3, p4, p5), self.heads):
            o = h(p)
            b, _, gh, gw = o.shape
            outs.append(o.view(b, NA, 4 + NC, gh, gw).permute(0, 1, 3, 4, 2))  # (B, A, H, W, 4+C)
        return outs


def decode_level(o, anchors, stride):
    """Box parameters -> xyxy pixels. Centre offset in (-0.5, 1.5) cells, size in (0, 4) x anchor."""
    _, _, gh, gw, _ = o.shape
    gy, gx = torch.meshgrid(torch.arange(gh, device=o.device), torch.arange(gw, device=o.device), indexing="ij")
    s = o[..., :4].sigmoid()
    cx = (s[..., 0] * 2 - 0.5 + gx) * stride
    cy = (s[..., 1] * 2 - 0.5 + gy) * stride
    w = (s[..., 2] * 2) ** 2 * anchors[None, :, 0, None, None]
    h = (s[..., 3] * 2) ** 2 * anchors[None, :, 1, None, None]
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def build_targets(outs, anchors, targets):
    """Assigns every ground-truth box to grid cells / anchors (Lec 17).
    targets: (N, 6) = image index, class, x1, y1, x2, y2.
    An anchor is responsible for a box if it has the highest shape-IoU with the box or a shape-IoU > 0.3;
    besides the cell holding the box centre, the two nearest neighbouring cells are also used (more positives)."""
    res = []
    if not len(targets):
        return [None] * 3
    wh = targets[:, 4:6] - targets[:, 2:4]
    ctr = (targets[:, 2:4] + targets[:, 4:6]) / 2
    flat = anchors.view(-1, 2)
    inter = torch.min(wh[:, None], flat[None]).prod(2)
    shape_iou = inter / (wh.prod(1)[:, None] + flat.prod(1)[None] - inter)
    best = shape_iou.argmax(1)
    for l, (o, stride) in enumerate(zip(outs, STRIDES)):
        _, _, gh, gw, _ = o.shape
        idx = []
        for a in range(NA):
            k = l * NA + a
            m = (shape_iou[:, k] > 0.3) | (best == k)
            ratio = torch.max(wh / anchors[l, a], anchors[l, a] / wh).max(1).values
            m &= ratio < 4  # box must be representable with this anchor (size up to 4x)
            if not m.any():
                continue
            t, c = targets[m], ctr[m] / stride
            g = c.floor()
            f = c - g
            offs = [torch.zeros_like(g)]
            offs.append(torch.where(f[:, :1] < 0.5, -1.0, 1.0) * torch.tensor([[1.0, 0.0]], device=g.device))
            offs.append(torch.where(f[:, 1:] < 0.5, -1.0, 1.0) * torch.tensor([[0.0, 1.0]], device=g.device))
            for d in offs:
                cell = g + d
                ok = (cell[:, 0] >= 0) & (cell[:, 0] < gw) & (cell[:, 1] >= 0) & (cell[:, 1] < gh)
                idx.append(torch.cat([t[ok], torch.full((int(ok.sum()), 1), a, device=g.device), cell[ok]], 1))
        res.append(torch.cat(idx) if idx else None)  # rows: img, cls, x1, y1, x2, y2, anchor, gx, gy
    return res


def focal_bce(logits, target, gamma=2.0, alpha=0.25):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    return ce * (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt) ** gamma


def giou_loss(p, g):
    lt, rb = torch.max(p[:, :2], g[:, :2]), torch.min(p[:, 2:], g[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    union = area(p) + area(g) - inter
    iou = inter / (union + 1e-9)
    hull = (torch.max(p[:, 2:], g[:, 2:]) - torch.min(p[:, :2], g[:, :2])).prod(1)
    return 1 - iou + (hull - union) / (hull + 1e-9)


def detection_loss(outs, anchors, targets, cls_mask, box_w=5.0):
    """Per-class sigmoid scores (multi-label, Lec 15) trained with focal binary cross-entropy.
    cls_mask (B, C) is 1 for the classes that the image's source dataset annotates; the classification loss of
    other classes is ignored, so e.g. unlabelled plates in the seat-belt images are not treated as background.
    Box loss: squared error on the normalised box parameters + GIoU."""
    tg = build_targets(outs, anchors, targets)
    lcls, lbox, npos = 0.0, 0.0, 0
    for l, (o, t) in enumerate(zip(outs, tg)):
        cls_t = torch.zeros_like(o[..., 4:])
        if t is not None:
            b, c, a, gx, gy = t[:, 0].long(), t[:, 1].long(), t[:, 6].long(), t[:, 7].long(), t[:, 8].long()
            cls_t[b, a, gy, gx, c] = 1.0
            pred = o[b, a, gy, gx, :4].float()
            stride = STRIDES[l]
            s = pred.sigmoid()
            cx = (s[:, 0] * 2 - 0.5 + gx) * stride
            cy = (s[:, 1] * 2 - 0.5 + gy) * stride
            w = (s[:, 2] * 2) ** 2 * anchors[l, a, 0]
            h = (s[:, 3] * 2) ** 2 * anchors[l, a, 1]
            pb = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            gb = t[:, 2:6]
            gw_, gh_ = gb[:, 2] - gb[:, 0], gb[:, 3] - gb[:, 1]
            tx = ((gb[:, 0] + gb[:, 2]) / 2 / stride - gx + 0.5) / 2
            ty = ((gb[:, 1] + gb[:, 3]) / 2 / stride - gy + 0.5) / 2
            tw = (gw_ / anchors[l, a, 0]).sqrt() / 2
            th = (gh_ / anchors[l, a, 1]).sqrt() / 2
            mse = ((s - torch.stack([tx, ty, tw, th], 1).clamp(0, 1)) ** 2).sum(1)
            lbox = lbox + (mse + giou_loss(pb, gb)).sum()
            npos += len(t)
        m = cls_mask[:, None, None, None, :].expand_as(cls_t)
        lcls = lcls + (focal_bce(o[..., 4:].float(), cls_t) * m).sum()
    npos = max(npos, 1)
    lbox_v = lbox.item() if torch.is_tensor(lbox) else lbox
    return (lcls + box_w * lbox) / npos, lcls.item() / npos, lbox_v / npos


@torch.no_grad()
def predict(model, x, conf=0.25, iou=0.5, max_det=100, all_scores=False):
    """Returns per image a tensor (K, 6) = x1, y1, x2, y2, score, class, after class-wise NMS.
    all_scores=True appends the scores of all NC classes at the same anchor (K, 6 + NC)."""
    outs = model(x)
    boxes = torch.cat([decode_level(o.float(), model.anchors[l], STRIDES[l]).flatten(1, 3) for l, o in enumerate(outs)], 1)
    scores = torch.cat([o[..., 4:].float().sigmoid().flatten(1, 3) for o in outs], 1)
    res = []
    for b, s in zip(boxes, scores):
        i, c = (s > conf).nonzero(as_tuple=True)
        top = s[i, c].topk(min(len(i), 3000)).indices  # bound the candidates fed to NMS
        i, c = i[top], c[top]
        bx, sc = b[i], s[i, c]
        keep = []
        for k in c.unique():  # NMS for every class separately
            m = (c == k).nonzero().flatten()
            keep.append(m[nms(bx[m], sc[m], iou)])
        keep = torch.cat(keep) if keep else torch.zeros(0, dtype=torch.long, device=x.device)
        keep = keep[sc[keep].argsort(descending=True)][:max_det]
        cols = [bx[keep], sc[keep, None], c[keep, None].float()] + ([s[i[keep]]] if all_scores else [])
        res.append(torch.cat(cols, 1))
    return res
