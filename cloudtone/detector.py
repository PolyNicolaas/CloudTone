"""Track detection: background subtraction -> onset pixels -> one measured
TrackEvent per new condensation trail."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import DetectConfig


@dataclass
class TrackEvent:
    id: int
    t: float
    x: float              # centroid, 0..1 across the frame
    y: float              # centroid, 0..1 down the frame
    length_mm: float      # straight extent along principal axis
    path_mm: float        # length along the (possibly curved) centreline
    width_mm: float
    angle_deg: float      # 0..180, orientation of principal axis
    wiggle: float         # centreline RMS deviation / length (0 = ruler straight)
    brightness: float     # mean grey levels above background (ionisation proxy)
    area_px: int
    clipped: bool         # touches the frame edge (true length unknown)
    bbox: tuple[int, int, int, int]   # full-frame pixels, even when an ROI is set
    kind: str = "unknown"
    confidence: float = 0.0
    extra: dict = field(default_factory=dict)


def measure(coords: np.ndarray, weights: np.ndarray, ppm: float) -> dict:
    """Shape features for one blob. coords: (N,2) as (x, y) pixels."""
    c = coords.astype(np.float64)
    mu = c.mean(0)
    d = c - mu
    evals, evecs = np.linalg.eigh(np.cov(d.T) if len(c) > 2 else np.eye(2))
    v1 = evecs[:, 1]
    v2 = evecs[:, 0]
    p, r = d @ v1, d @ v2
    length = float(p.max() - p.min()) + 1.0
    area = len(c)
    width = area / length
    # centreline: mean perpendicular offset in bins along the principal axis
    nb = int(np.clip(length / max(2.0, 1.5 * width), 3, 40))
    edges = np.linspace(p.min(), p.max() + 1e-6, nb + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, nb - 1)
    centres = []
    for i in range(nb):
        sel = idx == i
        if sel.any():
            centres.append(((edges[i] + edges[i + 1]) / 2, r[sel].mean()))
    cl = np.array(centres)
    path = float(np.sum(np.hypot(np.diff(cl[:, 0]), np.diff(cl[:, 1])))) if len(cl) > 1 else length
    # remove any constant tilt before measuring wiggle
    if len(cl) > 2:
        fit = np.polyfit(cl[:, 0], cl[:, 1], 1)
        resid = cl[:, 1] - np.polyval(fit, cl[:, 0])
    else:
        resid = np.zeros(1)
    wiggle = float(np.sqrt(np.mean(resid ** 2)) / length)
    ang = math.degrees(math.atan2(v1[1], v1[0])) % 180.0
    return dict(
        x=mu[0], y=mu[1],
        length_mm=length / ppm, path_mm=max(path, length) / ppm,
        width_mm=width / ppm, angle_deg=ang, wiggle=wiggle,
        brightness=float(weights.mean()), area_px=area,
    )


class TrackDetector:
    """Onset-based detector.

    Condensation trails appear within a frame or two and then only fade. So
    instead of following blobs (which merge and split as they fade and cross),
    we look for *onset* pixels: foreground now, but not foreground anywhere
    nearby in the last `forget_frames` frames. Onset pixels accumulate for a
    couple of frames while a trail is still forming; once a cluster stops
    growing it is measured and emitted exactly once. Fading trails make no
    onset pixels, and a new trail crossing an old one is still seen as new.
    """

    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self.bg: np.ndarray | None = None
        self.frame_no = 0
        self.next_id = 1
        self.last_fg: np.ndarray | None = None   # frame index each pixel was last (near) foreground
        self.birth: np.ndarray | None = None     # frame index an accumulating onset pixel appeared
        self.last_mask: np.ndarray | None = None
        k = max(1, cfg.close_kernel_px)
        self._close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = 2 * max(1, cfg.match_margin_px // 3) + 1
        self._halo = cv2.getStructuringElement(cv2.MORPH_RECT, (m, m))

    def process(self, gray: np.ndarray, t: float) -> list[TrackEvent]:
        cfg = self.cfg
        if cfg.roi:
            x, y, w, h = cfg.roi
            gray = gray[y:y + h, x:x + w]
        f = gray.astype(np.float32)
        self.frame_no += 1
        fn = self.frame_no
        if self.bg is None:
            self.bg = f.copy()
            self.last_fg = np.full(f.shape, -10_000, np.int32)
            self.birth = np.zeros(f.shape, np.int32)
            return []
        diff = f - self.bg
        mask = (diff > cfg.threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close)
        cv2.accumulateWeighted(f, self.bg, cfg.bg_alpha, mask=(1 - mask))
        self.last_mask = mask
        halo = cv2.dilate(mask, self._halo).astype(bool)

        fg = mask.astype(bool)
        forming = cv2.dilate((self.birth > 0).astype(np.uint8), self._halo).astype(bool)
        # new = dark for a long time, OR the growing tip of a trail still forming
        onset = fg & (self.birth == 0) & (((fn - self.last_fg) > cfg.forget_frames) | forming)
        self.birth[onset] = fn
        self.birth[~halo] = 0            # onset pixels that vanished were noise
        self.last_fg[halo] = fn

        acc = (self.birth > 0).astype(np.uint8)
        if not acc.any():
            return []
        acc_j = cv2.morphologyEx(acc, cv2.MORPH_CLOSE, self._close)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(acc_j, connectivity=8)
        H, W = mask.shape
        events: list[TrackEvent] = []
        for lab in range(1, n):
            x, y, w, h, _ = stats[lab]
            sub = labels[y:y + h, x:x + w] == lab
            b = self.birth[y:y + h, x:x + w][sub]
            b = b[b > 0]
            if not len(b):
                continue
            age = fn - int(b.min()) + 1
            growing = bool((b == fn).sum() > 0.05 * len(b))
            ready = age >= cfg.settle_frames and (not growing or age >= cfg.settle_frames + 4)
            if not ready:
                continue
            ys, xs = np.nonzero(sub & (self.birth[y:y + h, x:x + w] > 0))
            ys, xs = ys + y, xs + x
            self.birth[ys, xs] = 0       # consumed: never emits again
            if len(xs) < cfg.min_area_px:
                continue
            m = measure(np.column_stack([xs, ys]), diff[ys, xs], cfg.px_per_mm)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            clipped = bbox[0] <= 1 or bbox[1] <= 1 or bbox[0] + bbox[2] >= W - 1 or bbox[1] + bbox[3] >= H - 1
            ox, oy = (cfg.roi[0], cfg.roi[1]) if cfg.roi else (0, 0)
            bbox = (bbox[0] + ox, bbox[1] + oy, bbox[2], bbox[3])     # full-frame pixels
            events.append(TrackEvent(id=self.next_id, t=t, x=m.pop("x") / W, y=m.pop("y") / H,
                                     clipped=bool(clipped), bbox=bbox, **m))
            self.next_id += 1
        return events
