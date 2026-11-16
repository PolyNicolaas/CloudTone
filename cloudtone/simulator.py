"""Synthetic cloud chamber: renders fading tracks of known type onto a noisy
dark background. Lets the whole pipeline run (and be tested) with no camera."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

KINDS = ("alpha", "beta", "muon", "electron")


@dataclass
class SimTrack:
    kind: str
    points: np.ndarray      # (N, 2) int32 pixel polyline
    thickness: int
    brightness: float
    born: float
    grow: float             # seconds to fully form
    life: float             # seconds until fully faded


class ChamberSimulator:
    def __init__(self, width=640, height=480, px_per_mm=6.4, rate_hz=1.5, seed=None,
                 kinds=KINDS, weights=(0.3, 0.35, 0.15, 0.2)):
        self.w, self.h, self.ppm = width, height, px_per_mm
        self.rate = rate_hz
        self.rng = np.random.default_rng(seed)
        self.kinds, self.weights = list(kinds), np.array(weights[: len(kinds)], float)
        self.weights /= self.weights.sum()
        self.tracks: list[SimTrack] = []
        self.history: list[SimTrack] = []
        yy, xx = np.mgrid[0:height, 0:width]
        # uneven side lighting, like a real LED strip
        self.base = (18 + 10 * (xx / width)).astype(np.float32)

    # ---- track generators (lengths in mm, converted to px) ----
    def _line(self, length_mm, angle=None, start=None):
        L = length_mm * self.ppm
        a = self.rng.uniform(0, math.pi) if angle is None else angle
        if start is None:
            start = self.rng.uniform([0.15 * self.w, 0.15 * self.h], [0.85 * self.w, 0.85 * self.h])
        d = np.array([math.cos(a), math.sin(a)])
        return np.array([start - d * L / 2, start + d * L / 2])

    def _walk(self, length_mm, step_mm, turn_sd, start=None):
        n = max(2, int(length_mm / step_mm))
        a = self.rng.uniform(0, 2 * math.pi)
        p = np.array(start if start is not None else
                     self.rng.uniform([0.2 * self.w, 0.2 * self.h], [0.8 * self.w, 0.8 * self.h]))
        pts = [p.copy()]
        for _ in range(n):
            a += self.rng.normal(0, turn_sd)
            p = p + step_mm * self.ppm * np.array([math.cos(a), math.sin(a)])
            pts.append(p.copy())
        return np.array(pts)

    def make_track(self, kind: str, t: float, **kw) -> SimTrack:
        r = self.rng
        if kind == "alpha":
            pts, th, b = self._line(r.uniform(18, 50), **kw), int(round(r.uniform(1.6, 2.6) * self.ppm)), r.uniform(70, 110)
            grow, life = 0.05, r.uniform(1.2, 2.0)
        elif kind == "muon":
            pts = self._line(r.uniform(60, 110), **kw)
            th, b, grow, life = max(2, int(0.4 * self.ppm)), r.uniform(35, 55), 0.05, r.uniform(0.8, 1.4)
        elif kind == "beta":
            pts = self._walk(r.uniform(25, 70), 2.5, 0.22, **kw)
            th, b, grow, life = max(2, int(0.35 * self.ppm)), r.uniform(30, 50), 0.1, r.uniform(0.8, 1.3)
        elif kind == "electron":
            pts = self._walk(r.uniform(4, 9), 0.8, 0.9, **kw)
            th, b, grow, life = max(2, int(0.45 * self.ppm)), r.uniform(35, 55), 0.05, r.uniform(0.6, 1.0)
        else:
            raise ValueError(kind)
        return SimTrack(kind, pts.astype(np.int32), th, b, t, grow, life)

    def add(self, track: SimTrack):
        self.tracks.append(track)
        self.history.append(track)

    def frame(self, t: float, dt: float) -> np.ndarray:
        if self.rate > 0 and self.rng.random() < self.rate * dt:
            self.add(self.make_track(self.rng.choice(self.kinds, p=self.weights), t))
        img = self.base + self.rng.normal(0, 3.0, (self.h, self.w)).astype(np.float32)
        layer = np.zeros((self.h, self.w), np.float32)
        alive = []
        for tr in self.tracks:
            age = t - tr.born
            if age > tr.life:
                continue
            alive.append(tr)
            grow = min(1.0, age / tr.grow) if tr.grow > 0 else 1.0
            fade = max(0.0, 1.0 - max(0.0, age - tr.grow) / (tr.life - tr.grow))
            n = max(2, int(len(tr.points) * grow))
            cv2.polylines(layer, [tr.points[:n]], False, float(tr.brightness * fade), tr.thickness, cv2.LINE_AA)
        self.tracks = alive
        img += cv2.GaussianBlur(layer, (0, 0), 1.2)
        return np.clip(img, 0, 255).astype(np.uint8)
