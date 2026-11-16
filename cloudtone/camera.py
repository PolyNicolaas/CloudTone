"""Frame sources. Iterating one yields (gray_uint8_frame, timestamp_seconds).
`apply(cfg)` pushes live camera settings (exposure, gain, focus) where supported."""
from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .config import CameraConfig


class Source:
    live_controls = False

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]: ...
    def apply(self, cfg: CameraConfig) -> None: ...


class PiCameraSource(Source):
    live_controls = True

    def __init__(self, cfg: CameraConfig):
        from picamera2 import Picamera2

        self.cfg = cfg
        self.cam = Picamera2()
        conf = self.cam.create_video_configuration(
            main={"size": (cfg.width, cfg.height), "format": "YUV420"},
            controls={"FrameRate": cfg.fps, "AwbEnable": False},
            buffer_count=4,
        )
        self.cam.configure(conf)
        self.apply(cfg)
        self.cam.start()

    def apply(self, cfg: CameraConfig) -> None:
        # locked exposure: auto-exposure "breathing" would look like tracks
        ctl = {"AeEnable": False, "ExposureTime": int(cfg.exposure_us),
               "AnalogueGain": float(cfg.analogue_gain)}
        if cfg.lens_position is not None and "LensPosition" in self.cam.camera_controls:
            ctl.update(AfMode=0, LensPosition=float(cfg.lens_position))   # manual focus
        self.cam.set_controls(ctl)

    def __iter__(self):
        h, w = self.cfg.height, self.cfg.width
        try:
            while True:
                yuv = self.cam.capture_array("main")
                yield yuv[:h, :w], time.monotonic()      # Y plane = grayscale
        finally:
            self.cam.stop()


class OpenCVSource(Source):
    def __init__(self, src: str, cfg: CameraConfig, realtime: bool):
        import cv2

        self.cv2, self.src, self.cfg, self.realtime = cv2, src, cfg, realtime
        self.cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {src!r}")

    def __iter__(self):
        cv2, cfg = self.cv2, self.cfg
        fps = self.cap.get(cv2.CAP_PROP_FPS) or cfg.fps
        i, t0 = 0, time.monotonic()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            if gray.shape[1] != cfg.width:
                gray = cv2.resize(gray, (cfg.width, int(gray.shape[0] * cfg.width / gray.shape[1])))
            t = i / fps
            if self.realtime and not self.src.isdigit():
                time.sleep(max(0.0, t0 + t - time.monotonic()))
            i += 1
            yield gray, t


class SimSource(Source):
    def __init__(self, cfg: CameraConfig, px_per_mm: float, realtime: bool, seed=None, rate_hz=1.5):
        from .simulator import ChamberSimulator

        self.cfg, self.realtime = cfg, realtime
        self.sim = ChamberSimulator(cfg.width, cfg.height, px_per_mm, rate_hz=rate_hz, seed=seed)

    def __iter__(self):
        dt = 1.0 / self.cfg.fps
        t0, i = time.monotonic(), 0
        while True:
            t = i * dt
            if self.realtime:
                time.sleep(max(0.0, t0 + t - time.monotonic()))
            yield self.sim.frame(t, dt), t
            i += 1


def open_source(cfg: CameraConfig, px_per_mm: float, realtime=True, **sim_kw) -> Source:
    if cfg.source == "picamera":
        return PiCameraSource(cfg)
    if cfg.source == "sim":
        return SimSource(cfg, px_per_mm, realtime, **sim_kw)
    return OpenCVSource(cfg.source, cfg, realtime)
