"""Frame sources. Each yields (gray_uint8_frame, timestamp_seconds)."""
from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .config import CameraConfig


def picamera_frames(cfg: CameraConfig) -> Iterator[tuple[np.ndarray, float]]:
    from picamera2 import Picamera2

    cam = Picamera2()
    conf = cam.create_video_configuration(
        main={"size": (cfg.width, cfg.height), "format": "YUV420"},
        controls={
            "FrameRate": cfg.fps,
            # locked exposure: auto-exposure "breathing" would look like tracks
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": cfg.exposure_us,
            "AnalogueGain": cfg.analogue_gain,
        },
        buffer_count=4,
    )
    cam.configure(conf)
    if cfg.lens_position is not None and "LensPosition" in cam.camera_controls:
        cam.set_controls({"AfMode": 0, "LensPosition": cfg.lens_position})   # manual focus
    cam.start()
    try:
        while True:
            yuv = cam.capture_array("main")
            yield yuv[: cfg.height, : cfg.width], time.monotonic()   # Y plane = grayscale
    finally:
        cam.stop()


def opencv_frames(src: str, cfg: CameraConfig, realtime: bool) -> Iterator[tuple[np.ndarray, float]]:
    import cv2

    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {src!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps
    i = 0
    t0 = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if gray.shape[1] != cfg.width:
            gray = cv2.resize(gray, (cfg.width, int(gray.shape[0] * cfg.width / gray.shape[1])))
        t = i / fps
        if realtime and not src.isdigit():
            time.sleep(max(0.0, t0 + t - time.monotonic()))
        i += 1
        yield gray, t


def sim_frames(cfg: CameraConfig, px_per_mm: float, realtime: bool, seed=None, rate_hz=1.5,
               sim=None) -> Iterator[tuple[np.ndarray, float]]:
    from .simulator import ChamberSimulator

    sim = sim or ChamberSimulator(cfg.width, cfg.height, px_per_mm, rate_hz=rate_hz, seed=seed)
    dt = 1.0 / cfg.fps
    t0 = time.monotonic()
    i = 0
    while True:
        t = i * dt
        if realtime:
            time.sleep(max(0.0, t0 + t - time.monotonic()))
        yield sim.frame(t, dt), t
        i += 1


def open_source(cfg: CameraConfig, px_per_mm: float, realtime=True, **sim_kw):
    if cfg.source == "picamera":
        return picamera_frames(cfg)
    if cfg.source == "sim":
        return sim_frames(cfg, px_per_mm, realtime, **sim_kw)
    return opencv_frames(cfg.source, cfg, realtime)
