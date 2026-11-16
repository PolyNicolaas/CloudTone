"""Configuration: dataclasses with sensible defaults, optionally overridden by YAML."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    # "picamera" | "sim" | a video file path | a webcam index like "0"
    source: str = "picamera"
    width: int = 640
    height: int = 480
    fps: int = 20
    # Fixed exposure keeps the background stable (auto-exposure "breathing"
    # looks like tracks to the detector). Values are for Picamera2.
    exposure_us: int = 20000
    analogue_gain: float = 4.0
    # Camera Module 3 only: fixed focus in dioptres (1/metres). 10 = 10 cm away.
    # Autofocus hunts in the dark, so lock it. None = leave camera default.
    lens_position: float | None = 10.0


@dataclass
class DetectConfig:
    # Calibration: pixels per millimetre at the cold plate. A 100 mm chamber
    # filling 640 px wide is ~6.4 px/mm. Measure with a ruler on the plate.
    px_per_mm: float = 6.4
    bg_alpha: float = 0.03          # background learning rate (0..1)
    threshold: float = 14.0         # grey levels above background = "track"
    min_area_px: int = 25           # ignore specks smaller than this
    close_kernel_px: int = 5        # joins gaps in wispy beta tracks
    settle_frames: int = 2          # frames a trace must exist before it's measured
    forget_frames: int = 40         # a pixel must be dark this long before it can start a new track
    match_margin_px: int = 12       # bbox slack when matching across frames
    roi: list[int] | None = None    # optional [x, y, w, h] crop of the chamber


@dataclass
class ClassifyConfig:
    short_max_mm: float = 12.0          # below this: low-energy electron squiggle
    alpha_min_width_mm: float = 1.3     # alphas are fat and dense
    alpha_max_length_mm: float = 60.0   # ~5 MeV alphas range ~4-5 cm in air
    muon_min_length_mm: float = 35.0
    muon_max_wiggle: float = 0.012      # centreline RMS deviation / length
    muon_max_width_mm: float = 1.0


@dataclass
class AudioConfig:
    sample_rate: int = 44100         # Sonos-safe MP3 rate
    block_size: int = 1024
    root_hz: float = 110.0
    scale: str = "minor_pentatonic"   # see sonify.SCALES
    master_gain: float = 0.6
    max_voices: int = 16
    outputs: list[str] = field(default_factory=lambda: ["local"])  # "local", "sonos"


@dataclass
class SonosConfig:
    speaker: str | None = None   # room name, e.g. "Living Room"; None = first found
    port: int = 8765
    bitrate_kbps: int = 128
    volume: int | None = None    # set speaker volume on connect (0-100)
    host_ip: str | None = None   # override auto-detected LAN IP of the Pi


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    sonos: SonosConfig = field(default_factory=SonosConfig)


def _merge(obj: Any, data: dict) -> None:
    names = {f.name: f for f in fields(obj)}
    for key, value in (data or {}).items():
        if key not in names:
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path:
        import yaml

        with open(path) as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    return cfg
