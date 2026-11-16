"""Turn a classified TrackEvent into a note, and render notes to audio.

Mapping (all tweakable here):
  particle type  -> instrument + register   alpha=low bell, beta=pluck, muon=high glass, electron=tick
  orientation    -> pitch (scale degree; 0..180 deg sweeps two octaves)
  path length    -> note length
  ionisation     -> loudness (brighter, denser trail = louder)
  x position     -> stereo pan
  wiggle         -> vibrato depth (scattered tracks wobble)

A note on "speed": a cloud chamber trail forms almost instantly, so the camera
cannot measure particle speed. Ionisation density is the physical stand-in:
slow, heavy particles (alphas) ionise densely; fast, minimum-ionising ones
(muons, fast betas) leave faint trails. That's what drives loudness here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import TrackEvent

SCALES = {
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "chromatic": list(range(12)),
}

# octave offset above root, base gain, instrument
VOICES = {
    "alpha": (0, 1.0, "bell"),
    "beta": (1, 0.8, "pluck"),
    "muon": (2, 0.7, "glass"),
    "electron": (3, 0.5, "tick"),
    "unknown": (1, 0.5, "pluck"),
}


@dataclass
class Note:
    freq: float
    dur: float
    gain: float
    pan: float          # -1 left .. +1 right
    instrument: str
    vibrato_cents: float
    kind: str


INSTRUMENTS = ("bell", "pluck", "glass", "tick")


def event_to_note(e: TrackEvent, root_hz=110.0, scale="minor_pentatonic",
                  instruments: dict | None = None) -> Note | None:
    """Returns None when this particle type is muted."""
    octave, base_gain, inst = VOICES.get(e.kind, VOICES["unknown"])
    inst = (instruments or {}).get(e.kind, inst)
    if inst == "mute":
        return None
    steps = SCALES[scale]
    span = 2 * len(steps)
    deg = int(np.clip(e.angle_deg / 180.0 * span, 0, span - 1))
    semis = 12 * (octave + deg // len(steps)) + steps[deg % len(steps)]
    freq = root_hz * 2 ** (semis / 12)
    L = e.path_mm
    dur = {
        "alpha": 0.6 + np.clip(L / 50, 0, 1.5) * 1.2,
        "muon": 1.5 + np.clip(L / 100, 0, 1.2) * 1.5 + (0.5 if e.clipped else 0),
        "beta": 0.3 + np.clip(L / 80, 0, 1) * 0.9,
        "electron": 0.08 + np.clip(L / 12, 0, 1) * 0.15,
    }.get(e.kind, 0.5)
    loud = float(np.clip(e.brightness / 60.0, 0.3, 1.0))
    return Note(freq=float(freq), dur=float(dur), gain=base_gain * loud,
                pan=float(np.clip(e.x * 2 - 1, -1, 1)), instrument=inst,
                vibrato_cents=float(min(60.0, e.wiggle * 3000)), kind=e.kind)


def _env(n, sr, attack, decay_tau):
    t = np.arange(n) / sr
    a = np.minimum(1.0, t / max(attack, 1e-4))
    e = a * np.exp(-t / decay_tau)
    tail = min(n, int(0.02 * sr))     # 20 ms fade so nothing clicks
    e[-tail:] *= np.linspace(1, 0, tail)
    return e


def render(note: Note, sr: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return (N, 2) float32 stereo audio for one note."""
    rng = rng or np.random.default_rng()
    n = int(note.dur * sr)
    t = np.arange(n) / sr
    vib = 2 ** (note.vibrato_cents / 1200 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(note.freq * vib) / sr
    f, inst = note.freq, note.instrument
    if inst == "bell":       # inharmonic partials, each decaying faster
        y = np.zeros(n)
        for ratio, amp, tau in ((1, 1, 0.45), (2.0, 0.5, 0.3), (2.76, 0.35, 0.2), (5.4, 0.2, 0.1)):
            y += amp * np.sin(phase * ratio) * np.exp(-t / (tau * note.dur))
        y *= _env(n, sr, 0.004, note.dur)
    elif inst == "glass":    # pure, slow-swelling, shimmering
        y = np.sin(phase) + 0.15 * np.sin(3 * phase) + 0.3 * np.sin(phase * 1.003)
        y *= _env(n, sr, 0.08, note.dur * 0.6)
    elif inst == "pluck":    # harmonic series, upper harmonics die first
        y = np.zeros(n)
        for h in range(1, 7):
            if f * h > sr / 2.2:
                break
            y += (1 / h) * np.sin(phase * h) * np.exp(-t * h / (0.35 * note.dur))
        y *= _env(n, sr, 0.002, note.dur * 0.5)
    else:                    # tick: short bright blip with a breath of noise
        y = np.sin(phase * 2) + 0.3 * rng.normal(0, 1, n)
        y *= _env(n, sr, 0.001, max(0.02, note.dur * 0.3))
    y = y / (np.max(np.abs(y)) + 1e-9) * note.gain
    ang = (note.pan + 1) * np.pi / 4      # equal-power pan
    return np.column_stack([y * np.cos(ang), y * np.sin(ang)]).astype(np.float32)
