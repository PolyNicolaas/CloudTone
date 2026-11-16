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


INSTRUMENTS = ("orbit", "comet", "drift", "ping", "boom", "zap", "warble", "blip",
               "bell", "pluck", "glass", "tick")
ARCADE = ("boom", "zap", "warble", "blip")
RETRO = ("orbit", "comet", "drift", "ping")
SOUND_SETS = {
    "retro": {"alpha": "orbit", "beta": "drift", "muon": "comet", "electron": "ping"},
    "classic": {"alpha": "bell", "beta": "pluck", "muon": "glass", "electron": "tick"},
    "arcade": {"alpha": "boom", "beta": "warble", "muon": "zap", "electron": "blip"},
}


# typical length (mm) and the note length it gets, per particle; "fixed" uses the typical value
_TYPICAL = {"alpha": (35, 1.4), "muon": (80, 2.6), "beta": (40, 0.75), "electron": (6, 0.15)}


def _norm_log(v, lo, hi):
    return float(np.clip(np.log(max(v, lo) / lo) / np.log(hi / lo), 0, 1))


def pitch_position(e: TrackEvent, source: str) -> float:
    """0..1 position of this trace along the chosen pitch source."""
    if source == "length":
        return _norm_log(e.path_mm, 3, 120)      # log: short trails get room too
    if source == "height":
        return 1.0 - e.y                          # top of the chamber = high
    if source == "across":
        return e.x                                # left = low, like a keyboard
    return e.angle_deg / 180.0


def event_to_note(e: TrackEvent, root_hz=110.0, scale="minor_pentatonic",
                  instruments: dict | None = None, m=None) -> Note | None:
    """Returns None when this particle type is silent. `m` is a MappingConfig
    (defaults reproduce the original mapping)."""
    from .config import MappingConfig

    m = m or MappingConfig()
    base_oct, base_gain, inst = VOICES.get(e.kind, VOICES["unknown"])
    inst = (instruments or {}).get(e.kind, inst)
    if inst == "mute":
        return None
    octave = int(m.octaves.get(e.kind, base_oct))

    # pitch: chosen source -> scale degree across pitch_octaves
    steps = SCALES[scale]
    pos = pitch_position(e, m.pitch_from)
    if m.pitch_reverse:
        pos = 1.0 - pos
    span = max(1, int(round(m.pitch_octaves * len(steps))))
    deg = int(np.clip(pos * span, 0, span - 1))
    semis = 12 * (octave + deg // len(steps)) + steps[deg % len(steps)]
    freq = root_hz * 2 ** (semis / 12)

    # note length
    L = e.path_mm
    if m.length_from == "fixed":
        dur = _TYPICAL.get(e.kind, (30, 0.5))[1]
    else:
        dur = {
            "alpha": 0.6 + np.clip(L / 50, 0, 1.5) * 1.2,
            "muon": 1.5 + np.clip(L / 100, 0, 1.2) * 1.5 + (0.5 if e.clipped else 0),
            "beta": 0.3 + np.clip(L / 80, 0, 1) * 0.9,
            "electron": 0.08 + np.clip(L / 12, 0, 1) * 0.15,
        }.get(e.kind, 0.5)
    dur = float(np.clip(dur * m.length_scale, 0.04, 8.0))

    # loudness: source -> 0.3..1, then dynamics blends toward even
    if m.loudness_from == "length":
        loud = 0.3 + 0.7 * _norm_log(L, 3, 100)
    elif m.loudness_from == "even":
        loud = 1.0
    else:
        loud = float(np.clip(e.brightness / 60.0, 0.3, 1.0))
    loud = 1.0 - m.dynamics * (1.0 - loud)

    return Note(freq=float(freq), dur=dur, gain=base_gain * loud,
                pan=float(np.clip((e.x * 2 - 1) * m.stereo, -1, 1)), instrument=inst,
                vibrato_cents=float(min(120.0, e.wiggle * 3000 * m.wobble)), kind=e.kind)


def sample_event(kind: str) -> TrackEvent:
    """A typical trace of this particle, for the "hear it" button."""
    L = _TYPICAL.get(kind, (30, 0.5))[0]
    width = {"alpha": 2.2, "muon": 0.5, "beta": 0.6, "electron": 0.7}.get(kind, 0.6)
    wig = {"beta": 0.03, "electron": 0.05}.get(kind, 0.003)
    return TrackEvent(id=0, t=0, x=0.5, y=0.5, length_mm=L, path_mm=L, width_mm=width, angle_deg=70,
                      wiggle=wig, brightness=90 if kind == "alpha" else 45, area_px=0,
                      clipped=False, bbox=(0, 0, 0, 0), kind=kind, confidence=1.0)


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
    elif inst == "tick":     # short bright blip with a breath of noise
        y = np.sin(phase * 2) + 0.3 * rng.normal(0, 1, n)
        y *= _env(n, sr, 0.001, max(0.02, note.dur * 0.3))
    elif inst in RETRO:
        y = _retro(inst, note, t, n, sr)
    else:
        y = _arcade(inst, note, t, phase, n, sr, rng)
    y = y / (np.max(np.abs(y)) + 1e-9) * note.gain
    if inst in ARCADE:       # square waves sound much louder than sines at the same peak
        y *= 0.55
    ang = (note.pan + 1) * np.pi / 4      # equal-power pan
    return np.column_stack([y * np.cos(ang), y * np.sin(ang)]).astype(np.float32)


# ---------- arcade voices: square waves and clocked noise, like 1970s sound chips ----------

def _square(cycles: np.ndarray, duty=0.5) -> np.ndarray:
    return np.where((cycles % 1.0) < duty, 1.0, -1.0)


def _sweep_cycles(f_of_t: np.ndarray, sr: int) -> np.ndarray:
    return np.cumsum(f_of_t) / sr


def _clocked_noise(n, sr, rate_hz: np.ndarray, rng) -> np.ndarray:
    """Noise held at a changing clock rate, like the shift-register noise in an
    old arcade board: a fast clock hisses, a slow one rumbles."""
    ticks = np.floor(np.cumsum(rate_hz) / sr).astype(int)
    vals = rng.choice([-1.0, 1.0], ticks[-1] + 2)
    return vals[ticks]


def _crush(y: np.ndarray, levels=15, hold=4) -> np.ndarray:
    """4-bit-ish amplitude steps and an ~11 kHz sample-and-hold."""
    y = y / (np.max(np.abs(y)) + 1e-9)
    y = np.round(y * levels) / levels
    return np.repeat(y[::hold], hold)[: len(y)]


def _arcade(inst, note, t, phase, n, sr, rng) -> np.ndarray:
    f, d = note.freq, note.dur
    vib = 2 ** (note.vibrato_cents / 1200 * np.sin(2 * np.pi * 5.0 * t))   # the track's wobble
    if inst == "zap":        # muon: a falling "pew" that settles and rings
        fr = f * (1 + 3.0 * np.exp(-t / 0.05)) * vib
        duty = 0.5 - 0.375 * np.minimum(1, t / 0.15)            # thins out as it rings
        y = _square(_sweep_cycles(fr, sr), duty)
        y *= _env(n, sr, 0.002, d * 0.55)
    elif inst == "warble":   # beta: a flying-saucer warble
        fr = f * (1 + 0.07 * np.sin(2 * np.pi * 7.5 * t)) * vib
        y = _square(_sweep_cycles(fr, sr))
        y *= _env(n, sr, 0.01, d * 0.5)
    elif inst == "boom":     # alpha: thump plus a rumbling explosion
        thump_f = 0.5 * f * (1 + 1.5 * np.exp(-t / 0.03))
        thump = _square(_sweep_cycles(thump_f, sr)) * np.exp(-t / 0.12)
        rumble = _clocked_noise(n, sr, 300 + 5000 * np.exp(-t / (0.25 * d)), rng) * np.exp(-t / (0.45 * d))
        y = 0.8 * thump + 0.7 * rumble
        y *= _env(n, sr, 0.001, d)
    else:                    # blip: electron, a short chirp up
        fr = 2 * f * (1 + 0.5 * np.minimum(1, t / 0.03))
        y = _square(_sweep_cycles(fr, sr))
        y *= _env(n, sr, 0.001, max(0.03, d * 0.45))
    return _crush(y)


# ---------- retro-future voices: 1970s analog synth sci-fi ----------
# Additive "analog" oscillators: each harmonic is weighted by a moving low-pass
# (and a resonant bump at the cutoff), which is how a filter sweep sounds,
# without a per-sample loop. Band-limited by construction, so nothing fizzes.

def _analog(freq_t: np.ndarray, sr: int, shape: str, cutoff: np.ndarray, reso=0.0,
            max_h=18, detune_cents=0.0) -> np.ndarray:
    """freq_t: Hz per sample. cutoff: cutoff per sample, in multiples of the
    fundamental. shape: saw | tri | sine.

    Kept cheap for a Pi Zero: filter weights are computed at a control rate
    (every 64 samples) and harmonics come from the Chebyshev recurrence
    sin(k x) = 2 cos(x) sin((k-1) x) - sin((k-2) x), so there is one sin/cos
    per oscillator instead of one per harmonic."""
    n = len(freq_t)
    cyc = 2 * np.pi * np.cumsum(freq_t) / sr
    fmax = float(freq_t.max())
    K = max(1, min(max_h, int(sr / 2.3 / max(fmax, 1.0))))
    step = 64
    cut_c = np.asarray(cutoff, dtype=np.float64)[::step]
    idx = np.minimum(np.arange(n) // step, len(cut_c) - 1)
    oscs = [(1.0, 0.0)] if not detune_cents else [(0.5, -detune_cents), (0.5, detune_cents)]
    y = np.zeros(n)
    for weight, cents in oscs:
        ph = cyc * 2 ** (cents / 1200)
        s1, c2 = np.sin(ph), 2 * np.cos(ph)
        s_prev, s_k = np.zeros(n), s1          # sin(0), sin(1 x)
        for k in range(1, K + 1):
            if shape == "saw":
                amp = 1.0 / k
            elif shape == "tri":
                amp = 0.0 if k % 2 == 0 else (1.0 / k ** 2) * (1 if (k // 2) % 2 == 0 else -1)
            else:
                amp = 1.0 if k == 1 else 0.0
            if amp:
                lp = 1.0 / np.sqrt(1.0 + (k / cut_c) ** 4)                 # 2-pole roll-off
                if reso:
                    lp = lp * (1.0 + reso * np.exp(-((k - cut_c) / (0.35 * cut_c + 0.5)) ** 2))
                y += weight * amp * lp[idx] * s_k
            if shape == "sine":
                break
            s_prev, s_k = s_k, c2 * s_k - s_prev
    return y


def _retro(inst, note, t, n, sr) -> np.ndarray:
    f, d = note.freq, note.dur
    wob = 2 ** (note.vibrato_cents / 1200 * np.sin(2 * np.pi * 5.0 * t))     # the track's own wobble
    if inst == "orbit":      # alpha: detuned analog pair, filter sweeping down with a "wow"
        cutoff = 1.3 + 6.0 * np.exp(-t / (0.22 * d)) + 0.6 * np.sin(2 * np.pi * 0.7 * t) * np.exp(-t / d)
        y = _analog(f * wob * np.ones(n), sr, "saw", cutoff, reso=0.7, max_h=14, detune_cents=7)
        y *= _env(n, sr, 0.035, 0.6 * d)
    elif inst == "comet":    # muon: theremin-like glide down a fourth, vibrato easing in
        glide = 1 + (2 ** (5 / 12) - 1) * np.exp(-t / 0.12)
        vib = 2 ** (14 / 1200 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, np.maximum(0, t - 0.25) / 0.5))
        y = _analog(f * glide * vib * wob, sr, "tri", np.full(n, 2.2), max_h=7)
        y += 0.35 * _analog(f * glide * vib * wob * 2, sr, "sine", np.ones(n))     # airy octave
        y *= _env(n, sr, 0.07, 0.7 * d)
    elif inst == "drift":    # beta: soft triangle with slow tape wow
        wow = 1 + 0.006 * np.sin(2 * np.pi * 0.9 * t) + 0.003 * np.sin(2 * np.pi * 2.3 * t)
        y = _analog(f * wow * wob, sr, "tri", np.full(n, 3.5), max_h=9, detune_cents=5)
        y *= _env(n, sr, 0.015, 0.45 * d)
    else:                    # ping: electron, a gentle sonar ping
        fr = f * (1 + 0.04 * np.exp(-t / 0.02)) * wob
        y = _analog(fr, sr, "sine", np.ones(n)) + 0.12 * _analog(fr * 2.01, sr, "sine", np.ones(n))
        y *= _env(n, sr, 0.003, max(0.12, d))
    return y
