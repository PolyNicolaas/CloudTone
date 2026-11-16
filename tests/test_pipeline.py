import numpy as np
import pytest

from cloudtone.audio import Mixer
from cloudtone.classifier import Classifier
from cloudtone.config import Config
from cloudtone.detector import TrackDetector
from cloudtone.simulator import KINDS, ChamberSimulator
from cloudtone.sonify import event_to_note, render


def run_single(kind, seed, cfg=None):
    cfg = cfg or Config()
    sim = ChamberSimulator(rate_hz=0, seed=seed)
    det, cls = TrackDetector(cfg.detect), Classifier(cfg.classify)
    dt, events = 1 / 20, []
    for i in range(10):                       # learn the empty background
        det.process(sim.frame(i * dt, dt), i * dt)
    sim.add(sim.make_track(kind, 10 * dt))
    for i in range(10, 50):
        events += [cls.classify(e) for e in det.process(sim.frame(i * dt, dt), i * dt)]
    return events


@pytest.mark.parametrize("kind", KINDS)
def test_one_event_per_track(kind):
    counts = [len(run_single(kind, s)) for s in range(15)]
    assert counts.count(1) >= 14, counts


@pytest.mark.parametrize("kind,min_acc", [("alpha", 0.95), ("muon", 0.95), ("beta", 0.85), ("electron", 0.95)])
def test_classification_accuracy(kind, min_acc):
    hits = sum(any(e.kind == kind for e in run_single(kind, s)) for s in range(20))
    assert hits / 20 >= min_acc


def test_empty_chamber_is_silent():
    cfg = Config()
    sim = ChamberSimulator(rate_hz=0, seed=0)
    det = TrackDetector(cfg.detect)
    evs = [e for i in range(200) for e in det.process(sim.frame(i / 20, 1 / 20), i / 20)]
    assert evs == []


def test_realistic_rate_counts():
    cfg = Config()
    sim = ChamberSimulator(rate_hz=1.0, seed=3)
    det = TrackDetector(cfg.detect)
    n = sum(len(det.process(sim.frame(i / 20, 1 / 20), i / 20)) for i in range(600))
    assert abs(n - len(sim.history)) <= 0.15 * len(sim.history)


def test_notes_render_and_mix():
    cfg = Config()
    ev = run_single("alpha", 1)[0]
    note = event_to_note(ev, cfg.audio.root_hz, cfg.audio.scale)
    audio = render(note, cfg.audio.sample_rate)
    assert audio.shape[1] == 2 and np.isfinite(audio).all()
    assert len(audio) == int(note.dur * cfg.audio.sample_rate)
    mix = Mixer(cfg.audio.sample_rate)
    mix.play(audio)
    blk = mix.next_block(1024)
    assert np.abs(blk).max() > 0 and np.abs(blk).max() <= 1.0


@pytest.mark.parametrize("inst", ["boom", "zap", "warble", "blip"])
def test_arcade_voices_render(inst):
    from cloudtone.sonify import Note
    note = Note(freq=330, dur=0.8, gain=0.8, pan=0.3, instrument=inst, vibrato_cents=20, kind="beta")
    a = render(note, 44100, np.random.default_rng(0))
    assert a.shape == (int(0.8 * 44100), 2) and np.isfinite(a).all()
    assert 0.2 < np.abs(a).max() <= 0.8
    assert len(np.unique(np.round(a[:, 0] / (np.abs(a[:, 0]).max() + 1e-9), 3))) <= 40   # crushed to few levels


def _centroid(a, sr=44100):
    x = a[:, 0]
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return (X * f).sum() / X.sum()


@pytest.mark.parametrize("retro,arcade,freq", [("orbit", "boom", 165), ("comet", "zap", 523),
                                               ("drift", "warble", 330), ("ping", "blip", 1320)])
def test_retro_voices_are_mellow(retro, arcade, freq):
    from cloudtone.sonify import Note
    mk = lambda inst: render(Note(freq, 1.0, 0.8, 0, inst, 15, "x"), 44100, np.random.default_rng(0))
    r, a = mk(retro), mk(arcade)
    assert np.isfinite(r).all() and 0.2 < np.abs(r).max() <= 0.8
    assert _centroid(r) < 0.4 * _centroid(a)          # far less bright than the arcade voice
