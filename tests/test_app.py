import json
import threading
import time
import urllib.request

import numpy as np
import yaml

from cloudtone.app import App, Pipeline
from cloudtone.audio import Mixer
from cloudtone.camera import SimSource
from cloudtone.config import Config, load_config


def make_app(tmp_path):
    cfg = Config()
    cfg.camera.source = "sim"
    cfg.audio.outputs = []
    return App(cfg, str(tmp_path / "config.yaml"))


def test_settings_apply_live_and_persist(tmp_path):
    app = make_app(tmp_path)
    res = app.update({"audio.bpm": 120, "audio.muted": True, "detect.threshold": 999,
                      "audio.instruments.electron": "mute", "audio.scale": "nope", "bogus": 1})
    assert set(res["applied"]) == {"audio.bpm", "audio.muted", "detect.threshold", "audio.instruments.electron"}
    assert "audio.scale" in res["errors"] and "bogus" in res["errors"]
    assert app.cfg.detect.threshold == 60.0          # clamped to range
    assert app.pipe.mixer.gain == 0.0                # mute takes effect at once
    saved = load_config(tmp_path / "config.yaml")
    assert saved.audio.bpm == 120 and saved.audio.instruments["electron"] == "mute"


def test_preset_needs_restart(tmp_path):
    app = make_app(tmp_path)
    assert app.update({"camera.preset": "light"})["restart_needed"] is True
    assert (app.cfg.camera.width, app.cfg.camera.height) == (480, 360)
    assert app.update({"camera.preset": "standard"})["restart_needed"] is False


def test_calibration_and_crop(tmp_path):
    app = make_app(tmp_path)
    app.frame_size = (640, 480)
    r = app.action("calibrate_scale", {"p1": [0.1, 0.5], "p2": [0.6, 0.5], "mm": 50})
    assert abs(r["px_per_mm"] - 6.4) < 1e-6
    assert "error" in app.action("calibrate_scale", {"p1": [0.1, 0.5], "p2": [0.1, 0.5], "mm": 50})
    app.action("set_roi", {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5})
    assert app.cfg.detect.roi == [64, 48, 320, 240]
    app.action("clear_roi", {})
    assert app.cfg.detect.roi is None


def test_event_ids_survive_background_reset(tmp_path):
    app = make_app(tmp_path)
    src = SimSource(app.cfg.camera, app.cfg.detect.px_per_mm, realtime=False, seed=2, rate_hz=3)
    ids = []
    for i, (frame, t) in enumerate(src):
        if i == 150:
            app.reset_background()
        ids += [e.id for e in app.pipe.step(frame, t)]
        if i > 300:
            break
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_muted_particle_makes_no_sound():
    cfg = Config()
    cfg.audio.instruments = {k: "mute" for k in ("alpha", "beta", "muon", "electron")}
    pipe = Pipeline(cfg)
    src = SimSource(cfg.camera, cfg.detect.px_per_mm, realtime=False, seed=1, rate_hz=3)
    n = 0
    for i, (frame, t) in enumerate(src):
        n += len(pipe.step(frame, t))
        if i > 200:
            break
    assert n > 0 and pipe.mixer._voices == []


def test_beat_mode_delays_note_onset():
    m = Mixer(1000, echo=0)
    m.play(np.ones((50, 2), np.float32), delay_samples=300)
    blocks = np.concatenate([m.next_block(100) for _ in range(5)])
    nz = np.nonzero(blocks[:, 0])[0]
    assert nz[0] == 300 and len(nz) == 50


def test_web_endpoints(tmp_path):
    from cloudtone.web import serve
    app = make_app(tmp_path)
    srv = serve(app, 0)
    port = srv.server_address[1]
    threading.Thread(target=app.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        assert b"CloudTone" in urllib.request.urlopen(base + "/").read()
        time.sleep(1.0)
        st = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert "settings" in st and st["status"]["source"] == "sim"
        req = urllib.request.Request(base + "/api/settings", data=json.dumps({"audio.echo": 0.5}).encode(),
                                     headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read())["applied"] == ["audio.echo"]
        with urllib.request.urlopen(base + "/stream.mp3", timeout=5) as r:
            head = r.read(2)
            assert head[0] == 0xFF and head[1] & 0xE0 == 0xE0     # MP3 frame sync: audio is flowing
        with urllib.request.urlopen(base + "/preview.mjpg", timeout=5) as r:
            assert b"--frame" in r.read(2000)
        assert yaml.safe_load(open(tmp_path / "config.yaml"))["audio"]["echo"] == 0.5
    finally:
        app.stop()
        srv.shutdown()


def test_mapping_settings_change_the_note():
    from cloudtone.config import MappingConfig
    from cloudtone.sonify import event_to_note, sample_event
    e = sample_event("muon")
    base = event_to_note(e)
    m = MappingConfig(pitch_from="across", length_from="fixed", length_scale=2.0,
                      loudness_from="even", stereo=0.0, wobble=0.0)
    m.octaves["muon"] = 0
    n = event_to_note(e, m=m)
    assert n.freq < base.freq                  # two octaves lower register
    assert abs(n.dur - 2 * 2.6) < 1e-6         # fixed typical length, doubled
    assert n.pan == 0 and n.vibrato_cents == 0
    rev = event_to_note(e, m=MappingConfig(pitch_reverse=True))
    assert rev.freq != base.freq


def test_filters_and_limiter(tmp_path):
    app = make_app(tmp_path)
    app.update({"mapping.max_notes_per_sec": 1, "mapping.min_length_mm": 10})
    src = SimSource(app.cfg.camera, app.cfg.detect.px_per_mm, realtime=False, seed=4, rate_hz=8)
    evs = []
    for i, (frame, t) in enumerate(src):
        evs += app.pipe.step(frame, t)
        if i > 100:
            break
    skipped = {e.extra.get("skipped") for e in evs}
    assert "short" in skipped and "busy" in skipped
    assert all(e.path_mm >= 10 for e in evs if e.extra.get("note"))


def test_hear_action(tmp_path):
    app = make_app(tmp_path)
    assert "note" in app.action("hear", {"kind": "alpha"})
    app.update({"audio.instruments.alpha": "mute"})
    assert "error" in app.action("hear", {"kind": "alpha"})
