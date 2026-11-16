"""The running instrument: camera loop, pipeline, audio outputs, and the live
settings the phone page reads and writes. Every change takes effect at once
and is saved to config.yaml, so the file stays the single source of truth."""
from __future__ import annotations

import collections
import logging
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

from .audio import AudioEngine, LocalSink, Mixer
from .camera import open_source
from .classifier import Classifier
from .config import Config, save_config
from .detector import TrackDetector, TrackEvent
from .sonify import INSTRUMENTS, SCALES, event_to_note, render, sample_event

log = logging.getLogger("cloudtone")

KINDS = ("alpha", "beta", "muon", "electron")
COLORS_BGR = {"alpha": (92, 179, 255), "beta": (255, 200, 92), "muon": (225, 245, 207),
              "electron": (216, 111, 255), "unknown": (200, 200, 200)}
PRESETS = {"light": (480, 360), "standard": (640, 480), "detailed": (800, 600)}

# key -> (type, lo, hi) or (type, choices); "restart" marks settings applied on restart
SETTINGS = {
    "audio.master_gain": (float, 0.0, 1.0),
    "audio.muted": (bool,),
    "audio.scale": (str, tuple(SCALES)),
    "audio.root_hz": (float, 40.0, 880.0),
    "audio.mode": (str, ("free", "beat")),
    "audio.bpm": (int, 40, 200),
    "audio.echo": (float, 0.0, 0.8),
    **{f"audio.instruments.{k}": (str, INSTRUMENTS + ("mute",)) for k in KINDS},
    "mapping.pitch_from": (str, ("angle", "length", "height", "across")),
    "mapping.pitch_octaves": (float, 1.0, 4.0),
    "mapping.pitch_reverse": (bool,),
    "mapping.length_from": (str, ("length", "fixed")),
    "mapping.length_scale": (float, 0.25, 3.0),
    "mapping.loudness_from": (str, ("brightness", "length", "even")),
    "mapping.dynamics": (float, 0.0, 1.0),
    "mapping.stereo": (float, 0.0, 1.0),
    "mapping.wobble": (float, 0.0, 2.0),
    "mapping.min_confidence": (float, 0.0, 0.9),
    "mapping.min_length_mm": (float, 0.0, 30.0),
    "mapping.max_notes_per_sec": (int, 1, 20),
    **{f"mapping.octaves.{k}": (int, -1, 4) for k in KINDS},
    "detect.threshold": (float, 4.0, 60.0),
    "detect.px_per_mm": (float, 0.5, 50.0),
    "camera.exposure_us": (int, 100, 200_000),
    "camera.analogue_gain": (float, 1.0, 16.0),
    "camera.lens_position": (float, 0.0, 15.0),
    "camera.preset": (str, tuple(PRESETS), "restart"),
    "sonos.speaker": (str, None),
    "sonos.volume": (int, 0, 100),
    "outputs.sonos": (bool,),
    "outputs.local": (bool,),
}


def annotate(gray, recent, now, bg=None, roi=None):
    ox, oy = (roi[0], roi[1]) if roi else (0, 0)
    g = cv2.convertScaleAbs(gray, alpha=1.6, beta=-10)
    if bg is not None:                      # background-subtracted: trails pop out
        h, w = bg.shape
        sub = gray[oy:oy + h, ox:ox + w].astype(np.float32)
        if sub.shape == bg.shape:
            g[oy:oy + h, ox:ox + w] = np.clip((sub - bg) * 4 + 18, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if roi:
        cv2.rectangle(img, (ox, oy), (ox + roi[2], oy + roi[3]), (120, 110, 90), 1)
    for e, born in recent:
        if now - born > 2.5:          # roughly how long a trail stays visible
            continue
        x, y, w, h = e.bbox
        c = COLORS_BGR.get(e.kind, COLORS_BGR["unknown"])
        cv2.rectangle(img, (x - 3, y - 3), (x + w + 3, y + h + 3), c, 1, cv2.LINE_AA)
        cv2.putText(img, e.kind, (x - 3, max(12, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return img


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.det = TrackDetector(cfg.detect)
        self.cls = Classifier(cfg.classify)
        a = cfg.audio
        self.mixer = Mixer(a.sample_rate, 0.0 if a.muted else a.master_gain, a.max_voices, a.echo)
        self.recent: collections.deque = collections.deque(maxlen=200)
        self.counts = collections.Counter()
        self.rng = np.random.default_rng()
        self.event_times: collections.deque = collections.deque(maxlen=2000)
        self.played: collections.deque = collections.deque()   # note times, for the limiter
        self.next_id = 1          # survives detector resets, so the page never replays old events

    def step(self, frame, t) -> list[TrackEvent]:
        a = self.cfg.audio
        evs = [self.cls.classify(e) for e in self.det.process(frame, t)]
        m = self.cfg.mapping
        now = t          # frame time, so offline renders limit the same way as live
        for e in evs:
            e.id, self.next_id = self.next_id, self.next_id + 1
            e.extra["note"] = None
            note = None
            while self.played and not (0 <= now - self.played[0] <= 1.0):
                self.played.popleft()
            if e.confidence < m.min_confidence:
                e.extra["skipped"] = "uncertain"
            elif e.path_mm < m.min_length_mm:
                e.extra["skipped"] = "short"
            elif len(self.played) >= m.max_notes_per_sec:
                e.extra["skipped"] = "busy"
            else:
                note = event_to_note(e, a.root_hz, a.scale, a.instruments, m)
            if note is not None:
                self.played.append(now)
                delay = 0
                if a.mode == "beat":            # snap the onset to the next eighth note
                    step = a.sample_rate * 60.0 / a.bpm / 2
                    delay = int((-self.mixer.samples_out) % step)
                self.mixer.play(render(note, a.sample_rate, self.rng), delay)
                e.extra.update(note=f"{note.instrument} {note.freq:.0f} Hz", freq=note.freq,
                               dur=note.dur, delay=delay / a.sample_rate)
            self.recent.append((e, t))
            self.counts[e.kind] += 1
            self.event_times.append(time.monotonic())
            log.info("%-8s conf %.2f  %5.1f mm  w %.2f mm  %3.0f deg  -> %s",
                     e.kind, e.confidence, e.path_mm, e.width_mm, e.angle_deg,
                     e.extra["note"] or e.extra.get("skipped", "silent"))
        return evs


def _get(cfg, key):
    obj = cfg
    for part in key.split("."):
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    return obj


def _set(cfg, key, value):
    *path, last = key.split(".")
    obj = cfg
    for part in path:
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


class App:
    def __init__(self, cfg: Config, cfg_path: str | None):
        self.cfg, self.cfg_path = cfg, cfg_path
        self.pipe = Pipeline(cfg)
        self.paused = False
        self.restart_needed = False
        self.fps = 0.0
        self.frame_size = (cfg.camera.width, cfg.camera.height)
        self.preview_clients = 0
        self.jpeg: bytes | None = None
        self.jpeg_cond = threading.Condition()
        self.source = None
        self.out_state = {"sonos": "off", "local": "off"}
        self.coord = None
        self.local: LocalSink | None = None
        self._stop = threading.Event()
        self.hub = None
        try:
            from .sonos import StreamHub
            self.hub = StreamHub(cfg.audio.sample_rate, cfg.sonos.bitrate_kbps)
        except ImportError:
            log.warning("lameenc not installed: no Sonos / Listen here stream")
        self.engine = AudioEngine(self.pipe.mixer, [s for s in [self.hub] if s], cfg.audio.block_size)

    # ---------- settings ----------
    def settings(self) -> dict:
        c = self.cfg
        out = {k: _get(c, k) for k in SETTINGS if not k.startswith(("outputs.", "camera.preset"))}
        out["camera.preset"] = next((n for n, wh in PRESETS.items()
                                     if wh == (c.camera.width, c.camera.height)), "custom")
        out["outputs.sonos"] = "sonos" in c.audio.outputs
        out["outputs.local"] = "local" in c.audio.outputs
        out["detect.roi"] = c.detect.roi
        return out

    def update(self, changes: dict) -> dict:
        errors, applied = {}, []
        for key, raw in changes.items():
            spec = SETTINGS.get(key)
            if spec is None:
                errors[key] = "unknown setting"
                continue
            typ = spec[0]
            try:
                val = (raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "on")) \
                    if typ is bool else typ(raw)
            except (TypeError, ValueError):
                errors[key] = f"expected {typ.__name__}"
                continue
            if typ in (int, float) and len(spec) >= 3:
                val = typ(min(max(val, spec[1]), spec[2]))
            elif typ is str and len(spec) >= 2 and spec[1] is not None and val not in spec[1]:
                errors[key] = f"choose one of {', '.join(spec[1])}"
                continue
            self._apply(key, val)
            applied.append(key)
        if applied:
            self.save()
        return {"applied": applied, "errors": errors, "restart_needed": self.restart_needed}

    def _apply(self, key, val):
        c, mixer = self.cfg, self.pipe.mixer
        if key == "outputs.sonos" or key == "outputs.local":
            name = key.split(".")[1]
            outs = [o for o in c.audio.outputs if o != name] + ([name] if val else [])
            c.audio.outputs = outs
            self._set_output(name, val)
            return
        if key == "camera.preset":
            c.camera.width, c.camera.height = PRESETS[val]
            self.restart_needed = (c.camera.width, c.camera.height) != self.frame_size
            return
        _set(c, key, val)
        if key in ("audio.master_gain", "audio.muted"):
            mixer.gain = 0.0 if c.audio.muted else c.audio.master_gain
        elif key == "audio.echo":
            mixer.set_echo(val)
        elif key.startswith("camera.") and self.source is not None and self.source.live_controls:
            self.source.apply(c.camera)
        elif key == "sonos.volume" and self.coord is not None:
            threading.Thread(target=self._sonos_volume, args=(val,), daemon=True).start()
        elif key == "sonos.speaker" and "sonos" in c.audio.outputs:
            self._set_output("sonos", True)

    def save(self):
        if self.cfg_path:
            try:
                save_config(self.cfg, self.cfg_path)
            except OSError:
                log.exception("could not save %s", self.cfg_path)

    # ---------- actions ----------
    def action(self, name: str, args: dict) -> dict:
        W, H = self.frame_size
        if name == "pause":
            self.paused = True
        elif name == "resume":
            self.paused = False
            self.reset_background()
        elif name == "reset_background":
            self.reset_background()
        elif name == "calibrate_scale":
            (x1, y1), (x2, y2), mm = args["p1"], args["p2"], float(args["mm"])
            px = math.hypot((x2 - x1) * W, (y2 - y1) * H)
            if mm <= 0 or px < 10:
                return {"error": "Pick two points further apart and enter a distance above 0 mm."}
            self.cfg.detect.px_per_mm = round(px / mm, 3)
            self.save()
            return {"px_per_mm": self.cfg.detect.px_per_mm}
        elif name == "set_roi":
            x, y, w, h = (float(args[k]) for k in "xywh")
            roi = [int(x * W), int(y * H), int(w * W), int(h * H)]
            if roi[2] < 40 or roi[3] < 40:
                return {"error": "Drag a larger box: the crop must be at least 40 px each way."}
            self.cfg.detect.roi = roi
            self.reset_background()
            self.save()
        elif name == "clear_roi":
            self.cfg.detect.roi = None
            self.reset_background()
            self.save()
        elif name == "hear":
            kind = args.get("kind")
            if kind not in KINDS:
                return {"error": "Pick alpha, beta, muon or electron."}
            a = self.cfg.audio
            note = event_to_note(sample_event(kind), a.root_hz, a.scale, a.instruments, self.cfg.mapping)
            if note is None:
                return {"error": f"{kind.capitalize()} is set to silent."}
            self.pipe.mixer.play(render(note, a.sample_rate))
            return {"note": f"{note.instrument} {note.freq:.0f} Hz", "dur": round(note.dur, 2)}
        elif name == "sonos_rooms":
            try:
                from .sonos import list_speakers
                return {"rooms": list_speakers(timeout=4)}
            except Exception as exc:
                return {"rooms": [], "error": f"Sonos search failed: {exc}"}
        elif name == "restart":
            self.save()
            threading.Timer(0.5, lambda: os.execv(sys.executable, [sys.executable, "-m", "cloudtone"] + sys.argv[1:])).start()
        else:
            return {"error": f"unknown action {name}"}
        return {"ok": True}

    def reset_background(self):
        self.pipe.det = TrackDetector(self.cfg.detect)

    # ---------- outputs ----------
    def _set_output(self, name, on):
        if name == "local":
            if on:
                try:
                    if self.local is None:
                        self.local = LocalSink(self.cfg.audio.sample_rate, self.cfg.audio.block_size)
                        self.engine.sinks.append(self.local)
                    self.local.enabled = True
                    self.out_state["local"] = "playing"
                except Exception as exc:
                    self.out_state["local"] = f"error: no audio device ({exc.__class__.__name__})"
            else:
                if self.local:
                    self.local.enabled = False
                self.out_state["local"] = "off"
        else:
            if on:
                self.out_state["sonos"] = "connecting"
                threading.Thread(target=self._sonos_start, daemon=True).start()
            else:
                coord, self.coord = self.coord, None
                self.out_state["sonos"] = "off"
                if coord:
                    threading.Thread(target=coord.stop, daemon=True).start()

    def _sonos_start(self):
        try:
            from .sonos import start_on_sonos
            if self.hub is None:
                raise RuntimeError("lameenc not installed")
            self.coord = start_on_sonos(self.cfg.sonos, self.cfg.web.port)
            self.out_state["sonos"] = f"playing on {self.coord.player_name}"
        except Exception as exc:
            self.coord = None
            self.out_state["sonos"] = f"error: {exc}"
            log.warning("Sonos: %s", exc)

    def _sonos_volume(self, v):
        try:
            self.coord.volume = int(v)
        except Exception as exc:
            log.warning("Sonos volume: %s", exc)

    # ---------- status ----------
    def state(self, since: int = 0) -> dict:
        now = time.monotonic()
        rate = sum(1 for t in self.pipe.event_times if now - t <= 60)
        evs = [{"id": e.id, "kind": e.kind, "conf": e.confidence, "path_mm": round(e.path_mm, 1),
                "width_mm": round(e.width_mm, 2), "angle": round(e.angle_deg),
                "note": e.extra.get("note"), "freq": e.extra.get("freq"),
                "dur": e.extra.get("dur"), "x": round(e.x, 3), "skipped": e.extra.get("skipped"),
                "delay": e.extra.get("delay", 0)}
               for e, _ in list(self.pipe.recent) if e.id > since][-30:]
        return {
            "settings": self.settings(),
            "status": {
                "paused": self.paused, "fps": round(self.fps, 1), "per_min": rate,
                "counts": {k: self.pipe.counts.get(k, 0) for k in KINDS},
                "outputs": self.out_state, "source": self.cfg.camera.source,
                "live_camera": bool(self.source and self.source.live_controls),
                "frame": list(self.frame_size), "restart_needed": self.restart_needed,
            },
            "events": evs,
            "last_id": self.pipe.recent[-1][0].id if self.pipe.recent else since,
        }

    # ---------- main loop ----------
    def run(self, frames=None):
        c = self.cfg
        self.engine.start()
        for out in list(c.audio.outputs):
            self._set_output(out, True)
        self.source = frames or open_source(c.camera, c.detect.px_per_mm, realtime=True)
        n, t_fps, t_prev = 0, time.monotonic(), 0.0
        try:
            for frame, t in self.source:
                if self._stop.is_set():
                    break
                self.frame_size = (frame.shape[1], frame.shape[0])
                if not self.paused:
                    self.pipe.step(frame, t)
                n += 1
                now = time.monotonic()
                if now - t_fps >= 2:
                    self.fps, n, t_fps = n / (now - t_fps), 0, now
                if self.preview_clients and now - t_prev >= 1 / c.web.preview_fps:
                    t_prev = now
                    img = annotate(frame, list(self.pipe.recent)[-20:], t, self.pipe.det.bg, c.detect.roi)
                    pw = c.web.preview_width
                    img = cv2.resize(img, (pw, int(img.shape[0] * pw / img.shape[1])), interpolation=cv2.INTER_AREA)
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if ok:
                        with self.jpeg_cond:
                            self.jpeg = jpg.tobytes()
                            self.jpeg_cond.notify_all()
        finally:
            self.engine.stop()

    def stop(self):
        self._stop.set()
