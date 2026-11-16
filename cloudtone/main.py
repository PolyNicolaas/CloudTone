"""CloudTone command line.

  cloudtone run   [-c config.yaml] [--source sim|picamera|FILE|0] [--output local sonos]
  cloudtone demo  [--seconds 30] [--out demo]       offline: simulated chamber -> demo.wav + demo.mp4
  cloudtone sonos-list                              show Sonos rooms on the network
"""
from __future__ import annotations

import argparse
import collections
import logging
import signal
import threading
import time

import cv2
import numpy as np

from .audio import AudioEngine, Mixer, WavSink
from .camera import open_source
from .classifier import Classifier
from .config import load_config
from .detector import TrackDetector, TrackEvent
from .sonify import event_to_note, render

log = logging.getLogger("cloudtone")

COLORS = {"alpha": (60, 60, 255), "beta": (255, 200, 60), "muon": (80, 255, 80),
          "electron": (255, 80, 255), "unknown": (200, 200, 200)}


def annotate(gray: np.ndarray, recent: list[tuple[TrackEvent, float]], now: float,
             bg: np.ndarray | None = None) -> np.ndarray:
    if bg is not None and bg.shape == gray.shape:   # background-subtracted: trails pop out
        g = np.clip((gray.astype(np.float32) - bg) * 4 + 15, 0, 255).astype(np.uint8)
    else:
        g = cv2.convertScaleAbs(gray, alpha=2.0, beta=-20)
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for e, born in recent:
        if now - born > 1.5:
            continue
        x, y, w, h = e.bbox
        c = COLORS.get(e.kind, COLORS["unknown"])
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 1)
        cv2.putText(img, f"{e.kind} {e.path_mm:.0f}mm", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return img


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.det = TrackDetector(cfg.detect)
        self.cls = Classifier(cfg.classify)
        self.mixer = Mixer(cfg.audio.sample_rate, cfg.audio.master_gain, cfg.audio.max_voices)
        self.recent: collections.deque = collections.deque(maxlen=50)
        self.counts = collections.Counter()
        self.rng = np.random.default_rng()

    def step(self, frame, t) -> list[TrackEvent]:
        evs = [self.cls.classify(e) for e in self.det.process(frame, t)]
        for e in evs:
            note = event_to_note(e, self.cfg.audio.root_hz, self.cfg.audio.scale)
            self.mixer.play(render(note, self.cfg.audio.sample_rate, self.rng))
            self.recent.append((e, t))
            self.counts[e.kind] += 1
            log.info("%-8s conf %.2f  %5.1f mm  w %.2f mm  %3.0f deg  -> %s %.0f Hz %.2fs",
                     e.kind, e.confidence, e.path_mm, e.width_mm, e.angle_deg,
                     note.instrument, note.freq, note.dur)
        return evs

    def status(self):
        return {"counts": dict(self.counts),
                "recent": [{"kind": e.kind, "conf": e.confidence, "path_mm": round(e.path_mm, 1),
                            "width_mm": round(e.width_mm, 2), "angle": round(e.angle_deg),
                            "t": round(t, 2)} for e, t in list(self.recent)[-20:]]}


def cmd_run(args, cfg):
    if args.source:
        cfg.camera.source = args.source
    if args.output:
        cfg.audio.outputs = args.output
    pipe = Pipeline(cfg)
    sinks = []
    if "local" in cfg.audio.outputs:
        from .audio import LocalSink
        sinks.append(LocalSink(cfg.audio.sample_rate, cfg.audio.block_size))
    if "sonos" in cfg.audio.outputs:
        from .sonos import StreamHub, start_on_sonos
        sinks.append(StreamHub(cfg.audio.sample_rate, cfg.sonos.bitrate_kbps, cfg.sonos.port, pipe.status))
        start_on_sonos(cfg.sonos, cfg.sonos.port)
    engine = AudioEngine(pipe.mixer, sinks, cfg.audio.block_size)
    engine.start()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    frames = open_source(cfg.camera, cfg.detect.px_per_mm, realtime=True)
    n, t_last = 0, time.monotonic()
    try:
        for frame, t in frames:
            if stop.is_set():
                break
            pipe.step(frame, t)
            if args.preview:
                cv2.imshow("CloudTone", annotate(frame, list(pipe.recent), t, pipe.det.bg))
                if cv2.waitKey(1) == 27:
                    break
            n += 1
            if time.monotonic() - t_last > 30:
                log.info("%.1f fps, totals %s", n / (time.monotonic() - t_last), dict(pipe.counts))
                n, t_last = 0, time.monotonic()
    finally:
        engine.stop()


def cmd_demo(args, cfg):
    """Deterministic offline render: no camera, no sound card, no real-time."""
    cfg.camera.source = "sim"
    pipe = Pipeline(cfg)
    sr, fps = cfg.audio.sample_rate, cfg.camera.fps
    wav = WavSink(f"{args.out}.wav", sr)
    vid = cv2.VideoWriter(f"{args.out}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (cfg.camera.width, cfg.camera.height))
    frames = open_source(cfg.camera, cfg.detect.px_per_mm, realtime=False, seed=args.seed, rate_hz=args.rate)
    spf = sr / fps
    acc = 0.0
    for i, (frame, t) in enumerate(frames):
        if t >= args.seconds:
            break
        pipe.step(frame, t)
        vid.write(annotate(frame, list(pipe.recent), t, pipe.det.bg))
        acc += spf
        k = int(acc)
        acc -= k
        wav.write(pipe.mixer.next_block(k))
    wav.close()
    vid.release()
    print(f"wrote {args.out}.wav and {args.out}.mp4 — detected {dict(pipe.counts)}")


def cmd_sonos_list(args, cfg):
    from .sonos import list_speakers
    for s in list_speakers():
        print(s)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cloudtone", description="Cloud chamber -> music")
    ap.add_argument("-c", "--config", help="YAML config file")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="live: camera -> tones")
    r.add_argument("--source", help="picamera | sim | video file | webcam index")
    r.add_argument("--output", nargs="+", choices=["local", "sonos"], help="audio outputs")
    r.add_argument("--preview", action="store_true", help="show annotated video window")
    d = sub.add_parser("demo", help="offline render of a simulated chamber")
    d.add_argument("--seconds", type=float, default=30)
    d.add_argument("--rate", type=float, default=1.5, help="simulated tracks per second")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--out", default="demo")
    sub.add_parser("sonos-list", help="list Sonos rooms")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    {"run": cmd_run, "demo": cmd_demo, "sonos-list": cmd_sonos_list}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
