"""CloudTone command line.

  cloudtone run   [-c config.yaml] [--source sim|picamera|FILE|0] [--output local sonos]
                  phone page at http://<pi>:8765/
  cloudtone demo  [--seconds 30] [--out demo]       offline: simulated chamber -> demo.wav + demo.mp4
  cloudtone sonos-list                              show Sonos rooms on the network
"""
from __future__ import annotations

import argparse
import logging
import os
import signal

import cv2

from .app import App, Pipeline, annotate
from .audio import WavSink
from .camera import open_source
from .config import load_config

log = logging.getLogger("cloudtone")


def cmd_run(args, cfg, cfg_path):
    if args.source:
        cfg.camera.source = args.source
    if args.output is not None:
        cfg.audio.outputs = args.output
    app = App(cfg, cfg_path)
    if cfg.web.enabled:
        from .web import serve
        serve(app, cfg.web.port)
    signal.signal(signal.SIGINT, lambda *_: app.stop())
    signal.signal(signal.SIGTERM, lambda *_: app.stop())
    app.run()


def cmd_demo(args, cfg, _):
    """Deterministic offline render: no camera, no sound card, no real-time."""
    cfg.camera.source = "sim"
    if args.set:
        from .app import App
        res = App(cfg, None).update(dict(kv.split("=", 1) for kv in args.set))
        for k, err in res["errors"].items():
            raise SystemExit(f"--set {k}: {err}")
    pipe = Pipeline(cfg)
    sr, fps = cfg.audio.sample_rate, cfg.camera.fps
    wav = WavSink(f"{args.out}.wav", sr)
    vid = cv2.VideoWriter(f"{args.out}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (cfg.camera.width, cfg.camera.height))
    frames = open_source(cfg.camera, cfg.detect.px_per_mm, realtime=False, seed=args.seed, rate_hz=args.rate)
    spf, acc = sr / fps, 0.0
    for frame, t in frames:
        if t >= args.seconds:
            break
        pipe.step(frame, t)
        vid.write(annotate(frame, list(pipe.recent), t, pipe.det.bg, cfg.detect.roi))
        acc += spf
        k = int(acc)
        acc -= k
        wav.write(pipe.mixer.next_block(k))
    wav.close()
    vid.release()
    print(f"wrote {args.out}.wav and {args.out}.mp4 — detected {dict(pipe.counts)}")


def cmd_sonos_list(args, cfg, _):
    from .sonos import list_speakers
    for s in list_speakers():
        print(s)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cloudtone", description="Cloud chamber -> music")
    ap.add_argument("-c", "--config", default="config.yaml",
                    help="YAML config file (created on first save if missing)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="live: camera -> tones, with the phone page")
    r.add_argument("--source", help="picamera | sim | video file | webcam index")
    r.add_argument("--output", nargs="*", choices=["local", "sonos"], help="audio outputs at start")
    d = sub.add_parser("demo", help="offline render of a simulated chamber")
    d.add_argument("--seconds", type=float, default=30)
    d.add_argument("--rate", type=float, default=1.5, help="simulated tracks per second")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--out", default="demo")
    d.add_argument("--set", nargs="*", metavar="KEY=VALUE",
                   help="try settings, e.g. mapping.pitch_from=length audio.mode=beat")
    sub.add_parser("sonos-list", help="list Sonos rooms")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    {"run": cmd_run, "demo": cmd_demo, "sonos-list": cmd_sonos_list}[args.cmd](args, cfg, args.config)


if __name__ == "__main__":
    main()
