# CloudTone

Turn a cloud chamber into an instrument. A Raspberry Pi camera watches the
chamber; every new particle trail is detected, identified (alpha, beta, muon,
electron), and played as a note, on the Pi's audio out or on your Sonos.

See [docs/DESIGN.md](docs/DESIGN.md) for how it works and why.

## Try it without hardware

```bash
pip install -e ".[dev]"
python -m cloudtone demo --seconds 30 --out demo   # writes demo.wav + demo.mp4 from a simulated chamber
pytest
```

## On the Pi (Zero 2 W, Raspberry Pi OS Bookworm 64-bit)

```bash
git clone https://github.com/PolyNicolaas/CloudTone.git cloudtone && cd cloudtone
./deploy/install.sh
nano config.yaml                                  # Sonos room, px_per_mm
.venv/bin/cloudtone sonos-list
.venv/bin/cloudtone -c config.yaml run --source sim   # simulated chamber -> Sonos
.venv/bin/cloudtone -c config.yaml run               # the real thing
```

Run at boot: `sudo cp deploy/cloudtone.service /etc/systemd/system/ && sudo systemctl enable --now cloudtone`

## Commands

| Command | What it does |
|---|---|
| `cloudtone run [--source picamera\|sim\|file\|0] [--output local sonos] [--preview]` | live |
| `cloudtone demo [--seconds N] [--rate R] [--out NAME]` | offline render of a simulated chamber |
| `cloudtone sonos-list` | list Sonos rooms |

While running with Sonos output, `http://<pi>:8765/status` shows recent events.

## Setup tips

* **Light from the side**, at plate height, against a matte black plate.
* **Lock exposure** (the default): auto-exposure flicker looks like tracks.
* **Calibrate `px_per_mm`**: lay a ruler on the plate. All classifier limits are in mm.
* **Tune with recordings**: record a clip (`rpicam-vid -t 60000 -o test.h264`),
  then `cloudtone run --source test.h264 --output local --preview` on any computer.

## Layout

```
cloudtone/
  camera.py      Picamera2 / OpenCV / simulator frame sources
  detector.py    background model, onset detection, track features
  classifier.py  rule-based particle ID
  sonify.py      track -> note mapping and synthesis
  audio.py       mixer, local and WAV outputs
  sonos.py       MP3 stream server and Sonos control (SoCo)
  simulator.py   synthetic chamber for development and tests
  main.py        CLI
```
