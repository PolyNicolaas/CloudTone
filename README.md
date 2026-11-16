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

## Phone control page

Open `http://<pi-address>:8765/` on your phone (same Wi-Fi).

* **Play**: volume, sensitivity, scale and key, timing (as they arrive, or on the beat with a tempo),
  echo, an instrument per particle (or silent), Sonos room and speaker volume, Pi speaker.
* **Tone**: how a trace becomes a note. Pick what sets pitch (angle, length, height or
  left-to-right position), its range and direction; note length and loudness sources;
  stereo spread and wobble; sound and octave per particle with a "Hear it" button; and
  filters to skip uncertain or short trails or cap notes per second.
* **Setup**: measure the scale with a ruler, drag a crop around the chamber floor, reset the
  background, camera exposure/gain/focus, detail level.
* **Recreation**: switch the view between Camera, Recreation and Both. Each detected trail is
  redrawn from its real shape and glows in its particle's colour as its note plays: it streaks
  on with the attack, pulses with the note's envelope, wobbles if curvy, then fades like vapour.
  Timing matches whichever speaker you hear (Sonos, Listen here, or the Pi), set on the Tone tab.
  The default look is a **1970s green vector monitor**: phosphor beams with persistence and bloom,
  alphas as bundled beams, muons as razor lines, betas segmented with lit corners, and an
  Asteroids-style stroke font for labels and the score. "Colour glow" is the alternative.
* **Show mode**: tap the full-screen button, or open `http://<pi-address>:8765/show` on a TV
  or tablet next to the chamber.
* **Listen here** plays the same stream Sonos gets, on the phone itself.

Every change applies at once and is saved to `config.yaml`.

## Commands

| Command | What it does |
|---|---|
| `cloudtone run [--source picamera\|sim\|file\|0] [--output local sonos]` | live, with the phone page |
| `cloudtone demo [--seconds N] [--rate R] [--out NAME] [--set KEY=VALUE ...]` | offline render of a simulated chamber, optionally with different settings (e.g. `--set mapping.pitch_from=length audio.mode=beat`) |
| `cloudtone sonos-list` | list Sonos rooms |


## Setup tips

* **Light from the side**, at plate height, against a matte black plate.
* **Lock exposure** (the default): auto-exposure flicker looks like tracks.
* **Calibrate `px_per_mm`**: lay a ruler on the plate. All classifier limits are in mm.
* **Tune with recordings**: record a clip (`rpicam-vid -t 60000 -o test.h264`),
  then `cloudtone run --source test.h264` on any computer and open `http://localhost:8765/`.

## Layout

```
cloudtone/
  camera.py      Picamera2 / OpenCV / simulator frame sources
  detector.py    background model, onset detection, track features
  classifier.py  rule-based particle ID
  sonify.py      track -> note mapping and synthesis
  audio.py       mixer, local and WAV outputs
  sonos.py       MP3 stream encoder and Sonos control (SoCo)
  app.py         the running instrument: live settings, outputs, camera loop
  web.py         phone page server, preview, settings API, audio stream
  web/index.html the phone page
  simulator.py   synthetic chamber for development and tests
  main.py        CLI
```
