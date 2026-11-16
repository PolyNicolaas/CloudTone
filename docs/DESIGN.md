# CloudTone design (v0.2)

A Raspberry Pi watches a cloud chamber, spots each new condensation trail, works
out what particle probably made it, and plays a note shaped by the trail. The
audio goes to the Pi's own output or to a Sonos system on the LAN.

## Hardware

| Part | Recommendation | Why |
|---|---|---|
| Computer | **Raspberry Pi Zero 2 W** | Smallest board that can do this: 4× Cortex-A53, 512 MB, Wi-Fi, CSI camera port. The original Zero W (single ARMv6 core) is too slow for OpenCV at a usable frame rate. |
| Camera | Camera Module 3 (or v2) | Picamera2 support, manual exposure/focus. Needs the Zero-size (22-pin) ribbon. |
| Lighting | Bright white LED strip, side-lit at plate height | Trails scatter light toward the camera; direct light washes them out. |
| Background | Matte black plate / felt | Contrast is everything for detection. |
| Audio | Sonos over Wi-Fi, **or** a USB / I²S DAC | The Zero 2 W has no headphone jack. |

The Peltier stack should be on its own supply; the Pi draws about 1 W.

## Pipeline

```
camera (Y plane, locked exposure)
  -> background model (running average, learned only off-track)
  -> threshold + open/close  -> foreground mask
  -> ONSET detection: pixels that are bright now but were dark for ~2 s
  -> accumulate onset pixels while the trail is still forming (1-6 frames)
  -> measure once -> classify -> note -> mixer -> outputs
```

**Why onset detection rather than blob tracking.** Trails appear almost
instantly and then only fade. As they fade they break into fragments, and
fragments of crossing trails merge. Tracking blobs frame to frame led to tracks
being counted twice or swallowed. Asking "which pixels just switched on?"
sidesteps that: a fading trail can't produce new pixels, and a fresh trail
crossing an old one still registers as new. Accuracy in the simulator:

* one event per track: 160 of 160 isolated tracks
* realistic rate (~1 track/s): 37 events for 38 tracks
* at 3–6 tracks/s, crossings add roughly 20–25% short extra events (a known limitation; see roadmap)

### Measured features

| Feature | How it's measured |
|---|---|
| length | extent along the principal axis (PCA) |
| path length | length of the binned centreline (follows curves) |
| width | area / length |
| wiggle | RMS deviation of the centreline from a straight line, divided by length |
| brightness | mean grey levels above background (a proxy for ionisation density) |
| angle, x/y | orientation from 0 to 180°, and the centroid |
| clipped | touches the frame edge, so the true length is unknown |

### Classification (rule-based, `classifier.py`)

| Particle | Signature | Sound |
|---|---|---|
| alpha | short (1–5 cm), fat, dense, straight | low bell |
| muon | long, thin, ruler-straight, often edge to edge | high glass tone, long |
| beta | thin, medium to long, wandering or kinked | pluck |
| electron | very short curly squiggle (Compton electrons, delta rays) | tick |

Each class gets a soft score; the winner's margin over second place becomes the
confidence. Simulator accuracy is alpha, muon and electron 100% and beta about
93% (straight betas that leave the frame read as muons, the same ambiguity a
human has).

**Why rules and not ML:** there's no labelled data yet, rules are easy to tune
on the real chamber, and they cost almost no CPU on a Zero 2. Every event is
logged with its features, so a small learned classifier can come later (see
roadmap).

## Sound mapping (`sonify.py`)

| Track property | Musical parameter |
|---|---|
| particle type | instrument and register |
| angle | pitch (scale degree; 0 to 180° spans two octaves of the chosen scale) |
| path length | note length |
| ionisation (brightness) | loudness |
| x position | stereo pan |
| wiggle | vibrato depth |

All of this is adjustable on the phone page's Tone tab (`mapping:` in config.yaml):

| Setting | Options | Default |
|---|---|---|
| Pitch follows | angle, trail length (log), height in chamber, left to right | angle |
| Range / reverse | 1–4 octaves; flip direction | 2, off |
| Register per particle | octave offset −1 to +4 | alpha 0, beta 1, muon 2, electron 3 |
| Note length | trail length or same for all; ×0.25–3 | trail length, ×1 |
| Loudness follows | density (brightness), trail length, even; contrast 0–100 | density, 70 |
| Stereo spread | 0–100 (pan from position across) | 100 |
| Wobble | vibrato from curviness, 0–200 | 100 |
| Which trails play | skip below a confidence; skip shorter than N mm; max notes per second | off, off, 8 |

Skipped trails still count and still light the page; they just don't sound.

**On "speed":** a camera can't measure particle speed, because the trail forms
in milliseconds. Ionisation density is the physical stand-in: slow, heavy
particles (alphas) ionise densely, while fast, minimum-ionising ones (muons)
leave faint lines. So the "speed" information arrives as loudness and timbre.

## Sonos

The Pi runs a never-ending MP3 stream (`http://<pi>:8765/stream.mp3`, made with
`lameenc`, encoded only while someone is listening) and SoCo tells the chosen room's group coordinator to play it as
radio. Notes are mixed into the stream continuously.

| Option | Latency | Overlap / timing | Verdict |
|---|---|---|---|
| **Continuous stream (chosen)** | 2–4 s fixed buffer | perfect: every note, in rhythm | best for an ambient installation |
| One clip per event via `play_uri` | 1–2 s each | every play interrupts the last one | choppy |
| Sonos cloud audioClip API | ~1 s | ducks music, needs cloud OAuth | possible later as a "notification" mode |
| Local DAC on the Pi | ~20 ms | perfect | best if you want sound in sync with the chamber |

You can run both at once (`outputs: [local, sonos]`).

## Phone control page

Served by the Pi on the same port as the stream (`web.py`, standard library only).

| Group | Controls |
|---|---|
| Tone | everything in the mapping table above, plus sound per particle and a "Hear it" test note |
| Play | volume, mute, pause, sensitivity, scale, key, timing (free or on the beat plus tempo), echo, Sonos on/off, room, speaker volume, Pi speaker on/off, "Listen here" |
| Setup | scale calibration (tap two ruler marks), crop (drag a box), background reset, exposure, gain, focus, detail preset (restart) |
| Shown | live annotated view, a scrolling score of the last 30 s of notes, counts per particle, tracks per minute, fps |
| config.yaml only | classifier limits, filter internals, mapping curves |

Every change applies immediately and is written to `config.yaml` atomically.
The preview JPEGs are only rendered while a page is watching.

## Performance

On the dev box the detector runs at about 360 fps at 640×480. A Zero 2 W is
roughly 10–15× slower, so expect 20–35 fps (an estimate; to be confirmed on
hardware). MP3 encoding at 128 kbps and the synthesis are light. To save
headroom, drop to 480×360 or set an ROI.

## Roadmap / candidate features

1. ~~Phone web UI~~ and ~~calibration helper~~: done in v0.2.
2. **Event log to CSV/SQLite**: real data, plus training data for a learned classifier.
3. **MIDI / OSC output**: drive real synths or a DAW instead of the built-in voices.
4. **Musical modes**: a drone whose density follows the event rate.
5. **Physics extras**: alpha direction (Bragg peak = brighter end); kink and fork detection (decays, δ-rays); a magnet under the plate makes curvature reveal charge sign.
6. **Crossing-track de-duplication** for busy chambers.
7. Chamber telemetry (plate temperature) overlaid and logged.
