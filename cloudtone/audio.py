"""Real-time mixer and audio outputs ("sinks").

The mixer always produces audio (silence when nothing is happening), so a
network stream to Sonos stays open and notes play the moment they're mixed in.
"""
from __future__ import annotations

import logging
import threading
import time
import wave

import numpy as np

log = logging.getLogger(__name__)


class Mixer:
    def __init__(self, sr: int, master_gain=0.6, max_voices=16, echo=0.3):
        self.sr = sr
        self.gain = master_gain
        self.samples_out = 0                    # audio clock, used for beat snapping
        self.max_voices = max_voices
        self._voices: list[list] = []          # [audio, position]
        self._lock = threading.Lock()
        # simple stereo echo for a bit of space (cheap enough for a Pi Zero)
        self._echo = echo
        self._dl = int(0.31 * sr)
        self._dbuf = np.zeros((self._dl, 2), np.float32)
        self._dpos = 0

    def play(self, audio: np.ndarray, delay_samples: int = 0) -> None:
        with self._lock:
            if len(self._voices) >= self.max_voices:
                self._voices.pop(0)            # steal the oldest voice
            self._voices.append([audio, -int(delay_samples)])   # negative = not started yet

    def next_block(self, n: int) -> np.ndarray:
        out = np.zeros((n, 2), np.float32)
        with self._lock:
            alive = []
            for v in self._voices:
                a, p = v
                if p + n <= 0:                 # still waiting for its beat
                    v[1] = p + n
                    alive.append(v)
                    continue
                o = max(0, -p)                 # offset into this block where it starts
                p = max(0, p)
                k = min(n - o, len(a) - p)
                out[o:o + k] += a[p:p + k]
                v[1] = p + k
                if v[1] < len(a):
                    alive.append(v)
            self._voices = alive
        self.samples_out += n
        if self._echo > 0 and n <= self._dl:
            idx = (self._dpos + np.arange(n)) % self._dl
            d = self._dbuf[idx]
            self._dbuf[idx] = out + 0.35 * d[:, ::-1]   # ping-pong feedback
            out += self._echo * d
            self._dpos = (self._dpos + n) % self._dl
        return np.tanh(out * self.gain).astype(np.float32)   # soft clip

    def set_echo(self, amount: float) -> None:
        self._echo = float(np.clip(amount, 0, 0.8))


def to_int16(block: np.ndarray) -> bytes:
    return (np.clip(block, -1, 1) * 32767).astype("<i2").tobytes()


class Sink:
    clocked = False          # True if write() blocks at the audio rate (sets the pace)
    enabled = True

    def write(self, block: np.ndarray) -> None: ...
    def close(self) -> None: ...


class LocalSink(Sink):
    """Pi's own audio out (3.5 mm / HDMI / USB / I2S DAC) via sounddevice."""
    clocked = True

    def __init__(self, sr: int, block: int, device=None):
        import sounddevice as sd

        self.stream = sd.OutputStream(samplerate=sr, channels=2, dtype="float32",
                                      blocksize=block, device=device)
        self.stream.start()

    def write(self, block):
        self.stream.write(block)

    def close(self):
        self.stream.stop()
        self.stream.close()


class WavSink(Sink):
    def __init__(self, path: str, sr: int):
        self.w = wave.open(path, "wb")
        self.w.setnchannels(2)
        self.w.setsampwidth(2)
        self.w.setframerate(sr)

    def write(self, block):
        self.w.writeframes(to_int16(block))

    def close(self):
        self.w.close()


class AudioEngine:
    """Pulls blocks from the mixer in real time and feeds every sink."""

    def __init__(self, mixer: Mixer, sinks: list[Sink], block: int):
        self.mixer, self.sinks, self.block = mixer, sinks, block
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="audio", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        for s in self.sinks:
            s.close()

    def _run(self):
        period = self.block / self.mixer.sr
        next_t = time.monotonic()
        while not self._stop.is_set():
            clocked = any(s.clocked and s.enabled for s in self.sinks)
            blk = self.mixer.next_block(self.block)
            for s in list(self.sinks):
                if not s.enabled:
                    continue
                try:
                    s.write(blk)
                except Exception:          # one bad sink shouldn't kill the rest
                    log.exception("audio sink %s failed", type(s).__name__)
            if clocked:
                next_t = time.monotonic()
            else:
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()
