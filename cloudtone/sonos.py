"""Sonos output: the Pi serves a never-ending MP3 "radio station" over HTTP and
tells a Sonos speaker to play it.

Why a stream and not one clip per particle: Sonos takes ~1-2 s to start any new
URI and every play_uri() interrupts the last one, so per-event clips would be
laggy and choppy. A continuous stream costs a fixed buffer delay (typically
2-4 s on Sonos) but plays every note, overlapping, in time with each other.
"""
from __future__ import annotations

import logging
import queue
import socket
import threading

from .audio import Sink, to_int16

log = logging.getLogger(__name__)


class StreamHub(Sink):
    """Encodes mixer blocks to MP3 and fans them out to HTTP listeners
    (Sonos, or a phone's "Listen here"). Costs nothing while nobody listens."""

    def __init__(self, sr: int, bitrate_kbps=128):
        import lameenc

        self.enc = lameenc.Encoder()
        self.enc.set_bit_rate(bitrate_kbps)
        self.enc.set_in_sample_rate(sr)
        self.enc.set_channels(2)
        self.enc.set_quality(7)          # fast; fine for tones, easy on a Pi Zero
        self.clients: set[queue.Queue] = set()
        self.lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self.lock:
            self.clients.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            self.clients.discard(q)

    def write(self, block):
        if not self.clients:
            return
        data = bytes(self.enc.encode(to_int16(block)))
        if not data:
            return
        with self.lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(data)
                except queue.Full:          # slow listener: drop rather than stall the mixer
                    pass


def lan_ip(towards: str = "8.8.8.8") -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((towards, 80))       # no packets sent; just picks the outgoing interface
        return s.getsockname()[0]
    finally:
        s.close()


def find_speaker(name: str | None = None, timeout=5):
    import soco

    zones = soco.discover(timeout=timeout) or set()
    if not zones:
        raise RuntimeError("No Sonos speakers found. Is the Pi on the same network/VLAN?")
    if name:
        for z in zones:
            if z.player_name.lower() == name.lower():
                return z
        raise RuntimeError(f"Sonos room '{name}' not found. Seen: {sorted(z.player_name for z in zones)}")
    return sorted(zones, key=lambda z: z.player_name)[0]


def list_speakers(timeout=5) -> list[str]:
    import soco

    return sorted({z.player_name for z in (soco.discover(timeout=timeout) or [])})


def start_on_sonos(cfg, port: int):
    """Point a Sonos room (its group coordinator) at our stream."""
    spk = find_speaker(cfg.speaker)
    coord = spk.group.coordinator
    host = cfg.host_ip or lan_ip(coord.ip_address)
    url = f"http://{host}:{port}/stream.mp3"
    if cfg.volume is not None:
        coord.volume = int(cfg.volume)
    # force_radio makes Sonos treat it as live radio (no seeking, no end)
    coord.play_uri(url, title="CloudTone", force_radio=True)
    log.info("Sonos '%s' playing %s", coord.player_name, url)
    return coord
