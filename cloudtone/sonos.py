"""Sonos output: the Pi serves a never-ending MP3 "radio station" over HTTP and
tells a Sonos speaker to play it.

Why a stream and not one clip per particle: Sonos takes ~1-2 s to start any new
URI and every play_uri() interrupts the last one, so per-event clips would be
laggy and choppy. A continuous stream costs a fixed buffer delay (typically
2-4 s on Sonos) but plays every note, overlapping, in time with each other.
"""
from __future__ import annotations

import json
import logging
import queue
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .audio import Sink, to_int16

log = logging.getLogger(__name__)


class StreamHub(Sink):
    """Encodes mixer blocks to MP3 and fans them out to HTTP listeners."""

    def __init__(self, sr: int, bitrate_kbps=128, port=8765, status_fn=None):
        import lameenc

        self.enc = lameenc.Encoder()
        self.enc.set_bit_rate(bitrate_kbps)
        self.enc.set_in_sample_rate(sr)
        self.enc.set_channels(2)
        self.enc.set_quality(7)          # fast; fine for tones, easy on a Pi Zero
        self.clients: set[queue.Queue] = set()
        self.lock = threading.Lock()
        self.port = port
        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"   # Sonos is happiest with a plain close-delimited body

            def log_message(self, fmt, *args):
                log.debug("http: " + fmt, *args)

            def do_GET(self):
                if self.path.startswith("/status") and status_fn:
                    body = json.dumps(status_fn(), indent=1).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if not self.path.startswith("/stream.mp3"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("icy-name", "CloudTone")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q: queue.Queue = queue.Queue(maxsize=256)
                with hub.lock:
                    hub.clients.add(q)
                log.info("stream listener connected: %s", self.client_address[0])
                try:
                    while True:
                        self.wfile.write(q.get())
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with hub.lock:
                        hub.clients.discard(q)
                    log.info("stream listener left: %s", self.client_address[0])

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, name="http", daemon=True).start()

    def write(self, block):
        data = bytes(self.enc.encode(to_int16(block)))
        if not data:
            return
        with self.lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(data)
                except queue.Full:          # slow listener: drop rather than stall the mixer
                    pass

    def close(self):
        self.server.shutdown()


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

    return sorted(f"{z.player_name}  ({z.ip_address})" for z in (soco.discover(timeout=timeout) or []))


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
