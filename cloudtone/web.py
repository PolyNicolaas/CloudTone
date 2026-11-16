"""Tiny HTTP server for the phone page. Standard library only, so it costs
nothing extra on a Pi Zero.

  GET  /                 the control page
  GET  /show             full-screen recreation, for a TV or tablet by the chamber
  GET  /api/state?since= settings, status and new events (polled by the page)
  POST /api/settings     {"audio.bpm": 110, ...}  -> applied live and saved
  POST /api/action       {"action": "reset_background" | "pause" | ...}
  GET  /preview.mjpg     annotated live view (only rendered while someone watches)
  GET  /stream.mp3       the audio stream Sonos plays ("Listen here" on the page too)
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)


def serve(app, port: int) -> ThreadingHTTPServer:
    page = resources.files("cloudtone").joinpath("web/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"     # close-delimited streams; Sonos likes this

        def log_message(self, fmt, *args):
            log.debug("http %s " + fmt, self.client_address[0], *args)

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html", "/show"):
                self._send(200, page, "text/html; charset=utf-8")
            elif url.path == "/api/state":
                since = int(parse_qs(url.query).get("since", ["0"])[0] or 0)
                self._json(app.state(since))
            elif url.path == "/status":
                self._json(app.state()["status"])
            elif url.path == "/preview.mjpg":
                self._mjpeg(parse_qs(url.query).get("boxes", ["1"])[0] != "0")
            elif url.path == "/stream.mp3":
                self._mp3()
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "body must be JSON"}, 400)
            path = urlparse(self.path).path
            if path == "/api/settings":
                self._json(app.update(body))
            elif path == "/api/action":
                self._json(app.action(body.pop("action", ""), body))
            else:
                self._json({"error": "not found"}, 404)

        def _mjpeg(self, boxes=True):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            app.preview_clients[boxes] += 1
            last = None
            try:
                while True:
                    with app.jpeg_cond:
                        app.jpeg_cond.wait_for(lambda: app.jpeg[boxes] is not last, timeout=5)
                        jpg = app.jpeg[boxes]
                    if jpg is None or jpg is last:
                        continue
                    last = jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                app.preview_clients[boxes] -= 1

        def _mp3(self):
            if app.hub is None:
                return self._json({"error": "install lameenc for the audio stream"}, 503)
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("icy-name", "CloudTone")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = app.hub.subscribe()
            log.info("stream listener connected: %s", self.client_address[0])
            try:
                while True:
                    self.wfile.write(q.get())
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                app.hub.unsubscribe(q)
                log.info("stream listener left: %s", self.client_address[0])

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="web", daemon=True).start()
    log.info("control page: http://<this-pi>:%d/", port)
    return srv
