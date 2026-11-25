"""HTTP application that exposes the seam-carving engine to the web player.

The UI is a thin client: every button calls one of the endpoints below and the
:class:`SeamEngine` remains the single source of truth.  Failed solves return
HTTP 200 with ``ok=false`` and a human-readable status so the player can render
the solver's feedback instead of breaking the session.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .energy import EnergyConfig
from .engine import SeamEngine
from .flow import FlowConfig
from .seam_solver import SolveConfig
from . import synth
from .video_io import overlay_seams

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False


def _frame_jpeg(rgb: np.ndarray, quality: int = 82) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


class AppState:
    def __init__(self):
        self.lock = threading.RLock()
        self.engine: SeamEngine | None = None
        self.scenario = "moving_marker"

    def load_scenario(self, name: str) -> SeamEngine:
        if name == "moving_marker":
            frames, _ = synth.moving_marker(T=24, H=64, W=96, speed=2.2)
        elif name == "fast_occlusion":
            frames, _ = synth.fast_occlusion(T=24, H=64, W=112, speed=6.5)
        elif name == "shot_cut":
            frames, _ = synth.shot_cut(T=24, H=64, W=96, cut_at=12)
        else:
            raise ValueError(f"unknown scenario {name}")
        self.engine = SeamEngine(
            frames, energy_cfg=EnergyConfig(), flow_cfg=FlowConfig(),
            solve_cfg=SolveConfig(), name=name)
        self.engine.solve()
        self.scenario = name
        return self.engine

    def load_video(self, path: str, max_frames: int = 40, scale: float = 0.5) -> SeamEngine:
        from .video_io import read_video
        frames = read_video(path, max_frames=max_frames, scale=scale)
        self.engine = SeamEngine(frames, name=os.path.basename(path))
        self.engine.solve()
        self.scenario = self.engine.name
        return self.engine


STATE = AppState()


class Handler(BaseHTTPRequestHandler):
    server_version = "SeamCarver/1.0"

    def log_message(self, fmt, *args):  # quieter console
        pass

    # ------------------------------------------------------------- helpers
    def _json(self, obj, code: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def _bytes(self, ctype: str, body: bytes, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------- routing
    def do_GET(self):
        try:
            if self.path == "/" or self.path == "/index.html":
                return self._serve_static("index.html", "text/html; charset=utf-8")
            if self.path.startswith("/static/"):
                return self._serve_static(self.path.split("/static/", 1)[1], None)
            if self.path == "/api/state":
                with STATE.lock:
                    return self._json(self._state_payload())
            if self.path.startswith("/api/frame?"):
                return self._serve_frame()
            if self.path == "/api/energy":
                return self._serve_energy()
            self._json({"ok": False, "status": "not found"}, 404)
        except Exception as e:  # surface kernel failures to the UI
            self._json({"ok": False, "status": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        try:
            data = self._body()
            action = self.path
            with STATE.lock:
                if STATE.engine is None:
                    STATE.load_scenario("moving_marker")
                eng = STATE.engine
                if action == "/api/load":
                    eng = STATE.load_scenario(data.get("scenario", "moving_marker"))
                elif action == "/api/solve":
                    eng.solve()
                elif action == "/api/step":
                    kind = data.get("kind", "remove")
                    if kind not in ("remove", "insert"):
                        return self._json({"ok": False, "status": "kind must be remove|insert"})
                    eng.apply_step(kind)
                elif action == "/api/resize":
                    target = int(data["target_width"])
                    if not (1 <= target <= eng.project.original_width * 2):
                        return self._json({"ok": False,
                                           "status": "target width out of supported range"})
                    eng.resize_to(target)
                elif action == "/api/edit_seam":
                    eng.edit_seam(int(data["frame"]), int(data["row"]), int(data["col"]))
                elif action == "/api/protect":
                    eng.set_protected_rect(
                        int(data["frame"]), int(data["x0"]), int(data["y0"]),
                        int(data["x1"]), int(data["y1"]))
                elif action == "/api/reset":
                    STATE.load_scenario(STATE.scenario)
                elif action == "/api/upload":
                    path = data.get("path", "")
                    if not path or not os.path.exists(path):
                        return self._json({"ok": False, "status": "upload path not found"})
                    STATE.load_video(path)
                else:
                    return self._json({"ok": False, "status": f"unknown action {action}"}, 404)
                return self._json(self._state_payload())
        except Exception as e:
            self._json({"ok": False, "status": f"{type(e).__name__}: {e}"}, 500)

    # ------------------------------------------------------------- payloads
    def _state_payload(self) -> dict:
        eng = STATE.engine
        snap = eng.state_snapshot()
        last = eng.history[-1] if eng.history else {}
        return {
            "ok": True,
            "status": last.get("event", "ready"),
            "warnings": last.get("warnings", []),
            "flow_value": last.get("flow_value", 0.0),
            "elapsed": last.get("elapsed", 0.0),
            "seams": eng.last_seams.tolist() if eng.last_seams is not None else None,
            "shot_cuts": eng.energy.shot_cuts.tolist(),
            "cost_per_frame": last.get("cost_per_frame", []),
            **snap,
        }

    def _serve_frame(self):
        qs = self.path.split("?", 1)[1]
        params = dict(kv.split("=", 1) for kv in qs.split("&") if "=" in kv)
        f = int(params.get("f", 0))
        view = params.get("view", "seam")
        with STATE.lock:
            eng = STATE.engine
            f = max(0, min(f, eng.project.T - 1))
            if view == "carved":
                frame = eng.project.current_frames()[f]
            elif view == "original":
                frame = eng.project.original[f]
            elif view == "energy":
                e = eng.energy.unary[f]
                norm = np.clip(e / (np.percentile(e, 98) + 1e-6), 0, 1)
                frame = (np.stack([norm * 255] * 3, axis=-1)).astype(np.uint8)
            else:
                frame = eng.frames_with_seams()[f]
        self._bytes("image/jpeg", _frame_jpeg(frame))

    def _serve_energy(self):
        with STATE.lock:
            e = STATE.engine.energy
            payload = {
                "spatial_mean": e.spatial.mean(axis=(1, 2)).tolist(),
                "motion_mean": e.motion.mean(axis=(1, 2)).tolist(),
                "occlusion_mean": e.occlusion.mean(axis=(1, 2)).tolist(),
                "shot_cuts": e.shot_cuts.tolist(),
            }
        self._json({"ok": True, **payload})

    def _serve_static(self, name: str, ctype: str | None):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
        safe = os.path.normpath(os.path.join(root, name))
        if not safe.startswith(root + os.sep) or not os.path.isfile(safe):
            return self._bytes("text/plain", b"not found", 404)
        if ctype is None:
            ctype = "application/octet-stream"
            if safe.endswith(".js"):
                ctype = "text/javascript; charset=utf-8"
            elif safe.endswith(".css"):
                ctype = "text/css; charset=utf-8"
        with open(safe, "rb") as fh:
            self._bytes(ctype, fh.read())


def serve(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd
