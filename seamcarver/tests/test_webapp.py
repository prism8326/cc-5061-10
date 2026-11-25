"""Integration tests for the player HTTP API (in-process server).

These drive the exact endpoints ``web/app.js`` calls, so the UI contract that
"editing and export use the reconstructed path" and "missing segments are shown
for re-read" is verified end-to-end, not just at the module level.

Run directly: ``python tests/test_webapp.py``
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seamcarver import webapp
from seamcarver.webapp import STATE, Handler


def _start():
    from http.server import ThreadingHTTPServer
    webapp.STATE.load_scenario("shot_cut")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd, port


class _Client:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def get(self, path):
        return json.load(urllib.request.urlopen(self.base + path))

    def post(self, path, payload):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        return json.load(urllib.request.urlopen(req))


def test_api_state_exposes_store_and_reconstruction():
    httpd, port = _start()
    try:
        c = _Client(port)
        st = c.get("/api/state")
        assert st["ok"]
        assert st["store"] is not None
        assert st["store"]["format_version"] == 2
        assert st["store"]["missing"] == []
        r = c.post("/api/reconstruct", {"horizon": 8})
        assert r["ok"] and r["reconstruction"]["strategy"] == "import_rebuild"
        assert r["boundary_check"]["ok"], r["boundary_check"]
        print("ok test_api_state_exposes_store_and_reconstruction rev",
              st["store"]["revision"])
    finally:
        httpd.shutdown()


def test_api_clear_then_play_reconstructs():
    httpd, port = _start()
    try:
        c = _Client(port)
        r = c.post("/api/clear_reconstruct", {"horizon": 8})
        assert r["ok"]
        assert r["seams"] is not None and len(r["seams"]) == r["frames"]
        assert r["boundary_check"]["ok"]
        print("ok test_api_clear_then_play_reconstructs frames", r["frames"])
    finally:
        httpd.shutdown()


def test_api_edit_uses_and_bumps_revision():
    httpd, port = _start()
    try:
        c = _Client(port)
        before = c.get("/api/state")["store"]["revision"]
        r = c.post("/api/edit_seam", {"frame": 5, "row": 20, "col": 30})
        assert r["ok"]
        assert r["reconstruction"] is not None, "edit must re-derive reconstruction"
        assert r["store"]["revision"] > before, "edit must bump the store revision"
        print("ok test_api_edit_uses_and_bumps_revision",
              before, "->", r["store"]["revision"])
    finally:
        httpd.shutdown()


def test_api_export_reports_reread_for_missing_boundary():
    httpd, port = _start()
    try:
        c = _Client(port)
        # Simulate a lost temporal-adjacency record, as would happen with a
        # stale/missing chunk edge upload.
        with STATE.lock:
            STATE.engine.store.read_block(0, STATE.engine.store.T)
            STATE.engine.store.delete_boundary(9)
            STATE.engine.store.flush()
        r = c.post("/api/reconstruct", {"horizon": 8})
        assert 9 in r["reconstruction"]["missing_boundaries"]
        assert r["reconstruction"]["reread_frames"], "gap frames must be re-read"
        x = c.post("/api/export", {})
        segs = x["export"]["segments"]
        assert any(s["kind"] == "reread" for s in segs), segs
        assert x["export"]["must_reread"], "exporter must list re-read frames"
        print("ok test_api_export_reports_reread_for_missing_boundary",
              x["export"]["must_reread"])
    finally:
        httpd.shutdown()


def test_api_error_surfaced_not_500():
    httpd, port = _start()
    try:
        c = _Client(port)
        r = c.post("/api/load", {"scenario": "nope"})
        assert r["ok"] is False
        assert "status" in r
        print("ok test_api_error_surfaced_not_500:", r["status"])
    finally:
        httpd.shutdown()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"\nall {len(tests)} webapp tests passed")


if __name__ == "__main__":
    main()
