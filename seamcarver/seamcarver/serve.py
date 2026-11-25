"""Entry point for the seam-carving web player.

Start with:
    python -m seamcarver.serve --host 127.0.0.1 --port 8000
then open http://127.0.0.1:8000/ in a browser.
"""
from __future__ import annotations

import argparse

from .webapp import STATE, serve


def main() -> int:
    ap = argparse.ArgumentParser(description="Spatiotemporal seam-carving player")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--scenario", default="moving_marker",
                    choices=["moving_marker", "fast_occlusion", "shot_cut"])
    args = ap.parse_args()

    STATE.load_scenario(args.scenario)
    httpd = serve(args.host, args.port)
    print(f"Seam-carving player running at http://{args.host}:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
