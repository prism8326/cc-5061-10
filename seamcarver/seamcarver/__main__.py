"""Command-line entry point for the seam-carving tool.

The player and the graphical UI are the primary interface; this CLI is a thin
*assistant* for batch work that mirrors the exact same kernel:

* ``serve``       -- start the motion-analysis player (the graphical entry);
* ``migrate``     -- convert a legacy flat energy/path table into a chunked store;
* ``reconstruct`` -- rebuild seams across frames from a chunked store and report
  flicker / which frames must be re-read;
* ``verify``      -- integrity-check every chunk and boundary of a store.

Nothing here re-implements seam logic; it calls :mod:`seamcarver.migration`,
:mod:`seamcarver.reconstruct` and :mod:`seamcarver.seam_store` directly.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import synth
from .energy import EnergyConfig
from .flow import FlowConfig
from .migration import (
    build_two_transition_projects, make_moving_transition_clip,
    migrate_legacy,
)
from .reconstruct import compare_strategies, reconstruct
from .seam_solver import SolveConfig
from .seam_store import load_store


def _cmd_migrate(args) -> int:
    frames = None
    if args.frames:
        with np.load(args.frames) as npz:
            key = "frames" if "frames" in npz.files else (
                "arr_0" if "arr_0" in npz.files else None)
            if key is None:
                raise SystemExit(f"{args.frames} has no 'frames' array")
            frames = np.asarray(npz[key], np.uint8)
    # When --frames is omitted, migrate_legacy auto-loads the legacy sidecar
    # <path>.frames.npz if present.
    rep = migrate_legacy(args.legacy, args.target, frames=frames,
                         chunk_size=args.chunk_size)
    print(json.dumps(rep.public(), indent=2))
    return 0


def _cmd_reconstruct(args) -> int:
    store = load_store(args.store)
    rep = reconstruct(store, horizon=args.horizon, strategy=args.strategy)
    out = rep.public()
    if args.compare:
        out = {"compare": compare_strategies(store, horizon=args.horizon),
               "reconstruction": out}
    print(json.dumps(out, indent=2))
    return 0


def _cmd_verify(args) -> int:
    store = load_store(args.store)
    missing = store.verify()
    print(json.dumps({"store": args.store, "revision": store.revision,
                      "missing": missing}, indent=2))
    return 0 if not missing else 1


def _cmd_demo(args) -> int:
    """End-to-end migration demo on the built-in identical-pixel transition.

    Uses a synthetic drifting-corridor energy graph (a cheap corridor that moves
    faster than the temporal motion gate) so the lazy-vs-import comparison has a
    real cross-chunk seam flicker to measure.
    """
    from .seam_store import SeamStore
    T, H, W = 20, 24, 48
    cs = 4
    corridor = W // 2
    unary = np.ones((T, H, W), np.float32)
    unary[:, :, corridor] = 0.10                  # global cheap corridor
    offsets = [-4, 4, -4, 4, -4]
    for k in range((T + cs - 1) // cs):
        s, e = k * cs, min((k + 1) * cs, T)
        unary[s:e, :, corridor + offsets[k % len(offsets)]] = 0.06  # chunk-local trap
    seam = np.full((T, H), corridor, np.int32)
    # Static flow => strong temporal coupling; the correct seam stays on the
    # global corridor across every chunk boundary.  The lazy per-chunk solve
    # gets trapped by each chunk-local distractor and flickers.
    gap = np.zeros((T - 1, H, W), np.float32)
    store = SeamStore.from_energy(unary, seam, gap_motion=gap,
                                  shot_cuts=np.zeros(T - 1, bool),
                                  chunk_size=cs, root=f"{args.out}/chunked",
                                  created_from="demo")
    cmp = compare_strategies(store, horizon=args.horizon)
    rec = reconstruct(store, horizon=args.horizon)
    print(json.dumps({"compare": cmp, "reconstruction": rec.public()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Spatiotemporal seam-carving (CLI assistant)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the graphical player (primary interface)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--scenario", default="moving_marker",
                   choices=["moving_marker", "fast_occlusion", "shot_cut"])
    s.set_defaults(func=lambda a: _serve(a))

    m = sub.add_parser("migrate", help="convert a legacy flat table to a chunked store")
    m.add_argument("--legacy", required=True)
    m.add_argument("--target", required=True)
    m.add_argument("--frames", default=None, help="optional .npz with key 'frames'")
    m.add_argument("--chunk-size", type=int, default=8)
    m.set_defaults(func=_cmd_migrate)

    r = sub.add_parser("reconstruct", help="rebuild seams across frames from a store")
    r.add_argument("--store", required=True)
    r.add_argument("--horizon", type=int, default=0,
                   help="cross-frame backtracking horizon; 0 = full clip")
    r.add_argument("--strategy", default="auto",
                   choices=["auto", "lazy", "import_rebuild"])
    r.add_argument("--compare", action="store_true")
    r.set_defaults(func=_cmd_reconstruct)

    v = sub.add_parser("verify", help="integrity-check a store's chunks/boundaries")
    v.add_argument("--store", required=True)
    v.set_defaults(func=_cmd_verify)

    d = sub.add_parser("demo", help="run the built-in migration/reconstruction demo")
    d.add_argument("--out", default="/tmp/seam_demo")
    d.add_argument("--horizon", type=int, default=0)
    d.set_defaults(func=_cmd_demo)
    return ap


def _serve(args) -> int:
    from .serve import main as serve_main
    sys.argv = ["seamcarver", "--host", args.host, "--port", str(args.port),
                "--scenario", args.scenario]
    return serve_main()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
