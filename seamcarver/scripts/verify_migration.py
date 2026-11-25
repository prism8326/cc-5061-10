"""Clean-room evidence for the legacy -> chunked seam-data conversion.

Everything here is generated from pixels at run time: the synthetic clip, its
legacy flat energy/path table, the migrated store, the reconstructed seams and
the exporter's re-read map.  Nothing is read from a hand-edited fixture, and the
script refuses to write into a non-empty directory so the conversion provably
runs in a clean location.

It records, per frame, the reconstructed seam *trajectory* (hash + mean/min/max
column) and the *frame aspect* after carving, then asserts against the values
the execution actually produced.  If any check fails the script exits non-zero
instead of printing a success line.

Usage:
    python scripts/verify_migration.py [--scenario moving_marker] \
        [--out /tmp/seam_clean] [--chunk-size 4]

Artifacts (all under ``--out``):
    legacy/old.npz(+.json/.frames.npz)  the flat table handed to the migrator
    store/                              the chunked, versioned store
    evidence.json                       checks, trajectory & aspect evidence
    summary.txt                         human-readable summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seamcarver import synth
from seamcarver.energy import build_energy
from seamcarver.exporter import export_with_reread
from seamcarver.migration import migrate_legacy, write_legacy_table
from seamcarver.reconstruct import reconstruct
from seamcarver.seam_solver import SolveConfig, build_and_solve
from seamcarver.seam_store import load_store

SCENARIOS = {
    "moving_marker": lambda: synth.moving_marker(T=18, H=40, W=64, speed=2.0),
    "fast_occlusion": lambda: synth.fast_occlusion(T=18, H=40, W=72, speed=6.5),
    "shot_cut": lambda: synth.shot_cut(T=18, H=40, W=64, cut_at=9),
}


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _trajectory(seams: np.ndarray) -> list[dict]:
    """Per-frame seam trajectory evidence, straight from the solved array."""
    return [
        {
            "frame": t,
            "mean_col": round(float(seams[t].mean()), 4),
            "min_col": int(seams[t].min()),
            "max_col": int(seams[t].max()),
            "sha256": _sha(seams[t].astype(np.int32)),
        }
        for t in range(seams.shape[0])
    ]


def _aspect(frames: np.ndarray) -> dict:
    T, H, W = frames.shape[:3]
    return {"frames": int(T), "height": int(H), "width": int(W),
            "aspect": round(W / H, 6),
            "all_same_dims": bool(len({f.shape for f in frames}) == 1)}


def _check(cond: bool, label: str, detail: object, checks: list[dict]) -> None:
    checks.append({"check": label, "ok": bool(cond), "detail": detail})


def run(scenario: str, out_root: str, chunk_size: int) -> dict:
    frames, _truth = SCENARIOS[scenario]()
    T, H, W = frames.shape[:3]

    legacy_dir = os.path.join(out_root, "legacy")
    store_dir = os.path.join(out_root, "store")
    os.makedirs(legacy_dir, exist_ok=True)

    # 1. Build the legacy flat table the way the pre-chunk engine would: one
    #    flat energy volume + one cached seam, no chunks, no boundaries.
    ev = build_energy(frames.astype(np.float32), cfg=None)
    legacy_seam, _, _, _ = build_and_solve(
        ev.unary, ev.flows, ev.shot_cuts, SolveConfig())
    legacy_path = os.path.join(legacy_dir, "old.npz")
    write_legacy_table(legacy_path, ev.unary, legacy_seam,
                       shot_cuts=ev.shot_cuts, frames=frames)

    # 2. Convert in the clean directory (no pre-existing store to reuse).
    mrep = migrate_legacy(legacy_path, store_dir, chunk_size=chunk_size)
    store = load_store(store_dir)
    rev_after_migrate = store.revision

    # 3. Reconstruct the cross-frame path from the chunked store.  Fidelity is
    #    checked with unbounded backtracking (horizon=0) so the comparison is
    #    against the full temporal chain; a finite horizon is a deliberate
    #    truncation, so its flicker is measured separately below.
    rrep_full = reconstruct(store, horizon=0)
    rec_seams = rrep_full.seams
    rrep_finite = reconstruct(store, horizon=12)

    # 4. Export using the reconstructed path, proving pixels come from source.
    export = export_with_reread(frames, rrep_full, store.revision)

    # 5. Second, independent store in a sibling clean dir: drop one boundary and
    #    confirm the exporter flags a re-read instead of inheriting old energy.
    lose_dir = os.path.join(out_root, "store_lost_boundary")
    loserep = migrate_legacy(legacy_path, lose_dir, chunk_size=chunk_size)
    lose = load_store(lose_dir)
    lose.read_block(0, lose.T)
    dropped_gap = max(1, T // 2)
    lose.delete_boundary(dropped_gap)
    lose.flush()
    lose_rec = reconstruct(lose, horizon=12)
    lose_export = export_with_reread(frames, lose_rec, lose.revision)

    checks: list[dict] = []
    verified_missing = store.verify()
    _check(verified_missing == [], "migrated store integrity", verified_missing,
           checks)
    _check(mrep.recomputed_frames == T and mrep.inherited_frames == 0,
           "energy recomputed from pixels (not inherited)",
           {"recomputed": mrep.recomputed_frames,
            "inherited": mrep.inherited_frames}, checks)
    _check(mrep.boundaries_rebuilt == T - 1,
           "temporal adjacency rebuilt for every gap",
           {"rebuilt": mrep.boundaries_rebuilt, "expected": T - 1}, checks)
    _check(store.manifest["format_version"] == 2,
           "energy-storage migration version annotated",
           {"format_version": store.manifest["format_version"]}, checks)
    _check(np.array_equal(rec_seams, legacy_seam),
           "converted legacy project rebuilds the SAME seam",
           {"max_abs_diff": int(np.abs(rec_seams - legacy_seam).max()),
            "mismatched_rows": int((rec_seams != legacy_seam).sum())}, checks)
    _check(export.frames.shape[2] == W - 1,
           "export uses the reconstructed path at current aspect",
           {"exported_width": int(export.frames.shape[2]),
            "expected": W - 1}, checks)
    _check(export.used_revision == store.revision,
           "export bound to the reconstruction revision",
           {"export_revision": export.used_revision,
            "store_revision": store.revision}, checks)
    _check(not export.must_reread,
           "clean store needs no re-read", {"must_reread": export.must_reread},
           checks)
    _check(dropped_gap in lose_rec.missing_boundaries,
           "dropped boundary is reported missing, not zero-fabricated",
           {"dropped_gap": dropped_gap,
            "missing_boundaries": lose_rec.missing_boundaries}, checks)
    _check(dropped_gap in lose_export.must_reread
           and dropped_gap + 1 in lose_export.must_reread,
           "exporter flags both sides of the dropped gap for re-read",
           {"must_reread": lose_export.must_reread}, checks)
    _check(any(s["kind"] == "reread" for s in lose_export.segments),
           "exporter surfaces a re-read segment",
           [s for s in lose_export.segments if s["kind"] == "reread"], checks)

    ev_doc = {
        "clean_directory": out_root,
        "scenario": scenario,
        "generated_from_pixels": True,
        "hand_edited_inputs": False,
        "inputs": {
            "frames_sha256": _sha(frames),
            "legacy_unary_sha256": _sha(ev.unary),
            "legacy_seam_sha256": _sha(legacy_seam.astype(np.int32)),
        },
        "frame_aspect": {
            "source": _aspect(frames),
            "store": {"T": store.T, "H": store.H, "W": store.W,
                      "chunk_size": store.chunk_size,
                      "n_chunks": store.n_chunks},
            "carved": _aspect(export.frames),
        },
        "trajectory": {
            "legacy": _trajectory(legacy_seam),
            "reconstructed": _trajectory(rec_seams),
            "identical": bool(np.array_equal(rec_seams, legacy_seam)),
            "finite_horizon_12_identical": bool(
                np.array_equal(rrep_finite.seams, legacy_seam)),
        },
        "seam_trajectory_sha256": _sha(rec_seams.astype(np.int32)),
        "migration": mrep.public(),
        "store_revision_after_migrate": rev_after_migrate,
        "reconstruction": rrep_full.public(),
        "reconstruction_finite_horizon_12": {
            "flicker": rrep_finite.flicker,
            "strategy": rrep_finite.strategy,
            "reread_frames": rrep_finite.reread_frames,
        },
        "export": export.public(),
        "lost_boundary": {
            "dropped_gap": dropped_gap,
            "reconstruction": lose_rec.public(),
            "export": lose_export.public(),
        },
        "checks": checks,
        "ok": all(c["ok"] for c in checks),
    }
    return ev_doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="moving_marker", choices=sorted(SCENARIOS))
    ap.add_argument("--out", default="/tmp/seam_clean")
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--force", action="store_true",
                    help="remove a previous evidence dir before running")
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        if not args.force:
            raise SystemExit(
                f"{args.out} is not empty; refusing to mix evidence with "
                f"existing files (use --force to start clean)")
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    doc = run(args.scenario, args.out, args.chunk_size)
    with open(os.path.join(args.out, "evidence.json"), "w") as fh:
        json.dump(doc, fh, indent=2)

    lines = [f"scenario={doc['scenario']} clean_dir={doc['clean_directory']}"]
    for c in doc["checks"]:
        lines.append(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
    lines.append(f"trajectory identical={doc['trajectory']['identical']} "
                 f"sha256={doc['seam_trajectory_sha256'][:16]}...")
    fa = doc["frame_aspect"]
    lines.append(f"aspect source={fa['source']['width']}x{fa['source']['height']} "
                 f"carved={fa['carved']['width']}x{fa['carved']['height']}")
    summary = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(summary + "\n")
    print(summary)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
