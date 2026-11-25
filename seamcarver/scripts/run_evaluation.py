"""Run the full quantitative evaluation on the three synthetic scenarios.

Usage:
    python scripts/run_evaluation.py [--out outputs/eval]

Writes a JSON report and a human-readable text summary, including the uniform
scaling control numbers and the acceptability verdict for each resize ratio.
Hard cases (fast occlusion, shot cut) are kept on purpose; no best-looking
single frame is cherry-picked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seamcarver.engine import SeamEngine
from seamcarver.energy import EnergyConfig
from seamcarver.evaluate import (
    ScenarioReport, evaluate_scenario, track_red_marker, trajectory_jitter,
)
from seamcarver.flow import FlowConfig
from seamcarver.reconstruct import compare_strategies
from seamcarver.seam_solver import SolveConfig
from seamcarver import synth
from seamcarver.video_io import write_video, overlay_seams


SCENARIOS = {
    "moving_marker": lambda: synth.moving_marker(T=24, H=64, W=96, speed=2.2),
    "fast_occlusion": lambda: synth.fast_occlusion(T=24, H=64, W=112, speed=6.5),
    "shot_cut": lambda: synth.shot_cut(T=24, H=64, W=96, cut_at=12),
}


def uniform_scale(frames: np.ndarray, target_w: int) -> np.ndarray:
    import cv2
    T, H, W, C = frames.shape
    out = np.zeros((T, H, target_w, C), frames.dtype)
    for t in range(T):
        out[t] = cv2.resize(frames[t], (target_w, H), interpolation=cv2.INTER_AREA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--ratios", default="0.85,0.75,0.65")
    ap.add_argument("--flow", default="farneback", choices=["farneback", "blockmatch"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ratios = [float(x) for x in args.ratios.split(",")]

    flow_cfg = FlowConfig(method=args.flow)
    reports: list[dict] = []
    migration: dict[str, dict] = {}
    for name, factory in SCENARIOS.items():
        frames, truth = factory()
        write_video(os.path.join(args.out, f"{name}_original.mp4"), frames, fps=12)
        for ratio in ratios:
            rep = evaluate_scenario(
                name, frames, truth, target_ratio=ratio,
                energy_cfg=EnergyConfig(), flow_cfg=flow_cfg, solve_cfg=SolveConfig(),
            )
            reports.append(rep.__dict__.copy())
            print(f"[{name}] ratio={ratio:.2f} "
                  f"obj_width={rep.object_width_ratio_mean:.3f} "
                  f"jitter={rep.jitter_px:.3f} switches={rep.seam_side_switches} "
                  f"cuts={rep.n_shot_cuts} t={rep.total_seconds:.2f}s "
                  f"scale_baseline={rep.uniform_scale_object_ratio:.3f} "
                  f"acceptable={rep.acceptable}")

        # Persist one carved clip + seam overlay at the middle ratio for visual
        # continuity inspection across the whole sequence, not one frame.
        mid_ratio = ratios[len(ratios) // 2]
        eng = SeamEngine(frames, flow_cfg=flow_cfg, name=name)
        target = max(8, round(frames.shape[2] * mid_ratio))
        eng.resize_to(target)
        carved = eng.project.current_frames()
        write_video(os.path.join(args.out, f"{name}_carved.mp4"), carved, fps=12)
        overlay = overlay_seams(carved, eng.last_seams)
        write_video(os.path.join(args.out, f"{name}_seam_overlay.mp4"), overlay, fps=12)
        scaled = uniform_scale(frames, target)
        write_video(os.path.join(args.out, f"{name}_uniform_scale.mp4"), scaled, fps=12)

        # Point the engine at the chunked store and produce migration evidence:
        # reconstruction flicker, lazy-vs-import comparison, and the export
        # re-read map.  This proves editing/export use the NEW path, not the
        # legacy per-frame energy table.
        store_dir = os.path.join(args.out, "stores", name)
        eng.store_root = store_dir
        eng.recompute_energy()
        eng.solve()
        recon = eng.reconstruct(horizon=0)   # full clip: truncate only at shot cuts
        cmp = compare_strategies(eng.store, horizon=0)
        export = eng.export()
        migration[name] = {
            "store": {
                "format_version": eng.store.manifest["format_version"],
                "revision": eng.store.revision,
                "chunk_size": eng.store.chunk_size,
                "n_chunks": eng.store.n_chunks,
                "missing": eng.store.verify(),
            },
            "reconstruction": recon.public(),
            "strategy_compare": {
                "chosen": cmp["chosen"], "reason": cmp["reason"],
                "lazy_flicker_index": cmp["lazy"]["flicker"]["flicker_index"],
                "import_flicker_index": cmp["import_rebuild"]["flicker"]["flicker_index"],
                "lazy_chunk_edge_discontinuity": cmp["lazy"]["discontinuity_at_chunk_edges"],
                "import_chunk_edge_discontinuity": cmp["import_rebuild"]["discontinuity_at_chunk_edges"],
            },
            "export": export.public(),
        }
        print(f"[{name}] store rev={eng.store.revision} "
              f"recon_flicker={recon.flicker['flicker_index']:.4f} "
              f"strategy={cmp['chosen']} "
              f"reread={len(export.must_reread)}")

    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(reports, f, indent=2)
    with open(os.path.join(args.out, "migration.json"), "w") as f:
        json.dump(migration, f, indent=2)

    lines = ["scenario\tratio\tobj_width\tjitter\tswitches\tshot_cuts\t"
             "solve_s\ttotal_s\tscale_baseline\tacceptable"]
    for r in reports:
        lines.append("\t".join(str(x) for x in [
            r["scenario"], f"{r['width_ratio']:.2f}",
            f"{r['object_width_ratio_mean']:.3f}", f"{r['jitter_px']:.3f}",
            r["seam_side_switches"], r["n_shot_cuts"],
            f"{r['solve_seconds']:.3f}", f"{r['total_seconds']:.2f}",
            f"{r['uniform_scale_object_ratio']:.3f}", r["acceptable"],
        ]))
    with open(os.path.join(args.out, "report.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote", os.path.join(args.out, "report.tsv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
