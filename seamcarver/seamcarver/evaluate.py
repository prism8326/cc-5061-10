"""Quantitative evaluation on the synthetic scenarios.

Metrics
-------
* object width change : mean relative width of the tracked marker after
  resizing vs. its true original width - seam carving should preserve it much
  better than uniform scaling, which shrinks everything proportionally;
* trajectory jitter    : second difference of the tracked marker centre,
  measuring frame-to-frame wobble introduced by per-frame independent seams;
* temporal cost        : wall-clock solve/apply time and graph flow value;
* seam side switches   : how often the seam column jumps across the marker,
  the failure mode named in the task.

A uniform-scaling baseline is computed only as a control to prove the core
pipeline is not scaling; it is never used inside the carver.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .engine import SeamEngine
from .energy import EnergyConfig
from .flow import FlowConfig
from .seam_solver import SolveConfig


def track_red_marker(frames: np.ndarray) -> np.ndarray:
    """Return per-frame (cx, width, x0, x1) by thresholding the red marker."""
    T = frames.shape[0]
    out = np.zeros((T, 4), np.float32)
    for t, f in enumerate(frames):
        r, g, b = f[..., 0].astype(int), f[..., 1].astype(int), f[..., 2].astype(int)
        mask = (r > 170) & (r - g > 80) & (r - b > 60)
        cols = np.where(mask.any(axis=0))[0]
        if cols.size == 0:
            out[t] = (np.nan, 0, np.nan, np.nan)
            continue
        x0, x1 = cols[0], cols[-1] + 1
        out[t] = ((x0 + x1) / 2.0, x1 - x0, x0, x1)
    return out


def trajectory_jitter(centres: np.ndarray) -> float:
    """Mean |second difference| of the centre trajectory."""
    c = centres[~np.isnan(centres)]
    if c.size < 3:
        return float("nan")
    return float(np.mean(np.abs(c[2:] - 2 * c[1:-1] + c[:-2])))


@dataclass
class ScenarioReport:
    scenario: str
    original_width: int
    target_width: int
    width_ratio: float
    object_width_ratio_mean: float
    object_width_ratio_max: float
    jitter_px: float
    seam_side_switches: int
    n_shot_cuts: int
    solve_seconds: float
    total_seconds: float
    uniform_scale_object_ratio: float
    acceptable: bool
    notes: str


def _side_switches(seams_log, truth, original_width: int) -> int:
    """Count frame gaps where the seam crosses from one marker side to the other."""
    if not seams_log:
        return 0
    seams = seams_log[-1]
    mid = seams.mean(axis=1)
    x0, x1 = truth["x0"], truth["x1"]
    side_prev = None
    switches = 0
    for t in range(seams.shape[0]):
        if mid[t] < x0[t]:
            side = -1
        elif mid[t] > x1[t]:
            side = 1
        else:
            side = 0
        if side_prev is not None and side in (-1, 1) and side_prev == -side:
            switches += 1
        if side != 0:
            side_prev = side
    return switches


def evaluate_scenario(name: str, frames: np.ndarray, truth: dict,
                      target_ratio: float = 0.75,
                      energy_cfg: EnergyConfig | None = None,
                      flow_cfg: FlowConfig | None = None,
                      solve_cfg: SolveConfig | None = None) -> ScenarioReport:
    W0 = frames.shape[2]
    target = max(8, int(round(W0 * target_ratio)))

    engine = SeamEngine(
        frames,
        energy_cfg=energy_cfg or EnergyConfig(),
        flow_cfg=flow_cfg or FlowConfig(),
        solve_cfg=solve_cfg or SolveConfig(),
        name=name,
    )
    t0 = time.perf_counter()
    results = engine.resize_to(target)
    total = time.perf_counter() - t0
    solve_seconds = float(sum(r.elapsed for r in results))

    carved = engine.project.current_frames()
    track_orig = track_red_marker(frames)
    track_new = track_red_marker(carved)
    valid = ~np.isnan(track_new[:, 0])
    if valid.sum() >= 2:
        ratio = track_new[valid, 1] / truth["width"]
        ratio_mean, ratio_max = float(ratio.mean()), float(np.max(np.abs(1 - ratio)))
        jitter = trajectory_jitter(track_new[:, 0])
    else:
        ratio_mean, ratio_max, jitter = float("nan"), float("nan"), float("nan")

    switches = _side_switches(
        [engine.project.steps[i].seams for i in range(len(engine.project.steps))],
        truth, W0,
    )
    uniform_ratio = target / W0
    # Acceptability window: object keeps within +-12% of its true width and the
    # clip stays continuous (jitter under 1.5 px per frame at this resolution).
    acceptable = (
        np.isfinite(ratio_mean)
        and abs(1 - ratio_mean) <= 0.12
        and (np.isnan(jitter) or jitter <= 1.5)
    )
    notes = "; ".join(w for r in results for w in r.warnings) or "no solver warnings"
    return ScenarioReport(
        scenario=name, original_width=W0, target_width=target,
        width_ratio=target / W0,
        object_width_ratio_mean=ratio_mean,
        object_width_ratio_max=ratio_max,
        jitter_px=jitter,
        seam_side_switches=switches,
        n_shot_cuts=int(engine.energy.shot_cuts.sum()),
        solve_seconds=solve_seconds, total_seconds=total,
        uniform_scale_object_ratio=uniform_ratio,
        acceptable=acceptable, notes=notes,
    )
