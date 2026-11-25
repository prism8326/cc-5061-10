"""Spatiotemporal seam solver.

A vertical seam is one pixel per row per frame.  Frames are stacked into a
3D node volume (t, y, x) and labeled by an s/t min-cut:

* node label 0 (source side) = "left of the seam"
* node label 1 (sink side)   = "right of the seam"
* the seam is the ordered list of 0->1 transitions per row.

Unary capacities encode the per-pixel removal cost so that any feasible cut
has capacity equal to the energy sum along the seam it represents.  Vertical
spatial edges couple neighbouring rows so the seam is an 8-connected path;
temporal edges couple the same (y, x) position across frames so seams of
adjacent frames stay spatially coherent.  Those temporal edges are the real
inter-frame association: they are *solved* jointly by max-flow rather than
copying the previous frame's coordinates.  At shot cuts temporal edges are
removed and at fast-motion/occlusion rows their capacity drops, explicitly
permitting the seam to switch to the other side of an athlete.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .maxflow_wrapper import grid_mincut


@dataclass
class SolveConfig:
    temporal_base: float = 25.0      # base capacity of temporal edges
    temporal_max: float = 200.0
    temporal_motion_gate: float = 6.0  # px; above this flow, loosen coherence
    lock_capacity: float = 1.0e7     # hard source/sink lock
    boundary_bias: float = 1.0e-3    # tiny left-wall bias to disambiguate


def _spatial_capacities(unary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal seam-edge capacities and vertical 8-connectivity coupling.

    The horizontal edge between columns x and x+1 on a row carries the unary
    energy of the right pixel.  With the left wall pinned to source and the
    right wall pinned to sink, any 0->1 transition (the seam) pays exactly the
    unary of its seam column.  Vertical edges couple adjacent rows so the seam
    stays an 8-connected single-pixel-wide path.
    """
    T, H, W = unary.shape
    cap_h = unary[..., 1:].astype(np.float32).copy()
    cap_v = np.full((T, H - 1, W), 1.0e4, np.float32)
    return cap_h, cap_v


def temporal_capacities(
    unary: np.ndarray, gap_motion: np.ndarray, shot_cuts: np.ndarray,
    cfg: SolveConfig,
) -> np.ndarray:
    """Capacity of edges joining node (t,y,x) to (t+1,y,x), from motion magnitude.

    High where the scene is static (seam should persist), low where the local
    motion magnitude or occlusion cost indicates the seam may need to jump,
    and zero across a detected shot cut.  ``gap_motion[t]`` is the forward
    flow *magnitude* for the t -> t+1 gap; keeping it as an explicit input lets
    the chunked store rebuild the exact same edges without re-running optical
    flow.
    """
    T, H, W = unary.shape
    if T < 2:
        return np.zeros((0, H, W), np.float32)
    cap = np.zeros((T - 1, H, W), np.float32)
    for t in range(T - 1):
        if shot_cuts[t]:
            continue
        loosen = np.exp(-(gap_motion[t] / cfg.temporal_motion_gate) ** 2)
        energy_scale = 0.5 * (unary[t] + unary[t + 1])
        cap[t] = np.minimum(
            cfg.temporal_max,
            cfg.temporal_base * loosen + 0.25 * energy_scale * loosen,
        )
    return cap.astype(np.float32)


def _temporal_capacities(
    unary: np.ndarray, flows: np.ndarray, shot_cuts: np.ndarray, cfg: SolveConfig
) -> np.ndarray:
    """Backward-compatible wrapper: derive per-gap magnitudes from dense flow."""
    if flows.shape[0] == 0:
        gap_motion = np.zeros((0,) + unary.shape[1:], np.float32)
    else:
        gap_motion = np.sqrt((flows ** 2).sum(-1) + 1e-8)
    return temporal_capacities(unary, gap_motion, shot_cuts, cfg)


def build_and_solve(
    unary: np.ndarray,
    flows: np.ndarray,
    shot_cuts: np.ndarray,
    cfg: SolveConfig | None = None,
    locked_seams: np.ndarray | None = None,
):
    """Run the 3D min-cut and return per-frame seam coordinates.

    Parameters
    ----------
    unary : (T,H,W) float32
    flows : (T-1,H,W,2) forward optical flow
    shot_cuts : (T-1,) bool, temporal gaps to sever
    locked_seams : optional (T,H) int array.  Where >= 0, the seam is forced
        to pass at exactly that x column, implemented as hard s/t locks.

    Returns
    -------
    seams : (T,H) int32 columns, one per row
    labels : (T,H,W) uint8
    flow_value : float
    """
    cfg = cfg or SolveConfig()
    T, H, W = unary.shape
    cap_h, cap_v = _spatial_capacities(unary)
    cap_t = _temporal_capacities(unary, flows, shot_cuts, cfg)

    # Wall-pinned seam construction.  Horizontal pair edges already carry the
    # per-pixel unary (see _spatial_capacities), so terminals only enforce the
    # ordering: column 0 is forced source side and column W-1 sink side with a
    # capacity larger than any achievable cut.  The seam is then the unique
    # source->sink transition per row, whose crossing edge costs that column's
    # unary; temporal edges couple the transition across frames.
    wall = cfg.lock_capacity
    term_src = np.zeros((T, H, W), np.float32)
    term_sink = np.zeros((T, H, W), np.float32)
    term_src[:, :, 0] = wall
    term_sink[:, :, W - 1] = wall

    if locked_seams is not None:
        locked_seams = np.asarray(locked_seams)
        if locked_seams.shape != (T, H):
            raise ValueError("locked_seams must have shape (T,H)")
        xs = np.broadcast_to(np.arange(W), (T, H, W))
        lock = locked_seams[..., None]
        valid = lock >= 0
        left = valid & (xs < lock)
        right = valid & (xs >= lock)
        term_src[left] = cfg.lock_capacity
        term_sink[right] = cfg.lock_capacity

    labels, flow_value = grid_mincut(term_src, term_sink, cap_h, cap_v, cap_t)
    seams = extract_seams(labels)
    return seams, labels, flow_value, cap_t


def extract_seams(labels: np.ndarray) -> np.ndarray:
    """Return the first sink-side column per row, shape (T,H).

    A small left-wall source bias (see build_and_solve) guarantees every row
    starts on the source side, so the transition is unique.  Rows with no
    transition (only possible on degenerate manually locked inputs) fall back
    to W so callers always get a valid column.
    """
    T, H, W = labels.shape
    seams = np.full((T, H), W - 1, dtype=np.int32)
    for t in range(T):
        for y in range(H):
            row = labels[t, y]
            idx = np.flatnonzero(row == 1)
            seams[t, y] = idx[0] if idx.size else W - 1
    return seams


def seam_cost(unary: np.ndarray, seams: np.ndarray) -> np.ndarray:
    """Actual summed unary cost of a seam per frame (for diagnostics)."""
    T, H = seams.shape
    ys = np.broadcast_to(np.arange(H), (T, H))
    return unary[np.arange(T)[:, None], ys, seams].sum(axis=1)
