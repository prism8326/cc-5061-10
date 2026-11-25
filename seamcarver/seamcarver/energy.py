"""Spatiotemporal seam energy.

The energy of a pixel combines three inspectable terms:

* spatial term  E_sp  : color/intensity gradient magnitude (single-frame cost);
* motion term   E_mo  : dense optical-flow magnitude (moving objects are
  expensive to carve, so seams prefer static background);
* temporal term E_tc  : forward/backward color difference used to mark
  occlusion boundaries and disocclusion, plus an explicit temporal
  smoothness term that is realized as pairwise capacities in the graph
  (see ``seam_solver.build_graph``), not as another unary weight.

Protected ("do-not-cross") regions and shot boundaries are handled here:
protected pixels receive an effectively infinite unary barrier and shot
boundaries cut the temporal pairwise capacities so seams are allowed to
re-decide their side at a cut instead of being dragged across it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .flow import FlowConfig, compute_flows, flow_magnitude


@dataclass
class EnergyConfig:
    w_spatial: float = 1.0
    w_motion: float = 3.0
    w_occlusion: float = 4.0
    motion_sigma: float = 3.0          # px; soft scale for flow magnitude
    protected_inf: float = 1.0e6       # hard barrier capacity
    shot_threshold: float = 28.0       # mean abs gray diff on 0..255 scale
    shot_flow_ratio: float = 1.5       # also flag implausibly large median flow


@dataclass
class EnergyVolume:
    unary: np.ndarray                  # (T,H,W) float32 seam-removal cost
    spatial: np.ndarray
    motion: np.ndarray
    occlusion: np.ndarray
    protected: np.ndarray              # bool (T,H,W)
    shot_cuts: np.ndarray              # bool (T-1,) temporal edges to disable
    flows: np.ndarray
    config: EnergyConfig


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Vectorized Sobel magnitude for a frame stack (T,H,W)."""
    g = gray.astype(np.float32)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    pad = np.pad(g, ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    for i in range(3):
        for j in range(3):
            block = pad[:, i:i + g.shape[1], j:j + g.shape[2]]
            gx += kx[i, j] * block
            gy += kx[j, i] * block
    return np.sqrt(gx * gx + gy * gy + 1e-8)


def occlusion_map(frames: np.ndarray, flows: np.ndarray) -> np.ndarray:
    """Forward-backward consistency mask, shape (T,H,W), values in [0,1].

    A pixel whose forward-warped position does not map back onto itself is
    either occluded in the next frame or disoccluded; such locations are made
    expensive for seams.  Frame 0 and the last frame reuse their neighbour.
    """
    T = frames.shape[0]
    H, W = frames.shape[1:3]
    occ = np.zeros((T, H, W), np.float32)
    if T < 2:
        return occ
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for t in range(T - 1):
        fl = flows[t]
        # Warp the forward flow back by sampling the next frame flow grid with
        # bilinear nearest neighbour on integer coordinates for inspectability.
        nx = np.clip(np.round(xx + fl[..., 0]).astype(int), 0, W - 1)
        ny = np.clip(np.round(yy + fl[..., 1]).astype(int), 0, H - 1)
        back_dx = fl[ny, nx, 0]
        back_dy = fl[ny, nx, 1]
        err = np.sqrt((fl[..., 0] + back_dx) ** 2 + (fl[..., 1] + back_dy) ** 2)
        m = np.clip((err - 1.5) / 4.0, 0.0, 1.0).astype(np.float32)
        occ[t] = np.maximum(occ[t], m)
        occ[t + 1] = np.maximum(occ[t + 1], m)
    return occ


def detect_shot_cuts(gray: np.ndarray, flows: np.ndarray, cfg: EnergyConfig) -> np.ndarray:
    """Boolean per inter-frame gap: True means a hard temporal boundary."""
    T = gray.shape[0]
    cuts = np.zeros(max(T - 1, 1), dtype=bool)
    for t in range(T - 1):
        diff = np.abs(gray[t + 1] - gray[t]).mean()
        med_flow = np.median(np.sqrt((flows[t] ** 2).sum(-1))) if flows.shape[0] else 0.0
        cuts[t] = diff > cfg.shot_threshold or med_flow > cfg.shot_flow_ratio * gray.shape[2]
    return cuts[: T - 1] if T > 1 else cuts[:0]


def build_energy(
    frames: np.ndarray,
    protected: np.ndarray | None = None,
    cfg: EnergyConfig | None = None,
    flow_cfg: FlowConfig | None = None,
    existing_flows: np.ndarray | None = None,
) -> EnergyVolume:
    """Assemble the full unary energy volume and temporal boundary flags."""
    cfg = cfg or EnergyConfig()
    frames = frames.astype(np.float32)
    if frames.ndim == 3:
        gray = frames.copy()
    else:
        gray = 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
    T, H, W = gray.shape

    flows = existing_flows if existing_flows is not None else compute_flows(frames, flow_cfg)
    spatial = _sobel_magnitude(gray)
    spatial = spatial / (spatial.mean(axis=(1, 2), keepdims=True) + 1e-6)

    mag = flow_magnitude(flows)
    motion = 1.0 - np.exp(-(mag / cfg.motion_sigma) ** 2)
    motion = motion / (motion.mean(axis=(1, 2), keepdims=True) + 1e-6)

    occ = occlusion_map(frames, flows)
    shot_cuts = detect_shot_cuts(gray, flows, cfg)

    unary = (
        cfg.w_spatial * spatial
        + cfg.w_motion * motion
        + cfg.w_occlusion * occ
    ).astype(np.float32)

    if protected is None:
        protected = np.zeros((T, H, W), dtype=bool)
    else:
        protected = protected.astype(bool).reshape(T, H, W)
    unary = unary.copy()
    unary[protected] = cfg.protected_inf

    return EnergyVolume(
        unary=unary, spatial=spatial, motion=motion, occlusion=occ,
        protected=protected, shot_cuts=shot_cuts, flows=flows, config=cfg,
    )
