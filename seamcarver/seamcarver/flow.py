"""Optical-flow estimation with a pure-numpy block-matching fallback.

All public functions return dense flow as float32 arrays of shape (H, W, 2),
where channel 0 is the x displacement (columns) and channel 1 the y
displacement (rows) of each pixel between consecutive frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover - exercised only without OpenCV
    cv2 = None
    _HAS_CV2 = False


@dataclass
class FlowConfig:
    method: str = "farneback"          # "farneback" | "blockmatch"
    block: int = 9
    search: int = 12
    downscale: int = 2                 # estimate at reduced resolution


def _to_gray(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 4:
        f = frames.astype(np.float32)
        return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return frames.astype(np.float32)


def _blockmatch_flow(prev: np.ndarray, cur: np.ndarray, cfg: FlowConfig) -> np.ndarray:
    """Forward flow prev->cur on a coarse grid, nearest-neighbour upsampled.

    Block matching is intentionally simple and inspectable: it exhaustively
    searches integer shifts in a window and minimizes SAD.  It serves both as
    an OpenCV-free fallback and as documentation of the correspondence model.
    """
    H, W = prev.shape
    s = max(cfg.downscale, 1)
    h, w = max(H // s, 4), max(W // s, 4)
    if _HAS_CV2:
        pg = cv2.resize(prev, (w, h))
        cg = cv2.resize(cur, (w, h))
    else:
        pg = _resize_nn(prev, (h, w))
        cg = _resize_nn(cur, (h, w))
    b = max(cfg.block // s, 3)
    r = max(cfg.search // s, 2)
    half = b // 2
    flow_small = np.zeros((h, w, 2), np.float32)
    for y in range(half, h - half):
        for x in range(half, w - half):
            best, bd = (0.0, 0.0), np.inf
            ref = pg[y - half:y + half + 1, x - half:x + half + 1]
            for dy in range(-r, r + 1):
                yy = y + dy
                if not (half <= yy < h - half):
                    continue
                for dx in range(-r, r + 1):
                    xx = x + dx
                    if not (half <= xx < w - half):
                        continue
                    cand = cg[yy - half:yy + half + 1, xx - half:xx + half + 1]
                    d = np.abs(cand - ref).mean()
                    if d < bd:
                        bd, best = d, (float(dx), float(dy))
            flow_small[y, x] = best
    flow_small[..., 0] *= s
    flow_small[..., 1] *= s
    if _HAS_CV2:
        return cv2.resize(flow_small, (W, H)).astype(np.float32)
    return _resize_flow_nn(flow_small, (H, W))


def _resize_nn(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ys = (np.arange(h) * a.shape[0] / h).astype(int)
    xs = (np.arange(w) * a.shape[1] / w).astype(int)
    return a[np.ix_(ys, xs)]


def _resize_flow_nn(f: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ys = (np.arange(h) * f.shape[0] / h).astype(int)
    xs = (np.arange(w) * f.shape[1] / w).astype(int)
    return f[np.ix_(ys, xs)]


def compute_flows(frames: np.ndarray, cfg: FlowConfig | None = None) -> np.ndarray:
    """Return forward flows with shape (T-1, H, W, 2)."""
    cfg = cfg or FlowConfig()
    gray = _to_gray(frames)
    T = gray.shape[0]
    H, W = gray.shape[1:]
    flows = np.zeros((max(T - 1, 1), H, W, 2), np.float32)
    use_fb = _HAS_CV2 and cfg.method == "farneback"
    for t in range(T - 1):
        if use_fb:
            fl = cv2.calcOpticalFlowFarneback(
                gray[t], gray[t + 1], None,
                pyr_scale=0.5, levels=4, winsize=11,
                iterations=4, poly_n=5, poly_sigma=1.2, flags=0,
            ).astype(np.float32)
        else:
            fl = _blockmatch_flow(gray[t], gray[t + 1], cfg)
        flows[t] = fl
    return flows[: T - 1] if T > 1 else flows[:0]


def flow_magnitude(flows: np.ndarray) -> np.ndarray:
    """Per-pixel motion magnitude aligned to each frame (T, H, W).

    Frame 0 uses the first forward flow; frame T-1 the last; interior frames
    average the incoming (warped) and outgoing flow magnitudes.
    """
    Tm1 = flows.shape[0]
    if Tm1 == 0:
        return np.zeros((1, flows.shape[1], flows.shape[2]), np.float32)
    mag = np.sqrt((flows ** 2).sum(-1) + 1e-8)
    H, W = mag.shape[1:]
    out = np.zeros((Tm1 + 1, H, W), np.float32)
    out[0] = mag[0]
    out[-1] = mag[-1]
    for t in range(1, Tm1):
        out[t] = 0.5 * (mag[t - 1] + mag[t])
    return out
