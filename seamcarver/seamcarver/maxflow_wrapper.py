"""Thin Python wrapper around the compiled bkflow C extension.

The grid graph is a 3D volume T(frames) x H(rows) x W(cols) with one extra
source and sink node.  Every node carries unary source/sink capacities; every
spatial/temporal neighbor pair carries one undirected pairwise capacity.
"""
from __future__ import annotations

import numpy as np

from . import bkflow


def grid_mincut(
    term_src: np.ndarray,
    term_sink: np.ndarray,
    cap_h: np.ndarray,
    cap_v: np.ndarray,
    cap_t: np.ndarray,
):
    """Compute a min-cut on the grid.

    Arrays are float32 with shapes:
      term_src / term_sink : (T, H, W)
      cap_h : (T, H, W-1), cap_v : (T, H-1, W), cap_t : (T-1, H, W)
    Returns labels (uint8 (T,H,W), 0=source side / 1=sink side) and flow.
    """
    T, H, W = term_src.shape
    for name, a, shape in (
        ("term_src", term_src, (T, H, W)),
        ("term_sink", term_sink, (T, H, W)),
        ("cap_h", cap_h, (T, H, W - 1)),
        ("cap_v", cap_v, (T, H - 1, W)),
        ("cap_t", cap_t, (max(T - 1, 0), H, W)),
    ):
        if a.shape != shape:
            raise ValueError(f"{name} shape {a.shape} != {shape}")

    labels, flow = bkflow.grid_mincut(
        int(W), int(H), int(T),
        np.ascontiguousarray(term_src, np.float32).reshape(-1),
        np.ascontiguousarray(term_sink, np.float32).reshape(-1),
        np.ascontiguousarray(cap_h, np.float32).reshape(-1),
        np.ascontiguousarray(cap_v, np.float32).reshape(-1),
        np.ascontiguousarray(cap_t, np.float32).reshape(-1) if T > 1
        else np.zeros(0, np.float32),
    )
    return labels.reshape(T, H, W).astype(np.uint8), float(flow)
