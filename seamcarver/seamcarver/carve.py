"""Apply seams to frames and maintain recoverable coordinate mappings.

A :class:`CarveProject` keeps the ORIGINAL clip untouched for its whole life.
Each resizing step appends a :class:`SeamStep` describing the seam in the
current geometry and whether it was removed or inserted.  Forward and inverse
mapping tables are rebuilt from the step log, so any intermediate geometry and
the original clip are always recoverable and object tracks can be mapped
between original and current coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeamStep:
    kind: str                  # "remove" | "insert"
    seams: np.ndarray          # (T,h) int columns in the geometry at this step
    width_before: int
    width_after: int
    label: str = ""


@dataclass
class CarveProject:
    original: np.ndarray                    # (T,H,W0,C) uint8, never mutated
    steps: list[SeamStep] = field(default_factory=list)

    @property
    def T(self) -> int:
        return self.original.shape[0]

    @property
    def H(self) -> int:
        return self.original.shape[1]

    @property
    def original_width(self) -> int:
        return self.original.shape[2]

    @property
    def width(self) -> int:
        w = self.original_width
        for s in self.steps:
            w += -1 if s.kind == "remove" else 1
        return w

    def current_frames(self) -> np.ndarray:
        frames = self.original
        for step in self.steps:
            frames = apply_seams(frames, step.seams, step.kind)
        return frames

    def forward_map(self) -> np.ndarray:
        """Original-column -> current-column per frame and row, shape (T,H,W0).

        Columns that were themselves removed collapse onto the last surviving
        column; inserted pixels share the source column's target position and
        the extra duplicate is reachable through :meth:`inverse_map`.
        """
        T, H, W0 = self.T, self.H, self.original_width
        m = np.broadcast_to(np.arange(W0), (T, H, W0)).copy().astype(np.int64)
        for step in self.steps:
            m = _advance_forward(m, step.seams, step.kind)
        width = self.width
        return np.clip(m, 0, width - 1)

    def inverse_map(self) -> np.ndarray:
        """Current-column -> original-column per frame and row, shape (T,H,width)."""
        T, H, W0 = self.T, self.H, self.original_width
        m = np.broadcast_to(np.arange(W0), (T, H, W0)).copy().astype(np.int64)
        for step in self.steps:
            m = _advance_inverse(m, step.seams, step.kind)
        return m

    def map_points_forward(self, points: np.ndarray) -> np.ndarray:
        """Map (T,K,2) x,y points from original coordinates to current geometry."""
        fmap = self.forward_map()
        pts = np.asarray(points).copy()
        for t in range(self.T):
            for k in range(pts.shape[1]):
                y = int(np.clip(round(pts[t, k, 1]), 0, self.H - 1))
                x = int(np.clip(round(pts[t, k, 0]), 0, self.original_width - 1))
                pts[t, k, 0] = fmap[t, y, x]
        return pts

    def restore_original(self) -> np.ndarray:
        return self.original.copy()


def apply_seams(frames: np.ndarray, seams: np.ndarray, kind: str) -> np.ndarray:
    """Remove or insert one vertical seam per frame.

    frames: (T,H,W,C) uint8, seams: (T,H) columns in current geometry.
    Vectorized with a boolean keep/duplicate mask over columns.
    """
    if kind not in ("remove", "insert"):
        raise ValueError(f"unknown seam kind {kind}")
    T, H, W, C = frames.shape
    if seams.shape != (T, H):
        raise ValueError(f"seams {seams.shape} != {(T, H)}")
    cols = np.arange(W)
    seam_mask = cols[None, None, :] == seams[:, :, None]      # (T,H,W)
    if kind == "remove":
        keep = ~seam_mask                                       # (T,H,W)
        out = frames[keep].reshape(T, H, W - 1, C)
        return out
    # Insertion: duplicate each seam pixel.  Build per-row source column
    # indices of length W+1, then gather.  The duplicate is softened by
    # averaging with its right neighbour.
    idx = np.broadcast_to(cols, (T, H, W)).copy()
    insert_at = np.clip(seams, 0, W - 1)
    # source index table: columns before seam unchanged, from seam onward the
    # new row is [seam, seam, seam+1, ...]
    src = np.empty((T, H, W + 1), dtype=np.int64)
    order = np.broadcast_to(np.arange(W + 1), (T, H, W + 1))
    src = np.where(order <= insert_at[:, :, None],
                   np.minimum(order, W - 1),
                   order - 1)
    gathered = np.take_along_axis(frames, src[:, :, :, None], axis=2)
    # soften only the duplicated seam column (the inserted copy at seam+1)
    dup_pos = insert_at + 1
    is_dup = order == dup_pos[:, :, None]
    right = np.take_along_axis(
        frames, np.clip(src, 0, W - 1)[:, :, :, None], axis=2)
    blended = (gathered.astype(np.uint16) // 2 + right.astype(np.uint16) // 2).astype(np.uint8)
    out = np.where(is_dup[:, :, :, None], blended, gathered)
    return out


def _advance_forward(m: np.ndarray, seams: np.ndarray, kind: str) -> np.ndarray:
    """Per-row original->current column mapping through one seam step.

    m has shape (T,H,W) holding the current target column of every original
    column.  Each row deletes (or inserts next to) its own seam column, so the
    shift is computed per row against ``seams[t,y]`` in one broadcast.
    """
    T, H, W = m.shape
    cols = np.broadcast_to(np.arange(W), (T, H, W))
    shift = (cols >= seams[:, :, None]).astype(np.int64)
    if kind == "remove":
        return m - shift
    return m + shift


def _advance_inverse(m: np.ndarray, seams: np.ndarray, kind: str) -> np.ndarray:
    """Per-row current->original column mapping through one seam step."""
    T, H, W = m.shape
    if kind == "remove":
        cols = np.broadcast_to(np.arange(W), (T, H, W))
        keep = cols != seams[:, :, None]
        return m[keep].reshape(T, H, W - 1)
    # Build source indices of length W+1: repeat the seam column.
    insert_at = np.clip(seams, 0, W - 1)
    order = np.broadcast_to(np.arange(W + 1), (T, H, W + 1))
    src = np.where(order <= insert_at[:, :, None],
                   np.minimum(order, W - 1),
                   order - 1)
    return np.take_along_axis(m, src, axis=2)
