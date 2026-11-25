"""Reconstruct seam paths across frames from the chunked energy store.

This is the migration target: the legacy flat per-frame energy table is turned
into a *graph* that can backtrack across frames.  Each temporal gap becomes an
edge whose capacity comes from the stored forward-flow magnitude; a seam is
then chosen jointly so adjacent frames agree instead of each frame being solved
in isolation.

Two strategies are implemented because the task asks to compare them and pick
the better one:

``lazy``
    Stream chunks in order and only add a temporal edge once both endpoint
    energy chunks happen to be loaded.  Nothing is prepared ahead of time, so
    the solve restarts at every chunk boundary.

``import_rebuild``
    Read the whole finite-horizon cost graph up front from the chunks and solve
    it once, so temporal coupling survives chunk boundaries.

:func:`compare_strategies` runs both and returns the measured seam flicker.
``strategy='auto'`` measures once per project (keyed by geometry, chunking and
the boundary structure) and reuses that selection on later calls, so the
interactive edit path does not pay for a fresh comparison on every drag.

``horizon`` bounds how far cross-frame backtracking reaches.  ``0`` (the
default) means the full finite clip, so dependency is truncated **only** at
shot cuts, exactly as the migration contract requires.  A positive value severs
the chain every ``horizon`` gaps as a deliberate truncation; it is honestly
reported as extra flicker rather than silently preferred.

Invariants enforced here
------------------------
* Shot cuts sever the temporal dependency: the run is truncated at the cut and
  the two sides are solved independently (a cut is *legitimately* independent).
* A **missing** boundary is not a shot cut.  It must never be treated as a
  zero-capacity edge and silently inherited; instead the adjacent frames are
  marked ``reread`` and excluded from the verified set.  The exporter surfaces
  the corresponding segments so they are re-read from the original clip.
* A ``lock`` row (``>= 0``) pins the seam at that column; author locks always
  win, matching the migration contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import os

import numpy as np

from .seam_solver import SolveConfig, build_and_solve, temporal_capacities
from .seam_store import SeamStore


@dataclass
class ReconstructionReport:
    seams: np.ndarray                      # (T,H) int32
    flicker: dict                          # measured seam flicker
    reread_frames: list[int]               # frames whose seam must be re-read
    reread_segments: list[dict]            # merged runs for the exporter
    shot_cuts: np.ndarray                  # (T-1,) bool used
    missing_boundaries: list[int]          # gaps with no stored adjacency
    strategy: str
    revision: int
    horizon: int
    verified: np.ndarray                   # (T,) bool, seam trustworthy as-is
    locked_rows: int = 0

    def public(self) -> dict:
        return {
            "strategy": self.strategy,
            "revision": self.revision,
            "horizon": self.horizon,
            "flicker": self.flicker,
            "reread_frames": self.reread_frames,
            "reread_segments": self.reread_segments,
            "missing_boundaries": self.missing_boundaries,
            "shot_cuts": self.shot_cuts.tolist(),
            "locked_rows": self.locked_rows,
            "verified_frames": int(self.verified.sum()),
        }


def _assemble(store: SeamStore, strategy: str):
    """Read the store and return the tensors reconstruction needs.

    ``lazy`` only keeps one chunk of temporal edges at a time (the caller solves
    each chunk on its own).  ``import_rebuild`` returns a single assembled graph.
    Missing boundaries are reported, never fabricated.
    """
    T, H, W = store.T, store.H, store.W
    unary = np.full((T, H, W), np.nan, np.float32)
    lock = np.full((T, H), -1, np.int32)
    origin = np.full(T, 0, np.uint8)
    present = np.zeros(T, bool)
    missing_frames: list[int] = []

    for block in store.iter_blocks():
        for t, rec in block.frames.items():
            unary[t] = rec.unary
            lock[t] = rec.lock
            origin[t] = rec.origin
            present[t] = True
    # A frame is untrustworthy on its own only when it could not be read at all;
    # a missing *boundary* affects just the two frames it joins, which are added
    # below.  Flagging every frame of a chunk that lost one boundary would
    # over-report re-read segments.
    missing_frames = [t for t in range(T) if not present[t]]

    gap_motion = np.zeros((max(T - 1, 0), H, W), np.float32)
    shot = np.zeros(max(T - 1, 0), bool)
    broken = np.zeros(max(T - 1, 0), bool)   # severed because adjacency is unknown
    missing_bounds: list[int] = []
    for t in range(T - 1):
        b = store._boundaries.get(t)
        if b is None:
            # Try disk-backed lookup through a small read.
            blk = store.read_block(t, t + 1)
            b = blk.boundaries.get(t)
        if b is None:
            missing_bounds.append(t)
            # A missing boundary must NOT be inherited as a strong (zero-motion)
            # coupling; that would silently reuse the old relationship.  Sever
            # it so the two sides are decided independently and the exporter
            # can flag them for re-read.
            broken[t] = True
            continue
        gap_motion[t] = b.gap_motion
        shot[t] = b.shot_cut
    if broken.any():
        gap_motion[broken] = 0.0
    return (unary, lock, origin, present, missing_frames, gap_motion, shot,
            broken, missing_bounds)


def _solve_with_horizon(unary, gap_motion, shot, broken, lock, cfg, horizon):
    """Solve the 3D graph over the whole clip, optionally breaking every ``horizon`` gaps.

    ``horizon=0`` (default) keeps the full finite clip joined, so dependency is
    truncated only at shot cuts and at ``broken`` gaps whose adjacency is
    unknown.  A positive horizon additionally severs the chain every ``horizon``
    gaps; that is a deliberate truncation, not a way to hide a discontinuity.
    Shot cuts and ``broken`` gaps are severed via the solver's shot flags but
    remain reported separately.  Locks are applied as hard s/t pins.
    """
    T, H, W = unary.shape
    # build_and_solve derives its own temporal capacities from the flow, so the
    # ONLY way to sever a gap through the public solver is via the shot flags.
    # Merge broken (missing-adjacency) gaps into the solve flags but keep them
    # out of the reported shot_cuts so the two remain distinguishable.
    sever = shot | broken
    if horizon > 0:
        for t in range(T - 1):
            if (t + 1) % horizon == 0:
                sever[t] = True
    fake_flow = np.zeros((max(T - 1, 0), H, W, 2), np.float32)
    if T > 1:
        fake_flow[..., 0] = gap_motion
    seams, labels, flow_value, _ = build_and_solve(
        unary, fake_flow, sever, cfg, locked_seams=lock)
    cap_t = temporal_capacities(unary, gap_motion, sever, cfg) if T > 1 else np.zeros((0, H, W), np.float32)
    return seams, labels, flow_value, cap_t


def _solve_lazy(store, cfg, horizon):
    """Per-chunk independent solve: temporal edges exist only inside a chunk."""
    T, H, W = store.T, store.H, store.W
    (unary, lock, origin, present, missing_frames, gap_motion, shot, broken,
     missing_bounds) = _assemble(store, "lazy")
    seams = np.full((T, H), W - 1, np.int32)
    for cid in range(store.n_chunks):
        s = cid * store.chunk_size
        e = min(s + store.chunk_size, T)
        sub_u = unary[s:e]
        sub_g = gap_motion[s:e - 1] if e - 1 > s else np.zeros((0, H, W), np.float32)
        sub_shot = shot[s:e - 1] if e - 1 > s else np.zeros(0, bool)
        sub_broken = broken[s:e - 1] if e - 1 > s else np.zeros(0, bool)
        sub_lock = lock[s:e]
        if np.isnan(sub_u).any():
            continue
        fake = np.zeros((max(e - 1 - s, 0), H, W, 2), np.float32)
        if e - 1 > s:
            fake[..., 0] = sub_g
        # Sever broken (missing-adjacency) gaps through the shot flags, which
        # build_and_solve honours; report shot/broken separately upstream.
        sub_sever = sub_shot | sub_broken
        sub_seams, _, _, _ = build_and_solve(sub_u, fake, sub_sever, cfg,
                                             locked_seams=sub_lock)
        seams[s:e] = sub_seams
    reread = sorted(set(missing_frames) | {t for g in missing_bounds for t in (g, g + 1) if t < T})
    return seams, reread, missing_bounds, shot, origin


def _solve_import(store, cfg, horizon):
    T, H, W = store.T, store.H, store.W
    (unary, lock, origin, present, missing_frames, gap_motion, shot, broken,
     missing_bounds) = _assemble(store, "import")
    if np.isnan(unary).any():
        bad = np.isnan(unary).any(axis=(1, 2))
        # Fall back: fill unreadable frames with a large constant so they are
        # never preferred; they are flagged for re-read below.
        unary = np.where(np.isnan(unary), np.float32(1e5), unary).astype(np.float32)
        missing_frames = sorted(set(missing_frames) | set(np.flatnonzero(bad).tolist()))
    seams, _, _, _ = _solve_with_horizon(unary, gap_motion, shot, broken, lock,
                                         cfg, horizon)
    reread = sorted(set(missing_frames) | {t for g in missing_bounds for t in (g, g + 1) if t < T})
    return seams, reread, missing_bounds, shot, origin


def seam_flicker(seams: np.ndarray, width: int | None = None) -> dict:
    """Quantify seam flicker in the exported trajectory.

    * ``mean_step``   : mean per-row column change between adjacent frames;
    * ``max_step``    : worst adjacent-frame mean change;
    * ``side_flips``  : adjacent-frame swings larger than 25% of the width,
      which a human reads as the seam "jumping to the other side";
    * ``flicker_index``: ``mean_step`` normalised by width, the headline number.

    ``width`` should be the actual frame width; when omitted it is inferred from
    the largest seam column (an underestimate if the seam never reaches the
    right wall, which would inflate the index).
    """
    T, H = seams.shape
    if T < 2:
        return {"mean_step": 0.0, "max_step": 0.0, "side_flips": 0, "flicker_index": 0.0}
    step = np.abs(seams[1:].astype(np.int64) - seams[:-1].astype(np.int64)).mean(axis=1)
    w = int(width) if width else max(int(seams.max()) + 1, 1)
    return {
        "mean_step": float(step.mean()),
        "max_step": float(step.max()),
        "side_flips": int((step > 0.25 * w).sum()),
        "flicker_index": float(step.mean() / w),
    }


def _merge_runs(frames: list[int], T: int) -> list[dict]:
    if not frames:
        return []
    frames = sorted(set(frames))
    segs, start, prev = [], frames[0], frames[0]
    for t in frames[1:]:
        if t == prev + 1:
            prev = t
            continue
        segs.append({"start": start, "stop": prev + 1, "reason": "missing-boundary"})
        start = prev = t
    segs.append({"start": start, "stop": prev + 1, "reason": "missing-boundary"})
    return segs


_STRATEGY_CACHE: dict[tuple, str] = {}


def _strategy_fingerprint(store: SeamStore, horizon: int) -> tuple:
    """Content fingerprint of the things that decide which strategy is smoother.

    Only the temporal adjacency structure matters: frame geometry, chunking,
    which gaps exist, their motion magnitude and shot flags.  The seam/energy
    revision is deliberately excluded, so an edit that changes the seam does not
    force a fresh comparison on every drag (that would triple the interactive
    latency).
    """
    import hashlib

    h = hashlib.sha256()
    h.update(f"{store.T}|{store.H}|{store.W}|{store.chunk_size}|{horizon}".encode())
    listed = {int(b["after_frame"]): (b.get("sha256"), bool(b.get("shot_cut")))
              for b in store.manifest.get("boundaries", [])}
    disk = (store.root is not None
            and os.path.isfile(os.path.join(store.root, "manifest.json")))
    for t in range(store.T - 1):
        rec = store._boundaries.get(t)
        if rec is not None:
            h.update(f"b{t}".encode())
            h.update(np.asarray(rec.gap_motion, np.float32).tobytes())
            h.update(b"S" if rec.shot_cut else b"-")
        elif disk and t in listed:
            h.update(f"d{t}:{listed[t][0]}:{int(listed[t][1])}".encode())
        else:
            h.update(f"x{t}".encode())      # missing adjacency
    return h.hexdigest()


def chosen_strategy(store: SeamStore, cfg: SolveConfig | None = None,
                    horizon: int = 0) -> str:
    """Measured-better strategy for this store's structure (cached)."""
    key = _strategy_fingerprint(store, horizon)
    if key not in _STRATEGY_CACHE:
        if len(_STRATEGY_CACHE) > 64:
            _STRATEGY_CACHE.clear()
        _STRATEGY_CACHE[key] = compare_strategies(
            store, cfg, horizon=horizon)["chosen"]
    return _STRATEGY_CACHE[key]


def reconstruct(store: SeamStore, cfg: SolveConfig | None = None,
                horizon: int = 0, strategy: str = "auto") -> ReconstructionReport:
    """Rebuild seam paths across frames from the chunked store.

    ``strategy='auto'`` picks the smoother implementation (see
    :func:`chosen_strategy`); pass ``'lazy'`` or ``'import_rebuild'`` to force
    one.
    """
    cfg = cfg or SolveConfig()
    if strategy == "auto":
        strategy = chosen_strategy(store, cfg, horizon=horizon)
    if strategy == "lazy":
        seams, reread, missing, shot, origin = _solve_lazy(store, cfg, horizon)
    elif strategy == "import_rebuild":
        seams, reread, missing, shot, origin = _solve_import(store, cfg, horizon)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    T, H = seams.shape
    verified = np.ones(T, bool)
    verified[reread] = False
    locks = 0
    for cid in range(store.n_chunks):
        s = cid * store.chunk_size
        e = min(s + store.chunk_size, T)
        blk = store.read_block(s, e)
        for rec in blk.frames.values():
            locks += int((rec.lock >= 0).sum())
    return ReconstructionReport(
        seams=seams,
        flicker=seam_flicker(seams, width=store.W),
        reread_frames=reread,
        reread_segments=_merge_runs(reread, T),
        shot_cuts=shot,
        missing_boundaries=missing,
        strategy=strategy,
        revision=store.revision,
        horizon=horizon,
        verified=verified,
        locked_rows=locks,
    )


def compare_strategies(store: SeamStore, cfg: SolveConfig | None = None,
                       horizon: int = 0) -> dict:
    """Run lazy vs import-rebuild and report measured flicker for each."""
    cfg = cfg or SolveConfig()
    out = {}
    for strategy in ("lazy", "import_rebuild"):
        rep = reconstruct(store, cfg, horizon=horizon, strategy=strategy)
        out[strategy] = {
            "flicker": rep.flicker,
            "reread_frames": rep.reread_frames,
            "missing_boundaries": rep.missing_boundaries,
            "discontinuity_at_chunk_edges": _chunk_edge_discontinuity(store, rep.seams),
        }
    lazy_f = out["lazy"]["flicker"]["flicker_index"]
    imp_f = out["import_rebuild"]["flicker"]["flicker_index"]
    out["chosen"] = "import_rebuild" if imp_f <= lazy_f else "lazy"
    out["reason"] = (
        f"import_rebuild flicker_index={imp_f:.4f} vs lazy={lazy_f:.4f} "
        f"(lower is smoother; chunk-edge discontinuity "
        f"import={out['import_rebuild']['discontinuity_at_chunk_edges']:.2f} "
        f"lazy={out['lazy']['discontinuity_at_chunk_edges']:.2f})"
    )
    return out


def _chunk_edge_discontinuity(store: SeamStore, seams: np.ndarray) -> float:
    """Mean seam change exactly at chunk boundaries (the seams the lazy path breaks)."""
    vals = []
    cs = store.chunk_size
    for cid in range(1, store.n_chunks):
        t = cid * cs
        if 0 < t < seams.shape[0]:
            vals.append(float(np.abs(seams[t].astype(int) - seams[t - 1].astype(int)).mean()))
    return float(np.mean(vals)) if vals else 0.0
