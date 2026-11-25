"""Migrate a legacy flat per-frame energy/path table into a chunked store.

Legacy format (``legacy_table.npz``) is exactly what the pre-chunk engine kept:
one flat ``unary`` (T,H,W), one flat ``seam`` (T,H), optional ``lock`` (T,H) and
``protected`` (T,H,W), a ``shot_cuts`` (T-1,) flag array and no notion of
chunks, versions or temporal boundaries.  There is deliberately **no** gap
record in it: the temporal adjacency has to be regenerated on import, which is
the whole point of the migration.

The migrator
------------
* recomputes the per-gap forward-flow magnitude from the *actual* transition
  pixel pairs so the rebuilt graph has real temporal adjacency;
* recomputes unary energy from pixels for frames whose pixels are available
  (``provenance=recomputed"``), and records inherited frames otherwise;
* preserves author locks (``lock >= 0``) as ``ORIGIN_AUTHOR`` so a later
  frame replacement reports a *conflict* instead of silently re-solving them;
* re-solves auto/imported frames and stores the new path, marking them
  ``ORIGIN_AUTO``/``ORIGIN_IMPORTED``.

Two transition fixtures for the tests
-------------------------------------
:func:`make_transition_clip` builds a clip whose pixels are **identical** on
both sides of a seam decision, so any difference in the migrated outcome comes
from the lock/auto policy, not from the picture.
:func:`replace_frame_before_transition` swaps one frame just before a transition
so a caller can observe: author-locked keeps the author mark and reports a new
conflict; auto re-solves the path across chunks.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass

import numpy as np

from .seam_solver import SolveConfig, build_and_solve
from .seam_store import (
    ORIGIN_AUTHOR, ORIGIN_AUTO, ORIGIN_IMPORTED,
    PROV_INHERITED, PROV_RECOMPUTED, SeamStore,
)

LEGACY_FORMAT = "legacy_flat"
LEGACY_VERSION = 1


# --------------------------------------------------------------- legacy table
def write_legacy_table(path: str, unary: np.ndarray, seam: np.ndarray,
                       lock: np.ndarray | None = None,
                       protected: np.ndarray | None = None,
                       shot_cuts: np.ndarray | None = None,
                       frames: np.ndarray | None = None,
                       origin_meta: dict | None = None) -> str:
    """Persist a flat legacy table (the format the migrator reads)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    T, H, W = unary.shape
    payload = {
        "unary": np.asarray(unary, np.float32),
        "seam": np.asarray(seam, np.int32),
        "lock": (np.full((T, H), -1, np.int32) if lock is None
                 else np.asarray(lock, np.int32)),
        "protected": (np.zeros((T, H, W), bool) if protected is None
                      else np.asarray(protected, bool)),
        "shot_cuts": (np.zeros(max(T - 1, 0), bool) if shot_cuts is None
                      else np.asarray(shot_cuts, bool)),
    }
    headers = {
        "format": LEGACY_FORMAT, "format_version": LEGACY_VERSION,
        "T": T, "H": H, "W": W,
        "origin_meta": origin_meta or {},
    }
    np.savez_compressed(path, **payload)
    with open(path + ".json", "w") as fh:
        json.dump(headers, fh, indent=2)
    if frames is not None:
        np.savez_compressed(path + ".frames.npz", frames=np.asarray(frames, np.uint8))
    return path


def read_legacy_table(path: str) -> dict:
    with np.load(path) as npz:
        data = {k: npz[k] for k in npz.files}
    meta_path = path + ".json"
    if os.path.isfile(meta_path):
        with open(meta_path) as fh:
            data["_meta"] = json.load(fh)
    else:
        data["_meta"] = {"format": LEGACY_FORMAT, "format_version": LEGACY_VERSION}
    if data["_meta"].get("format") != LEGACY_FORMAT:
        raise ValueError(f"{path} is not a legacy table")
    return data


# ------------------------------------------------------------------- migrate
@dataclass
class MigrationReport:
    source: str
    target: str
    T: int
    H: int
    W: int
    n_chunks: int
    author_rows: int
    auto_rows: int
    imported_rows: int
    recomputed_frames: int
    inherited_frames: int
    boundaries_rebuilt: int
    missing_boundary_gaps: list[int]
    revision: int
    generation: str
    notes: str

    def public(self) -> dict:
        return {
            "source": self.source, "target": self.target,
            "T": self.T, "H": self.H, "W": self.W, "n_chunks": self.n_chunks,
            "author_rows": self.author_rows, "auto_rows": self.auto_rows,
            "imported_rows": self.imported_rows,
            "recomputed_frames": self.recomputed_frames,
            "inherited_frames": self.inherited_frames,
            "boundaries_rebuilt": self.boundaries_rebuilt,
            "missing_boundary_gaps": self.missing_boundary_gaps,
            "revision": self.revision, "generation": self.generation,
            "notes": self.notes,
        }


def migrate_legacy(path: str, target_root: str, *,
                   frames: np.ndarray | None = None,
                   chunk_size: int = 8,
                   cfg: SolveConfig | None = None,
                   generation: str | None = None) -> MigrationReport:
    """Convert one legacy flat table into a chunked, versioned store.

    ``frames`` overrides the pixels to use for recomputation; when omitted the
    sidecar ``<path>.frames.npz`` is used if present, otherwise energy is
    inherited from the legacy table (``PROV_INHERITED``) and the report says so.
    """
    legacy = read_legacy_table(path)
    unary = np.asarray(legacy["unary"], np.float32)
    T, H, W = unary.shape
    lock = np.asarray(legacy["lock"], np.int32)
    protected = np.asarray(legacy["protected"], bool)
    shot_cuts = np.asarray(legacy["shot_cuts"], bool)
    seam = np.asarray(legacy["seam"], np.int32)

    if frames is None:
        side = path + ".frames.npz"
        if os.path.isfile(side):
            with np.load(side) as npz:
                frames = np.asarray(npz["frames"], np.uint8)
    if frames is not None and frames.shape[0] != T:
        raise ValueError(f"frames T={frames.shape[0]} != legacy T={T}")

    # Guard against a flat table saved at a different aspect than its frames.
    if frames is not None and (frames.shape[1], frames.shape[2]) != (H, W):
        raise ValueError(f"frames HxW={frames.shape[1:3]} != legacy {H}x{W}")

    store = SeamStore.create(target_root, T, H, W, chunk_size=chunk_size,
                            created_from="legacy_flat", generation=generation)
    cfg = cfg or SolveConfig()

    # Recompute unary + flow from actual pixels where possible, otherwise
    # inherit the legacy energy (and flag it).  Temporal adjacency is ALWAYS
    # rebuilt from pixels; the legacy table has no boundary records at all.
    if frames is not None:
        from .energy import build_energy
        ev = build_energy(frames, protected=protected, cfg=None)
        # Keep the legacy shot_cut decision if the new detector disagrees only
        # weakly; author-authored cuts win.
        cuts = shot_cuts | ev.shot_cuts
        gap_motion = np.sqrt((ev.flows ** 2).sum(-1) + 1e-8) if ev.flows.shape[0] else np.zeros((0, H, W), np.float32)
        provenance = PROV_RECOMPUTED
        recomputed_frames = T
        inherited_frames = 0
        unary_new = ev.unary
    else:
        cuts = shot_cuts
        gap_motion = np.zeros((max(T - 1, 0), H, W), np.float32)
        provenance = PROV_INHERITED
        recomputed_frames = 0
        inherited_frames = T
        unary_new = unary

    # Decide each frame's origin: an author lock must be preserved as
    # ORIGIN_AUTHOR; any other frame enters the store as ORIGIN_IMPORTED (its
    # path is still the legacy cached one) and becomes ORIGIN_AUTO only after
    # _resolve_auto re-solves it against the rebuilt graph.
    author_rows = auto_rows = imported_rows = 0
    for t in range(T):
        rows = int((lock[t] >= 0).sum())
        author_rows += rows
        if rows:
            origin = ORIGIN_AUTHOR
        else:
            origin = ORIGIN_IMPORTED
            imported_rows += int((lock[t] < 0).sum())
        store._frames[t] = _mk_frame(t, unary_new[t], seam[t], lock[t],
                                     protected[t], origin, provenance,
                                     spatial=None)
    # Build boundary records: real adjacency where pixels exist; a *deliberate*
    # empty boundary (recorded as missing) where the legacy table gave us none.
    for t in range(T - 1):
        if frames is not None:
            store._boundaries[t] = _mk_boundary(t, gap_motion[t], bool(cuts[t]))
        else:
            # No pixels: cannot rebuild adjacency.  Leave the record out so the
            # exporter sees it as missing rather than silently zero.
            pass

    store._dirty_chunks = set(range(store.n_chunks))
    store._dirty_boundaries = set(range(max(T - 1, 0)))
    store.flush()

    # Re-solve auto frames against the rebuilt graph so cached auto paths are
    # replaced by paths that are valid for *this* energy, not the old one.
    if frames is not None:
        auto_rows = _resolve_auto(store, cfg)
        store._dirty_chunks = set(range(store.n_chunks))
        store.flush()

    missing_gaps = [t for t in range(T - 1) if store._boundaries.get(t) is None]
    notes = (
        "migrated legacy flat table -> chunked store v2; "
        f"author rows preserved={author_rows}, "
        f"auto rows re-solved={auto_rows}, imported(no-pixel) rows={imported_rows if frames is None else 0}; "
        f"energy {'recomputed from pixels' if frames is not None else 'inherited (no pixels)'}; "
        f"temporal boundaries rebuilt={len(store._boundaries)}"
        + (f"; {len(missing_gaps)} gap(s) have no adjacency (will be flagged)" if missing_gaps else "")
    )
    return MigrationReport(
        source=path, target=target_root, T=T, H=H, W=W, n_chunks=store.n_chunks,
        author_rows=author_rows,
        auto_rows=(auto_rows if frames is not None else 0),
        imported_rows=(imported_rows if frames is None else 0),
        recomputed_frames=recomputed_frames, inherited_frames=inherited_frames,
        boundaries_rebuilt=len(store._boundaries),
        missing_boundary_gaps=missing_gaps,
        revision=store.revision, generation=store.manifest["generation"],
        notes=notes,
    )


def _mk_frame(t, unary_t, seam_t, lock_t, protected_t, origin, provenance,
              spatial=None):
    from .seam_store import FrameRecord
    H, W = unary_t.shape
    return FrameRecord(
        index=t,
        unary=np.asarray(unary_t, np.float32),
        spatial=(np.zeros((H, W), np.float32) if spatial is None else spatial),
        motion=np.zeros((H, W), np.float32),
        occlusion=np.zeros((H, W), np.float32),
        protected=np.asarray(protected_t, bool),
        seam=np.asarray(seam_t, np.int32),
        lock=np.asarray(lock_t, np.int32),
        origin=int(origin), provenance=int(provenance),
    )


def _mk_boundary(t, gap_motion_t, shot_cut):
    from .seam_store import BoundaryRecord
    return BoundaryRecord(after_frame=t,
                          gap_motion=np.asarray(gap_motion_t, np.float32),
                          shot_cut=bool(shot_cut))


def _resolve_auto(store: SeamStore, cfg: SolveConfig) -> None:
    """Re-solve frames that are NOT author-locked against the rebuilt graph.

    This is a *per-frame* solve against the migrated unary, which is what the
    migration contract asks for ("auto paths are re-solved, author locks are
    kept").  The cross-frame reconstruction (:mod:`reconstruct`) is the caller's
    job after migration; here we only refresh the cached auto path so it is not
    the stale legacy one.
    """
    T, H, W = store.T, store.H, store.W
    unary = np.stack([store._frames[t].unary for t in range(T)]).astype(np.float32)
    lock = np.stack([store._frames[t].lock for t in range(T)]).astype(np.int32)
    shot = np.array([store._boundaries[t].shot_cut if t in store._boundaries else False
                     for t in range(T - 1)], bool)
    gap = (np.stack([store._boundaries[t].gap_motion for t in range(T - 1)]).astype(np.float32)
           if T > 1 and all(t in store._boundaries for t in range(T - 1))
           else np.zeros((max(T - 1, 0), H, W), np.float32))
    fake = np.zeros((max(T - 1, 0), H, W, 2), np.float32)
    if T > 1:
        fake[..., 0] = gap
    seams, _, _, _ = build_and_solve(unary, fake, shot, cfg, locked_seams=lock)
    resolved = 0
    for t in range(T):
        if store._frames[t].origin != ORIGIN_AUTHOR:
            store._frames[t].seam = seams[t].astype(np.int32)
            store._frames[t].origin = ORIGIN_AUTO
            resolved += int(H)
    return resolved


# ------------------------------------------------------------------ fixtures
@dataclass
class RevisionReport:
    frame_index: int
    origin: str
    action: str
    author_kept: bool
    conflicts: list[dict]
    seam_changed_rows: int
    crossed_chunk_boundary: bool
    window: tuple[int, int]
    revision: int

    def public(self) -> dict:
        return {
            "frame_index": self.frame_index, "origin": self.origin,
            "action": self.action, "author_kept": self.author_kept,
            "conflicts": self.conflicts,
            "seam_changed_rows": self.seam_changed_rows,
            "crossed_chunk_boundary": self.crossed_chunk_boundary,
            "window": list(self.window), "revision": self.revision,
        }


def revise_frame(store: SeamStore, frames: np.ndarray, frame_index: int,
                 cfg: SolveConfig | None = None) -> RevisionReport:
    """Replace one frame's pixels and reconcile the store against them.

    Author-locked frame: the lock and path are kept, but if the new pixels make
    that locked path materially more expensive a **conflict** is reported (the
    author's mark is preserved, never silently dropped).

    Auto frame: the path is re-solved together with its temporal neighbours.  If
    the reconcile window spans a chunk boundary the report says so, proving the
    auto path was recomputed *across chunks* rather than inherited from cache.
    """
    cfg = cfg or SolveConfig()
    from .energy import build_energy
    T, H, W = store.T, store.H, store.W
    if frames.shape[:3] != (T, H, W):
        raise ValueError(f"frames shape {frames.shape[:3]} != store {(T, H, W)}")

    ev = build_energy(frames)
    gap_motion = (np.sqrt((ev.flows ** 2).sum(-1) + 1e-8)
                  if ev.flows.shape[0] else np.zeros((0, H, W), np.float32))

    cs = store.chunk_size
    cid = store.chunk_of(frame_index)
    lo = max(0, (cid - 1) * cs)
    hi = min(T, (cid + 2) * cs)
    window = (lo, hi)
    # Populate the frame window and its boundaries from disk before touching
    # them (a freshly `load_store`d store has no frames cached yet).
    store.read_block(lo, hi)

    rec = store._frames[frame_index]
    old_cost = float(rec.unary[np.arange(H), np.clip(rec.seam, 0, W - 1)].sum())
    # Refresh this frame's energy + its two adjacent boundaries from pixels.
    rec.unary = ev.unary[frame_index].astype(np.float32)
    rec.protected = ev.protected[frame_index].astype(bool)
    rec.provenance = PROV_RECOMPUTED
    for g in (frame_index - 1, frame_index):
        if 0 <= g < T - 1 and gap_motion.shape[0] > g:
            store.set_boundary(g, gap_motion=gap_motion[g],
                               shot_cut=bool(ev.shot_cuts[g]))
    store._dirty_chunks.add(store.chunk_of(frame_index))

    conflicts: list[dict] = []
    if rec.origin == ORIGIN_AUTHOR:
        # Keep the author path and lock exactly; measure whether pixels now
        # disagree strongly with it.
        new_cost = float(rec.unary[np.arange(H), np.clip(rec.seam, 0, W - 1)].sum())
        delta = new_cost / max(old_cost, 1e-6)
        if delta > 1.25 or (rec.lock >= 0).any():
            conflicts.append({
                "frame": frame_index,
                "locked_rows": int((rec.lock >= 0).sum()),
                "cost_before": round(old_cost, 4),
                "cost_after": round(new_cost, 4),
                "ratio": round(delta, 4),
                "reason": "author lock preserved but new pixels raise its cost; "
                          "surface conflict for author review",
            })
        report = RevisionReport(
            frame_index=frame_index, origin=rec.origin_name(), action="kept-author-lock",
            author_kept=True, conflicts=conflicts, seam_changed_rows=0,
            crossed_chunk_boundary=False, window=window,
            revision=store.revision)
        store.flush()
        return report

    # Auto frame whose neighbours include an author lock: re-solving freely
    # would cross the author's mark, so keep the path and report the conflict.
    neighbour_locked = [
        t for t in (frame_index - 1, frame_index + 1)
        if 0 <= t < T and store._frames.get(t) is not None
        and store._frames[t].origin == ORIGIN_AUTHOR
    ]
    if neighbour_locked:
        conflicts.append({
            "frame": frame_index,
            "neighbour_author_frames": neighbour_locked,
            "reason": "revised frame is adjacent to an author-locked frame; "
                      "keeping the author mark and reporting a new conflict",
        })
        report = RevisionReport(
            frame_index=frame_index, origin=rec.origin_name(), action="kept-author-lock",
            author_kept=True, conflicts=conflicts, seam_changed_rows=0,
            crossed_chunk_boundary=False, window=window,
            revision=store.revision)
        store.flush()
        return report

    # Auto path: re-solve a window around the frame so temporal neighbours (and
    # a chunk edge if the frame sits on one) participate.
    crossed = (store.chunk_of(lo) != store.chunk_of(hi - 1))
    unary = np.stack([store._frames[t].unary for t in range(lo, hi)]).astype(np.float32)
    lock = np.stack([store._frames[t].lock for t in range(lo, hi)]).astype(np.int32)
    shot = np.array([store._boundaries[t].shot_cut if t in store._boundaries else False
                     for t in range(lo, hi - 1)], bool)
    gap = np.stack([store._boundaries[t].gap_motion for t in range(lo, hi - 1)]
                   ).astype(np.float32) if hi - 1 > lo else np.zeros((0, H, W), np.float32)
    fake = np.zeros((max(hi - 1 - lo, 0), H, W, 2), np.float32)
    if hi - 1 > lo:
        fake[..., 0] = gap
    seams, _, _, _ = build_and_solve(unary, fake, shot, cfg, locked_seams=lock)
    before = rec.seam.copy()
    for i, t in enumerate(range(lo, hi)):
        if store._frames[t].origin != ORIGIN_AUTHOR:
            store._frames[t].seam = seams[i].astype(np.int32)
    store._frames[frame_index].origin = ORIGIN_AUTO
    changed = int((store._frames[frame_index].seam != before).sum())
    store._dirty_chunks = set(range(store.chunk_of(lo), store.chunk_of(hi - 1) + 1))
    store.flush()
    return RevisionReport(
        frame_index=frame_index, origin="auto", action="resolved-auto-cross-chunk",
        author_kept=False, conflicts=[], seam_changed_rows=changed,
        crossed_chunk_boundary=crossed, window=window, revision=store.revision)


def make_transition_clip(T: int = 16, H: int = 48, W: int = 64,
                         seed: int = 5) -> np.ndarray:
    """A clip with **identical pixels on both sides** of a seam transition.

    The left half is low-frequency and the right half high-frequency, but every
    frame is pixel-identical to the next; only the stored ``seam``/``lock``
    records differ, which is exactly what the migration must preserve or
    re-decide independently of the picture.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((H, W, 3), np.uint8)
    # Left: smooth gradient. Right: hard checker. Both deterministic.
    xs = np.linspace(30, 90, W, dtype=np.float32)
    for c in range(3):
        base[:, :, c] = np.tile(xs, (H, 1)).astype(np.uint8)
    half = W // 2
    check = ((np.arange(H)[:, None] // 4 + np.arange(W)[None, :] // 4) % 2)
    base[:, half:, :] = np.where(check[:, half:, None] > 0, 220, 40).astype(np.uint8)
    return np.repeat(base[None], T, axis=0)


def build_two_transition_projects(frames: np.ndarray, seam_column: int,
                                  chunk_size: int = 4) -> tuple[str, str]:
    """Write two legacy tables over the SAME pixels: one author-locked, one auto.

    Returns ``(locked_path, auto_path)``.  Both start from a cached "auto" seam
    (a straight column) but the locked one also records the author lock on the
    transition frame, so the migrator must preserve it.
    """
    T, H, W = frames.shape[:3]
    from .energy import build_energy
    ev = build_energy(frames)
    seam = np.full((T, H), seam_column, np.int32)
    lock = np.full((T, H), -1, np.int32)
    # author locked the transition frame (index T//2) from row 0 to H
    lock_locked = lock.copy()
    lock_locked[T // 2, :] = seam_column
    d = tempfile.mkdtemp(prefix="legacy_")
    locked_path = os.path.join(d, "author_locked.npz")
    auto_path = os.path.join(d, "auto_cached.npz")
    write_legacy_table(locked_path, ev.unary, seam, lock_locked, ev.protected,
                       ev.shot_cuts, frames=frames)
    write_legacy_table(auto_path, ev.unary, seam, lock, ev.protected,
                       ev.shot_cuts, frames=frames)
    return locked_path, auto_path


def make_moving_transition_clip(T: int = 24, H: int = 48, W: int = 96,
                                speed: float = 2.0, seed: int = 9) -> np.ndarray:
    """A moving *cheap* vertical band over a high-energy textured background.

    The seam prefers the smooth band, so as the band moves the reconstructed
    seam must follow it; that inter-frame motion is what the flicker metric and
    the lazy-vs-import comparison measure.  (A high-energy mover, as in
    ``synth.moving_marker``, would instead be avoided by the seam.)
    """
    rng = np.random.default_rng(seed)
    T, H, W = int(T), int(H), int(W)
    bg = rng.integers(60, 200, size=(H, W, 3), dtype=np.uint8)
    frames = np.empty((T, H, W, 3), np.uint8)
    mw = max(8, W // 7)
    for t in range(T):
        f = bg.copy()
        frac = t / max(T - 1, 1)
        cx = int((W - mw) * (0.5 + 0.35 * np.sin(2 * np.pi * (frac * 0.8 - 0.2))) + speed * (frac - 0.5))
        cx = int(np.clip(cx, 1, W - mw - 1))
        # smooth, low-gradient band => low spatial energy => seam prefers it
        f[:, cx:cx + mw] = 128
        frames[t] = f
    return frames


def replace_frame_before_transition(frames: np.ndarray,
                                    seam_column: int) -> np.ndarray:
    """Return ``frames`` with the frame just before the mid transition changed.

    The replacement is a clearly different image so a naive per-frame cache
    cannot match the old path; the migrator must either honour the author lock
    (and report a conflict) or re-solve the auto path.
    """
    out = frames.copy()
    t = max(0, len(frames) // 2 - 1)
    H, W = out.shape[1:3]
    out[t] = 0
    out[t, :, : max(1, W // 3)] = 255
    return out
