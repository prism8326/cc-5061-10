"""Chunked, versioned storage for seam energy and seam paths.

The legacy layout kept one flat per-frame energy table and one flat seam table
for the whole clip; reading it required the whole table in memory and there was
no way to say "this frame/energy is stale" or "this neighbour was never
recorded".  :class:`SeamStore` instead splits a project into fixed-size frame
*chunks* plus explicit *boundary* records that carry the temporal adjacency
between the last frame of one chunk and the first frame of the next.

Design contract
---------------
* Every chunk and every boundary is content-hashed in ``manifest.json``.  A
  read that finds a missing file or a hash mismatch reports the gap in
  :class:`BlockRead.missing` — it never substitutes zeros, because a zero
  energy makes the seam prefer the gap and a zero temporal capacity silently
  severs correspondence.
* Per-frame records carry the full unary energy decomposition, the protected
  ("object") mask and a per-row ``lock`` column: ``-1`` means the row was
  auto-selected, ``>= 0`` is an author-locked column.  ``origin`` and
  ``provenance`` distinguish an author path from a cached auto path and a
  pixel-recomputed frame from one inherited from a legacy table.
* ``manifest.revision`` increases on every mutation so an exporter can prove a
  given seam path was produced against the energy it is claimed to use.

The manifest is plain JSON so it is inspectable without this package.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field

import numpy as np

FORMAT = "seamstore"
# 1 = legacy flat table (import source only); 2 = chunked graph store.
FORMAT_VERSION = 2

ORIGIN_AUTO = 0        # auto-selected, may be re-solved across chunks
ORIGIN_AUTHOR = 1      # author-locked rows, must survive migration
ORIGIN_IMPORTED = 2    # cached path inherited from a legacy table, never re-locked

PROV_INHERITED = 0     # energy copied from the legacy flat table
PROV_RECOMPUTED = 1    # energy recomputed from actual pixels

_ORIGIN_NAMES = {ORIGIN_AUTO: "auto", ORIGIN_AUTHOR: "author",
                 ORIGIN_IMPORTED: "imported"}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class FrameRecord:
    """One frame's energy/path/protection record."""
    index: int
    unary: np.ndarray          # (H,W) float32
    spatial: np.ndarray        # (H,W)
    motion: np.ndarray         # (H,W)
    occlusion: np.ndarray      # (H,W)
    protected: np.ndarray      # (H,W) bool, object-protection mask
    seam: np.ndarray           # (H,) int32 candidate path columns
    lock: np.ndarray           # (H,) int32, -1 free else author column
    origin: int = ORIGIN_AUTO
    provenance: int = PROV_RECOMPUTED

    def origin_name(self) -> str:
        return _ORIGIN_NAMES.get(self.origin, "?")


@dataclass
class BoundaryRecord:
    """Temporal adjacency for the gap after frame ``after_frame``."""
    after_frame: int
    gap_motion: np.ndarray     # (H,W) float32 forward-flow magnitude
    shot_cut: bool = False


@dataclass
class BlockRead:
    start: int
    stop: int
    frames: dict[int, FrameRecord]
    boundaries: dict[int, BoundaryRecord]
    missing: list[dict] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing

    def ordered_frames(self) -> list[FrameRecord]:
        return [self.frames[t] for t in sorted(self.frames)]


class SeamStore:
    """A chunked seam project.

    Use :meth:`create`/:meth:`from_energy` to build one.  The same object works
    in memory (``root=None``) or on disk (``root`` a directory); disk mode adds
    content hashing and manifest versioning so missing/stale data is always
    detectable.
    """

    def __init__(self, root: str | None, manifest: dict,
                 frames: dict[int, FrameRecord] | None = None,
                 boundaries: dict[int, BoundaryRecord] | None = None):
        self.root = root
        self.manifest = manifest
        self._frames: dict[int, FrameRecord] = frames or {}
        self._boundaries: dict[int, BoundaryRecord] = boundaries or {}
        self._dirty_chunks: set[int] = set()
        self._dirty_boundaries: set[int] = set()

    # ---------------------------------------------------------------- coords
    @property
    def T(self) -> int:
        return int(self.manifest["T"])

    @property
    def H(self) -> int:
        return int(self.manifest["H"])

    @property
    def W(self) -> int:
        return int(self.manifest["W"])

    @property
    def chunk_size(self) -> int:
        return int(self.manifest["chunk_size"])

    @property
    def revision(self) -> int:
        return int(self.manifest["revision"])

    @property
    def n_chunks(self) -> int:
        return max(1, (self.T + self.chunk_size - 1) // self.chunk_size)

    def chunk_of(self, index: int) -> int:
        return index // self.chunk_size

    # --------------------------------------------------------------- create
    @classmethod
    def create(cls, root: str | None, T: int, H: int, W: int,
               chunk_size: int = 8, created_from: str = "fresh",
               generation: str | None = None) -> "SeamStore":
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        manifest = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "created_from": created_from,
            "chunk_size": int(chunk_size),
            "T": int(T), "H": int(H), "W": int(W),
            "revision": 1,
            "generation": generation or uuid.uuid4().hex,
            "chunks": [],
            "boundaries": [],
        }
        if root is not None:
            os.makedirs(root, exist_ok=True)
        return cls(root, manifest)

    @classmethod
    def from_energy(cls, unary: np.ndarray, seam: np.ndarray,
                    spatial: np.ndarray | None = None,
                    motion: np.ndarray | None = None,
                    occlusion: np.ndarray | None = None,
                    protected: np.ndarray | None = None,
                    lock: np.ndarray | None = None,
                    origin: int | np.ndarray = ORIGIN_AUTO,
                    provenance: int | np.ndarray = PROV_RECOMPUTED,
                    gap_motion: np.ndarray | None = None,
                    shot_cuts: np.ndarray | None = None,
                    chunk_size: int = 8,
                    root: str | None = None,
                    created_from: str = "fresh") -> "SeamStore":
        """Build an in-memory (or on-disk) store from already-computed tensors.

        ``origin``/``provenance`` may be a scalar or a per-frame array so a
        re-sync can carry author locks forward.
        """
        T, H, W = unary.shape
        origin_arr = (np.full(T, int(origin), np.uint8) if np.isscalar(origin)
                      else np.asarray(origin, np.uint8))
        prov_arr = (np.full(T, int(provenance), np.uint8) if np.isscalar(provenance)
                    else np.asarray(provenance, np.uint8))
        store = cls.create(root, T, H, W, chunk_size=chunk_size,
                           created_from=created_from)
        z = np.zeros((H, W), np.float32)
        for t in range(T):
            store._frames[t] = FrameRecord(
                index=t,
                unary=np.asarray(unary[t], np.float32),
                spatial=(z.copy() if spatial is None else np.asarray(spatial[t], np.float32)),
                motion=(z.copy() if motion is None else np.asarray(motion[t], np.float32)),
                occlusion=(z.copy() if occlusion is None else np.asarray(occlusion[t], np.float32)),
                protected=(np.zeros((H, W), bool) if protected is None else np.asarray(protected[t], bool)),
                seam=np.asarray(seam[t], np.int32),
                lock=(np.full(H, -1, np.int32) if lock is None else np.asarray(lock[t], np.int32)),
                origin=int(origin_arr[t]),
                provenance=int(prov_arr[t]),
            )
        if gap_motion is not None:
            for t in range(min(T - 1, gap_motion.shape[0])):
                store._boundaries[t] = BoundaryRecord(
                    after_frame=t,
                    gap_motion=np.asarray(gap_motion[t], np.float32),
                    shot_cut=bool(shot_cuts[t]) if shot_cuts is not None and t < len(shot_cuts) else False,
                )
        elif shot_cuts is not None:
            for t in range(T - 1):
                store._boundaries[t] = BoundaryRecord(
                    after_frame=t, gap_motion=np.zeros((H, W), np.float32),
                    shot_cut=bool(shot_cuts[t]))
        store._dirty_chunks = set(range(store.n_chunks))
        store._dirty_boundaries = set(store._boundaries)
        if root is not None:
            store.flush()
        return store

    # ---------------------------------------------------------------- mutate
    def set_frame(self, index: int, **kwargs) -> None:
        """Patch fields of one frame record (keys match :class:`FrameRecord`)."""
        rec = self._frames[index]
        for key, val in kwargs.items():
            if not hasattr(rec, key):
                raise AttributeError(key)
            setattr(rec, key, val)
        self._dirty_chunks.add(self.chunk_of(index))

    def set_boundary(self, after_frame: int, gap_motion: np.ndarray | None = None,
                     shot_cut: bool | None = None) -> None:
        rec = self._boundaries.get(after_frame)
        if gap_motion is not None:
            if rec is None:
                H, W = self.H, self.W
                rec = BoundaryRecord(after_frame, np.zeros((H, W), np.float32))
                self._boundaries[after_frame] = rec
            rec.gap_motion = np.asarray(gap_motion, np.float32)
        if shot_cut is not None and rec is not None:
            rec.shot_cut = bool(shot_cut)
        self._dirty_boundaries.add(after_frame)

    def delete_boundary(self, after_frame: int) -> None:
        """Drop the temporal adjacency record for a gap (simulates a lost edge)."""
        self._boundaries.pop(after_frame, None)
        self._dirty_boundaries.add(after_frame)

    def bump_revision(self) -> None:
        self.manifest["revision"] = int(self.manifest["revision"]) + 1

    # --------------------------------------------------------------- read io
    def read_block(self, start: int, stop: int) -> BlockRead:
        """Read the half-open frame range ``[start, stop)`` including boundaries.

        On disk, each overlapping chunk is hash-checked; unreadable chunks are
        listed in ``missing`` and their frames are absent from ``frames``.
        """
        start = max(0, int(start))
        stop = min(self.T, int(stop))
        frames: dict[int, FrameRecord] = {}
        boundaries: dict[int, BoundaryRecord] = {}
        missing: list[dict] = []

        if self.root is None or not os.path.isfile(os.path.join(self.root, "manifest.json")):
            for t in range(start, stop):
                if t in self._frames:
                    frames[t] = self._frames[t]
            for t in range(start, stop):
                if t in self._boundaries:
                    boundaries[t] = self._boundaries[t]
            return BlockRead(start, stop, frames, boundaries, missing)

        # Disk mode: read each overlapping chunk that the manifest lists.
        listed = {c["id"]: c for c in self.manifest.get("chunks", [])}
        first_cid = self.chunk_of(start)
        last_cid = self.chunk_of(stop - 1) if stop > start else first_cid - 1
        for cid in range(first_cid, last_cid + 1):
            entry = listed.get(cid)
            if entry is None:
                missing.append({"kind": "chunk", "id": cid,
                                "reason": "no manifest entry"})
                continue
            path = os.path.join(self.root, entry["file"])
            if not os.path.isfile(path):
                missing.append({"kind": "chunk", "id": cid, "file": entry["file"],
                                "reason": "chunk file missing"})
                continue
            if _sha256(path) != entry.get("sha256"):
                missing.append({"kind": "chunk", "id": cid, "file": entry["file"],
                                "reason": "chunk hash mismatch (stale)"})
                continue
            with np.load(path) as npz:
                for k in range(int(entry["n"])):
                    t = int(entry["start"]) + k
                    if not (start <= t < stop):
                        continue
                    frames[t] = _record_from_arrays(npz, k)
                    self._frames[t] = frames[t]

        listed_b = {b["after_frame"]: b for b in self.manifest.get("boundaries", [])}
        for t in range(start, stop):
            if t >= self.T - 1:
                continue
            entry = listed_b.get(t)
            if entry is None:
                missing.append({"kind": "boundary", "after_frame": t,
                                "reason": "no boundary record"})
                continue
            path = os.path.join(self.root, entry["file"])
            if not os.path.isfile(path) or _sha256(path) != entry.get("sha256"):
                missing.append({"kind": "boundary", "after_frame": t,
                                "file": entry["file"], "reason": "boundary missing or stale"})
                continue
            with np.load(path) as npz:
                boundaries[t] = BoundaryRecord(
                    after_frame=t,
                    gap_motion=np.asarray(npz["gap_motion"], np.float32),
                    shot_cut=bool(npz["shot_cut"]),
                )
                self._boundaries[t] = boundaries[t]
        return BlockRead(start, stop, frames, boundaries, missing)

    def iter_blocks(self):
        for cid in range(self.n_chunks):
            s = cid * self.chunk_size
            yield self.read_block(s, min(s + self.chunk_size, self.T))

    def frame(self, index: int) -> FrameRecord | None:
        return self.read_block(index, index + 1).frames.get(index)

    # ----------------------------------------------------------------- write
    def flush(self) -> str:
        """Persist dirty chunks/boundaries and rewrite the manifest atomically."""
        if self.root is None:
            self.bump_revision()
            return "<memory>"
        os.makedirs(self.root, exist_ok=True)
        chunks = {c["id"]: c for c in self.manifest.get("chunks", [])}
        for cid in sorted(self._dirty_chunks):
            s = cid * self.chunk_size
            e = min(s + self.chunk_size, self.T)
            recs = [self._frames[t] for t in range(s, e) if t in self._frames]
            if not recs:
                continue
            name = f"chunk_{cid:04d}.npz"
            _write_chunk(os.path.join(self.root, name), recs)
            chunks[cid] = {"id": cid, "start": s, "stop": e, "n": len(recs),
                           "file": name, "sha256": _sha256(os.path.join(self.root, name))}
        boundaries = {b["after_frame"]: b for b in self.manifest.get("boundaries", [])}
        for t in sorted(self._dirty_boundaries):
            rec = self._boundaries.get(t)
            name = f"boundary_{t:04d}.npz"
            path = os.path.join(self.root, name)
            if rec is None:
                if os.path.isfile(path):
                    os.remove(path)
                boundaries.pop(t, None)
                continue
            np.savez_compressed(path, gap_motion=rec.gap_motion.astype(np.float32),
                                shot_cut=np.array(rec.shot_cut))
            boundaries[t] = {"after_frame": t, "file": name,
                             "sha256": _sha256(path),
                             "shot_cut": bool(rec.shot_cut)}
        self.manifest["chunks"] = [chunks[k] for k in sorted(chunks)]
        self.manifest["boundaries"] = [boundaries[k] for k in sorted(boundaries)]
        self.bump_revision()
        tmp = os.path.join(self.root, "manifest.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(self.manifest, fh, indent=2)
        os.replace(tmp, os.path.join(self.root, "manifest.json"))
        self._dirty_chunks.clear()
        self._dirty_boundaries.clear()
        return os.path.join(self.root, "manifest.json")

    # ----------------------------------------------------------------- verify
    def verify(self) -> list[dict]:
        """Full integrity scan: return every missing/stale chunk or boundary."""
        missing: list[dict] = []
        if self.root is None:
            for t in range(self.T):
                if t not in self._frames:
                    missing.append({"kind": "chunk", "index": t,
                                    "reason": "frame absent from memory store"})
            return missing
        listed = {c["id"] for c in self.manifest.get("chunks", [])}
        for cid in range(self.n_chunks):
            if cid not in listed:
                missing.append({"kind": "chunk", "id": cid,
                                "reason": "no manifest entry"})
        for c in self.manifest.get("chunks", []):
            path = os.path.join(self.root, c["file"])
            if not os.path.isfile(path):
                missing.append({"kind": "chunk", "id": c["id"],
                                "reason": "file missing"})
            elif _sha256(path) != c["sha256"]:
                missing.append({"kind": "chunk", "id": c["id"],
                                "reason": "hash mismatch"})
        for b in self.manifest.get("boundaries", []):
            path = os.path.join(self.root, b["file"])
            if not os.path.isfile(path) or _sha256(path) != b["sha256"]:
                missing.append({"kind": "boundary", "after_frame": b["after_frame"],
                                "reason": "boundary missing or stale"})
        return missing


# --------------------------------------------------------------------- codecs
def _record_from_arrays(npz, k: int) -> FrameRecord:
    return FrameRecord(
        index=int(npz["index"][k]),
        unary=np.asarray(npz["unary"][k], np.float32),
        spatial=np.asarray(npz["spatial"][k], np.float32),
        motion=np.asarray(npz["motion"][k], np.float32),
        occlusion=np.asarray(npz["occlusion"][k], np.float32),
        protected=np.asarray(npz["protected"][k], bool),
        seam=np.asarray(npz["seam"][k], np.int32),
        lock=np.asarray(npz["lock"][k], np.int32),
        origin=int(npz["origin"][k]),
        provenance=int(npz["provenance"][k]),
    )


def _write_chunk(path: str, recs: list[FrameRecord]) -> None:
    np.savez_compressed(
        path,
        index=np.array([r.index for r in recs], np.int32),
        unary=np.stack([r.unary for r in recs]).astype(np.float32),
        spatial=np.stack([r.spatial for r in recs]).astype(np.float32),
        motion=np.stack([r.motion for r in recs]).astype(np.float32),
        occlusion=np.stack([r.occlusion for r in recs]).astype(np.float32),
        protected=np.stack([r.protected for r in recs]).astype(bool),
        seam=np.stack([r.seam for r in recs]).astype(np.int32),
        lock=np.stack([r.lock for r in recs]).astype(np.int32),
        origin=np.array([r.origin for r in recs], np.uint8),
        provenance=np.array([r.provenance for r in recs], np.uint8),
    )


def load_store(root: str) -> SeamStore:
    """Open an existing store and validate manifest format/version."""
    path = os.path.join(root, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no seamstore manifest at {path}")
    with open(path) as fh:
        manifest = json.load(fh)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"not a seamstore: format={manifest.get('format')!r}")
    if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
        raise ValueError(
            f"store format_version {manifest.get('format_version')} is newer than "
            f"this reader ({FORMAT_VERSION})")
    return SeamStore(root, manifest)
