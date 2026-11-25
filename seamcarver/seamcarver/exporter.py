"""Export a carved clip while proving which segments were re-read.

The exporter never silently inherits old energy: it walks a
:class:`~seamcarver.reconstruct.ReconstructionReport`, carves every frame with
the reconstructed seam, and records a manifest segment that says exactly which
frames could not have their cross-frame relationship verified (missing temporal
boundary, stale chunk, etc.).  Those frames are flagged ``reread`` so a consumer
maintaining cached carved frames falls back to the source clip for them instead
of reusing the migrated store's stale adjacency.

The result is a small, inspectable dict (``segments``) plus optional on-disk
video.  ``must_reread`` lists only frames that genuinely lacked trust; frames
covered by a verified reconstructed seam are reported as ``verified``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .carve import apply_seams
from .reconstruct import ReconstructionReport


@dataclass
class ExportResult:
    frames: np.ndarray                 # (T,H,W-1,C) uint8 after seam removal
    segments: list[dict]               # re-read / verified runs with reasons
    must_reread: list[int]
    verified: list[int]
    used_revision: int
    applied_path: str                  # human note proving the new path was used

    def public(self) -> dict:
        return {
            "segments": self.segments,
            "must_reread": self.must_reread,
            "verified_frames": len(self.verified),
            "used_revision": self.used_revision,
            "applied_path": self.applied_path,
        }


def export_with_reread(source: np.ndarray, report: ReconstructionReport,
                       store_revision: int,
                       out_path: str | None = None,
                       fps: float = 12.0) -> ExportResult:
    """Apply the reconstructed seam, annotating segments that must be re-read.

    ``source`` is the pristine pixel clip at the current geometry.  Every frame
    is carved with the reconstructed path (never the legacy cached path), so the
    pixels always come from ``source``.  Frames in ``report.reread_frames`` are
    additionally marked in the returned ``segments``/``must_reread`` because
    their cross-frame relationship could not be verified (a missing temporal
    boundary or stale chunk); a downstream consumer that maintains its own cached
    carved frames must fall back to re-reading those spans from source rather
    than inheriting the old energy.
    """
    T, H, W = source.shape[:3]
    if report.seams.shape != (T, H):
        raise ValueError(f"seam shape {report.seams.shape} != {(T, H)}")
    if store_revision != report.revision:
        raise ValueError(
            f"export revision {store_revision} != reconstruction revision "
            f"{report.revision}: path was built against different energy")

    # Every frame uses the reconstructed path (never the legacy cached path).
    carved = apply_seams(source, report.seams.astype(np.int32), "remove")

    reread = set(report.reread_frames)
    verified = [t for t in range(T) if t not in reread]
    segments: list[dict] = []

    # Contiguous verified / reread runs so the UI can point at exact spans.
    t = 0
    while t < T:
        is_reread = t in reread
        j = t
        while j + 1 < T and ((j + 1) in reread) == is_reread:
            j += 1
        if is_reread:
            reason = "missing temporal boundary / stale chunk - re-read from source clip"
        else:
            reason = "reconstructed seam verified against energy"
        segments.append({"start": t, "stop": j + 1, "kind":
                         "reread" if is_reread else "verified", "reason": reason})
        t = j + 1

    if out_path:
        from .video_io import write_video
        write_video(out_path, carved, fps=fps)

    return ExportResult(
        frames=carved, segments=segments,
        must_reread=sorted(reread), verified=verified,
        used_revision=report.revision,
        applied_path=f"reconstructed({report.strategy}) rev {report.revision}",
    )
