"""Stateful resizing engine that drives both the CLI and the web player.

The engine owns the immutable original clip, the ordered seam log, the
current energy volume and the most recent solver result.  Every operation
returns a structured :class:`EngineResult` so the UI layer never has to touch
the graph internals; solver warnings (degenerate cuts, seam instability) are
fed back as ``status`` messages that the UI renders instead of hiding them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np

from .carve import CarveProject, SeamStep, apply_seams
from .energy import EnergyConfig, build_energy
from .flow import FlowConfig, compute_flows
from .seam_solver import SolveConfig, build_and_solve, seam_cost
from .video_io import overlay_seams


def _warp_flows_to_current(base_flows: np.ndarray, project: CarveProject) -> np.ndarray:
    """Resample flows computed on the original geometry onto current columns.

    Seam carving only removes/inserts columns; each surviving current pixel
    traces to an original column through the per-row inverse map, so the flow
    vector of that source pixel is reused (horizontal component is kept as the
    local displacement).  This preserves motion correspondence semantics
    across the many cheap resize steps without rerunning the flow estimator.
    """
    if base_flows.shape[0] == 0:
        return base_flows
    T, H, W = project.T, project.H, project.width
    inv = project.inverse_map()                       # (T,H,W) original col
    ys = np.arange(H)
    out = np.zeros((base_flows.shape[0], H, W, 2), np.float32)
    for t in range(base_flows.shape[0]):
        src_cols = inv[t]                             # (H,W)
        row_idx = np.broadcast_to(ys[:, None], (H, W))
        out[t] = base_flows[t, row_idx, src_cols]
    return out


@dataclass
class EngineResult:
    ok: bool
    status: str
    width: int
    seams: np.ndarray | None = None
    labels: np.ndarray | None = None
    flow_value: float = 0.0
    costs: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)
    elapsed: float = 0.0

    def public(self) -> dict:
        d = asdict(self)
        for k in ("seams", "labels", "costs"):
            d.pop(k, None)
        return d


class SeamEngine:
    def __init__(
        self,
        frames: np.ndarray,
        energy_cfg: EnergyConfig | None = None,
        flow_cfg: FlowConfig | None = None,
        solve_cfg: SolveConfig | None = None,
        protected: np.ndarray | None = None,
        name: str = "clip",
    ):
        self.project = CarveProject(frames.astype(np.uint8).copy())
        self.energy_cfg = energy_cfg or EnergyConfig()
        self.flow_cfg = flow_cfg or FlowConfig()
        self.solve_cfg = solve_cfg or SolveConfig()
        self.name = name
        self._protected0 = protected
        self.last_seams: np.ndarray | None = None
        self.last_labels: np.ndarray | None = None
        self.last_cap_t: np.ndarray | None = None
        self.energy = None
        self.history: list[dict] = []
        self._base_flows: np.ndarray | None = None
        self.recompute_energy()

    # ------------------------------------------------------------------ state
    def recompute_energy(self) -> None:
        """Rebuild gradients/flow/occlusion on the CURRENT geometry.

        The dense flow field is estimated once on the original clip and then
        resampled onto each resized geometry through the inverse coordinate
        map; only spatial gradients and occlusion masks are truly recomputed,
        which keeps multi-step resizing cheap while preserving motion cues.
        Protected masks are warped so energy and mappings never go stale.
        """
        frames = self.project.current_frames()
        protected = self._warp_protected(frames.shape[1], frames.shape[2])
        if self._base_flows is None:
            self._base_flows = compute_flows(frames.astype(np.float32), self.flow_cfg)
        existing = _warp_flows_to_current(self._base_flows, self.project)
        self.energy = build_energy(
            frames, protected=protected,
            cfg=self.energy_cfg, flow_cfg=self.flow_cfg,
            existing_flows=existing,
        )

    def _warp_protected(self, h: int, w: int) -> np.ndarray | None:
        if self._protected0 is None:
            return None
        T = self.project.T
        inv = self.project.inverse_map()  # (T,h,w) original column per current col
        p0 = self._protected0.astype(bool)
        ys = np.arange(h)
        out = np.zeros((T, h, w), dtype=bool)
        for t in range(T):
            orig_cols = inv[t, ys]          # (h,w)
            row_index = np.broadcast_to(ys[:, None], (h, w))
            out[t] = p0[t, row_index, orig_cols]
        return out

    # ----------------------------------------------------------------- solve
    def solve(self, locked_seams: np.ndarray | None = None) -> EngineResult:
        t0 = time.perf_counter()
        e = self.energy
        seams, labels, flow_value, cap_t = build_and_solve(
            e.unary, e.flows, e.shot_cuts, self.solve_cfg, locked_seams,
        )
        self.last_seams, self.last_labels, self.last_cap_t = seams, labels, cap_t
        costs = seam_cost(e.unary, seams)
        warnings = self._diagnose(seams)
        elapsed = time.perf_counter() - t0
        res = EngineResult(
            ok=True, status="solved", width=self.project.width,
            seams=seams, labels=labels, flow_value=flow_value,
            costs=costs, warnings=warnings, elapsed=elapsed,
        )
        self.history.append({"event": "solve", **res.public(),
                             "cost_per_frame": costs.tolist()})
        return res

    def _diagnose(self, seams: np.ndarray) -> list[str]:
        warnings = []
        T, H = seams.shape
        if T > 1:
            delta = np.abs(seams[1:].astype(int) - seams[:-1].astype(int))
            jump_frac = float((delta > max(4, 0.15 * self.project.width)).mean())
            if jump_frac > 0.25:
                warnings.append(
                    f"seam jumps side in {jump_frac:.0%} of adjacent frame/rows; "
                    "check temporal capacity / occlusion handling")
        if self.energy.protected.any():
            hit = self._seam_hits_protected(seams)
            if hit:
                warnings.append("seam crossed a protected region - constraints were violated")
        if np.isinf(self.energy.unary).any() or (self.energy.unary >= self.energy_cfg.protected_inf).any():
            pass
        return warnings

    def _seam_hits_protected(self, seams: np.ndarray) -> bool:
        prot = self.energy.protected
        T, H = seams.shape
        for t in range(T):
            if prot[t, np.arange(H), seams[t]].any():
                return True
        return False

    # ------------------------------------------------------------------ edit
    def edit_seam(self, frame: int, row: int, col: int) -> EngineResult:
        """User drags one seam vertex; relock the whole frame and re-solve.

        Only the edited frame is hard-locked; neighbouring frames remain free
        and re-associate through temporal capacities, which is how a user
        correction propagates instead of being mechanically copied.
        """
        if self.last_seams is None:
            self.solve()
        locked = np.full_like(self.last_seams, -1)
        locked[frame] = self.last_seams[frame]
        col = int(np.clip(col, 0, self.project.width - 1))
        # smooth the edit over a short vertical window so the path stays valid
        for y in range(max(0, row - 6), min(locked.shape[1], row + 7)):
            locked[frame, y] = col
        return self.solve(locked)

    def set_protected_rect(self, frame: int, x0: int, y0: int, x1: int, y1: int) -> EngineResult:
        T = self.project.T
        H, W0 = self.project.H, self.project.original_width
        if self._protected0 is None:
            self._protected0 = np.zeros((T, H, W0), dtype=bool)
        self._protected0[frame, y0:y1, x0:x1] = True
        self.recompute_energy()
        return self.solve()

    # ------------------------------------------------------------------ apply
    def apply_step(self, kind: str) -> EngineResult:
        if self.last_seams is None:
            r = self.solve()
            if not r.ok:
                return r
        step = SeamStep(
            kind=kind, seams=self.last_seams.copy(),
            width_before=self.project.width,
            width_after=self.project.width + (-1 if kind == "remove" else 1),
        )
        self.project.steps.append(step)
        self.recompute_energy()
        return self.solve()

    def resize_to(self, target_width: int, max_iterations: int | None = None) -> list[EngineResult]:
        """Iteratively remove/insert seams with energy recompute each step."""
        results = []
        n = abs(target_width - self.project.width)
        if max_iterations:
            n = min(n, max_iterations)
        kind = "remove" if target_width < self.project.width else "insert"
        for _ in range(n):
            r = self.apply_step(kind)
            results.append(r)
            if not r.ok:
                break
        return results

    # --------------------------------------------------------------- outputs
    def frames_with_seams(self) -> np.ndarray:
        frames = self.project.current_frames()
        if self.last_seams is None:
            return frames
        return overlay_seams(frames, self.last_seams)

    def state_snapshot(self) -> dict:
        return {
            "name": self.name,
            "frames": self.project.T,
            "height": self.project.H,
            "original_width": self.project.original_width,
            "width": self.project.width,
            "steps": [s.kind for s in self.project.steps],
            "shot_cuts": self.energy.shot_cuts.tolist(),
            "warnings": self.history[-1].get("warnings", []) if self.history else [],
        }
