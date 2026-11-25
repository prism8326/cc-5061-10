"""Tests for the chunked energy/path store, legacy migration and cross-frame
reconstruction.

Run directly: ``python tests/test_migration.py`` (no third-party runner needed).

Covers the migration contract from the task:
* legacy flat table -> chunked, versioned store with temporal boundaries;
* identical-pixel transition fixtures: author-locked vs cached-auto;
* replacing the frame before the transition: author lock survives + conflict
  reported, auto path re-solved across chunks;
* deleting a chunk-edge boundary: exporter flags the segment for re-read and
  never treats the gap as zero cost;
* import-rebuild beats per-chunk lazy solve on seam flicker;
* shot cuts truncate the temporal dependency; a missing neighbour is NOT a cut;
* engine edit + export genuinely use the reconstructed path and a new revision.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seamcarver.seam_store import (
    SeamStore, load_store, ORIGIN_AUTHOR, FORMAT_VERSION,
)
from seamcarver.migration import (
    write_legacy_table, read_legacy_table, migrate_legacy, revise_frame,
    build_two_transition_projects, make_transition_clip,
    make_moving_transition_clip, replace_frame_before_transition,
)
from seamcarver.reconstruct import (
    reconstruct, compare_strategies, seam_flicker,
)
from seamcarver.exporter import export_with_reread


def _tmp():
    return tempfile.mkdtemp(prefix="seamtest_")


def test_chunk_roundtrip_and_versioning():
    T, H, W = 10, 6, 8
    rng = np.random.default_rng(0)
    unary = rng.random((T, H, W)).astype(np.float32)
    seam = rng.integers(0, W, (T, H)).astype(np.int32)
    gap = rng.random((T - 1, H, W)).astype(np.float32)
    cuts = np.zeros(T - 1, bool)
    root = os.path.join(_tmp(), "store")
    st = SeamStore.from_energy(unary, seam, gap_motion=gap, shot_cuts=cuts,
                               chunk_size=3, root=root)
    assert st.verify() == []
    # reload from disk
    st2 = load_store(root)
    blk = st2.read_block(0, T)
    assert blk.complete and not blk.missing
    assert len(blk.frames) == T
    for t in range(T):
        assert np.allclose(blk.frames[t].unary, unary[t])
        assert np.array_equal(blk.frames[t].seam, seam[t])
    assert st2.manifest["format_version"] == FORMAT_VERSION
    # flush() is the single owner of the revision counter
    rev = st2.revision
    st2.set_frame(0, seam=seam[0].copy())
    st2.flush()
    assert load_store(root).revision == rev + 1
    print("ok test_chunk_roundtrip_and_versioning rev", rev)


def test_missing_chunk_is_reported_not_zeroed():
    T, H, W = 9, 5, 6
    unary = np.ones((T, H, W), np.float32)
    seam = np.zeros((T, H), np.int32)
    root = os.path.join(_tmp(), "store")
    st = SeamStore.from_energy(unary, seam, chunk_size=3, root=root)
    # delete a chunk file without touching the manifest
    victim = os.path.join(root, "chunk_0001.npz")
    assert os.path.isfile(victim)
    os.remove(victim)
    blk = st.read_block(0, T)
    assert not blk.complete
    assert any(m["kind"] == "chunk" for m in blk.missing)
    # frames from the missing chunk are ABSENT, not zero-filled
    assert len(blk.frames) < T
    assert st.verify()  # non-empty
    print("ok test_missing_chunk_is_reported_not_zeroed", blk.missing)


def test_author_lock_preserved_and_conflict_reported():
    frames = make_transition_clip(T=16, H=48, W=64)
    locked_path, _ = build_two_transition_projects(frames, seam_column=32,
                                                   chunk_size=4)
    root = os.path.join(_tmp(), "L")
    rep = migrate_legacy(locked_path, root, frames=frames, chunk_size=4)
    assert rep.author_rows > 0, "author lock must survive migration"
    st = load_store(root)
    mid = 8
    rec = st.read_block(mid, mid + 1).frames[mid]
    assert rec.origin == ORIGIN_AUTHOR
    assert (rec.lock >= 0).all(), "author lock rows preserved verbatim"
    # replace the frame just before the transition
    frames2 = replace_frame_before_transition(frames, 32)
    rrep = revise_frame(st, frames2, mid - 1)
    assert rrep.author_kept, "author mark must be kept, not dropped"
    assert rrep.conflicts, "a new conflict must be reported"
    assert any("author" in c["reason"] for c in rrep.conflicts)
    print("ok test_author_lock_preserved_and_conflict_reported", rrep.action)


def test_auto_path_resolves_across_chunks():
    frames = make_transition_clip(T=16, H=48, W=64)
    _, auto_path = build_two_transition_projects(frames, seam_column=32,
                                                 chunk_size=4)
    root = os.path.join(_tmp(), "A")
    rep = migrate_legacy(auto_path, root, frames=frames, chunk_size=4)
    assert rep.author_rows == 0, "no author locks in the auto project"
    st = load_store(root)
    mid = 8
    assert st.read_block(mid, mid + 1).frames[mid].origin != ORIGIN_AUTHOR
    frames2 = replace_frame_before_transition(frames, 32)
    rrep = revise_frame(st, frames2, mid - 1)
    assert not rrep.author_kept
    assert rrep.action == "resolved-auto-cross-chunk"
    assert rrep.crossed_chunk_boundary, "auto path must re-solve across chunks"
    print("ok test_auto_path_resolves_across_chunks window", rrep.window)


def test_import_rebuild_beats_lazy_on_chunk_edges():
    """A globally cheap corridor + chunk-local traps: the per-chunk lazy solve
    flickers at chunk edges; the import-time rebuild holds a steady seam."""
    T, H, W = 20, 24, 48
    cs = 4
    corridor = W // 2
    unary = np.ones((T, H, W), np.float32)
    unary[:, :, corridor] = 0.10
    offsets = [-4, 4, -4, 4, -4]
    for k in range((T + cs - 1) // cs):
        s, e = k * cs, min((k + 1) * cs, T)
        unary[s:e, :, corridor + offsets[k % len(offsets)]] = 0.06
    seam = np.full((T, H), corridor, np.int32)
    gap = np.zeros((T - 1, H, W), np.float32)   # static => strong coupling
    store = SeamStore.from_energy(unary, seam, gap_motion=gap,
                                  shot_cuts=np.zeros(T - 1, bool),
                                  chunk_size=cs, root=os.path.join(_tmp(), "s"))
    cmp = compare_strategies(store, horizon=99)
    assert cmp["chosen"] == "import_rebuild"
    lazy_f = cmp["lazy"]["flicker"]["flicker_index"]
    imp_f = cmp["import_rebuild"]["flicker"]["flicker_index"]
    assert imp_f < lazy_f
    assert cmp["lazy"]["discontinuity_at_chunk_edges"] > 0
    assert cmp["import_rebuild"]["discontinuity_at_chunk_edges"] == 0
    print("ok test_import_rebuild_beats_lazy_on_chunk_edges",
          f"lazy_flicker={lazy_f:.4f} import_flicker={imp_f:.4f}")


def test_missing_boundary_flagged_not_zero_inherited():
    frames = make_moving_transition_clip(T=20, H=32, W=64)
    locked_path, _ = build_two_transition_projects(frames, seam_column=32,
                                                   chunk_size=5)
    root = os.path.join(_tmp(), "M")
    migrate_legacy(locked_path, root, frames=frames, chunk_size=5)
    st = load_store(root)
    st.read_block(0, st.T)
    st.delete_boundary(9)
    st.flush()
    rep = reconstruct(st, horizon=10)
    assert 9 in rep.missing_boundaries, "deleted boundary must be reported"
    assert 9 in rep.reread_frames and 10 in rep.reread_frames
    assert not rep.verified[9] and not rep.verified[10], "gap frames not trusted"
    ex = export_with_reread(frames, rep, st.revision)
    kinds = [s["kind"] for s in ex.segments]
    assert "reread" in kinds, "exporter must surface a re-read segment"
    reread_seg = next(s for s in ex.segments if s["kind"] == "reread")
    assert reread_seg["start"] <= 9 < reread_seg["stop"]
    # the exporter must NEVER silently zero-inherit the gap
    assert 9 in ex.must_reread and 10 in ex.must_reread
    print("ok test_missing_boundary_flagged_not_zero_inherited",
          ex.must_reread)


def test_missing_gap_severs_coupling_but_is_not_a_cut():
    """A missing boundary must not be inherited as strong coupling; it is
    severed (like a cut) but reported separately so the exporter re-reads it."""
    T, H, W = 16, 12, 24
    cs = 4
    # cheap corridor jumps at the gap we will delete (t=7) and is static flow
    unary = np.ones((T, H, W), np.float32)
    unary[:8, :, 6] = 0.05
    unary[8:, :, 18] = 0.05
    seam = np.full((T, H), 12, np.int32)
    gap = np.zeros((T - 1, H, W), np.float32)     # static => would strongly couple
    store = SeamStore.from_energy(unary, seam, gap_motion=gap,
                                  shot_cuts=np.zeros(T - 1, bool),
                                  chunk_size=cs, root=os.path.join(_tmp(), "g"))
    # With the boundary present, strong coupling keeps the first side's seam.
    base = reconstruct(store, horizon=99)
    store.read_block(0, T)
    store.delete_boundary(7)
    store.flush()
    rep = reconstruct(store, horizon=99)
    assert 7 in rep.missing_boundaries
    assert not rep.shot_cuts[7], "a missing gap is NOT a shot cut"
    assert not rep.verified[7] and not rep.verified[8], "gap frames need re-read"
    # Severing lets the post-gap side move to its own cheap corridor (18),
    # which inheritance-as-zero-motion would have prevented.
    assert np.median(rep.seams[9:]) > 12, np.median(rep.seams[9:])
    print("ok test_missing_gap_severs_coupling_but_is_not_a_cut",
          "post-gap median", float(np.median(rep.seams[9:])))


def test_shot_cut_truncates_temporal_dependency():
    """Across a shot cut the two sides are solved independently; that is a
    legitimate truncation, not a missing neighbour."""
    T, H, W = 12, 16, 32
    unary = np.ones((T, H, W), np.float32)
    # different cheap corridors before/after the cut at t=6
    unary[:6, :, 6] = 0.05
    unary[6:, :, 26] = 0.05
    seam = np.full((T, H), 16, np.int32)
    gap = np.zeros((T - 1, H, W), np.float32)
    cuts = np.zeros(T - 1, bool)
    cuts[5] = True
    store = SeamStore.from_energy(unary, seam, gap_motion=gap, shot_cuts=cuts,
                                  chunk_size=4, root=os.path.join(_tmp(), "c"))
    rep = reconstruct(store, horizon=99)
    assert rep.shot_cuts[5]
    assert 5 not in rep.missing_boundaries, "a cut is not a missing boundary"
    assert rep.verified.all(), "shot-cut frames remain trusted"
    before = rep.seams[:6].mean()
    after = rep.seams[6:].mean()
    assert before < 12 < after, (before, after)
    print("ok test_shot_cut_truncates_temporal_dependency",
          round(float(before), 1), "->", round(float(after), 1))


def test_flicker_metric_matches_expected():
    steady = np.tile(np.arange(10), (6, 1)).astype(np.int32)
    assert seam_flicker(steady)["mean_step"] == 0.0
    jumpy = steady.copy()
    jumpy[2:, :] += 5
    f = seam_flicker(jumpy)
    assert f["mean_step"] > 0 and f["side_flips"] >= 1
    print("ok test_flicker_metric_matches_expected", f)


def test_export_uses_new_path_and_revision():
    frames = make_moving_transition_clip(T=12, H=24, W=48)
    locked_path, _ = build_two_transition_projects(frames, seam_column=24,
                                                   chunk_size=4)
    root = os.path.join(_tmp(), "E")
    migrate_legacy(locked_path, root, frames=frames, chunk_size=4)
    st = load_store(root)
    rep = reconstruct(st, horizon=8)
    ex = export_with_reread(frames, rep, st.revision)
    # width shrinks by exactly one seam
    assert ex.frames.shape[2] == frames.shape[2] - 1
    assert ex.used_revision == rep.revision
    # exporting against a stale revision must refuse
    try:
        export_with_reread(frames, rep, st.revision + 99)
        raise AssertionError("stale revision export should fail")
    except ValueError:
        pass
    print("ok test_export_uses_new_path_and_revision rev", ex.used_revision)


def test_legacy_without_pixels_flags_missing_boundaries():
    """A legacy table with no sidecar pixels cannot rebuild adjacency: every gap
    must be reported missing, never fabricated as a zero-cost edge."""
    T, H, W = 8, 6, 10
    unary = np.ones((T, H, W), np.float32)
    seam = np.zeros((T, H), np.int32)
    path = os.path.join(_tmp(), "flat.npz")
    write_legacy_table(path, unary, seam)   # no frames sidecar
    root = os.path.join(_tmp(), "NP")
    rep = migrate_legacy(path, root, frames=None, chunk_size=4)
    assert rep.inherited_frames == T
    assert len(rep.missing_boundary_gaps) == T - 1
    st = load_store(root)
    rrec = reconstruct(st, horizon=8)
    assert rrec.missing_boundaries, "all gaps missing"
    assert not rrec.verified.any(), "no frame should be trusted without adjacency"
    print("ok test_legacy_without_pixels_flags_missing_boundaries",
          len(rep.missing_boundary_gaps), "gaps")


def test_export_geometry_matches_store_after_resize():
    """After resize steps the store/energy/seam live in CURRENT geometry; export
    must carve current frames (not the untouched original) or widths diverge."""
    from seamcarver.engine import SeamEngine
    from seamcarver import synth
    frames, _ = synth.moving_marker(T=10, H=32, W=48, speed=2.2)
    eng = SeamEngine(frames, name="geom",
                     store_root=os.path.join(_tmp(), "s"), store_chunk_size=5)
    eng.resize_to(40, max_iterations=3)
    assert eng.project.width < eng.project.original_width
    assert eng.store.W == eng.project.width, "store must track current width"
    rep = eng.reconstruct(horizon=8)
    assert rep.seams.max() < eng.store.W
    bc = eng.boundary_check()
    assert bc["ok"], bc
    assert bc["width"] == eng.project.width - 1, bc
    ex = eng.export()
    assert ex.frames.shape[2] == eng.project.width - 1, ex.frames.shape
    print("ok test_export_geometry_matches_store_after_resize",
          "current", eng.project.width, "exported", ex.frames.shape[2])


def test_engine_edit_author_lock_survives_resync():
    """An edit marks a frame ORIGIN_AUTHOR; a later solve/step re-syncs the
    store and must carry that author lock forward, never silently dropping it."""
    from seamcarver.engine import SeamEngine
    from seamcarver import synth
    from seamcarver.seam_store import ORIGIN_AUTHOR
    frames, _ = synth.moving_marker(T=10, H=32, W=48, speed=2.2)
    eng = SeamEngine(frames, name="lock",
                     store_root=os.path.join(_tmp(), "s"), store_chunk_size=5)
    eng.solve()
    eng.edit_seam(4, 16, 22)
    blk = eng.store.read_block(0, eng.store.T)
    assert blk.frames[4].origin == ORIGIN_AUTHOR
    assert (blk.frames[4].lock >= 0).any()
    eng.apply_step("remove")
    blk = eng.store.read_block(0, eng.store.T)
    assert blk.frames[4].origin == ORIGIN_AUTHOR, "author lock dropped on re-sync"
    print("ok test_engine_edit_author_lock_survives_resync")


def test_carried_lock_clamps_after_shrink():
    """An author lock recorded near the right wall must clamp into the current
    width after aggressive shrink, never leaving an off-grid lock column."""
    from seamcarver.engine import SeamEngine
    from seamcarver import synth
    from seamcarver.seam_store import ORIGIN_AUTHOR
    frames, _ = synth.moving_marker(T=8, H=32, W=48, speed=3.0)
    eng = SeamEngine(frames, name="clamp",
                     store_root=os.path.join(_tmp(), "s"), store_chunk_size=4)
    eng.solve()
    eng.edit_seam(4, 16, 46)
    for _ in range(10):
        eng.apply_step("remove")
    blk = eng.store.read_block(0, eng.store.T)
    assert blk.frames[4].origin == ORIGIN_AUTHOR
    assert int(blk.frames[4].lock.max()) < eng.store.W, "lock must clamp to width"
    rep = eng.reconstruct(horizon=6)
    assert (rep.seams >= 0).all() and (rep.seams < eng.store.W).all()
    print("ok test_carried_lock_clamps_after_shrink width", eng.store.W,
          "lock", int(blk.frames[4].lock.max()))


def test_missing_boundary_flags_only_adjacent_frames():
    """A single lost boundary must flag exactly the two frames it joins, not the
    whole chunk it happens to live in (over-reporting would mark verified seams
    as needing a re-read)."""
    T, H, W = 16, 12, 32
    unary = np.ones((T, H, W), np.float32)
    seam = np.full((T, H), W // 2, np.int32)
    store = SeamStore.from_energy(unary, seam,
                                  gap_motion=np.zeros((T - 1, H, W), np.float32),
                                  shot_cuts=np.zeros(T - 1, bool),
                                  chunk_size=4, root=os.path.join(_tmp(), "adj"))
    store.read_block(0, T)
    store.delete_boundary(6)        # inside chunk 1 (frames 4..7)
    store.flush()
    rep = reconstruct(store, horizon=0)
    assert rep.reread_frames == [6, 7], rep.reread_frames
    assert rep.verified.sum() == T - 2
    print("ok test_missing_boundary_flags_only_adjacent_frames", rep.reread_frames)


def test_converted_legacy_rebuilds_same_seam():
    """The migration contract: after conversion the chunked store reconstructs
    the SAME seam as the legacy solve when given the full temporal chain."""
    from seamcarver import synth
    from seamcarver.energy import build_energy
    from seamcarver.seam_solver import build_and_solve, SolveConfig
    frames, _ = synth.moving_marker(T=14, H=32, W=48, speed=2.0)
    ev = build_energy(frames.astype(np.float32), cfg=None)
    legacy_seam, _, _, _ = build_and_solve(ev.unary, ev.flows, ev.shot_cuts,
                                            SolveConfig())
    path = os.path.join(_tmp(), "old.npz")
    write_legacy_table(path, ev.unary, legacy_seam, shot_cuts=ev.shot_cuts,
                       frames=frames)
    root = os.path.join(_tmp(), "mig")
    rep = migrate_legacy(path, root, chunk_size=4)
    assert rep.inherited_frames == 0 and rep.recomputed_frames == 14
    st = load_store(root)
    full = reconstruct(st, horizon=0)
    assert np.array_equal(full.seams, legacy_seam), "same-seam contract broken"
    # A finite horizon is a deliberate truncation and may differ; it must still
    # be reported (flicker), just not claimed identical.
    finite = reconstruct(st, horizon=6)
    assert finite.flicker["flicker_index"] >= 0
    print("ok test_converted_legacy_rebuilds_same_seam",
          "finite_identical=", bool(np.array_equal(finite.seams, legacy_seam)))


def test_auto_strategy_selection_is_cached_per_structure():
    """``auto`` measures once per adjacency structure and reuses the choice; a
    changed structure (dropped boundary) must trigger a fresh comparison."""
    from seamcarver.reconstruct import chosen_strategy, _STRATEGY_CACHE
    T, H, W = 12, 10, 20
    unary = np.ones((T, H, W), np.float32)
    seam = np.full((T, H), W // 2, np.int32)
    root = os.path.join(_tmp(), "cache")
    st = SeamStore.from_energy(unary, seam,
                               gap_motion=np.zeros((T - 1, H, W), np.float32),
                               shot_cuts=np.zeros(T - 1, bool),
                               chunk_size=4, root=root)
    _STRATEGY_CACHE.clear()
    first = chosen_strategy(st, horizon=8)
    assert len(_STRATEGY_CACHE) == 1
    assert chosen_strategy(st, horizon=8) == first
    assert len(_STRATEGY_CACHE) == 1, "selection must be reused, not recomputed"
    st.read_block(0, T)
    st.delete_boundary(5)
    st.flush()
    chosen_strategy(st, horizon=8)
    assert len(_STRATEGY_CACHE) == 2, "changed structure must re-measure"
    print("ok test_auto_strategy_selection_is_cached_per_structure", first)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"\nall {len(tests)} migration tests passed")


if __name__ == "__main__":
    main()
