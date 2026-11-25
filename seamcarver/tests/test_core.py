"""Correctness tests for the max-flow cut, seam extraction and mappings.

Run directly:  python tests/test_core.py
No third-party test runner is required.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seamcarver.maxflow_wrapper import grid_mincut
from seamcarver.seam_solver import build_and_solve, SolveConfig, extract_seams
from seamcarver.carve import CarveProject, SeamStep, apply_seams
from seamcarver.energy import build_energy, EnergyConfig
from seamcarver.flow import FlowConfig


def brute_force_mincut(ts, tk, pair_edges):
    """Enumerate all 2^N labelings on a tiny graph for an oracle.

    Label 0 = source side, label 1 = sink side.  A source->node unary arc
    (term_src) is cut when the node is on the sink side; a node->sink unary
    arc (term_sink) is cut when the node is on the source side.
    """
    N = ts.size
    best = (np.inf, None)
    for labels in itertools.product((0, 1), repeat=N):
        lab = np.array(labels)
        cost = ts[lab == 1].sum() + tk[lab == 0].sum()
        for u, v, c in pair_edges:
            if lab[u] != lab[v]:
                cost += c
        if cost < best[0]:
            best = (cost, lab.copy())
    return best


def test_grid_mincut_matches_bruteforce():
    rng = np.random.default_rng(7)
    T, H, W = 2, 2, 3
    N = T * H * W
    ts = rng.random((T, H, W)).astype(np.float32) * 5
    tk = rng.random((T, H, W)).astype(np.float32) * 5
    ch = rng.random((T, H, W - 1)).astype(np.float32) * 4
    cv = rng.random((T, H - 1, W)).astype(np.float32) * 4
    ct = rng.random((T - 1, H, W)).astype(np.float32) * 4
    labels, flow = grid_mincut(ts, tk, ch, cv, ct)

    edges = []
    for t in range(T):
        for y in range(H):
            for x in range(W):
                n = t * H * W + y * W + x
                if x + 1 < W:
                    edges.append((n, n + 1, ch[t, y, x]))
                if y + 1 < H:
                    edges.append((n, n + W, cv[t, y, x]))
                if t + 1 < T:
                    edges.append((n, n + H * W, ct[t, y, x]))
    oracle_cost, oracle = brute_force_mincut(ts.reshape(-1), tk.reshape(-1), edges)
    lab = labels.reshape(-1)
    tsf, tkf = ts.reshape(-1), tk.reshape(-1)
    got = tsf[lab == 1].sum() + tkf[lab == 0].sum()
    got += sum(c for u, v, c in edges if lab[u] != lab[v])
    assert abs(got - oracle_cost) < 1e-3, (got, oracle_cost, flow)
    # flow equals cut value up to single-precision accumulation error
    assert abs(flow - oracle_cost) < 1e-2 * max(1.0, oracle_cost), (flow, oracle_cost)
    assert np.array_equal(lab, oracle) or abs(got - oracle_cost) < 1e-3
    print("ok test_grid_mincut_matches_bruteforce flow=%.3f oracle=%.3f" % (flow, oracle_cost))


def test_hard_protected_barrier():
    T, H, W = 1, 6, 8
    unary = np.ones((T, H, W), np.float32)
    unary[0, 2:4, 3:5] = 1e6
    flows = np.zeros((0, H, W, 2), np.float32)
    seams, labels, _, _ = build_and_solve(unary, flows, np.zeros(0, bool), SolveConfig())
    for y in range(H):
        x = seams[0, y]
        if 2 <= y < 4:
            assert not (3 <= x < 5), f"seam entered protected cell at y={y}, x={x}"
    print("ok test_hard_protected_barrier seams=", seams[0].tolist())


def test_seam_is_valid_path():
    rng = np.random.default_rng(3)
    T, H, W = 3, 10, 12
    unary = rng.random((T, H, W)).astype(np.float32) + 0.1
    flows = np.zeros((T - 1, H, W, 2), np.float32)
    seams, labels, flow, cap_t = build_and_solve(
        unary, flows, np.zeros(T - 1, bool), SolveConfig())
    assert seams.shape == (T, H)
    assert ((seams >= 0) & (seams < W)).all()
    for t in range(T):
        assert np.all(np.abs(np.diff(seams[t])) <= 1), "seam must be 8-connected"
    # static scene: temporal coupling should keep seams identical across frames
    assert np.all(seams[1:] == seams[0]), "static scene seams must persist"
    print("ok test_seam_is_valid_path flow=%.3f" % flow)


def test_temporal_loosening_allows_side_change():
    """A moving cheap corridor must be trackable; strong temporal coupling on a
    static scene must not freeze the seam when the corridor jumps."""
    T, H, W = 4, 8, 10
    unary = np.ones((T, H, W), np.float32)
    corridor = [2, 2, 7, 7]
    for t, cx in enumerate(corridor):
        unary[t, :, cx] = 0.05
    flows = np.zeros((T - 1, H, W, 2), np.float32)
    # large motion between frames 1 and 2 => loosened temporal edge
    flows[1, ..., 0] = 9.0
    cfg = SolveConfig(temporal_base=40.0, temporal_motion_gate=3.0)
    seams, _, _, _ = build_and_solve(unary, flows, np.zeros(T - 1, bool), cfg)
    assert seams[0].mean() < 4 and seams[-1].mean() > 5, seams
    print("ok test_temporal_loosening_allows_side_change means=",
          [round(float(seams[t].mean()), 2) for t in range(T)])


def test_apply_and_mappings_roundtrip():
    rng = np.random.default_rng(11)
    T, H, W, C = 2, 5, 7, 3
    frames = rng.integers(0, 255, (T, H, W, C), np.uint8)
    proj = CarveProject(frames)
    for k in range(3):
        cur = proj.current_frames()
        h, w = cur.shape[1:3]
        # one legal seam per frame: monotone path, distinct columns across rows
        seams = np.zeros((T, h), np.int32)
        for t in range(T):
            col = rng.integers(1, w - 1)
            for y in range(h):
                col = int(np.clip(col + rng.integers(-1, 2), 0, w - 1))
                seams[t, y] = col
        proj.steps.append(SeamStep("remove", seams, w, w - 1))
    assert proj.width == W - 3
    carved = proj.current_frames()
    assert carved.shape[2] == W - 3
    fmap = proj.forward_map()
    imap = proj.inverse_map()
    assert imap.shape == (T, H, W - 3)
    # every current column maps back to a valid original column
    assert ((imap >= 0) & (imap < W)).all()
    # forward map stays within the current width and never goes negative
    assert (fmap >= 0).all() and (fmap < carved.shape[2]).all()
    # pixel-level round trip: each current pixel traces back to the exact
    # original pixel that survived carving
    for t in range(T):
        for y in range(H):
            for xc in range(carved.shape[2]):
                xo = imap[t, y, xc]
                assert np.array_equal(carved[t, y, xc], frames[t, y, xo]), (t, y, xc, xo)
    # original is untouched and restorable
    assert np.array_equal(proj.restore_original(), frames)
    print("ok test_apply_and_mappings_roundtrip width", W, "->", proj.width)


def test_insertion_grows_width():
    rng = np.random.default_rng(4)
    frames = rng.integers(0, 255, (2, 5, 6, 3), np.uint8)
    seams = np.zeros((2, 5), np.int32) + 2
    out = apply_seams(frames, seams, "insert")
    assert out.shape == (2, 5, 7, 3)
    out2 = apply_seams(frames, seams, "remove")
    assert out2.shape == (2, 5, 5, 3)
    print("ok test_insertion_grows_width")


def test_shot_cut_severs_temporal_edge():
    from seamcarver.synth import shot_cut
    frames, truth = shot_cut(T=12, H=40, W=60, cut_at=6)
    cfg = EnergyConfig(shot_threshold=15.0)
    energy = build_energy(frames, cfg=cfg, flow_cfg=FlowConfig(method="blockmatch"))
    assert energy.shot_cuts[5], "shot boundary must be detected at the known cut"
    seams, _, _, cap_t = build_and_solve(
        energy.unary, energy.flows, energy.shot_cuts, SolveConfig())
    assert cap_t[5].sum() == 0.0, "temporal capacity at shot cut must be zero"
    print("ok test_shot_cut_severs_temporal_edge cuts=", energy.shot_cuts.tolist())


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"\nall {len(tests)} core tests passed")


if __name__ == "__main__":
    main()
