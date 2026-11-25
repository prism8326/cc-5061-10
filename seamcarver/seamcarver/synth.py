"""Deterministic synthetic test clips with ground-truth object geometry.

Three scenarios required by the task are generated without external assets:

* ``moving_marker``  : a bright geometric marker (the tracked "athlete")
  translating horizontally across a textured background, with a mild vertical
  bob, so true object width and centre are known exactly;
* ``fast_occlusion`` : the marker moves fast enough to cross frame-to-frame
  correspondence windows and passes behind an occluder bar;
* ``shot_cut``       : two static scenes of different texture joined by a hard
  cut mid-clip, to verify temporal edges are severed at the cut.

Each generator returns (frames uint8 (T,H,W,3), truth dict) where truth holds
the per-frame marker bounding box (x0,x1,y0,y1) and centre in ORIGINAL coords.
"""
from __future__ import annotations

import numpy as np


def _textured_background(rng: np.random.Generator, H: int, W: int) -> np.ndarray:
    base = rng.integers(40, 90, size=(H, W), dtype=np.uint8)
    img = np.repeat(base[..., None], 3, axis=2)
    # low-frequency stripes and checker patches give real gradient structure
    for y in range(0, H, 16):
        img[y:y + 2, :] += 35
    for x in range(0, W, 24):
        img[:, x:x + 2] += 25
    for y in range(0, H, 32):
        for x in range(0, W, 32):
            if ((x // 32 + y // 32) % 2) == 0:
                img[y:y + 16, x:x + 16] = np.clip(
                    img[y:y + 16, x:x + 16].astype(int) + 18, 0, 255)
    return img.astype(np.uint8)


def _draw_marker(frame: np.ndarray, cx: float, cy: float,
                 mw: int, mh: int) -> tuple[int, int, int, int]:
    H, W, _ = frame.shape
    x0 = int(round(cx - mw / 2))
    y0 = int(round(cy - mh / 2))
    x1, y1 = x0 + mw, y0 + mh
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x1, W), min(y1, H)
    if xb > xa and yb > ya:
        frame[ya:yb, xa:xb] = np.array([235, 70, 60], np.uint8)
        frame[ya:yb, xa, :] = 255
    return x0, x1, y0, y1


def moving_marker(T: int = 24, H: int = 80, W: int = 120,
                  speed: float = 2.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    bg = _textured_background(rng, H, W)
    frames = np.repeat(bg[None], T, axis=0).copy()
    mw, mh = 16, 28
    truth = {"cx": [], "x0": [], "x1": [], "y0": [], "y1": [], "width": mw}
    for t in range(T):
        cx = 18 + speed * t
        cy = H / 2 + 4.0 * np.sin(2 * np.pi * t / 12.0)
        x0, x1, y0, y1 = _draw_marker(frames[t], cx, cy, mw, mh)
        truth["cx"].append(cx)
        truth["x0"].append(x0)
        truth["x1"].append(x1)
        truth["y0"].append(y0)
        truth["y1"].append(y1)
    for k in ("cx", "x0", "x1", "y0", "y1"):
        truth[k] = np.array(truth[k], np.float32)
    return frames.astype(np.uint8), truth


def fast_occlusion(T: int = 24, H: int = 80, W: int = 140,
                   speed: float = 6.5, seed: int = 1):
    frames, truth = moving_marker(T, H, W, speed=speed, seed=seed)
    # vertical occluder bar in the middle of the scene
    bar_x0, bar_x1 = W // 2 - 5, W // 2 + 5
    for t in range(T):
        frames[t][:, bar_x0:bar_x1] = np.array([40, 40, 160], np.uint8)
    truth["occluder"] = (bar_x0, bar_x1)
    return frames, truth


def shot_cut(T: int = 24, H: int = 80, W: int = 120, cut_at: int = 12, seed: int = 2):
    rng = np.random.default_rng(seed)
    frames_a, truth_a = moving_marker(cut_at, H, W, speed=1.5, seed=seed)
    rng2 = np.random.default_rng(seed + 100)
    bg2 = _textured_background(rng2, H, W)
    # invert tone so the cut is unmistakable
    bg2 = 255 - bg2
    frames_b = np.repeat(bg2[None], T - cut_at, axis=0).copy()
    frames_b, truth_b = moving_marker(T - cut_at, H, W, speed=1.5, seed=seed + 50)
    frames_b = frames_b.astype(np.int16)
    frames_b = np.clip(255 - frames_b, 0, 255).astype(np.uint8)
    frames = np.concatenate([frames_a, frames_b], axis=0)
    truth = {
        "cx": np.concatenate([truth_a["cx"], truth_b["cx"]]),
        "x0": np.concatenate([truth_a["x0"], truth_b["x0"]]),
        "x1": np.concatenate([truth_a["x1"], truth_b["x1"]]),
        "y0": np.concatenate([truth_a["y0"], truth_b["y0"]]),
        "y1": np.concatenate([truth_a["y1"], truth_b["y1"]]),
        "width": truth_a["width"],
        "cut_at": cut_at,
    }
    return frames.astype(np.uint8), truth
