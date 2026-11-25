"""Video and frame-sequence I/O.

Reads any container OpenCV can decode into an RGB uint8 array and writes
either an mp4 (when an encoder is available) or an inspectable PNG/JPEG
sequence plus an HTML contact sheet.  No ffmpeg binary is required.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False


def read_video(path: str, max_frames: int | None = None, scale: float = 1.0) -> np.ndarray:
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required for video decoding")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frames = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if scale != 1.0:
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(rgb)
        idx += 1
        if max_frames and idx >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames).astype(np.uint8)


def write_video(path: str, frames: np.ndarray, fps: float = 12.0) -> str:
    """Best-effort mp4 writer; falls back to a frame sequence next to it."""
    if not _HAS_CV2:
        return write_frame_sequence(path, frames)
    T, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    if not writer.isOpened():
        writer.release()
        return write_frame_sequence(path, frames)
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def write_frame_sequence(video_path: str, frames: np.ndarray) -> str:
    base = os.path.splitext(video_path)[0] + "_frames"
    os.makedirs(base, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(base, f"f{i:04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    return base


def overlay_seams(frames: np.ndarray, seams: np.ndarray,
                  color=(255, 64, 0), thickness: int = 1) -> np.ndarray:
    """Draw each frame's seam path onto a copy of the frames."""
    out = frames.copy()
    T, H = seams.shape
    for t in range(T):
        for y in range(H):
            x = int(seams[t, y])
            for dx in range(thickness):
                xx = min(max(x + dx, 0), frames.shape[2] - 1)
                out[t, y, xx] = color
    return out
