"""映像の活動量計測とカメラズレ検出(OpenCV・決定的処理).

フレームを間引いて縮小グレースケール化し、隣接サンプル間の差分から
「画面がどれだけ動いているか」の時系列を作る。プレー中は選手が走るので
活動量が上がり、投球間・イニング間は下がる。

固定カメラ前提。打球直撃などで画角がズレた場合は位相相関で検出できる
(グローバルシフトが大きいサンプルを camera_shifts として報告)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MotionResult:
    times: np.ndarray        # 各サンプルの秒
    activity: np.ndarray     # 平均絶対差分(0-255 スケール)
    camera_shifts: list[tuple[float, float]]  # (秒, シフト量 px)


def measure_activity(
    video_path: str | Path,
    sample_fps: float = 2.0,
    resize_width: int = 160,
    shift_threshold_px: float = 4.0,
) -> MotionResult:
    """動画全体の活動量時系列を計測する.

    フレームの取得は VideoCapture.grab()(デコードなし・軽量)で間引き、
    サンプルだけ retrieve() でデコードする。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / sample_fps))

    times: list[float] = []
    activity: list[float] = []
    shifts: list[tuple[float, float]] = []
    prev: np.ndarray | None = None

    frame_idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = frame_idx / src_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale = resize_width / gray.shape[1]
            small = cv2.resize(gray, None, fx=scale, fy=scale)
            if prev is not None:
                diff = cv2.absdiff(small, prev)
                times.append(t)
                activity.append(float(diff.mean()))
                # 位相相関でグローバルシフトを推定(カメラズレ検出)
                shift, _resp = cv2.phaseCorrelate(
                    prev.astype(np.float32), small.astype(np.float32)
                )
                magnitude = float(np.hypot(*shift)) / scale  # 元解像度換算
                if magnitude >= shift_threshold_px:
                    shifts.append((t, magnitude))
            prev = small
        frame_idx += 1

    cap.release()
    return MotionResult(
        times=np.array(times),
        activity=np.array(activity),
        camera_shifts=shifts,
    )


def activity_bursts(
    result: MotionResult,
    z_threshold: float = 2.5,
    min_duration_sec: float = 2.0,
) -> list[tuple[float, float]]:
    """活動量が閾値を超え続けた区間 (start, end) を列挙する."""
    if len(result.activity) == 0:
        return []
    a = result.activity
    median = float(np.median(a))
    mad = float(np.median(np.abs(a - median))) or 1e-9
    hot = (a - median) / (1.4826 * mad) >= z_threshold

    bursts: list[tuple[float, float]] = []
    start: float | None = None
    for t, h in zip(result.times, hot):
        if h and start is None:
            start = float(t)
        elif not h and start is not None:
            if t - start >= min_duration_sec:
                bursts.append((start, float(t)))
            start = None
    if start is not None and result.times[-1] - start >= min_duration_sec:
        bursts.append((start, float(result.times[-1])))
    return bursts
