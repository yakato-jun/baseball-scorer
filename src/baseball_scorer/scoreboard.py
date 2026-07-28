"""スコアボード領域の検出(OpenCV・決定的処理).

原理: スコアボード(球場の掲示板・放送のスコアバグ)は画面上の位置が固定で、
時間方向の画素変化が背景(選手・観客・芝)より小さい。かつ文字が密集する
のでエッジ密度が高い。この 2 条件を満たす矩形領域を候補として返す。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Region:
    """画面上の矩形領域(ピクセル座標)."""

    x: int
    y: int
    w: int
    h: int

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


def detect_scoreboard_region(
    frame_paths: list[Path],
    sample_count: int = 30,
    variance_percentile: float = 25.0,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.15,
) -> Region | None:
    """フレーム集合からスコアボード領域を推定する.

    1. sample_count 枚を等間隔サンプリングしグレースケール化
    2. 画素ごとの時間方向分散を計算 → 低分散マスク
    3. 各フレームのエッジ(Canny)の時間平均 → 高エッジ密度マスク
    4. 両マスクの AND を取り、モルフォロジーで整形して最大連結成分の外接矩形を返す

    検出できない場合は None(→ LLM フォールバックへ)。
    """
    if len(frame_paths) < 3:
        return None

    step = max(1, len(frame_paths) // sample_count)
    sampled = frame_paths[::step][:sample_count]

    grays: list[np.ndarray] = []
    edges_acc: np.ndarray | None = None
    for p in sampled:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        grays.append(img.astype(np.float32))
        e = cv2.Canny(img, 80, 160).astype(np.float32) / 255.0
        edges_acc = e if edges_acc is None else edges_acc + e

    if len(grays) < 3 or edges_acc is None:
        return None

    stack = np.stack(grays)  # (N, H, W)
    variance = stack.var(axis=0)
    edge_density = edges_acc / len(grays)

    low_var = variance <= np.percentile(variance, variance_percentile)
    # エッジが安定して出ている(= 常に文字が表示されている)画素
    high_edge = edge_density >= 0.5

    mask = (low_var & high_edge).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h_img, w_img = mask.shape
    total_area = h_img * w_img
    best = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    area_ratio = (w * h) / total_area
    if not (min_area_ratio <= area_ratio <= max_area_ratio):
        return None

    # OCR しやすいよう少しマージンを付ける
    margin = 4
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(w_img - x, w + 2 * margin)
    h = min(h_img - y, h + 2 * margin)
    return Region(x, y, w, h)


def preprocess_for_ocr(crop: np.ndarray, scale: int = 3) -> np.ndarray:
    """OCR 前処理: 拡大 → グレースケール → 大津の二値化."""
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 白背景・黒文字に正規化(Tesseract の得意な形)
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)
    return binary
