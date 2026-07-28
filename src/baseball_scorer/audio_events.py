"""音声からの打撃音・捕球音検出(DSP・決定的処理).

金属バットの接触音やミットの捕球音は、広帯域(特に高域)のエネルギーが
短時間に立ち上がる「インパルス的」な音。STFT の高域帯エネルギーの
時間変化(スペクトラルフラックス)のピークを拾う。

全処理は NumPy / SciPy のベクトル演算(Python ループはピーク後処理のみ)。
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal


@dataclass
class AudioImpact:
    """打撃音・捕球音などのインパルス性イベント候補."""

    timestamp: float  # 秒
    strength: float   # 正規化強度(中央値絶対偏差ベースの z 値)


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """16bit PCM WAV をモノラル float32 [-1, 1] で読み込む."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)
    return samples, sr


def detect_impacts(
    samples: np.ndarray,
    sample_rate: int,
    band_hz: tuple[float, float] = (2000.0, 7500.0),
    frame_sec: float = 0.032,
    z_threshold: float = 6.0,
    min_gap_sec: float = 1.0,
) -> list[AudioImpact]:
    """高域スペクトラルフラックスのピークからインパルス性イベントを検出する.

    z_threshold は中央値絶対偏差(MAD)ベースのロバスト z 値。歓声や風などの
    定常ノイズはベースラインに吸収され、鋭い立ち上がりだけが残る。
    """
    nperseg = int(sample_rate * frame_sec)
    hop = nperseg // 2
    freqs, times, stft = signal.stft(
        samples, fs=sample_rate, nperseg=nperseg, noverlap=nperseg - hop
    )
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    mag = np.abs(stft[band, :])  # (F_band, T)

    # スペクトラルフラックス: 正方向のエネルギー増分のみ合算
    flux = np.maximum(np.diff(mag, axis=1), 0.0).sum(axis=0)
    flux = np.concatenate([[0.0], flux])

    # ロバスト z 値化
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median))) or 1e-9
    z = (flux - median) / (1.4826 * mad)

    frame_hop_sec = hop / sample_rate
    min_gap_frames = max(1, int(min_gap_sec / frame_hop_sec))
    peaks, props = signal.find_peaks(z, height=z_threshold, distance=min_gap_frames)

    return [
        AudioImpact(timestamp=float(times[p]), strength=float(h))
        for p, h in zip(peaks, props["peak_heights"])
    ]


def activity_envelope(
    samples: np.ndarray,
    sample_rate: int,
    window_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """1 秒粒度の音量エンベロープ(times, rms)。タイムライン表示の背景用."""
    win = int(sample_rate * window_sec)
    n = len(samples) // win
    if n == 0:
        return np.array([]), np.array([])
    trimmed = samples[: n * win].reshape(n, win)
    rms = np.sqrt((trimmed**2).mean(axis=1))
    times = np.arange(n) * window_sec
    return times, rms
