"""audio_events.py のテスト(合成音声で検証)."""

import numpy as np

from baseball_scorer.audio_events import activity_envelope, detect_impacts

SR = 16000
rng = np.random.default_rng(42)


def make_noise(duration_sec: float, level: float = 0.01) -> np.ndarray:
    return rng.normal(0, level, int(SR * duration_sec)).astype(np.float32)


def add_impact(samples: np.ndarray, at_sec: float, freq: float = 4000.0) -> None:
    """減衰する高周波バースト(金属バット音の模擬)を合成する."""
    dur = 0.05
    t = np.arange(int(SR * dur)) / SR
    burst = (0.8 * np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.01)).astype(np.float32)
    start = int(at_sec * SR)
    samples[start : start + len(burst)] += burst


class TestDetectImpacts:
    def test_finds_synthetic_impacts(self):
        samples = make_noise(30.0)
        for at in (5.0, 12.0, 25.0):
            add_impact(samples, at)
        impacts = detect_impacts(samples, SR)
        found = sorted(i.timestamp for i in impacts)
        assert len(found) == 3
        for expected, actual in zip((5.0, 12.0, 25.0), found):
            assert abs(expected - actual) < 0.2

    def test_no_false_positives_on_noise(self):
        samples = make_noise(30.0)
        assert detect_impacts(samples, SR) == []

    def test_min_gap_merges_close_impacts(self):
        samples = make_noise(20.0)
        add_impact(samples, 10.0)
        add_impact(samples, 10.3)  # min_gap_sec=1.0 内 → 1 件に統合される
        impacts = detect_impacts(samples, SR, min_gap_sec=1.0)
        assert len(impacts) == 1

    def test_steady_loud_noise_is_not_impact(self):
        # 定常的に大きい音(歓声・風)はベースラインに吸収される
        samples = make_noise(30.0, level=0.2)
        assert detect_impacts(samples, SR) == []


class TestActivityEnvelope:
    def test_envelope_shape(self):
        samples = make_noise(10.0)
        times, rms = activity_envelope(samples, SR, window_sec=1.0)
        assert len(times) == len(rms) == 10
        assert times[0] == 0.0 and times[-1] == 9.0

    def test_loud_section_has_higher_rms(self):
        quiet = make_noise(5.0, level=0.01)
        loud = make_noise(5.0, level=0.3)
        samples = np.concatenate([quiet, loud])
        _, rms = activity_envelope(samples, SR)
        assert rms[5:].mean() > rms[:5].mean() * 5
