"""motion.py と timeline.py のテスト(合成動画・合成イベント)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from baseball_scorer.audio_events import AudioImpact
from baseball_scorer.motion import MotionResult, activity_bursts, measure_activity
from baseball_scorer.speech import SpeechEvent
from baseball_scorer.timeline import TimelineEvent, fmt_ts, fuse_events, render_html

rng = np.random.default_rng(7)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> Path:
    """静止背景 30 秒、うち 10-20 秒だけ動く矩形がある合成動画(10fps)."""
    path = tmp_path_factory.mktemp("video") / "synth.avi"
    fps, w, h = 10, 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    background = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    for i in range(fps * 30):
        frame = background.copy()
        t = i / fps
        if 10 <= t < 20:  # 活動区間: 動く矩形
            x = int((t - 10) / 10 * (w - 40))
            cv2.rectangle(frame, (x, 100), (x + 40, 140), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


class TestMeasureActivity:
    def test_detects_activity_burst(self, synthetic_video):
        result = measure_activity(synthetic_video, sample_fps=2.0)
        bursts = activity_bursts(result, z_threshold=2.0)
        assert len(bursts) >= 1
        start, end = bursts[0]
        assert 8.0 <= start <= 12.0
        assert 18.0 <= end <= 22.0

    def test_static_video_no_shifts(self, synthetic_video):
        result = measure_activity(synthetic_video, sample_fps=2.0)
        assert result.camera_shifts == []


class TestFuseEvents:
    def test_sorted_and_filtered(self):
        events = fuse_events(
            impacts=[AudioImpact(30.0, 8.0), AudioImpact(10.0, 7.0)],
            speech=[
                SpeechEvent(20.0, 21.0, "アウト", keywords=[("judgment", "アウト")]),
                SpeechEvent(25.0, 26.0, "こんにちは"),  # 語彙なし → 除外
            ],
        )
        assert [e.timestamp for e in events] == [10.0, 20.0, 30.0]
        assert all(e.kind in ("impact", "speech") for e in events)

    def test_motion_bursts_included(self):
        motion = MotionResult(
            times=np.arange(0, 30, 0.5),
            activity=np.array([1.0] * 20 + [50.0] * 10 + [1.0] * 30),
            camera_shifts=[(25.0, 10.0)],
        )
        events = fuse_events(motion=motion)
        kinds = {e.kind for e in events}
        assert "burst" in kinds and "camera_shift" in kinds


class TestRenderHtml:
    def test_youtube_links(self):
        events = [TimelineEvent(75.0, "impact", "打撃/捕球音")]
        html_out = render_html(events, "テスト", youtube_id="abc123DEF-_")
        assert "https://youtu.be/abc123DEF-_?t=75" in html_out
        assert "1:15" in html_out

    def test_no_video_id_no_links(self):
        events = [TimelineEvent(75.0, "impact", "打撃/捕球音")]
        html_out = render_html(events, "テスト")
        assert "youtu.be" not in html_out

    def test_detail_is_escaped(self):
        events = [TimelineEvent(1.0, "speech", "アウト", detail="<script>x</script>")]
        html_out = render_html(events, "テスト")
        assert "<script>x</script>" not in html_out


class TestFmtTs:
    def test_formats(self):
        assert fmt_ts(75) == "1:15"
        assert fmt_ts(3671) == "1:01:11"
        assert fmt_ts(5) == "0:05"
