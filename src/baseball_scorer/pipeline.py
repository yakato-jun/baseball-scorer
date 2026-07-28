"""パイプライン全体のオーケストレーション.

フレーム抽出 → 領域検出 → OCR → ルール検証 → (LLM フォールバック) → 出力
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from . import rules, video
from .models import PlayEvent, ScoreboardReading
from .scoreboard import detect_scoreboard_region, preprocess_for_ocr


@dataclass
class PipelineResult:
    video_info: video.VideoInfo
    readings: list[ScoreboardReading]
    events: list[PlayEvent] = field(default_factory=list)
    region_found: bool = False

    def to_dict(self) -> dict:
        return {
            "video": {
                "path": str(self.video_info.path),
                "duration_sec": self.video_info.duration_sec,
                "width": self.video_info.width,
                "height": self.video_info.height,
            },
            "region_found": self.region_found,
            "readings": [r.to_dict() for r in self.readings],
            "events": [e.to_dict() for e in self.events],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def analyze(
    video_path: str | Path,
    fps: float = 1.0,
    use_llm: bool = True,
    work_dir: str | Path | None = None,
) -> PipelineResult:
    """動画を解析してスコア時系列(+ プレー候補)を返す."""
    info = video.probe(video_path)

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="baseball_scorer_"))
    frames_dir = Path(work_dir) / "frames"
    frame_paths = video.extract_frames(video_path, frames_dir, fps=fps)

    region = detect_scoreboard_region(frame_paths)
    result = PipelineResult(video_info=info, readings=[], region_found=region is not None)
    if region is None:
        # v1 では領域が取れなければ終了(将来: LLM に領域を尋ねるフォールバック)
        return result

    from .ocr import read_digits  # tesseract 未導入でも他機能を使えるよう遅延 import

    readings: list[ScoreboardReading] = []
    for i, p in enumerate(frame_paths, start=1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        crop = region.crop(img)
        ocr = read_digits(preprocess_for_ocr(crop))
        reading = _parse_ocr_text(
            ocr.text, ocr.confidence, video.frame_timestamp(i, fps)
        )
        if reading is not None:
            readings.append(reading)

    readings = rules.annotate_readings(readings)

    if use_llm:
        readings = _llm_fallback(readings, frame_paths, region, fps)
        readings = rules.annotate_readings(
            [ScoreboardReading(r.timestamp, r.state, r.confidence, r.source) for r in readings]
        )

    confirmed = [r for r in rules.dedupe_readings(readings) if not r.is_suspect]
    result.readings = confirmed

    if use_llm and len(confirmed) >= 2:
        from .llm import describe_play

        for prev, curr in zip(confirmed, confirmed[1:]):
            if prev.state != curr.state:
                result.events.append(describe_play(prev, curr))

    return result


def _parse_ocr_text(
    text: str, confidence: float, timestamp: float
) -> ScoreboardReading | None:
    """OCR の生テキストを GameState に変換する(v1 の素朴な実装).

    現状は「得点 2 つ」を想定した最小実装。レイアウトテンプレート導入までの
    つなぎであり、解釈できないものは低信頼度で返してフォールバックに委ねる。
    """
    from .models import GameState

    numbers = [int(t) for t in text.split() if t.isdigit()]
    if len(numbers) < 2:
        return None
    return ScoreboardReading(
        timestamp=timestamp,
        state=GameState(score_away=numbers[0], score_home=numbers[1]),
        confidence=confidence * 0.8,  # レイアウト未確定分のペナルティ
        source="ocr",
    )


def _llm_fallback(
    readings: list[ScoreboardReading],
    frame_paths: list[Path],
    region,
    fps: float,
) -> list[ScoreboardReading]:
    """疑わしい読み取りを Claude Vision で再判読して差し替える."""
    from .llm import read_scoreboard_image

    last_good = None
    fixed: list[ScoreboardReading] = []
    for r in readings:
        if not r.is_suspect:
            last_good = r.state
            fixed.append(r)
            continue
        idx = round(r.timestamp * fps) + 1
        if not (1 <= idx <= len(frame_paths)):
            continue
        img = cv2.imread(str(frame_paths[idx - 1]))
        if img is None:
            continue
        ok, png = cv2.imencode(".png", region.crop(img))
        if not ok:
            continue
        redo = read_scoreboard_image(png.tobytes(), r.timestamp, context=last_good)
        if redo is not None:
            fixed.append(redo)
    return fixed
