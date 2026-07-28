"""ffmpeg / ffprobe による動画メタデータ取得とフレーム抽出(決定的処理)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FfmpegNotFoundError(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise FfmpegNotFoundError(
            f"{cmd[0]} が見つかりません。`sudo apt install ffmpeg` を実行してください"
        ) from e


@dataclass
class VideoInfo:
    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float


def probe(path: str | Path) -> VideoInfo:
    """ffprobe で動画メタデータを取得する."""
    path = Path(path)
    proc = _run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:format=duration",
        "-of", "json", str(path),
    ])
    data = json.loads(proc.stdout)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return VideoInfo(
        path=path,
        duration_sec=float(data["format"]["duration"]),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=float(num) / float(den),
    )


def extract_frames(
    path: str | Path,
    out_dir: str | Path,
    fps: float = 1.0,
    start: float | None = None,
    end: float | None = None,
) -> list[Path]:
    """固定間隔でフレームを PNG として抽出し、パスのリストを返す.

    スコアボードの変化は秒単位なのでデフォルト 1fps で十分。
    ファイル名は連番だが、fps とオフセットからタイムスタンプを復元できる。
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-to", str(end - (start or 0.0))]
    cmd += ["-vf", f"fps={fps}", str(out_dir / "frame_%06d.png")]
    _run(cmd)

    return sorted(out_dir.glob("frame_*.png"))


def frame_timestamp(index: int, fps: float, start: float = 0.0) -> float:
    """extract_frames の連番 index(1 始まり)から動画内秒数を復元する."""
    return start + (index - 1) / fps
