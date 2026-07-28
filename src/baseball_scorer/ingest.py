"""入力の取得: YouTube ダウンロードと音声トラック抽出."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def youtube_id(url: str) -> str | None:
    m = YOUTUBE_ID_RE.search(url)
    return m.group(1) if m else None


def download(url: str, out_dir: str | Path, max_height: int = 480) -> Path:
    """yt-dlp で動画を取得する。既に存在すればスキップ."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = youtube_id(url) or "video"
    out_path = out_dir / f"{vid}.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp が見つかりません: pip install yt-dlp")
    subprocess.run(
        [
            "yt-dlp",
            "-f", f"b[height<={max_height}]/bv*[height<={max_height}]+ba/b",
            "-o", str(out_path),
            "--no-part",
            url,
        ],
        check=True,
    )
    return out_path


def extract_audio(
    video_path: str | Path,
    out_path: str | Path | None = None,
    sample_rate: int = 16000,
) -> Path:
    """ffmpeg でモノラル 16kHz WAV を抽出する(オンセット検出・Whisper 共用)."""
    video_path = Path(video_path)
    if out_path is None:
        out_path = video_path.with_suffix(".wav")
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(out_path),
        ],
        check=True,
    )
    return out_path
