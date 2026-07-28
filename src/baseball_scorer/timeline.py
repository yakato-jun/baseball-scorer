"""イベント統合とタイムライン HTML の生成.

各検出器(音インパクト・キーワード・活動バースト)の出力を時刻軸上で
統合し、YouTube の該当時刻へジャンプできる HTML を出力する。
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

from .audio_events import AudioImpact
from .motion import MotionResult, activity_bursts
from .speech import SpeechEvent


@dataclass
class TimelineEvent:
    timestamp: float
    kind: str      # "impact" | "speech" | "burst" | "camera_shift"
    label: str
    detail: str = ""
    strength: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "label": self.label,
            "detail": self.detail,
            "strength": self.strength,
        }


def fuse_events(
    impacts: list[AudioImpact] | None = None,
    speech: list[SpeechEvent] | None = None,
    motion: MotionResult | None = None,
) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []

    for imp in impacts or []:
        events.append(TimelineEvent(
            timestamp=imp.timestamp,
            kind="impact",
            label="打撃/捕球音",
            strength=imp.strength,
        ))

    for seg in speech or []:
        if not seg.has_keywords:
            continue  # 語彙に一致しないセグメントは幻聴の可能性があるため捨てる
        words = " / ".join(w for _cat, w in seg.keywords)
        events.append(TimelineEvent(
            timestamp=seg.start,
            kind="speech",
            label=words,
            detail=seg.text,
        ))

    if motion is not None:
        for start, end in activity_bursts(motion):
            events.append(TimelineEvent(
                timestamp=start,
                kind="burst",
                label=f"活動区間 ({end - start:.0f}秒)",
                strength=end - start,
            ))
        for t, magnitude in motion.camera_shifts:
            events.append(TimelineEvent(
                timestamp=t,
                kind="camera_shift",
                label=f"カメラズレ ({magnitude:.0f}px)",
                strength=magnitude,
            ))

    events.sort(key=lambda e: e.timestamp)
    return events


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


_KIND_STYLE = {
    "impact": ("🔊", "#d97706"),
    "speech": ("🗣️", "#2563eb"),
    "burst": ("🏃", "#059669"),
    "camera_shift": ("📷", "#9333ea"),
}


def render_html(
    events: list[TimelineEvent],
    title: str,
    youtube_id: str | None = None,
) -> str:
    """イベント一覧をジャンプリンク付き HTML にする."""
    rows = []
    for e in events:
        icon, color = _KIND_STYLE.get(e.kind, ("•", "#666"))
        if youtube_id:
            link = f"https://youtu.be/{youtube_id}?t={int(e.timestamp)}"
            ts_cell = f'<a href="{link}" target="_blank">{fmt_ts(e.timestamp)}</a>'
        else:
            ts_cell = fmt_ts(e.timestamp)
        detail = html.escape(e.detail) if e.detail else ""
        rows.append(
            f'<tr><td class="ts">{ts_cell}</td>'
            f'<td><span class="badge" style="background:{color}">{icon} {html.escape(e.label)}</span></td>'
            f'<td class="detail">{detail}</td></tr>'
        )

    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 900px; padding: 0 1rem; }}
  h1 {{ font-size: 1.3rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td {{ padding: .35rem .6rem; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
  .ts {{ white-space: nowrap; font-variant-numeric: tabular-nums; }}
  .ts a {{ text-decoration: none; font-weight: 600; }}
  .badge {{ color: #fff; border-radius: .4rem; padding: .1rem .5rem; font-size: .85rem; white-space: nowrap; }}
  .detail {{ color: #555; font-size: .9rem; }}
  .summary {{ color: #666; margin-bottom: 1rem; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="summary">検出イベント {len(events)} 件。時刻クリックで動画の該当箇所へジャンプします。</p>
<table><tbody>
{chr(10).join(rows)}
</tbody></table>
</body></html>
"""


def save_outputs(
    events: list[TimelineEvent],
    out_html: str | Path,
    title: str,
    youtube_id: str | None = None,
) -> None:
    out_html = Path(out_html)
    out_html.write_text(render_html(events, title, youtube_id), encoding="utf-8")
    out_json = out_html.with_suffix(".json")
    out_json.write_text(
        json.dumps([e.to_dict() for e in events], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
