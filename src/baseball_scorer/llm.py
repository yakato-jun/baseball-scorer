"""Claude によるフォールバック処理(曖昧さの解決専用).

呼ばれるのは 2 箇所だけ:
  1. read_scoreboard_image: OCR が低信頼 or ルール違反を出したフレームの判読
  2. describe_play: 確定した状態遷移から「何が起きたか」の候補生成

どちらも構造化出力(JSON Schema)で返答形式を強制する。
ANTHROPIC_API_KEY が未設定の環境でも import できるよう、クライアント生成は遅延させる。
"""

from __future__ import annotations

import base64
import json

from .models import GameState, PlayEvent, ScoreboardReading

MODEL = "claude-opus-5"

_READING_SCHEMA = {
    "type": "object",
    "properties": {
        "readable": {
            "type": "boolean",
            "description": "スコアボードとして判読できたか",
        },
        "inning": {"type": "integer"},
        "top": {"type": "boolean", "description": "イニングの表なら true"},
        "balls": {"type": "integer"},
        "strikes": {"type": "integer"},
        "outs": {"type": "integer"},
        "score_away": {"type": "integer", "description": "先攻チームの得点"},
        "score_home": {"type": "integer", "description": "後攻チームの得点"},
        "confidence": {
            "type": "number",
            "description": "読み取り全体の確信度 0.0-1.0",
        },
    },
    "required": [
        "readable", "inning", "top", "balls", "strikes", "outs",
        "score_away", "score_home", "confidence",
    ],
    "additionalProperties": False,
}

_PLAY_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "この区間で起きたプレーの推定(日本語・簡潔に)",
        },
        "confidence": {"type": "number", "description": "確信度 0.0-1.0"},
    },
    "required": ["description", "confidence"],
    "additionalProperties": False,
}


def _client():
    import anthropic

    return anthropic.Anthropic()


def _parse_json_response(response) -> dict:
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def read_scoreboard_image(
    png_bytes: bytes,
    timestamp: float,
    context: GameState | None = None,
) -> ScoreboardReading | None:
    """スコアボードのクロップ画像を Claude Vision で判読する.

    context には直前の確定状態を渡す(誤読の抑制に効く)。
    判読不能なら None。
    """
    hint = ""
    if context is not None:
        hint = (
            f"\n参考: 直前の確定状態は {context.inning}回"
            f"{'表' if context.top else '裏'}, "
            f"B{context.balls} S{context.strikes} O{context.outs}, "
            f"{context.score_away}-{context.score_home} でした。"
            "ただし画像に映っている値を優先してください。"
        )

    response = _client().messages.create(
        model=MODEL,
        max_tokens=1024,
        output_config={"format": {"type": "json_schema", "schema": _READING_SCHEMA}},
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(png_bytes).decode(),
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "これは野球のスコアボードの切り抜き画像です。"
                        "表示されている試合状態を読み取ってください。"
                        "判読できない項目があれば readable を false にしてください。"
                        + hint
                    ),
                },
            ],
        }],
    )
    if response.stop_reason == "refusal":
        return None

    data = _parse_json_response(response)
    if not data["readable"]:
        return None
    return ScoreboardReading(
        timestamp=timestamp,
        state=GameState(
            inning=data["inning"],
            top=data["top"],
            balls=data["balls"],
            strikes=data["strikes"],
            outs=data["outs"],
            score_away=data["score_away"],
            score_home=data["score_home"],
        ),
        confidence=float(data["confidence"]),
        source="llm",
    )


def describe_play(
    from_reading: ScoreboardReading,
    to_reading: ScoreboardReading,
) -> PlayEvent:
    """確定済みの 2 状態間で「何が起きたか」の候補を生成する(人間レビュー前提)."""
    a, b = from_reading.state, to_reading.state
    prompt = (
        "野球の試合で、スコアボードが次のように変化しました。\n"
        f"変化前: {a.inning}回{'表' if a.top else '裏'} "
        f"B{a.balls} S{a.strikes} O{a.outs} 得点 {a.score_away}-{a.score_home}\n"
        f"変化後: {b.inning}回{'表' if b.top else '裏'} "
        f"B{b.balls} S{b.strikes} O{b.outs} 得点 {b.score_away}-{b.score_home}\n"
        f"経過時間: 約{to_reading.timestamp - from_reading.timestamp:.0f}秒\n\n"
        "この間に起きたプレーとして最も可能性が高いものを 1 つ、簡潔に推定してください。"
    )
    response = _client().messages.create(
        model=MODEL,
        max_tokens=1024,
        output_config={"format": {"type": "json_schema", "schema": _PLAY_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    data = _parse_json_response(response)
    return PlayEvent(
        start_ts=from_reading.timestamp,
        end_ts=to_reading.timestamp,
        from_state=a,
        to_state=b,
        description=data["description"],
        confidence=float(data["confidence"]),
    )
