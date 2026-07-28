"""野球ルールに基づく決定的検証(状態機械).

パイプラインの肝。OCR の読み取り結果の時系列に対してルール制約を適用し、
「ありえない」遷移・状態を機械的に検出する。ここで疑わしいとマークされた
読み取りだけが LLM フォールバックに回る。

このモジュールは純粋関数のみで構成する(I/O なし・副作用なし)。
"""

from __future__ import annotations

from .models import GameState, ScoreboardReading

MAX_BALLS = 3  # 表示上の最大値(4 ボールになった瞬間はリセットされる)
MAX_STRIKES = 2
MAX_OUTS = 2
# 1 プレーで入る得点の上限(満塁本塁打)。これを超える増分はまとめ読み飛ばしか誤読
MAX_RUNS_PER_GAP = 4


def validate_state(state: GameState) -> list[str]:
    """単一状態の静的検証。違反理由のリストを返す(空なら正常)."""
    errors: list[str] = []
    if state.inning < 1:
        errors.append(f"inning {state.inning} < 1")
    if not (0 <= state.balls <= MAX_BALLS):
        errors.append(f"balls {state.balls} out of range 0-{MAX_BALLS}")
    if not (0 <= state.strikes <= MAX_STRIKES):
        errors.append(f"strikes {state.strikes} out of range 0-{MAX_STRIKES}")
    if not (0 <= state.outs <= MAX_OUTS):
        errors.append(f"outs {state.outs} out of range 0-{MAX_OUTS}")
    if state.score_away < 0 or state.score_home < 0:
        errors.append("negative score")
    return errors


def validate_transition(prev: GameState, curr: GameState) -> list[str]:
    """連続する 2 読み取り間の遷移検証。違反理由のリストを返す.

    フレーム間で複数プレーが起きている可能性はあるため、
    「単一プレーとして説明可能か」ではなく「物理的にありえるか」だけを見る。
    """
    errors: list[str] = []

    # 得点は減らない
    if curr.score_away < prev.score_away:
        errors.append(
            f"away score decreased {prev.score_away} -> {curr.score_away}"
        )
    if curr.score_home < prev.score_home:
        errors.append(
            f"home score decreased {prev.score_home} -> {curr.score_home}"
        )

    # イニングは逆行しない
    if curr.half_inning_index() < prev.half_inning_index():
        errors.append(
            f"inning went backwards: {prev.inning}{'表' if prev.top else '裏'}"
            f" -> {curr.inning}{'表' if curr.top else '裏'}"
        )

    # 同一ハーフイニング内では、守備側チームの得点は増えない
    if curr.half_inning_index() == prev.half_inning_index():
        if prev.top and curr.score_home > prev.score_home:
            errors.append("home (fielding) score increased during top half")
        if not prev.top and curr.score_away > prev.score_away:
            errors.append("away (fielding) score increased during bottom half")

        # アウトカウントは同一ハーフイニング内で減らない
        if curr.outs < prev.outs:
            errors.append(
                f"outs decreased {prev.outs} -> {curr.outs} within same half inning"
            )

    # 1 遷移あたりの得点増分の妥当性(サンプリング間隔が短い前提のゆるい上限)
    diff = (curr.score_away - prev.score_away) + (curr.score_home - prev.score_home)
    if diff > MAX_RUNS_PER_GAP:
        errors.append(f"score jumped by {diff} (> {MAX_RUNS_PER_GAP}) in one step")

    return errors


def annotate_readings(
    readings: list[ScoreboardReading],
    min_confidence: float = 0.6,
) -> list[ScoreboardReading]:
    """読み取り時系列にルール検証を適用し、suspect_reasons を付与して返す.

    遷移検証は「直前の正常な読み取り」を基準に行う。疑わしい読み取りを
    基準にすると誤読が連鎖的に後続を汚染するため。
    """
    last_good: GameState | None = None
    for r in readings:
        reasons = list(r.suspect_reasons)
        reasons += validate_state(r.state)
        if r.confidence < min_confidence:
            reasons.append(f"low confidence {r.confidence:.2f}")
        if last_good is not None and not reasons:
            reasons += validate_transition(last_good, r.state)
        r.suspect_reasons = reasons
        if not reasons:
            last_good = r.state
    return readings


def dedupe_readings(readings: list[ScoreboardReading]) -> list[ScoreboardReading]:
    """状態が変化した読み取りだけを残す(連続する同一状態を圧縮)."""
    result: list[ScoreboardReading] = []
    for r in readings:
        if not result or result[-1].state != r.state:
            result.append(r)
    return result
