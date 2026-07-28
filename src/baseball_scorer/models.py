"""ゲーム状態と読み取り結果のデータモデル."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass(frozen=True)
class GameState:
    """ある瞬間の試合状態(スコアボードから読み取れる情報)."""

    inning: int = 1
    top: bool = True  # 表 = True / 裏 = False
    balls: int = 0
    strikes: int = 0
    outs: int = 0
    score_away: int = 0  # 先攻
    score_home: int = 0  # 後攻
    # (一塁, 二塁, 三塁)。読み取れない場合は None
    runners: tuple[bool, bool, bool] | None = None

    @property
    def batting_team(self) -> Literal["away", "home"]:
        return "away" if self.top else "home"

    def half_inning_index(self) -> int:
        """表裏を通し番号にしたもの。進行方向の比較に使う."""
        return (self.inning - 1) * 2 + (0 if self.top else 1)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoreboardReading:
    """OCR または LLM による 1 フレーム分の生読み取り."""

    timestamp: float  # 動画内の秒数
    state: GameState
    confidence: float  # 0.0-1.0
    source: Literal["ocr", "llm"] = "ocr"
    suspect_reasons: list[str] = field(default_factory=list)

    @property
    def is_suspect(self) -> bool:
        return bool(self.suspect_reasons)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "state": self.state.to_dict(),
            "confidence": self.confidence,
            "source": self.source,
            "suspect_reasons": self.suspect_reasons,
        }


@dataclass
class PlayEvent:
    """状態遷移から推定されたプレー内容の候補(LLM 生成、人間レビュー前提)."""

    start_ts: float
    end_ts: float
    from_state: GameState
    to_state: GameState
    description: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "from_state": self.from_state.to_dict(),
            "to_state": self.to_state.to_dict(),
            "description": self.description,
            "confidence": self.confidence,
        }
