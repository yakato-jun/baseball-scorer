"""rules.py(決定的なルール検証)のテスト."""

from baseball_scorer.models import GameState, ScoreboardReading
from baseball_scorer.rules import (
    annotate_readings,
    dedupe_readings,
    validate_state,
    validate_transition,
)


def reading(ts, state, conf=0.9):
    return ScoreboardReading(timestamp=ts, state=state, confidence=conf)


class TestValidateState:
    def test_valid_default(self):
        assert validate_state(GameState()) == []

    def test_valid_full_count(self):
        assert validate_state(GameState(balls=3, strikes=2, outs=2)) == []

    def test_invalid_balls(self):
        assert validate_state(GameState(balls=4))  # 4 ボール表示はありえない

    def test_invalid_strikes(self):
        assert validate_state(GameState(strikes=3))

    def test_invalid_outs(self):
        assert validate_state(GameState(outs=3))

    def test_invalid_inning(self):
        assert validate_state(GameState(inning=0))

    def test_negative_score(self):
        assert validate_state(GameState(score_home=-1))


class TestValidateTransition:
    def test_normal_count_progress(self):
        a = GameState(balls=1, strikes=1)
        b = GameState(balls=2, strikes=1)
        assert validate_transition(a, b) == []

    def test_score_cannot_decrease(self):
        a = GameState(score_away=3)
        b = GameState(score_away=2)
        assert any("decreased" in e for e in validate_transition(a, b))

    def test_inning_cannot_go_backwards(self):
        a = GameState(inning=5, top=False)
        b = GameState(inning=5, top=True)
        assert any("backwards" in e for e in validate_transition(a, b))

    def test_top_to_bottom_is_forward(self):
        a = GameState(inning=3, top=True, outs=2)
        b = GameState(inning=3, top=False, outs=0)
        assert validate_transition(a, b) == []

    def test_fielding_team_cannot_score(self):
        # 3 回表(away の攻撃)中に home の得点が増えるのは誤読
        a = GameState(inning=3, top=True, score_home=1)
        b = GameState(inning=3, top=True, score_home=2)
        assert any("fielding" in e for e in validate_transition(a, b))

    def test_batting_team_can_score(self):
        a = GameState(inning=3, top=True, score_away=1)
        b = GameState(inning=3, top=True, score_away=3)
        assert validate_transition(a, b) == []

    def test_outs_cannot_decrease_within_half(self):
        a = GameState(inning=2, top=True, outs=2)
        b = GameState(inning=2, top=True, outs=1)
        assert any("outs decreased" in e for e in validate_transition(a, b))

    def test_outs_reset_on_new_half(self):
        a = GameState(inning=2, top=True, outs=2)
        b = GameState(inning=2, top=False, outs=0)
        assert validate_transition(a, b) == []

    def test_grand_slam_is_plausible(self):
        a = GameState(score_away=0)
        b = GameState(score_away=4)  # 満塁本塁打
        assert validate_transition(a, b) == []

    def test_absurd_score_jump(self):
        a = GameState(score_away=0)
        b = GameState(score_away=8)  # 一遷移で 8 点は誤読(例: 0 を 8 と読んだ)
        assert any("jumped" in e for e in validate_transition(a, b))


class TestAnnotateReadings:
    def test_clean_sequence_stays_clean(self):
        rs = annotate_readings([
            reading(0, GameState(balls=0)),
            reading(1, GameState(balls=1)),
            reading(2, GameState(balls=2)),
        ])
        assert all(not r.is_suspect for r in rs)

    def test_low_confidence_is_suspect(self):
        rs = annotate_readings([reading(0, GameState(), conf=0.3)])
        assert rs[0].is_suspect

    def test_bad_transition_marks_only_offender(self):
        # 得点 2 → 7 → 2: 中央の「7」が誤読。前後は正常のまま残る
        rs = annotate_readings([
            reading(0, GameState(score_away=2)),
            reading(1, GameState(score_away=7)),
            reading(2, GameState(score_away=2)),
        ])
        assert not rs[0].is_suspect
        assert rs[1].is_suspect  # 2 -> 7 は +5 でありえない
        assert not rs[2].is_suspect  # 直前の「正常な」読み取り(=2)と比較される

    def test_suspect_does_not_poison_baseline(self):
        # 静的に不正な状態は遷移基準にならない
        rs = annotate_readings([
            reading(0, GameState(outs=1)),
            reading(1, GameState(outs=5)),  # 静的違反
            reading(2, GameState(outs=2)),
        ])
        assert rs[1].is_suspect
        assert not rs[2].is_suspect


class TestDedupeReadings:
    def test_consecutive_identical_states_are_merged(self):
        rs = dedupe_readings([
            reading(0, GameState(balls=1)),
            reading(1, GameState(balls=1)),
            reading(2, GameState(balls=2)),
            reading(3, GameState(balls=2)),
        ])
        assert len(rs) == 2
        assert [r.timestamp for r in rs] == [0, 2]


class TestGameState:
    def test_half_inning_index_is_monotonic(self):
        seq = [
            GameState(inning=1, top=True),
            GameState(inning=1, top=False),
            GameState(inning=2, top=True),
            GameState(inning=2, top=False),
        ]
        indices = [s.half_inning_index() for s in seq]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)

    def test_batting_team(self):
        assert GameState(top=True).batting_team == "away"
        assert GameState(top=False).batting_team == "home"
