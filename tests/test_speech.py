"""speech.py のキーワードスポッティングのテスト(モデル不要の純ロジック)."""

from baseball_scorer.speech import SpeechEvent, spot_keywords


class TestSpotKeywords:
    def test_umpire_call(self):
        hits = spot_keywords("バッターアウト!")
        words = [w for _c, w in hits]
        assert "バッターアウト" in words

    def test_longest_match_priority(self):
        # 「フォアボール」が「ボール」として二重計上されない
        hits = spot_keywords("フォアボールです")
        words = [w for _c, w in hits]
        assert "フォアボール" in words
        assert "ボール" not in words

    def test_fielder_shout_gives_direction(self):
        hits = spot_keywords("ショート!ショート!")
        assert ("position", "ショート") in hits

    def test_nice_play_cheer(self):
        hits = spot_keywords("ナイスバッティング〜")
        assert ("cheer", "ナイスバッティング") in hits

    def test_multiple_categories(self):
        hits = spot_keywords("レフトフライ、アウト")
        cats = {c for c, _w in hits}
        assert "position" in cats and "judgment" in cats

    def test_no_baseball_vocab(self):
        assert spot_keywords("今日はいい天気ですね") == []

    def test_foul_variants(self):
        assert spot_keywords("ファールボール")  # ファール表記
        assert spot_keywords("ファウルです")    # ファウル表記


class TestSpeechEvent:
    def test_has_keywords(self):
        assert not SpeechEvent(0, 1, "こんにちは").has_keywords
        e = SpeechEvent(0, 1, "アウト", keywords=[("judgment", "アウト")])
        assert e.has_keywords
