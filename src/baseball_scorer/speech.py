"""音声認識 + 野球語彙キーワードスポッティング.

ユーザーが人力採点で使っている判断材料(審判のコール、守備の掛け声)を
そのまま機械化する層。faster-whisper(ローカル)で文字起こしし、
野球語彙辞書に一致した断片だけをイベントとして採用する。

Whisper は歓声・雑音区間で幻聴を起こすため、フル文字起こしは信用せず
「語彙ホワイトリスト一致 + セグメント時刻」だけを下流に渡す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# カテゴリ -> 表記ゆれを含むキーワード群
# 判定系は打席結果の確定材料、守備位置系は打球方向の推定材料になる
KEYWORDS: dict[str, list[str]] = {
    "judgment": [
        "アウト", "セーフ", "ファール", "ファウル", "ストライク", "ボール",
        "フォアボール", "デッドボール", "三振", "ホームラン", "タイム",
        "バッターアウト", "フェア",
    ],
    "position": [
        "ピッチャー", "キャッチャー", "ファースト", "セカンド", "サード",
        "ショート", "レフト", "センター", "ライト",
    ],
    "play": [
        "盗塁", "牽制", "ゲッツー", "ダブルプレー", "タッチアップ",
        "振り逃げ", "バント", "スクイズ", "チェンジ",
    ],
    "cheer": [
        "ナイスバッティング", "ナイスバッチ", "ナイスピッチ", "ナイスボール",
        "ナイスキャッチ", "ナイスラン", "ナイスカバー",
    ],
}


@dataclass
class SpeechEvent:
    start: float
    end: float
    text: str                      # Whisper が出したセグメント全文
    keywords: list[tuple[str, str]] = field(default_factory=list)  # (category, word)

    @property
    def has_keywords(self) -> bool:
        return bool(self.keywords)


def spot_keywords(text: str, vocab: dict[str, list[str]] | None = None) -> list[tuple[str, str]]:
    """テキストから野球語彙を抽出する(部分一致・長い語優先)."""
    vocab = vocab or KEYWORDS
    hits: list[tuple[str, str]] = []
    consumed = text
    # 長い語から照合して「フォアボール」を「ボール」より優先する
    all_words = sorted(
        ((cat, w) for cat, ws in vocab.items() for w in ws),
        key=lambda cw: -len(cw[1]),
    )
    for cat, word in all_words:
        if word in consumed:
            hits.append((cat, word))
            consumed = consumed.replace(word, "◆" * len(word))
    return hits


def transcribe(
    wav_path: str | Path,
    model_size: str = "small",
    extra_vocab: list[str] | None = None,
) -> list[SpeechEvent]:
    """faster-whisper で文字起こしし、キーワード付きセグメントを返す.

    extra_vocab には選手名などを渡す(initial_prompt として Whisper に
    ヒントを与えると固有名詞の認識率が上がる)。
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    prompt = "草野球の試合。アウト、セーフ、ストライク、ファール、ナイスバッティング。"
    if extra_vocab:
        prompt += " " + "、".join(extra_vocab) + "。"

    segments, _info = model.transcribe(
        str(wav_path),
        language="ja",
        vad_filter=True,
        initial_prompt=prompt,
        condition_on_previous_text=False,  # 幻聴の連鎖を防ぐ
    )

    events: list[SpeechEvent] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        events.append(
            SpeechEvent(
                start=float(seg.start),
                end=float(seg.end),
                text=text,
                keywords=spot_keywords(text),
            )
        )
    return events
