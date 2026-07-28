"""Tesseract によるスコアボード読み取り(ほぼ決定的処理).

v1 は数字(得点)の読み取りに絞った素朴な実装。B/S/O のランプ表示や
イニング矢印はスコアボードのレイアウト依存が強く、レイアウト定義
(将来: 対象球場ごとのテンプレート)を導入した段階で拡張する。
読めなかった・信頼度が低い場合は confidence を下げて返し、
ルール検証(rules.py)と LLM フォールバック(llm.py)に委ねる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pytesseract

    _HAS_TESSERACT = True
except ImportError:  # pragma: no cover
    _HAS_TESSERACT = False


@dataclass
class OcrResult:
    text: str
    confidence: float  # 0.0-1.0


def read_digits(binary_image: np.ndarray) -> OcrResult:
    """前処理済み画像から数字を読み取る."""
    if not _HAS_TESSERACT:
        return OcrResult(text="", confidence=0.0)

    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(
        binary_image, config=config, output_type=pytesseract.Output.DICT
    )
    words: list[str] = []
    confs: list[float] = []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        if not text:
            continue
        conf = float(conf)
        if conf < 0:  # Tesseract は非テキスト行に -1 を返す
            continue
        words.append(text)
        confs.append(conf / 100.0)

    if not words:
        return OcrResult(text="", confidence=0.0)
    return OcrResult(text=" ".join(words), confidence=float(np.mean(confs)))
