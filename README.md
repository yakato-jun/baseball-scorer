# baseball-scorer

野球の試合動画をアップロードすると、スコア収集(スコアブック作成)を支援するシステム。

## コンセプト

**決定的アルゴリズム(ホワイトボックス)でできることは全てコードで行い、曖昧な判断だけを LLM に委ねる。**

| 処理 | 担当 | 理由 |
|---|---|---|
| 動画の分割・フレーム抽出 | ffmpeg | 決定的・高速・無料 |
| シーン変化検出 | OpenCV(フレーム差分) | 決定的 |
| スコアボード領域の特定 | OpenCV(時間方向の分散解析) | 決定的 |
| スコア・カウントの読み取り | OCR(Tesseract) | ほぼ決定的 |
| 読み取り結果の整合性検証 | 野球ルールの状態機械 | 完全に決定的 |
| 低信頼度フレームの判読 | **Claude(Vision)** | 曖昧・文脈依存 |
| プレー内容の推定(何が起きたか) | **Claude** | 曖昧・文脈依存 |

## パイプライン

```
動画 → [1] フレーム抽出 (ffmpeg)
     → [2] シーン分割 (OpenCV)
     → [3] スコアボード領域検出 (OpenCV)
     → [4] OCR 読み取り (Tesseract)
     → [5] ルール検証 (状態機械) ──✓──→ スコアデータ (JSON)
                │
                ✗ 矛盾・低信頼度
                ↓
     → [6] LLM フォールバック (Claude Vision + 構造化出力)
     → [7] プレー推定 (状態遷移の差分から LLM が「何が起きたか」を推定)
```

詳細は [docs/DESIGN.md](docs/DESIGN.md) を参照。

## セットアップ

```bash
# システム依存
sudo apt install ffmpeg tesseract-ocr

# Python
pip install -e ".[dev]"

# LLM フォールバックを使う場合
export ANTHROPIC_API_KEY=sk-ant-...
```

## 使い方

```bash
# 動画からスコアを抽出
baseball-scorer analyze game.mp4 -o result.json

# LLM フォールバックなし(決定的処理のみ)
baseball-scorer analyze game.mp4 --no-llm

# テスト
pytest
```

## 開発状況

- [x] ゲーム状態モデル + ルール検証(状態機械)
- [x] フレーム抽出(ffmpeg ラッパー)
- [x] スコアボード領域検出(時間分散方式)
- [x] OCR / LLM フォールバックの骨格
- [ ] 実動画でのチューニング(サンプル動画が必要)
- [ ] レビュー用 UI(人間による最終確認)
