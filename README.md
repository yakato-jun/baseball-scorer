# baseball-scorer

草野球のホームビデオ(YouTube)から**打席タイムライン**を生成する、スコア収集の補助ツール。

## 背景

スコアは人間がつける。その際「試合動画を早回しして結果を1つずつ確認する」作業に
時間がかかるので、**何かが起きた時刻の一覧(クリックで動画の該当箇所へジャンプ)**を
自動生成して、この作業を置き換えるのが第一目標。

## コンセプト

**決定的アルゴリズム(ホワイトボックス)でできることは全てコードで行い、曖昧な判断だけを LLM に委ねる。**

人力採点で使っている判断材料 —— 審判のコール(「アウト!」)、守備の掛け声
(「ショート!」「ナイスレフト!」)、打撃音 —— をそのまま機械化する。

| シグナル | 手法 | 性質 |
|---|---|---|
| 打撃音・捕球音 | DSP(高域スペクトラルフラックスのピーク検出) | 完全に決定的 |
| 審判コール・掛け声 | ローカル Whisper + 野球語彙キーワードスポッティング | ほぼ決定的 |
| フィールドの動き | OpenCV(領域差分の活動量) | 完全に決定的 |
| カメラズレ | OpenCV(位相相関) | 完全に決定的 |
| 曖昧な区間の裁定 | LLM(クリップ+文脈 → 構造化出力)※未実装 | LLM |

## セットアップ

```bash
sudo apt install ffmpeg   # macOS: brew install ffmpeg
pip install -e ".[dev]"
```

## 使い方

```bash
# YouTube URL から直接(ローカル環境で実行)
baseball-scorer timeline "https://www.youtube.com/watch?v=XXXX" -o timeline.html

# ローカルの動画ファイルから
baseball-scorer timeline game.mp4 -o timeline.html

# 高速モード(音声認識をスキップ、打撃音+映像のみ)
baseball-scorer timeline game.mp4 --no-speech

# 選手名を追加語彙として渡す(認識率向上)
baseball-scorer timeline game.mp4 --vocab players.txt
```

出力は `timeline.html`(時刻クリックで YouTube の該当秒へジャンプ)と
同名の `.json`(機械可読)。

> **注**: YouTube のダウンロードにはローカル環境を推奨。データセンター IP からは
> YouTube のボット対策(PO Token / IP バインディング)により取得できないことがある。

## MVP ロードマップ

- [x] **MVP①**: 打撃音検出 + キーワードスポッティング + 活動量 → ジャンプ可能なタイムライン
- [ ] **MVP②**: 投球検出 + 打者の人物クラスタリングによる打席グルーピング
- [ ] **MVP③**: ファール vs 打席結果の分類(打者連続性 + 音声 + 曖昧ケースのみ LLM)
- [ ] **MVP④**: 攻守交代(イニング境界)検出
- [ ] **MVP⑤**: 投球間イベント(盗塁・牽制)の差し込み
- [ ] 手動スコア(`data/ground_truth/`)とのアラインメントによる評価・タイムスタンプ付与

設計の詳細は [docs/DESIGN.md](docs/DESIGN.md)。

## 評価データ

`data/ground_truth/` に人力スコア 4 試合分(対応する YouTube 動画あり)。
検出精度の測定と、スコア⇔動画のアラインメント開発に使う。

## テスト

```bash
pytest   # 合成音声・合成動画による決定的テスト
```
