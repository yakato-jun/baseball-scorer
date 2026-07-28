"""コマンドラインインターフェース."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="baseball-scorer",
        description="草野球動画から打席タイムラインを生成する補助ツール",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tl = sub.add_parser(
        "timeline",
        help="動画(ローカルファイル or YouTube URL)からイベントタイムラインを生成する",
    )
    p_tl.add_argument("input", help="動画ファイルパス または YouTube URL")
    p_tl.add_argument("-o", "--output", default="timeline.html", help="出力 HTML パス")
    p_tl.add_argument("--work-dir", default="work", help="ダウンロード・中間ファイル置き場")
    p_tl.add_argument("--no-speech", action="store_true", help="音声認識をスキップ(高速)")
    p_tl.add_argument("--no-motion", action="store_true", help="映像の活動量計測をスキップ")
    p_tl.add_argument(
        "--whisper-model", default="small",
        help="faster-whisper モデルサイズ (tiny/base/small/medium)",
    )
    p_tl.add_argument(
        "--vocab", default=None,
        help="追加語彙(選手名など)のテキストファイル。1行1語",
    )

    args = parser.parse_args(argv)
    if args.command == "timeline":
        return _run_timeline(args)
    return 0


def _run_timeline(args) -> int:
    from . import ingest
    from .audio_events import detect_impacts, load_wav
    from .timeline import fuse_events, save_outputs

    work_dir = Path(args.work_dir)
    youtube_id = None

    if args.input.startswith(("http://", "https://")):
        youtube_id = ingest.youtube_id(args.input)
        print(f"ダウンロード中: {args.input}")
        video_path = ingest.download(args.input, work_dir)
    else:
        video_path = Path(args.input)
        if not video_path.exists():
            print(f"エラー: {video_path} が見つかりません", file=sys.stderr)
            return 1

    print("音声抽出中...")
    wav_path = ingest.extract_audio(video_path)

    print("打撃音・捕球音の検出中...")
    samples, sr = load_wav(wav_path)
    impacts = detect_impacts(samples, sr)
    print(f"  インパルス候補 {len(impacts)} 件")

    speech = None
    if not args.no_speech:
        from .speech import transcribe

        extra_vocab = None
        if args.vocab:
            extra_vocab = [
                line.strip()
                for line in Path(args.vocab).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        print(f"音声認識中 (whisper {args.whisper_model}, ローカルCPU)... 実時間の数割かかります")
        speech = transcribe(wav_path, model_size=args.whisper_model, extra_vocab=extra_vocab)
        kw = sum(1 for s in speech if s.has_keywords)
        print(f"  発話セグメント {len(speech)} 件(うち野球語彙一致 {kw} 件)")

    motion = None
    if not args.no_motion:
        from .motion import measure_activity

        print("映像の活動量計測中...")
        motion = measure_activity(video_path)
        print(f"  サンプル {len(motion.times)} 点 / カメラズレ {len(motion.camera_shifts)} 回")

    events = fuse_events(impacts=impacts, speech=speech, motion=motion)
    title = f"タイムライン: {video_path.stem}"
    save_outputs(events, args.output, title, youtube_id=youtube_id)
    print(f"イベント {len(events)} 件 -> {args.output} (+ .json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
