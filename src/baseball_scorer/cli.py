"""コマンドラインインターフェース."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="baseball-scorer",
        description="野球動画からスコアを収集する",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="動画を解析してスコア時系列を出力する")
    p_analyze.add_argument("video", help="入力動画ファイル")
    p_analyze.add_argument("-o", "--output", default="result.json", help="出力 JSON パス")
    p_analyze.add_argument("--fps", type=float, default=1.0, help="サンプリング fps(デフォルト 1.0)")
    p_analyze.add_argument("--no-llm", action="store_true", help="LLM フォールバックを使わない")
    p_analyze.add_argument("--work-dir", default=None, help="中間ファイルの出力先(デバッグ用)")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        from .pipeline import analyze

        result = analyze(
            args.video,
            fps=args.fps,
            use_llm=not args.no_llm,
            work_dir=args.work_dir,
        )
        result.save(args.output)
        if not result.region_found:
            print(
                "警告: スコアボード領域を検出できませんでした。"
                "動画にスコア表示が映っているか確認してください。",
                file=sys.stderr,
            )
            return 1
        print(f"読み取り {len(result.readings)} 件 / プレー候補 {len(result.events)} 件 -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
