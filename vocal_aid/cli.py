"""Command-line entry point for the vocal pitch-correction engine."""
from __future__ import annotations

import argparse
import logging

from .pipeline import PipelineConfig, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vocal_aid",
        description=(
            "UTAU-style note-mapping vocal pitch-correction engine. "
            "WORLD F0 replacement + frame time-stretch only -- no SVS/SVC, no neural nets."
        ),
    )
    parser.add_argument("--source", required=True, help="Source vocal WAV file")
    parser.add_argument(
        "--melody", required=True, help="Melody file (.mid/.midi or .xml/.musicxml/.mxl)"
    )
    parser.add_argument(
        "--mode",
        choices=["a", "b"],
        default="b",
        help="Mapping strategy: a=auto syllable split, b=single sample repeat (default: b)",
    )
    parser.add_argument(
        "--flatten",
        type=float,
        default=1.0,
        help="0.0-1.0 pitch flatten strength; 1.0=fully locked to note (default: 1.0)",
    )
    parser.add_argument(
        "--portamento-ms",
        type=float,
        default=30.0,
        help="Entry portamento glide duration in ms (default: 30.0)",
    )
    parser.add_argument(
        "--vibrato-rate", type=float, default=0.0, help="Vibrato rate in Hz; 0=disabled (default: 0.0)"
    )
    parser.add_argument(
        "--vibrato-depth",
        type=float,
        default=0.0,
        help="Vibrato depth in semitones; 0=disabled (default: 0.0)",
    )
    parser.add_argument(
        "--preserve-attack",
        type=float,
        default=0.0,
        help="Attack/release window in ms held fixed during time-stretch (default: 0.0)",
    )
    parser.add_argument(
        "--frame-period-ms", type=float, default=5.0, help="WORLD analysis frame period in ms"
    )
    parser.add_argument("--out", required=True, help="Output WAV path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = PipelineConfig(
        mode=args.mode,
        flatten=args.flatten,
        portamento_ms=args.portamento_ms,
        vibrato_rate_hz=args.vibrato_rate,
        vibrato_depth_semitones=args.vibrato_depth,
        preserve_attack_ms=args.preserve_attack,
        frame_period_ms=args.frame_period_ms,
    )
    run(args.source, args.melody, args.out, config)


if __name__ == "__main__":
    main()
