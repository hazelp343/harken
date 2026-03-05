"""Command-line interface for harken.

Run ``harken --help`` for usage. The ``answer`` and ``evaluate`` commands work
fully offline with the built-in dummy encoder and tiny language model, so the
pipeline can be exercised without downloading anything; pass ``--encoder`` to
use a real pretrained encoder (requires the ``hf`` extra).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from harken import __version__


def cmd_info(args: argparse.Namespace) -> int:
    from harken.encoders import list_encoders
    from harken.projectors import list_projectors

    print(f"harken {__version__}")
    print("encoders:   " + ", ".join(list_encoders()))
    print("projectors: " + ", ".join(list_projectors()))
    return 0


def _build_pipeline(args: argparse.Namespace):
    """Build an (untrained) model + processor from CLI options.

    Uses the built-in tiny LM so the command runs without model downloads.
    """
    from harken.config import AudioQAConfig
    from harken.modeling import build_model
    from harken.processor import AudioQAProcessor
    from harken.testing import TinyTokenizer

    tokenizer = TinyTokenizer()
    config = AudioQAConfig(
        encoder_name=args.encoder,
        projector=args.projector,
        encoder_dim=args.encoder_dim,
        llm_dim=args.llm_dim,
        num_audio_tokens=args.num_audio_tokens,
    )
    model = build_model(config, audio_token_id=tokenizer.audio_token_id)
    processor = AudioQAProcessor(
        tokenizer, config.num_audio_tokens, sample_rate=config.sample_rate
    )
    return model, processor


def cmd_answer(args: argparse.Namespace) -> int:
    from harken.audio_io import load_audio

    model, processor = _build_pipeline(args)
    audio = load_audio(args.audio, sr=processor.sample_rate)
    options = args.option or None
    answer = model.answer(
        processor,
        args.question,
        audio,
        options=options,
        max_new_tokens=args.max_new_tokens,
    )
    print(answer)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    import json

    from harken.data import load_examples
    from harken.evaluation import evaluate_dataset

    model, processor = _build_pipeline(args)
    examples = load_examples(args.dataset)
    result = evaluate_dataset(
        model, processor, examples, max_new_tokens=args.max_new_tokens
    )
    print(json.dumps(result["metrics"], indent=2))
    return 0


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder", default="dummy", help="registered encoder name")
    parser.add_argument("--projector", default="mlp", help="projector name")
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--llm-dim", type=int, default=64)
    parser.add_argument("--num-audio-tokens", type=int, default=8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harken",
        description="Audio question answering: pretrained audio encoders + LLMs.",
    )
    parser.add_argument(
        "--version", action="version", version=f"harken {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

    info_parser = subparsers.add_parser(
        "info", help="show version and registered components"
    )
    info_parser.set_defaults(func=cmd_info)

    answer_parser = subparsers.add_parser(
        "answer", help="answer a question about an audio file"
    )
    answer_parser.add_argument("--audio", required=True, help="path to an audio file")
    answer_parser.add_argument("--question", required=True, help="the question to ask")
    answer_parser.add_argument(
        "--option", action="append", help="multiple-choice option (repeatable)"
    )
    answer_parser.add_argument("--max-new-tokens", type=int, default=32)
    _add_model_options(answer_parser)
    answer_parser.set_defaults(func=cmd_answer)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate on a JSONL dataset of audio questions"
    )
    evaluate_parser.add_argument(
        "--dataset", required=True, help="path to a JSONL dataset"
    )
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=16)
    _add_model_options(evaluate_parser)
    evaluate_parser.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
