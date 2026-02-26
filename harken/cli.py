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
