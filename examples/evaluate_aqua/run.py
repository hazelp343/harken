"""Evaluate the (download-free) stack on a tiny AQUA-style dataset.

The dataset mixes ``solvable`` and ``unanswerable`` questions grouped by clip,
so the abstention metrics (including conditional accuracy) are exercised.
Examples here have no audio, which the runner handles gracefully.
"""

import json
from pathlib import Path

from harken import AudioQAConfig, AudioQAProcessor, build_model
from harken.data import load_examples
from harken.evaluation import evaluate_dataset
from harken.testing import TinyTokenizer


def main() -> None:
    examples = load_examples(Path(__file__).parent / "sample.jsonl")

    tokenizer = TinyTokenizer()
    config = AudioQAConfig(encoder_dim=256, llm_dim=64, num_audio_tokens=8)
    model = build_model(config, audio_token_id=tokenizer.audio_token_id)
    processor = AudioQAProcessor(tokenizer, config.num_audio_tokens)

    result = evaluate_dataset(model, processor, examples, max_new_tokens=8)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
