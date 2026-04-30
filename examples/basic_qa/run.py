"""Minimal audio question-answering example.

Runs entirely offline with the built-in dummy encoder and tiny language model,
so it needs no downloads. The answer is not meaningful (the model is untrained)
-- the point is to show how the pieces fit together.
"""

import numpy as np

from harken import AudioQAConfig, AudioQAProcessor, build_model
from harken.testing import TinyTokenizer


def main() -> None:
    tokenizer = TinyTokenizer()
    config = AudioQAConfig(encoder_dim=256, llm_dim=64, num_audio_tokens=8)
    model = build_model(config, audio_token_id=tokenizer.audio_token_id)
    processor = AudioQAProcessor(tokenizer, config.num_audio_tokens)

    # One second of a 440 Hz tone standing in for a real recording.
    sr = config.sample_rate
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    question = "What do you hear in this recording?"
    answer = model.answer(processor, question, tone, max_new_tokens=12)
    print("Q:", question)
    print("A:", answer)


if __name__ == "__main__":
    main()
