# harken

**Connect pretrained audio encoders to large language models for audio question
answering.**

harken wires a frozen audio encoder (CLAP, Whisper, AST, …) to a causal LLM
through a small trainable projector, then splices the projected audio into the
prompt so the model can answer questions about what it hears. It is a compact,
hackable framework — encoder, projector and LLM are independent pieces you can
swap freely — with a fully tested core that runs on CPU without downloading any
weights.

```python
import numpy as np
from harken import AudioQAConfig, build_model, AudioQAProcessor
from harken.testing import TinyTokenizer

tokenizer = TinyTokenizer()
config = AudioQAConfig(encoder_dim=256, llm_dim=64, num_audio_tokens=8)
model = build_model(config, audio_token_id=tokenizer.audio_token_id)
processor = AudioQAProcessor(tokenizer, config.num_audio_tokens)

audio = np.zeros(16000, dtype=np.float32)
print(model.answer(processor, "What do you hear?", audio))
```

## Features

- **Pluggable encoders** — `dummy`, `clap`, `whisper`, `ast`, plus a registry to
  add your own.
- **A projector zoo** — `linear`, `mlp`, `stack` (temporal compression) and
  `query` (Q-Former-style), all sharing one interface.
- **LLaVA-style splicing** — audio embeddings replace placeholder tokens in the
  prompt; attention masks and labels stay correct by construction.
- **Abstention-aware evaluation** — exact-match / token-F1 plus conditional
  accuracy on unanswerable questions, inspired by AQUA-Bench.
- **Download-free core** — a dummy encoder and a tiny LM exercise the full
  pipeline (including a real training step) in CI on CPU.
- **Typed, linted, tested** — ruff, mypy and pytest, shipping a `py.typed`
  marker.

## Install

```bash
pip install harken            # core: numpy, torch, soundfile
pip install "harken[hf]"      # real encoders (transformers, torchaudio)
pip install "harken[audio]"   # librosa + resampy for nicer audio I/O
```

## Command line

```bash
harken info
harken answer --audio clip.wav --question "What animal is this?"
harken evaluate --dataset aqua.jsonl
```

## How it works

```
waveform → AudioEncoder (frozen) → Projector → audio tokens ┐
prompt "<audio>\nWhat do you hear?" → tokenizer → embeds ───┴→ Causal LM → answer
```

See [docs/architecture.md](docs/architecture.md) for the full picture.

## Documentation

- [Usage guide](docs/usage.md)
- [Architecture](docs/architecture.md)
- [Design notes](docs/design-notes.md)
- [API reference](docs/api-reference.md)
- Runnable [examples](examples/)

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
mypy harken
```

## License

[MIT](LICENSE)
