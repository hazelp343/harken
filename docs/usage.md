# Usage

## Install

```bash
pip install harken            # core (numpy, torch, soundfile)
pip install "harken[hf]"      # + transformers/torchaudio for real encoders
pip install "harken[audio]"   # + librosa/resampy for nicer audio I/O
```

## A first answer (no downloads)

The built-in dummy encoder and tiny language model let you exercise the whole
pipeline offline. The answer is not meaningful (the model is untrained) — this
is for wiring, tests and demos.

```python
import numpy as np
from harken import AudioQAConfig, build_model, AudioQAProcessor
from harken.testing import TinyTokenizer

tokenizer = TinyTokenizer()
config = AudioQAConfig(encoder_dim=256, llm_dim=64, num_audio_tokens=8)
model = build_model(config, audio_token_id=tokenizer.audio_token_id)
processor = AudioQAProcessor(tokenizer, config.num_audio_tokens)

audio = np.zeros(16000, dtype=np.float32)        # one second of silence
print(model.answer(processor, "What do you hear?", audio))
```

## Loading audio

```python
from harken import load_audio

wave = load_audio("clip.wav", sr=16000)   # float32 mono at 16 kHz
```

`load_audio` reads any libsndfile-supported format, downmixes to mono and
resamples to the requested rate.

## Using a real encoder + LLM

With the `hf` extra you can plug in pretrained towers from the Hugging Face hub:

```python
from harken.encoders import get_encoder

encoder = get_encoder("whisper", model_id="openai/whisper-base")
# pass `encoder` and your own `language_model` to harken.modeling.AudioQAModel
```

Available encoders: `dummy`, `clap`, `whisper`, `ast` (see
`harken.list_encoders()`).

## Choosing a projector

```python
AudioQAConfig(projector="mlp")     # default, LLaVA-style
AudioQAConfig(projector="linear")  # cheapest
AudioQAConfig(projector="stack", stack_factor=4)  # temporal compression
AudioQAConfig(projector="query")   # fixed-width Q-Former-style
```

## Evaluating a dataset

```python
from harken.data import load_examples
from harken.evaluation import evaluate_dataset

examples = load_examples("aqua.jsonl")
result = evaluate_dataset(model, processor, examples)
print(result["metrics"])
```

## Command line

```bash
harken info
harken answer --audio clip.wav --question "What animal is this?"
harken evaluate --dataset aqua.jsonl
```
