# Design notes

Some decisions and trade-offs behind harken.

## Placeholder expansion instead of mask surgery

Rather than inserting audio embeddings at runtime and patching attention masks,
harken keeps `num_audio_tokens` real placeholder tokens in `input_ids`. The
model only *overwrites the embeddings* at those positions. This keeps the
attention mask, position ids and label alignment trivially correct, at the cost
of fixing the audio length to `num_audio_tokens` per clip. For audio QA — where
a compact pooled representation is usually enough — this is a good trade.

## A numpy front end

The log-mel front end (`harken.features`) is plain numpy, so importing harken
and preprocessing audio needs no torchaudio/librosa. Encoder wrappers that
prefer their own feature extractor (Whisper, AST) still use it via the `hf`
extra; the numpy path is the dependency-light default.

## Frozen encoder, trainable projector

The default recipe freezes the encoder and trains only the projector (and
optionally the LLM via LoRA, externally). `AudioQAModel` freezes the encoder by
default; `trainable_parameters()` reports what is left. This matches the
data-efficient two-stage recipe from the vision-language literature.

## Projector zoo

Four projectors cover the common designs:

- **linear / mlp** — per-frame projection then adaptive pooling to a fixed
  token count. `mlp` is the default.
- **stack** — concatenate consecutive frames before projecting (temporal
  downsampling), following Gazelle.
- **query** — learned query tokens cross-attend to the frames, giving a fixed
  output width independent of input length (a lightweight Q-Former).

All share Xavier weight initialisation via `Projector._init_weights`.

## Abstention-aware evaluation

Real audio QA includes questions that *cannot* be answered from the clip. Borrowing
from AQUA-Bench, `score_abstention` reports per-type accuracy and a
**conditional** accuracy on unanswerable items, counted only where the paired
solvable item was answered correctly. This separates "knows when to abstain"
from "can recognise the sound at all".

## Testing stand-ins

`DummyAudioEncoder` and `TinyCausalLM` implement the production interfaces with
no weights, so CI exercises the full path — including a real back-propagation
step — without downloads. They live in `harken.testing` and are intentionally
importable by users writing their own tests.
