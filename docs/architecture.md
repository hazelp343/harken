# Architecture

harken connects a **pretrained audio encoder** to a **causal language model** so
the LLM can answer questions about sound. The design follows the
encoder → projector → LLM recipe popularised by vision-language models such as
LLaVA, adapted for audio.

```
 waveform ──► AudioEncoder ──► frame embeddings ──► Projector ──► audio tokens
                (frozen)        (B, T, Denc)                       (B, N, Dllm)
                                                                       │
 prompt "<audio>\nWhat do you hear?" ──► tokenizer ──► input embeds ──┘ (splice)
                                                                       │
                                                              Causal LM ──► answer
```

## Components

| Component | Module | Responsibility |
|-----------|--------|----------------|
| `AudioEncoder` | `harken.encoders` | Turn a waveform into frame embeddings. Usually a frozen pretrained model (CLAP, Whisper, AST). |
| `Projector` | `harken.projectors` | Map `(B, T, Denc)` frame embeddings to a fixed `(B, N, Dllm)` sequence in the LLM's space. |
| `AudioQAProcessor` | `harken.processor` | Expand the `<audio>` placeholder, tokenize, and batch into padded tensors. |
| `AudioQAModel` | `harken.modeling` | Encode + project audio, splice it into the prompt embeddings, run the LM. |
| evaluation | `harken.evaluation` | Exact-match / token-F1 plus abstention-aware scoring. |

## How audio enters the token stream

1. The prompt contains a single `<audio>` placeholder.
2. The processor expands it into `num_audio_tokens` copies of the audio token.
3. At `forward` time the model embeds the token ids, encodes and projects the
   audio into exactly `num_audio_tokens` embeddings, and **replaces** the audio
   token positions with those embeddings (`_merge_audio_embeddings`).
4. The combined `inputs_embeds` are fed to the language model.

Because the audio tokens are real positions in `input_ids`, the attention mask
produced by the processor already covers them — no separate mask surgery is
required.

## Why a separable design

The encoder, projector and LLM are independent objects wired together by
`build_model`. This makes it cheap to swap an encoder (e.g. CLAP → Whisper),
change the projector (`mlp`, `linear`, `stack`, `query`), or point at a
different LLM, without touching the merging logic. A frozen encoder plus a small
trainable projector is the standard, data-efficient training recipe.

## Testing without downloads

A `DummyAudioEncoder` and a `TinyCausalLM` (see `harken.testing`) implement the
same interfaces as the real components, so the whole pipeline — including
training a projector by back-propagation — runs on CPU in milliseconds with no
model downloads. The unit tests rely on these stand-ins.
