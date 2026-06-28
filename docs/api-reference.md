# API reference

A summary of the public surface. Import everything top-level from `harken`
unless noted.

## Models

### `AudioQAConfig(...)`
Dataclass describing a model. Key fields: `encoder_name`, `llm_name`,
`projector`, `encoder_dim`, `llm_dim`, `num_audio_tokens`, `projector_hidden_dim`,
`stack_factor`, `sample_rate`, `audio_token`, `system_prompt`,
`encoder_kwargs`, `projector_kwargs`.
Methods: `to_dict()`, `from_dict(d)`, `save(path)`, `load(path)`.

### `build_model(config, audio_token_id, language_model=None, freeze_encoder=True)`
Assemble an `AudioQAModel`. Uses the built-in tiny LM when `language_model` is
`None`.

### `AudioQAModel`
- `encode_audio(audio_values) -> (B, N, Dllm)`
- `forward(input_ids, audio_values=None, attention_mask=None, labels=None)`
- `generate(input_ids, audio_values=None, attention_mask=None, max_new_tokens=32, **kw)`
- `answer(processor, question, audio, options=None, max_new_tokens=32) -> str`
- `trainable_parameters() -> int`

## Processing

### `AudioQAProcessor(tokenizer, num_audio_tokens, *, audio_token="<audio>", sample_rate=16000)`
Callable: `processor(text, audio=None) -> {input_ids, attention_mask[, audio_values]}`.
`text` may be a string or list of strings.

### `build_prompt(question, options=None, audio_token="<audio>", instruction=None)`
Build a prompt with the audio placeholder. From `harken.templates`:
`format_options`, `build_chat`.

### `load_audio(path, sr=16000, mono=True) -> np.ndarray`

## Encoders (`harken.encoders`)

- `get_encoder(name, **kwargs) -> AudioEncoder`
- `list_encoders() -> list[str]` — `dummy`, `clap`, `whisper`, `ast`
- `register_encoder(name)` — decorator to add your own
- `AudioEncoder` — base class (`forward`, `output_dim`, `sampling_rate`)

## Projectors (`harken.projectors`)

- `build_projector(name, in_dim, out_dim, num_tokens=8, **kwargs) -> Projector`
- `list_projectors() -> list[str]` — `linear`, `mlp`, `stack`, `query`

## Data (`harken.data`)

- `AudioQAExample(question, answer="", audio=None, options=None, type="solvable", group=None, id=None)`
- `load_examples(path)`, `save_examples(examples, path)`
- `Collator(processor, audio_loader=None)`

## Evaluation (`harken.evaluation`)

- `normalize_answer`, `exact_match`, `token_f1`, `best_exact_match`, `best_token_f1`
- `is_abstention`, `score_abstention`
- `aggregate(predictions, references)`
- `evaluate_dataset(model, processor, examples, *, audio_loader=None, max_new_tokens=32)`

## Testing helpers (`harken.testing`)

- `TinyTokenizer`, `TinyCausalLM`, `TinyCausalLMConfig`
