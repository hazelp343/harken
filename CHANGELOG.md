# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-15

Initial release.

### Added
- `AudioQAModel`, `AudioQAConfig` and `build_model` wiring an audio encoder, a
  projector and a causal LM.
- Encoder registry with a download-free `dummy` encoder and lazy `clap`,
  `whisper` and `ast` wrappers (via the `hf` extra).
- Projector zoo: `linear`, `mlp`, `stack` and `query`.
- `AudioQAProcessor` with audio-token expansion and batching.
- Pure-numpy log-mel front end, audio I/O and chunking helpers.
- `harken.testing` with `TinyTokenizer` and `TinyCausalLM`.

[Unreleased]: https://github.com/hazelp343/harken/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hazelp343/harken/releases/tag/v0.1.0
