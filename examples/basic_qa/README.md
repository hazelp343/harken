# Basic audio QA

A minimal, download-free example: build a model from the dummy encoder + tiny
language model, feed it a synthetic tone, and print an answer.

```bash
python examples/basic_qa/run.py
```

The output is gibberish because the tiny model is untrained — swap in a real
encoder (`get_encoder("whisper", ...)`) and a trained projector/LLM to get
meaningful answers. This example exists to show the wiring.
