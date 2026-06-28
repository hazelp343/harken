# Abstention-aware evaluation

Evaluate a model on a small [AQUA](https://github.com/kuan2jiu99/aqua-bench)-style
dataset that mixes answerable and unanswerable questions.

```bash
python examples/evaluate_aqua/run.py
```

`sample.jsonl` groups a `solvable` and an `unanswerable` question per clip, so
the runner reports per-type accuracy and **conditional** unanswerable accuracy
(scored only where the solvable question was answered correctly). The dataset
omits audio here for simplicity; add an `"audio": "path/to/clip.wav"` field to
each record to run with real recordings.
