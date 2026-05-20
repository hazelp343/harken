# Contributing to harken

Thanks for your interest in improving harken! Contributions of all kinds are
welcome — bug reports, docs, new encoders/projectors, and tests.

## Development setup

```bash
git clone https://github.com/hazelp343/harken
cd harken
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

Run the same checks CI runs:

```bash
pytest                       # tests must pass
ruff check .                 # lint
ruff format --check .        # formatting
mypy harken                  # type checks
```

`pre-commit run --all-files` runs ruff and mypy for you.

## Guidelines

- Keep pull requests focused; one logical change per PR.
- Add or update tests for any behaviour change. The dummy encoder and
  `TinyCausalLM` in `harken.testing` let you test the full pipeline on CPU.
- Public functions should have type hints and a short docstring.
- New encoders subclass `harken.encoders.AudioEncoder` and register with
  `@register_encoder("name")`; new projectors subclass `Projector` and register
  with `@projectors.register("name")`.
- Heavyweight tests that download pretrained models should be marked
  `@pytest.mark.slow`.

## Reporting bugs

Open an issue with a minimal reproduction, your harken/Python/torch versions,
and the full traceback. The bug report template will prompt for these.
