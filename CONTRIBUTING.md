# Contributing

Issues and pull requests are welcome.

## Set up

```bash
python3 -m venv .venv && source .venv/bin/activate    # Python 3.11 or 3.12
pip install -e ".[dev]"
pre-commit install    # optional; runs ruff before each commit
```

## Before opening a pull request

```bash
ruff check .
ruff format --check .
mypy
pytest --cov --cov-fail-under=80
```

CI runs the same commands on Python 3.11 and 3.12.

## Ground rules

- Tests must pass offline, without PyTorch, weights or a GPU. Fake the model at the
  `ModelManager` loader or `Predictor` boundary, as `tests/conftest.py` does.
- Never commit real invoices, personal data, weights or credentials. Use synthetic images.
- Any number in the README must come from a command in this repository that others can run,
  or be labelled as an illustrative example.
- Use conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `build:`, `chore:`).

By contributing, you agree that your contributions are licensed under the MIT licence.
