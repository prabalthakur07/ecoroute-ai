# Contributing

1. `pip install -e ".[dev]" && pre-commit install`
2. Branch from `main`: `feat/<topic>` or `fix/<topic>`
3. Use Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`)
4. `ruff check . && pytest -q` must pass; add a test for new behaviour
5. Any new metric must be scored against ground truth or a documented proxy
