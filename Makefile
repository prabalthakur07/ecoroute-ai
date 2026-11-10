.PHONY: install test lint format demo
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check .
format:
	ruff format .
demo:
	ecoroute demo --seed 0 --trips 300 --hour 18
