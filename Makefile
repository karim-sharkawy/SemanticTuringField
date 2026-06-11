.PHONY: install install-dev run-demo test lint format clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

install-dev:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .[dev,llm]

run-demo:
	python scripts/run_pipeline.py

test:
	pytest -q

lint:
	ruff check src tests
	black --check src tests

format:
	black src tests
	ruff check --fix src tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info outputs