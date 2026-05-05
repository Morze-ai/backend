setup:
	uv sync --group dev
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
	uv run pre-commit install --hook-type commit-msg

# compare-experiments

# evaluate-model

# fetch-data

# predict

# preprocess-data

# report-summary

# run-experiment

# train-model

# visualize

# pipeline
# full pipeline: fetch-data -> preprocess-data -> train-model -> evaluate-model -> report-summary -> visualize

test:
	uv run pytest

ruff:
	uv run ruff check .

ruff-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

pyright:
	uv run pyright

checks:
	uv run ruff format . --check
	uv run ruff check .
	uv run pyright
	uv run pytest
