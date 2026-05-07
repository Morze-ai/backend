setup:
	uv sync --group dev
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
	uv run pre-commit install --hook-type commit-msg

# CLI Commands
# Note: Each command requires a CONFIG path argument

fetch-data:
	@echo "Usage: make fetch-data CONFIG=path/to/config.yaml [SOURCE_CSV=path/to/source.csv]"
	uv run python -m src.cli.fetch_data $(CONFIG) $(if $(SOURCE_CSV),--source-csv $(SOURCE_CSV))

preprocess-data:
	@echo "Usage: make preprocess-data CONFIG=path/to/config.yaml"
	uv run python -m src.cli.preprocess_data $(CONFIG)

train-model:
	@echo "Usage: make train-model CONFIG=path/to/config.yaml"
	uv run python -m src.cli.train_model $(CONFIG)

evaluate-model:
	@echo "Usage: make evaluate-model CONFIG=path/to/config.yaml"
	uv run python -m src.cli.evaluate_model $(CONFIG)

predict:
	@echo "Usage: make predict CONFIG=path/to/config.yaml VALUES_JSON='{\"feature_name\": value}'"
	uv run python -m src.cli.predict $(CONFIG) --values-json '$(VALUES_JSON)'

run-experiment:
	@echo "Usage: make run-experiment CONFIG=path/to/config.yaml"
	uv run python -m src.cli.run_experiment $(CONFIG)

compare-experiments:
	@echo "Usage: make compare-experiments CONFIG=path/to/comparison_config.yaml"
	uv run python -m src.cli.compare_experiments $(CONFIG)

report-summary:
	@echo "Usage: make report-summary [REPORTS_ROOT=reports] [OUTPUT_CSV=reports/global/experiment_summary.csv]"
	uv run python -m src.cli.report_summary $(if $(REPORTS_ROOT),--reports-root $(REPORTS_ROOT)) $(if $(OUTPUT_CSV),--output-csv $(OUTPUT_CSV))

visualize:
	@echo "Usage: make visualize CONFIG=path/to/config.yaml [INCLUDE_EXPLORATORY=1]"
	uv run python -m src.cli.visualize $(CONFIG) $(if $(INCLUDE_EXPLORATORY),--include-exploratory)

# Pipeline: full workflow from raw data to report
# Usage: make pipeline CONFIG=path/to/config.yaml COMPARISON_CONFIG=path/to/comparison.yaml
pipeline: fetch-data preprocess-data train-model evaluate-model report-summary visualize
	@echo "Full pipeline completed: fetch-data -> preprocess-data -> train-model -> evaluate-model -> report-summary -> visualize"

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
