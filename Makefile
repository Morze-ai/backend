setup:
	uv sync --group dev
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
	uv run pre-commit install --hook-type commit-msg

install: setup

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

# ============================================================================
# Automated Complete Pipeline: make run
# ============================================================================
# One command to do everything: synchronize data, train all models, compare,
# generate reports, and make sample predictions. No parameters needed.
# Usage: make run
# ============================================================================

.PHONY: run
run: synchronize-data prepare-training-data train-all-models compare-and-report sample-predict
	@echo ""
	@echo "=========================================="
	@echo "✓ Complete pipeline finished successfully"
	@echo "=========================================="
	@echo ""
	@echo "📊 Results Summary:"
	@echo "  Models trained:"
	@echo "    • models/linear_water_level/"
	@echo "    • models/logistic_water_level/"
	@echo "    • models/mlp_water_level/"
	@echo ""
	@echo "  Comparison reports:"
	@echo "    • reports/comparisons/model_comparison.csv"
	@echo "    • reports/comparisons/model_comparison.json"
	@echo "    • reports/comparisons/model_comparison.png"
	@echo ""
	@echo "  Model evaluation & predictions:"
	@echo "    • reports/linear_water_level/"
	@echo "    • reports/logistic_water_level/"
	@echo "    • reports/mlp_water_level/"
	@echo ""
	@echo "  Summary report:"
	@echo "    • reports/global/experiment_summary.csv"
	@echo ""

.PHONY: synchronize-data
synchronize-data:
	@echo "📥 Synchronizing water level data..."
	@if [ ! -f data/processed/water_level_synchronized_hourly.csv ]; then \
		uv run python scripts/synchronize_and_resample.py; \
	else \
		echo "  ✓ Data already synchronized at data/processed/water_level_synchronized_hourly.csv"; \
	fi

.PHONY: prepare-training-data
prepare-training-data:
	@echo "🏷️  Preparing labeled training data..."
	@uv run python scripts/prepare_training_data.py
	@echo "  ✓ Training dataset ready at data/processed/water_level_training.csv"

.PHONY: train-all-models
train-all-models:
	@echo ""
	@echo "🤖 [1/3] Training linear model..."
	@uv run python -m src.cli.run_experiment configs/linear_water_level.yaml
	@echo "  ✓ Linear model trained"
	@echo ""
	@echo "🤖 [2/3] Training logistic model..."
	@uv run python -m src.cli.run_experiment configs/logistic_water_level.yaml
	@echo "  ✓ Logistic model trained"
	@echo ""
	@echo "🤖 [3/3] Training mlp model..."
	@uv run python -m src.cli.run_experiment configs/mlp_water_level.yaml
	@echo "  ✓ MLP model trained"
	@echo ""
	@echo "✓ All models training complete"

.PHONY: compare-and-report
compare-and-report: compare-models report-all visualize-all
	@echo "✓ Comparison and reporting complete"

.PHONY: compare-models
compare-models:
	@echo "📊 Comparing all models..."
	@uv run python -m src.cli.compare_experiments configs/compare_all_models.yaml
	@echo "  ✓ Comparison reports generated"

.PHONY: report-all
report-all:
	@echo "📋 Generating summary reports..."
	@uv run python -m src.cli.report_summary
	@echo "  ✓ Summary report generated"

.PHONY: visualize-all
visualize-all:
	@echo "🎨 Generating visualizations..."
	@uv run python -m src.cli.visualize configs/linear_water_level.yaml --include-exploratory > /dev/null 2>&1 || true
	@uv run python -m src.cli.visualize configs/logistic_water_level.yaml --include-exploratory > /dev/null 2>&1 || true
	@uv run python -m src.cli.visualize configs/mlp_water_level.yaml --include-exploratory > /dev/null 2>&1 || true
	@echo "  ✓ Visualizations generated"

.PHONY: sample-predict
sample-predict:
	@echo "📝 To make predictions with trained models, run:"
	@echo "   make predict CONFIG=configs/linear_water_level.yaml VALUES_JSON='{\"water_level_m\": 0.5, \"rain_1h_sum\": 0.0}'"
	@echo ""
	@echo "   Available models for prediction:"
	@echo "     • configs/linear_water_level.yaml"
	@echo "     • configs/logistic_water_level.yaml"
	@echo "     • configs/mlp_water_level.yaml"
	@echo ""
	@echo "   View test predictions CSV files in reports/{model_name}/test_predictions.csv"

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
