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

explain:
	@echo "Usage: make explain CONFIG=path/to/config.yaml"
	uv run python -m src.cli.explain $(CONFIG)

visualize:
	@echo "Usage: make visualize CONFIG=path/to/config.yaml [INCLUDE_EXPLORATORY=1]"
	uv run python -m src.cli.visualize $(CONFIG) $(if $(INCLUDE_EXPLORATORY),--include-exploratory)

continuous-eval:
	@echo "Usage: make continuous-eval CONFIG=path/to/config.yaml"
	uv run python -m src.cli.continuous_evaluation $(CONFIG)

continuous-scheduler:
	@echo "Usage: make continuous-scheduler CONFIG=path/to/config.yaml [MODE=daily|once] [HOUR=6] [MINUTE=0] [TIMEZONE=Europe/Warsaw]"
	uv run python -m src.cli.continuous_scheduler $(CONFIG) --mode $(if $(MODE),$(MODE),daily) --hour $(if $(HOUR),$(HOUR),6) --minute $(if $(MINUTE),$(MINUTE),0) --timezone $(if $(TIMEZONE),$(TIMEZONE),Europe/Warsaw)

api-server:
	@echo "Usage: make api-server [HOST=0.0.0.0] [PORT=8000]"
	uv run uvicorn src.api.main:app --host $(if $(HOST),$(HOST),0.0.0.0) --port $(if $(PORT),$(PORT),8000)

run-today:
	@echo "Run run_today for all configs in configs/ (no parameters required)"
	uv run python -m src.cli.run_today

# Zero-parameter continuous evaluation + prediction
# Runs continuous evaluation for all configs in the configs/ directory
continuous-predict:
	@echo "🔄 Running continuous evaluation and prediction for all configs"
	@for config in configs/*.yaml; do \
		echo "=========================================="; \
		echo "  Running continuous evaluation for $$config"; \
		echo "=========================================="; \
		uv run python -m src.cli.continuous_evaluation $$config; \
	done
	@echo ""
	@echo "  ✓ Continuous evaluation complete for all configs"
	@echo "  Run 'make api-server' to serve results via REST API"
	@echo ""
	@echo "  API Endpoints:"
	@echo "    GET  /continuous/forecast           — Current + horizon forecasts"
	@echo "    GET  /continuous/status              — Source health & freshness"
	@echo "    GET  /continuous/predictions/{+1d|+3d|+7d} — Per-horizon detail"
	@echo "    GET  /continuous/latest              — Full raw evaluation result"
	@echo "    POST /continuous/refresh             — Trigger new evaluation"

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
	@echo "  Visual Explainability Reports (PDF):"
	@echo "    • reports/linear_water_level/explainability_report.pdf"
	@echo "    • reports/logistic_water_level/explainability_report.pdf"
	@echo "    • reports/mlp_water_level/explainability_report.pdf"
	@echo ""

.PHONY: synchronize-data
synchronize-data:
	@echo "📥 Synchronizing water level data..."
	@if [ ! -f data/processed/water_level_synchronized_hourly.csv ]; then \
		PYTHONPATH=. uv run python scripts/synchronize_and_resample.py; \
	else \
		echo "  ✓ Data already synchronized at data/processed/water_level_synchronized_hourly.csv"; \
	fi

.PHONY: prepare-training-data
prepare-training-data:
	@echo "🏷️  Preparing labeled training data..."
	@PYTHONPATH=. uv run python scripts/prepare_training_data.py
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
compare-and-report: compare-models report-all visualize-all explain-all
	@echo "✓ Comparison and reporting complete (including PDF explainability)"

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
	@uv run python -m src.cli.visualize configs/linear_water_level.yaml --include-exploratory
	@uv run python -m src.cli.visualize configs/logistic_water_level.yaml --include-exploratory
	@uv run python -m src.cli.visualize configs/mlp_water_level.yaml --include-exploratory
	@echo "  ✓ Visualizations generated"

.PHONY: explain-all
explain-all: calculate-confidence
	@echo "🧠 Generating SHAP explainability reports (PDF)..."
	@uv run python -m src.cli.explain configs/linear_water_level.yaml
	@uv run python -m src.cli.explain configs/logistic_water_level.yaml
	@uv run python -m src.cli.explain configs/mlp_water_level.yaml
	@echo "  ✓ Explainability reports generated"

.PHONY: calculate-confidence
calculate-confidence:
	@echo "📊 Calculating historical rule confidence..."
	@uv run python -m scripts.calculate_historical_confidence configs/linear_water_level.yaml
	@uv run python -m scripts.calculate_historical_confidence configs/logistic_water_level.yaml
	@uv run python -m scripts.calculate_historical_confidence configs/mlp_water_level.yaml
	@echo "  ✓ Historical confidence calculated"

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

pytest: test

ruff:
	uv run ruff check .

ruff-fix:
	uv run ruff check . --fix

fix: ruff-fix

format:
	uv run ruff format .

pyright:
	uv run pyright

checks:
	uv run ruff format . --check
	uv run ruff check .
	uv run pyright
	uv run pytest
