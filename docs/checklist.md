# Project checklist

## Goal

Zbudować pipeline, który:

wykrywa epizody wysokiej wody
wskazuje główny czynnik (opad / nasycenie / roztopy / inne)
generuje komunikat ryzyka

## Dane które będą potrzebne

Input datasets (IMGW-PIB, Copernicus Climate Data Store - ERA5, port + martwa Wisła):
poziom wody (port + Martwa Wisła, opcjonalnie IMGW-PIB)
nasycenie gleby (IMGW-PIB)
dane meteorologiczne:

- opad
- temperatura
- ciśnienie
- wiatr (opcjonalnie)

## Etapy projektu

### 1. **Zbieranie danych**

Pobranie danych z IMGW-PIB, Copernicus Climate Data Store - ERA5 oraz portu + Martwa Wisła.

### 2. **Przygotowanie danych**

Oczyszczenie, uzupełnienie brakujących wartości i standaryzacja danych.

### 3. **Synchronizacja danych**

Dopasowanie danych do wspólnego formatu i osi czasu.

### 4. **Resampling danych**

Przekształcenie danych do odpowiedniej częstotliwości (godzinowej i dziennej).

### 5. **Agregacja danych**

Obliczenie średnich, sum i innych statystyk dla różnych okresów czasu.

- poziom wody - średnia, max, min
- opad - suma, średnia
- temperatura - średnia, max, min
- ciśnienie - średnia, max, min
- wiatr - średnia, max, min

### 6. **Feature engineering**

Stworzenie nowych cech na podstawie istniejących danych, takich jak wskaźniki nasycenia gleby, wskaźniki opadów itp.

- opad - rain_1h_sum, rain_3h_sum, rain_6h_sum, rain_12h_sum, rain_24h_sum
- temperatura - temp_mean, temp_delta_24h, thaw_flag (czy temperatura przekroczyła 0°C w ciągu ostatnich 24h)
- wiatr - speed, direction

### 7. **Lagi**

Dodanie opóźnień do cech, aby uwzględnić wpływ wcześniejszych warunków na poziom wody.

- rain_lag_1h, rain_lag_3h, rain_lag_6h, rain_lag_12h, rain_lag_24h, rain_lag_48h, rain_lag_72h
- temp_lag
- pressure_lag

### 8. **Sezonowość**

Dodanie cech sezonowych, takich jak miesiąc, dzień tygodnia, pora roku itp.

### 9. **Exploracyjna analiza danych**

Analiza korelacji między cechami a poziomem wody, identyfikacja wzorców i trendów.

- sezonowość poziomu wody
- zależność meteo vs poziom wody
- zależność nasycenia gleby vs poziom wody
- analiza opóźnień (lagów)

> [!NOTE]
> Kluczowe pytania:
> Jakie cechy są najbardziej istotne dla przewidywania poziomu wody?
> Po ilu godzinach od opadów poziom wody zaczyna rosnąć?
> Jakie są typowe wzorce sezonowe dla poziomu wody?

### 10. **Analiza statyczna**

Przeprowadzenie testów statystycznych, takich jak testy korelacji, testy t-Studenta itp., aby ocenić istotność cech.

- Korelacje z lagami
- Testy t-Studenta dla różnych grup (np. wysokie vs niskie poziomy wody)
- Porównanie (np: high rain vs low rain)

### 11. **Definicja epizodów wysokiej wody**

Określenie progu, powyżej którego poziom wody jest uważany za wysoki.

```python
threshold = water_level.quantile(0.95)
```

### 12. **System czujników**

Wskazać czy warunki sprzyjają wzrostowi wody oraz określenie głównego czynnika (opad, nasycenie, roztopy itp.) na podstawie analizy cech.

Przykładowe reguły:

**O1: Epizod opadowy (krótki i intensywny)**
a. warunek: intensywność opadu > próg (np. percentyl 90) i/lub opad 24h > próg
sezonowy
b. komunikat: "Wysoka intensywność opadu sprzyja gwałtownemu wzrostowi poziomu
wody."

**O2: Długotrwałe opady + nasycenie zlewni**
a. warunek: opad skumulowany 72h / 7 dni > próg + opad bieżący > próg
b. komunikat: "Zlewnia jest nasycona – nawet umiarkowany opad może podnieść poziom
wody."

**O3: Roztopy (mechanizm sezonowy)**
a. warunek: utrzymujące się temperatury dodatnie (lub szybki wzrost T) w sezonie
zimowo-wiosennym + brak/mały opad (lub opad deszczu na śnieg, jeśli da się uchwycić)
b. komunikat: "Warunki roztopowe mogą powodować wzrost poziomu wody."

**O4: Zależności sezonowe (różne mechanizmy w różnych porach roku)**
a. warunek: inne relacje meteo→poziom wody w różnych sezonach (np. latem dominują
wezbrania opadowe, a wiosną roztopowe)
b. komunikat: "W tym sezonie dominują inne czynniki podnoszące poziom wody"

### 13. **Wyliczenie confidence**

Na podstawie przypadków historycznych, które spełniały warunki O1-O4, można oszacować prawdopodobieństwo wystąpienia epizodu wysokiej wody i oszacować % pewności.

## Machine learning

### Modele

- Logistic Regression
- Random Forest / XGBoost

### Wymogi modelu

- interpretowalność (ważne cechy, wpływ na decyzję)
- możliwość oszacowania prawdopodobieństwa (confidence)
- SHAP values do analizy wpływu cech

### Walidacja modelu

Podział na zbiór treningowy i testowy:

- Zbiór treningowy: 2021-2023
- Zbiór testowy: 2024-2025

### Metryki oceny

- Recall (czułość) - ważne, aby model wykrywał jak najwięcej epizodów wysokiej wody
- Precision (precyzja) - ważne, aby unikać fałszywych alarmów
- Onset error - czas między przewidywanym a rzeczywistym początkiem epizodu wysokiej wody
- MAE / RMSE dla regresji poziomu wody (opcjonalnie)

### Interpretacja stabilności modelu

- Porównania rok do roku, sezon do sezonu
- Analiza per sezon

## Outputy

### Raport PDF

- Sezonowość
- Zależność Meteo -> poziom wody
- Interpretacja wyników modelu (ważne cechy, wpływ na decyzję)

### CSV

- Timestamp / Sezon
- Czynnik
- Threshold
- Lag

### Notebook

Pipeline:

1. Zbieranie danych
2. Przygotowanie danych
3. Feature engineering
4. Wykrywanie eventów
5. Analiza czynników
6. Generowanie outputu

## Checklista

### Faza 1: Przygotowanie Danych (Completed ✅)

- [x] **1. Zbieranie danych** — IMGW-PIB, ERA5, port, Martwa Wisła (scripts: [fetch_imgw_dataset.py](../scripts/fetch_imgw_dataset.py), [fetch_era5_data.py](../data/fetch_era5_data.py))
- [x] **2. Przygotowanie danych** — Oczyszczenie, braki, standaryzacja ([src/data/preprocessing.py](../src/data/preprocessing.py), strategy: valid-zero / invalid-zero)
- [x] **3. Synchronizacja danych** — Merge czasowy, alignment check ([src/data/synchronization.py](../src/data/synchronization.py))
- [x] **4. Resampling danych** — Godzinowy i dobowy ([src/data/synchronization.py](../src/data/synchronization.py) `create_daily_aggregations()`)
- [x] **5. Agregacja danych** — Mean, max, min dla wszystkich zmiennych (woda, opad, temp, ciśnienie, wiatr)

### Faza 2: Inżynieria Cech (Completed ✅)

- [x] **6. Feature engineering (domeny)** — Opad, temp_delta, thaw_flag, soil_saturation, wind_u/v ([src/data/feature_engineering.py](../src/data/feature_engineering.py) `engineer_features()`)
- [x] **7. Lagi** — rain, temp, pressure 1–72h, auto-drop warmup ([src/data/feature_engineering.py](../src/data/feature_engineering.py) `generate_lag_features()`, integrated in [src/experiments/base.py](../src/experiments/base.py))
- [x] **8. Sezonowość** — Miesiąc, dzień roku, dzień tygodnia, godzina, pora roku, is_weekend, is_growing_season ([src/data/feature_engineering.py](../src/data/feature_engineering.py) `generate_seasonal_features()`)

### Faza 3: Eksploracyjna Analiza Danych (Completed ✅)

- [x] **9. Exploracyjna analiza danych** — Sezonowość, korelacje, lagi, zależności ([src/visualization/plots.py](../src/visualization/plots.py), [src/cli/visualize.py](../src/cli/visualize.py))
  - Sezonowość poziomu wody ✅
  - Zależność meteo vs poziom wody ✅
  - Analiza opóźnień (lagów) ✅

### Faza 4: Analiza Statystyczna (Partially ⚠️)

- [ip] **10. Analiza statyczna** — Korelacje z lagami, testy istotności
  - Brier score: ✅ ([src/events/evaluator.py](../src/events/evaluator.py) line 265)
  - Korelacje z lagami: ⚠️ (dostępne w eksploracji; szczegółowe per-sezon hypothesis tests brakuje)
  - Testy t-Studenta / Mann-Whitney: ❌ (brakuje)

### Faza 5: Event Detection & Rule System (Partially ⚠️)

- [ip] **11. Definicja epizodów wysokiej wody** — Próg percentyla (95%) lub stat. ([src/data/preprocessing.py](../src/data/preprocessing.py), [src/events/evaluator.py](../src/events/evaluator.py))
  - Threshold def: ✅
  - Event matching & onset error: ✅ ([src/events/evaluator.py](../src/events/evaluator.py) `_match_spans()`)

- [ip] **12. System czujników (O1-O4)** — Reguły dla epizodów
  - **O1: Epizod opadowy** — Regel schema: ✅ ([src/events/rules.py](../src/events/rules.py) `FLASH_FLOOD_RULE`); Detection logic: ❌ placeholder ([src/events/detectors/rainfall.py](../src/events/detectors/rainfall.py))
  - **O2: Długotrwałe opady + nasycenie** — Regel schema: ✅ (`LONG_RAINFALL_RULE`); Detection logic: ❌ placeholder
  - **O3: Roztopy** — Regel schema: ✅ (`THAW_RULE`); Detection logic: ❌ placeholder ([src/events/detectors/thaw.py](../src/events/detectors/thaw.py))
  - **O4: Zależności sezonowe** — Regel schema: ✅ (`SEASONAL_DEPENDENCY_RULE`); Detection logic: ❌ placeholder ([src/events/detectors/seasonal.py](../src/events/detectors/seasonal.py))
  - Message generation: ✅ (all rules have `response_message`)

- [ ] **13. Wyliczenie confidence** — Calibration, event-level confidence
  - Model probabilities: ✅ ([src/experiments/base.py](../src/experiments/base.py) lines 220–222)
  - Calibration (temp scaling, isotonic): ❌ Not implemented
  - Per-event confidence (historical co-occurrence): ❌ Not implemented

### Faza 6: Machine Learning (Completed ✅)

- [x] **Modele** — Logistic Regression, Random Forest / XGBoost (MLP, Linear Classifier) ([src/models/](../src/models/))
  - Logistic Regression: ✅ ([src/models/logistic_regression.py](../src/models/logistic_regression.py))
  - Linear Classifier: ✅ ([src/models/linear.py](../src/models/linear.py))
  - MLP: ✅ ([src/models/mlp.py](../src/models/mlp.py))

- [x] **Wymogi modelu** — Interpretowalność, probability, SHAP
  - Interpretability: ✅ (SHAP: [src/explain/shap_explainer.py](../src/explain/shap_explainer.py))
  - Probability output: ✅
  - SHAP values: ✅

- [x] **Walidacja modelu** — Temporal split train/val/test
  - Split strategy: ✅ ([src/data/preprocessing.py](../src/data/preprocessing.py) `split_dataset()`)
  - Train 2021–2023, test 2024–2025: ✅ (configurable via config YAML)

- [x] **Metryki oceny** — Recall, Precision, Onset error
  - Row-level metrics: ✅ (accuracy, precision, recall, F1)
  - Event-level metrics: ✅ (event_recall, event_precision, onset_error_hours, false_alarm_rate)
  - Tests: ✅ ([tests/test_event_evaluator.py](../tests/test_event_evaluator.py))

### Faza 7: Raportowanie & Interpretacja (Partially ⚠️)

- [ip] **Wyniki interpretacji** — Feature importance, seasonal breakdown
  - SHAP feature importance: ✅ ([src/explain/feature_importance.py](../src/explain/feature_importance.py))
  - Markdown report: ✅ ([src/explain/report.py](../src/explain/report.py) `generate_markdown_report()`)
  - Seasonal breakdown (per-year, per-season): ✅ ([src/events/evaluator.py](../src/events/evaluator.py) `summarize_by_period()`)
  - Factor attribution (mapping features to O1-O4): ⚠️ Partially (SHAP shows top features; explicit O1-O4 mapping missing)

### Faza 8: Outputy (Partially ⚠️)

- [x] **Raport PDF** — Sezonowość, meteo→woda, czynniki
  - Current output: ⚠️ Markdown + CSV only (PDF generation not implemented)
  - Required per project sheet: PDF/DOCX report with figures, tables, narrative (❌ Not started)

- [ip] **CSV** — Threshold, czynnik, lag per sezon
  - Feature importance CSV: ✅ ([src/explain/report.py](../src/explain/report.py) `save_feature_importance_csv()`)
  - Factor/threshold tables per season: ⚠️ Partial (metrics available; structured table generation not yet implemented)

- [ip] **Notebook** — Pipeline pełny
  - CLI pipeline: ✅ ([src/cli/run_experiment.py](../src/cli/run_experiment.py))
  - Jupyter notebook: ⚠️ Exists but unclear if current ([notebooks/seaData_pipeline.ipynb](../notebooks/seaData_pipeline.ipynb))
  - User guide: ❌ [docs/USAGE.md](docs/USAGE.md) not yet created

---

## Podsumowanie Stanu (Summary of Status)

| Komponent | Status | Uwagi |
| ----------- | -------- | ------- |
| **Data Prep** | ✅ Complete | Fetch, clean, sync, resample, aggregate all done. |
| **Features** | ✅ Complete | Domain, lag, seasonal, rolling all implemented & tested. |
| **Training** | ✅ Complete | Models, trainer, temporal split, metrics all done. |
| **EDA & Visualization** | ✅ Complete | Plots, seasonality, correlation analysis available. |
| **Event Rules (Schema)** | ✅ Complete | O1–O4 rules, messages, thresholds defined. |
| **Event Detection (Logic)** | ❌ Placeholder | Detectors return `detected=False`; need threshold checks. |
| **Confidence** | ⚠️ Partial | Probabilities exist; calibration & event-level confidence missing. |
| **SHAP Explainability** | ✅ Complete | SHAP values, ranking, markdown report all done. |
| **Statistical Tests** | ⚠️ Partial | Basic metrics computed; seasonal hypothesis tests missing. |
| **PDF Reporting** | ❌ Not Started | Current: markdown + CSV; need PDF/DOCX generation. |
| **Notebook & Docs** | ⚠️ Partial | CLI commands exist; consolidated notebook & user guide missing. |

---

## Następne Kroki (Next Steps)

**Phase 1: Event Detection** (High impact)

1. Implement `detect_long_rainfall()`, `detect_flash_flood()`, `detect_thaw()`, `detect_seasonal_dependencies()` with actual logic.
2. Test detectors against historical data.
3. Wire into evaluation reports.

**Phase 2: Confidence & Interpretability** (High impact)

1. Add probability calibration (temperature scaling).
2. Compute event-level confidence (historical co-occurrence).
3. Map top SHAP features to O1–O4 categories.

**Phase 3: Statistical Analysis** (Medium impact)

1. Add seasonal hypothesis tests (t-test, Mann-Whitney).
2. Lag sensitivity analysis per season.
3. Cross-validation & onset error distribution.

**Phase 4: Reporting** (High impact)

1. PDF generation with figures, tables, narrative sections.
2. Factor threshold tables per season (CSV + PDF).
3. Jupyter notebook walkthrough.
4. [docs/USAGE.md](docs/USAGE.md) user guide.
