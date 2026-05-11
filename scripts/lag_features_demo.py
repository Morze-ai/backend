#!/usr/bin/env python3
"""
Quick demonstration of lag features in the water level prediction pipeline.

This script shows:
1. How lag features are automatically generated
2. The effect of lag features on the dataset
3. How to run an experiment with lag features
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feature_engineering import (
    generate_lag_features,
    generate_rolling_features,
    generate_seasonal_features,
    drop_initial_lag_rows,
)


def demo_basic_lag_generation():
    """Demonstrate basic lag feature generation."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Lag Feature Generation")
    print("="*80)
    
    # Create sample data
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=10, freq="h"),
        "rainfall_mm": [0, 1, 2, 3, 4, 3, 2, 1, 0, 0],
        "temperature_c": [5, 6, 7, 8, 9, 10, 9, 8, 7, 6],
    })
    
    print("\nOriginal DataFrame (first 5 rows):")
    print(df.head())
    
    # Generate lags
    df_with_lags = generate_lag_features(
        df,
        timestamp_column="timestamp",
        lag_columns={"rainfall_mm": 3, "temperature_c": 3},
    )
    
    print(f"\nAfter lag generation: {len(df_with_lags.columns)} columns")
    print("\nColumns created:")
    for col in df_with_lags.columns:
        if "lag" in col:
            print(f"  - {col}")
    
    print("\nDataFrame with lags (all columns):")
    print(df_with_lags.head(6))
    
    print("\n📊 Observation:")
    print("  - Row 0: rainfall_mm_lag_1h is NaN (no history)")
    print("  - Row 1: rainfall_mm_lag_1h = 0 (rainfall from row 0)")
    print("  - Row 2: rainfall_mm_lag_1h = 1, rainfall_mm_lag_2h = 0")
    print("  - Row 3: rainfall_mm_lag_1h = 2, rainfall_mm_lag_2h = 1, rainfall_mm_lag_3h = 0")


def demo_rolling_aggregates():
    """Demonstrate rolling feature generation."""
    print("\n" + "="*80)
    print("DEMO 2: Rolling Aggregates (mean, max, std)")
    print("="*80)
    
    # Create sample data with varying values
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=20, freq="h"),
        "rainfall_mm": np.array([0, 2, 4, 3, 1, 0, 1, 5, 6, 4, 2, 1, 0, 1, 2, 3, 2, 1, 0, 0]),
    })
    
    print("\nOriginal rainfall data:")
    print(df)
    
    # Generate rolling features
    df_with_rolling = generate_rolling_features(
        df,
        window_hours=[3, 6],
        agg_functions=["mean", "max", "std"],
        columns_to_aggregate=["rainfall_mm"],
    )
    
    rolling_cols = [col for col in df_with_rolling.columns if "rainfall_mm_" in col and "_" in col.split("rainfall_mm_")[1]]
    
    print(f"\nRolling features created:")
    for col in rolling_cols:
        print(f"  - {col}")
    
    print("\nRolling aggregates (rows 5-10):")
    print(df_with_rolling[["timestamp", "rainfall_mm"] + rolling_cols[4:8]].iloc[5:10])
    
    print("\n📊 Observation:")
    print("  - rainfall_mm_mean_3h: average rainfall over last 3 hours")
    print("  - rainfall_mm_max_3h: maximum rainfall in last 3 hours")
    print("  - rainfall_mm_std_6h: variability of rainfall over last 6 hours")


def demo_seasonal_features():
    """Demonstrate seasonal feature generation."""
    print("\n" + "="*80)
    print("DEMO 3: Seasonal Features (month, season, growing_season)")
    print("="*80)
    
    # Create data spanning entire year
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=365, freq="D"),
        "water_level_m": np.random.randn(365) + 5,
    })
    
    # Generate seasonal features
    df_seasonal = generate_seasonal_features(df)
    
    seasonal_cols = ["month", "day_of_year", "season", "is_growing_season"]
    
    print("\nSeasonal features added to DataFrame:")
    
    # Show examples from different seasons
    examples = [
        ("Winter (Jan 1)", 0),
        ("Spring (Apr 1)", 90),
        ("Summer (Jul 1)", 181),
        ("Autumn (Oct 1)", 273),
    ]
    
    for label, idx in examples:
        row = df_seasonal.iloc[idx]
        print(f"\n{label}:")
        print(f"  month: {row['month']}, season: {row['season']}, is_growing_season: {row['is_growing_season']}")
    
    print("\n📊 Observation:")
    print("  - Growing season (April-October): is_growing_season = 1")
    print("  - Hydrological seasons: winter/spring/summer/autumn")
    print("  - Useful for capturing seasonal flooding patterns")


def demo_warmup_period():
    """Demonstrate warmup period dropping."""
    print("\n" + "="*80)
    print("DEMO 4: Dropping Warmup Period")
    print("="*80)
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=100, freq="h"),
        "rainfall_mm": np.random.rand(100) * 10,
    })
    
    # Generate 72h lags
    df_with_lags = generate_lag_features(
        df,
        lag_columns={"rainfall_mm": 72},
    )
    
    print(f"\nBefore dropping warmup:")
    print(f"  - Rows: {len(df_with_lags)}")
    print(f"  - First row has lags: {not df_with_lags.iloc[0][['rainfall_mm_lag_1h', 'rainfall_mm_lag_72h']].isna().all()}")
    
    # Drop warmup period
    df_clean = drop_initial_lag_rows(df_with_lags, max_lag_hours=72)
    
    print(f"\nAfter dropping warmup (72 rows):")
    print(f"  - Rows: {len(df_clean)}")
    print(f"  - First row now has complete lags: {not df_clean.iloc[0][['rainfall_mm_lag_1h', 'rainfall_mm_lag_72h']].isna().all()}")
    print(f"  - Rows dropped: {len(df_with_lags) - len(df_clean)}")
    
    print("\n📊 Key Point:")
    print("  - Lags for first N rows are NaN (not enough history)")
    print("  - Drop these rows to avoid model training on incomplete data")
    print("  - This happens automatically in BaseExperiment.preprocess()")


def show_integration_example():
    """Show how integration works in the actual pipeline."""
    print("\n" + "="*80)
    print("DEMO 5: Integration in Pipeline (What Happens Automatically)")
    print("="*80)
    
    code = '''
# In src/experiments/base.py, the preprocess() method does this:

def preprocess(self, frame: pd.DataFrame) -> None:
    processed_frame = frame.copy()
    
    # Auto-detect weather columns
    weather_columns = {
        "rainfall_mm": 72,
        "temperature_c": 72,
        "pressure_hpa": 72
    }
    available_weather = {
        col: lags for col, lags in weather_columns.items()
        if col in frame.columns
    }
    
    # If weather data exists, generate lags
    if available_weather:
        logger.info(f"Generating lag features for {list(available_weather.keys())}")
        
        # Generate lags
        processed_frame = generate_lag_features(
            processed_frame,
            timestamp_column="timestamp",
            lag_columns=available_weather,
        )
        
        # Drop warmup period
        max_lags = max(available_weather.values())
        processed_frame = drop_initial_lag_rows(
            processed_frame,
            max_lag_hours=max_lags,
        )
        
        # Add lag columns to feature list
        for col in available_weather:
            for lag_hour in range(1, available_weather[col] + 1):
                lag_col_name = f"{col}_lag_{lag_hour}h"
                self.config.data.feature_columns.append(lag_col_name)
    
    # Continue with splitting, normalization, etc...
    '''
    
    print(code)
    
    print("\n✨ Summary:")
    print("  1. System detects available weather columns")
    print("  2. Generates lags automatically (if weather data exists)")
    print("  3. Drops first 72 rows (warmup period)")
    print("  4. Updates feature_columns list automatically")
    print("  5. Model trains with 200+ features instead of just 1!")


def show_commands():
    """Show useful commands."""
    print("\n" + "="*80)
    print("USEFUL COMMANDS")
    print("="*80)
    
    commands = [
        ("Run full pipeline with lag features", 
         "uv run python -m src.cli.run_experiment configs/EXAMPLE_with_lag_features.yaml"),
        
        ("Just preprocess (generate lags)", 
         "uv run python -m src.cli.preprocess_data configs/EXAMPLE_with_lag_features.yaml"),
        
        ("Run tests", 
         "uv run pytest tests/test_feature_engineering.py -v"),
        
        ("Run all tests", 
         "uv run pytest tests/ -q"),
        
        ("View lag features guide", 
         "cat docs/LAG_FEATURES_GUIDE.md"),
        
        ("Run this demo", 
         "cd scripts && python lag_features_demo.py"),
    ]
    
    for desc, cmd in commands:
        print(f"\n{desc}:")
        print(f"  $ {cmd}")


if __name__ == "__main__":
    print("\n" + "🌊 "*40)
    print("LAG FEATURES DEMONSTRATION")
    print("Water Level Prediction with Temporal Context")
    print("🌊 "*40)
    
    demo_basic_lag_generation()
    demo_rolling_aggregates()
    demo_seasonal_features()
    demo_warmup_period()
    show_integration_example()
    show_commands()
    
    print("\n" + "="*80)
    print("✅ Lag features are fully implemented and integrated!")
    print("="*80)
    print("\nKey files:")
    print("  - src/data/feature_engineering.py (implementation)")
    print("  - src/experiments/base.py (integration)")
    print("  - tests/test_feature_engineering.py (tests)")
    print("  - docs/LAG_FEATURES_GUIDE.md (detailed guide)")
    print("  - configs/EXAMPLE_with_lag_features.yaml (example config)")
    print("\n")
