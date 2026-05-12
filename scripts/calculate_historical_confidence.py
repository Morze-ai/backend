"""Script to calculate historical co-occurrence confidence for O1-O4 rules."""

from __future__ import annotations

import pandas as pd

from src.cli import build_experiment, load_raw_frame
from src.events.detectors.rainfall import detect_flash_flood, detect_long_rainfall
from src.events.detectors.seasonal import detect_seasonal_dependencies
from src.events.detectors.thaw import detect_thaw
from src.events.rules import EventType
from src.utils.io import write_json


def calculate_confidence(config_path: str):
    config, experiment = build_experiment(config_path)
    frame = load_raw_frame(config)

    # Preprocess to get all features
    experiment.preprocess(frame)
    train_df = experiment._train_frame

    target_col = config.data.target_column
    positive_label = config.data.class_names[-1]

    # Binary target
    y_true = train_df[target_col].astype(str).eq(positive_label).astype(int)

    detectors = {
        EventType.FLASH_FLOOD: detect_flash_flood,
        EventType.LONG_RAINFALL: detect_long_rainfall,
        EventType.THAW: detect_thaw,
        EventType.SEASONAL_DEPENDENCY: detect_seasonal_dependencies,
    }

    confidences = {}

    # Define vectorized rules for historical confidence calculation
    for etype in detectors:
        mask = pd.Series(False, index=train_df.index)

        if etype == EventType.FLASH_FLOOD:
            # Simple vectorized version: rainfall > 95th percentile
            rainfall = train_df.get("rainfall_mm", pd.Series(0, index=train_df.index))
            threshold = rainfall[rainfall > 0].quantile(0.95) if (rainfall > 0).any() else 1.0
            mask = rainfall >= threshold

        elif etype == EventType.LONG_RAINFALL:
            # Simple vectorized version: 72h rolling sum > threshold
            rainfall = train_df.get("rainfall_mm", pd.Series(0, index=train_df.index))
            rolling_72h = rainfall.rolling(window=72, min_periods=1).sum()
            threshold = rolling_72h.quantile(0.95)
            mask = rolling_72h >= threshold

        elif etype == EventType.THAW:
            # Simple vectorized version: temp > 0 and previous min <= 0
            temp = train_df.get("temperature_c", pd.Series(0, index=train_df.index))
            prev_min = temp.rolling(window=48, min_periods=1).min().shift(1)
            mask = (temp > 2.0) & (prev_min <= 0.0)

        elif etype == EventType.SEASONAL_DEPENDENCY:
            # Simple vectorized version: season-based high rainfall or high temp
            if "season" in train_df.columns:
                is_winter_spring = train_df["season"].isin(["winter", "spring"])
                temp = train_df.get("temperature_c", pd.Series(0, index=train_df.index))
                rain = train_df.get("rainfall_mm", pd.Series(0, index=train_df.index))
                mask = (is_winter_spring & (temp > 5.0)) | (~is_winter_spring & (rain > 5.0))

        # Calculate P(High Water | Rule Active)
        if mask.sum() > 0:
            confidence = y_true[mask].mean()
            confidences[etype.value] = float(confidence)
        else:
            confidences[etype.value] = 0.5  # Default fallback

    output_path = config.paths.evaluation_json.parent / "historical_confidence.json"
    write_json(output_path, confidences)
    print(f"Historical confidence saved to {output_path}")
    return confidences


if __name__ == "__main__":
    import sys

    config_p = sys.argv[1] if len(sys.argv) > 1 else "configs/mlp_water_level.yaml"
    calculate_confidence(config_p)
