#!/usr/bin/env python3
"""Train a teaching-only thermal-risk model from the captured heat-gun run.

This one-run model demonstrates the data -> windows -> labels -> model pipeline.
It is not an independently validated fire detector.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


SOURCE = Path("data/heatgun_box_001.csv")
ARTIFACTS = Path("artifacts")
WINDOW_SAMPLES = 10
SAMPLE_SECONDS = 2
FORECAST_HORIZON_SAMPLES = 30  # 60 seconds
HIGH_TEMP_C = 50.7
ELEVATED_TEMP_C = 35.0
LABELS = ["NORMAL", "ELEVATED_THERMAL_RISK", "HIGH_THERMAL_RISK"]


def make_training_data(frame):
    readings = frame[["temp_c", "humidity_pct"]].to_numpy(dtype=np.float32)
    features, labels = [], []

    last_start = len(readings) - WINDOW_SAMPLES - FORECAST_HORIZON_SAMPLES
    for start in range(last_start + 1):
        window = readings[start:start + WINDOW_SAMPLES]
        end_temp = float(window[-1, 0])
        future_peak = float(readings[start + WINDOW_SAMPLES:
                                     start + WINDOW_SAMPLES + FORECAST_HORIZON_SAMPLES, 0].max())

        # These labels encode the one controlled run: high is already hot;
        # elevated is warming or will reach the known high-temperature point.
        if end_temp >= HIGH_TEMP_C:
            label = 2
        elif end_temp >= ELEVATED_TEMP_C or future_peak >= HIGH_TEMP_C:
            label = 1
        else:
            label = 0

        features.append(window.reshape(-1))
        labels.append(label)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing {SOURCE}")

    frame = pd.read_csv(SOURCE)
    if set(["sequence", "temp_c", "humidity_pct"]) - set(frame.columns):
        raise ValueError("CSV must contain sequence,temp_c,humidity_pct")

    X, y = make_training_data(frame.dropna())
    if len(np.unique(y)) != len(LABELS):
        raise ValueError("Captured run does not produce all three teaching labels")

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(scaler.transform(X), y)
    predicted = model.predict(scaler.transform(X))

    ARTIFACTS.mkdir(exist_ok=True)
    model_data = {
        "kind": "teaching_logistic_regression",
        "source": str(SOURCE),
        "warning": "One controlled heat-gun run; not a validated fire detector.",
        "labels": LABELS,
        "sample_seconds": SAMPLE_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "forecast_horizon_seconds": FORECAST_HORIZON_SAMPLES * SAMPLE_SECONDS,
        "high_temperature_c": HIGH_TEMP_C,
        "elevated_temperature_c": ELEVATED_TEMP_C,
        "feature_order": [f"{name}_{index}" for index in range(WINDOW_SAMPLES)
                          for name in ("temp_c", "humidity_pct")],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercepts": model.intercept_.tolist(),
    }
    report = {
        "source_rows": int(len(frame)),
        "windows": int(len(X)),
        "class_counts": {LABELS[index]: int((y == index).sum()) for index in range(len(LABELS))},
        "training_confusion_matrix": confusion_matrix(y, predicted).tolist(),
        "warning": "Training-set fit only. Do not treat this as real-world accuracy.",
    }

    (ARTIFACTS / "heatgun_teaching_model.json").write_text(json.dumps(model_data, indent=2))
    (ARTIFACTS / "heatgun_teaching_model_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
