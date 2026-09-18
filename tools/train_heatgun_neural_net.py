#!/usr/bin/env python3
"""Train a small dense neural network from captured DHT data.

Teaching-only model: Dense(20 -> 12) + ReLU + Dense(12 -> 3) + Softmax.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("data/supplied_training_data.csv")
ARTIFACTS = Path("artifacts")
WINDOW_SAMPLES = 10
SAMPLE_SECONDS = 2
FORECAST_HORIZON_SAMPLES = 30
HIGH_TEMP_C = 50.7
ELEVATED_TEMP_C = 35.0
LABELS = ["NORMAL", "ELEVATED_THERMAL_RISK", "HIGH_THERMAL_RISK"]
HIDDEN_UNITS = 12
EPOCHS = 350
BATCH_SIZE = 64
LEARNING_RATE = 0.003
RANDOM_SEED = 42
SYNTHETIC_WINDOWS_PER_RISK_CLASS = 1200


def make_training_data(frame):
    readings = frame[["temp_c", "humidity_pct"]].to_numpy(dtype=np.float32)
    features, labels = [], []
    last_start = len(readings) - WINDOW_SAMPLES - FORECAST_HORIZON_SAMPLES
    for start in range(last_start + 1):
        window = readings[start:start + WINDOW_SAMPLES]
        end_temp = float(window[-1, 0])
        future_peak = float(readings[start + WINDOW_SAMPLES:
                                     start + WINDOW_SAMPLES + FORECAST_HORIZON_SAMPLES, 0].max())
        label = 2 if end_temp >= HIGH_TEMP_C else (
            1 if end_temp >= ELEVATED_TEMP_C or future_peak >= HIGH_TEMP_C else 0
        )
        features.append(window.reshape(-1))
        labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8)


def rise_curve(profile):
    """Return one of the explainable, smooth 20-second heating shapes."""
    progress = np.linspace(0.0, 1.0, WINDOW_SAMPLES, dtype=np.float32)
    if profile == "gradual_rise":
        return progress
    if profile == "late_rise":
        return progress ** 2
    if profile == "two_stage_rise":
        second_stage = np.clip((progress - 0.45) / 0.55, 0.0, 1.0)
        return 0.30 * progress + 0.70 * second_stage ** 2
    raise ValueError(f"Unknown profile: {profile}")


def add_synthetic_risks(features, labels, rng):
    """Make labeled teaching examples without changing the original CSV.

    Each example starts with a real normal DHT window. Temperature receives a
    smooth heating curve and humidity falls modestly, as it did in the heated
    box. These are classroom augmentation examples, not measured fire data.
    """
    normal = features[labels == 0].reshape(-1, WINDOW_SAMPLES, 2)
    profiles = ("gradual_rise", "late_rise", "two_stage_rise")
    synthetic, synthetic_labels, metadata = [], [], []

    for label, name in ((1, "ELEVATED_THERMAL_RISK"), (2, "HIGH_THERMAL_RISK")):
        for index in range(SYNTHETIC_WINDOWS_PER_RISK_CLASS):
            base = normal[rng.integers(len(normal))].copy()
            profile = profiles[index % len(profiles)]
            curve = rise_curve(profile)
            # Target ending temperature determines the risk class.
            end_temp = (rng.uniform(ELEVATED_TEMP_C, HIGH_TEMP_C - 0.8)
                        if label == 1 else rng.uniform(HIGH_TEMP_C + 0.5, 61.8))
            temperature_gain = max(0.0, end_temp - float(base[-1, 0]))
            base[:, 0] += temperature_gain * curve
            # Heat-gun test showed humidity moving down as temperature rose.
            base[:, 1] -= rng.uniform(0.10, 0.45) * temperature_gain * curve
            synthetic.append(base.reshape(-1))
            synthetic_labels.append(label)
            metadata.append(("synthetic", profile, name))

    synthetic = np.asarray(synthetic, dtype=np.float32)
    all_features = np.vstack((features, synthetic))
    all_labels = np.concatenate((labels, np.asarray(synthetic_labels, dtype=np.int8)))
    return all_features, all_labels, metadata


def save_augmented_windows(features, labels, original_count, metadata):
    """Save the derived labels/windows so students can inspect them."""
    columns = [f"{field}_{index}" for index in range(WINDOW_SAMPLES)
               for field in ("temp_c", "humidity_pct")]
    derived = pd.DataFrame(features, columns=columns)
    derived["label"] = [LABELS[label] for label in labels]
    derived["source"] = ["measured"] * original_count + [item[0] for item in metadata]
    derived["profile"] = ["measured_window"] * original_count + [item[1] for item in metadata]
    ARTIFACTS.mkdir(exist_ok=True)
    derived.to_csv(ARTIFACTS / "heatgun_augmented_windows.csv", index=False)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def main():
    frame = pd.read_csv(SOURCE).dropna()
    measured_X, measured_y = make_training_data(frame)
    rng = np.random.default_rng(RANDOM_SEED)
    X, y, synthetic_metadata = add_synthetic_risks(measured_X, measured_y, rng)
    save_augmented_windows(X, y, len(measured_X), synthetic_metadata)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    X = (X - mean) / scale

    w1 = rng.normal(0, np.sqrt(2 / X.shape[1]), (X.shape[1], HIDDEN_UNITS)).astype(np.float32)
    b1 = np.zeros(HIDDEN_UNITS, dtype=np.float32)
    w2 = rng.normal(0, np.sqrt(2 / HIDDEN_UNITS), (HIDDEN_UNITS, len(LABELS))).astype(np.float32)
    b2 = np.zeros(len(LABELS), dtype=np.float32)
    parameters = [w1, b1, w2, b2]
    first_moment = [np.zeros_like(value) for value in parameters]
    second_moment = [np.zeros_like(value) for value in parameters]
    class_counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
    class_weights = len(y) / (len(LABELS) * class_counts)
    step = 0

    for _ in range(EPOCHS):
        order = rng.permutation(len(X))
        for start in range(0, len(X), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            xb, yb = X[indices], y[indices]
            hidden_linear = xb @ w1 + b1
            hidden = np.maximum(hidden_linear, 0)
            probabilities = softmax(hidden @ w2 + b2)
            gradient_logits = probabilities
            gradient_logits[np.arange(len(yb)), yb] -= 1
            gradient_logits *= class_weights[yb, None] / len(yb)
            gradient_w2 = hidden.T @ gradient_logits
            gradient_b2 = gradient_logits.sum(axis=0)
            gradient_hidden = gradient_logits @ w2.T
            gradient_hidden[hidden_linear <= 0] = 0
            gradients = [xb.T @ gradient_hidden, gradient_hidden.sum(axis=0), gradient_w2, gradient_b2]
            step += 1
            for index, gradient in enumerate(gradients):
                first_moment[index] = 0.9 * first_moment[index] + 0.1 * gradient
                second_moment[index] = 0.999 * second_moment[index] + 0.001 * gradient ** 2
                corrected_first = first_moment[index] / (1 - 0.9 ** step)
                corrected_second = second_moment[index] / (1 - 0.999 ** step)
                parameters[index] -= LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + 1e-8)

    probabilities = softmax(np.maximum(X @ w1 + b1, 0) @ w2 + b2)
    prediction = probabilities.argmax(axis=1)
    confusion = [[int(((y == actual) & (prediction == predicted)).sum())
                  for predicted in range(len(LABELS))] for actual in range(len(LABELS))]
    report = {
        "source_file": str(SOURCE), "source_rows": int(len(frame)),
        "measured_windows": int(len(measured_X)), "synthetic_windows": int(len(synthetic_metadata)),
        "windows": int(len(X)),
        "synthetic_profiles": ["gradual_rise", "late_rise", "two_stage_rise"],
        "structure": [20, HIDDEN_UNITS, 3],
        "class_counts": {LABELS[i]: int(class_counts[i]) for i in range(len(LABELS))},
        "training_accuracy": float((prediction == y).mean()),
        "training_confusion_matrix": confusion,
        "warning": "One controlled run plus synthetic teaching augmentation; training-set result only, not real-world accuracy.",
    }
    model = {
        "kind": "teaching_dense_neural_network",
        "warning": report["warning"], "labels": LABELS,
        "sample_seconds": SAMPLE_SECONDS, "window_samples": WINDOW_SAMPLES,
        "forecast_horizon_seconds": FORECAST_HORIZON_SAMPLES * SAMPLE_SECONDS,
        "structure": [20, HIDDEN_UNITS, 3],
        "activation": ["relu", "softmax"],
        "feature_order": [f"{name}_{i}" for i in range(WINDOW_SAMPLES) for name in ("temp_c", "humidity_pct")],
        "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(),
        "dense_1_weights": w1.tolist(), "dense_1_bias": b1.tolist(),
        "dense_2_weights": w2.tolist(), "dense_2_bias": b2.tolist(),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "heatgun_dense_nn.json").write_text(json.dumps(model, indent=2))
    (ARTIFACTS / "heatgun_dense_nn_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
