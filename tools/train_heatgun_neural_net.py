#!/usr/bin/env python3
"""Train a small dense neural network from captured DHT data.

This file intentionally implements the network with NumPy instead of hiding
the learning steps inside a large ML framework. Students can follow the whole
pipeline here:

1. turn DHT readings into ten-reading windows;
2. attach teaching labels and add clearly marked synthetic rises;
3. normalize every input feature;
4. train Dense(20 -> 12) + ReLU + Dense(12 -> 3) + Softmax;
5. save both the learned model and an honest training-only report.

This is a classroom thermal-risk model, not a validated fire detector.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Files are relative to the repository root. Run this script from that folder.
SOURCE = Path("data/supplied_training_data.csv")
ARTIFACTS = Path("artifacts")

# Ten readings at two seconds each give the model a 20-second history.
WINDOW_SAMPLES = 10
SAMPLE_SECONDS = 2

# Measured windows receive an elevated label when the known controlled capture
# reaches the high boundary within the next 30 readings (60 seconds). This is a
# teaching label made with future knowledge, not proof of future-fire prediction.
FORECAST_HORIZON_SAMPLES = 30
HIGH_TEMP_C = 50.7
ELEVATED_TEMP_C = 35.0
LABELS = ["NORMAL", "ELEVATED_THERMAL_RISK", "HIGH_THERMAL_RISK"]

# Small network/training settings chosen to finish quickly on student laptops.
HIDDEN_UNITS = 12
EPOCHS = 350
BATCH_SIZE = 64
LEARNING_RATE = 0.003

# A fixed seed makes every workshop run repeatable.
RANDOM_SEED = 42
SYNTHETIC_WINDOWS_PER_RISK_CLASS = 1200


def make_training_data(frame):
    """Convert sequential DHT rows into overlapping windows and class numbers.

    Each output feature row contains:
    [temp_0, humidity_0, ..., temp_9, humidity_9].
    Labels are stored as 0, 1, or 2 so they can index the three model outputs.
    """
    readings = frame[["temp_c", "humidity_pct"]].to_numpy(dtype=np.float32)
    features, labels = [], []

    # Stop early enough that every window still has the full future horizon.
    last_start = len(readings) - WINDOW_SAMPLES - FORECAST_HORIZON_SAMPLES
    for start in range(last_start + 1):
        window = readings[start:start + WINDOW_SAMPLES]
        end_temp = float(window[-1, 0])

        # Future values are used only to construct labels for this known
        # recording. They are never included in the model's 20 input values.
        future_peak = float(readings[start + WINDOW_SAMPLES:
                                     start + WINDOW_SAMPLES + FORECAST_HORIZON_SAMPLES, 0].max())
        label = 2 if end_temp >= HIGH_TEMP_C else (
            1 if end_temp >= ELEVATED_TEMP_C or future_peak >= HIGH_TEMP_C else 0
        )

        # Flatten 10 rows x 2 sensors into the 20-number network input.
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
    # Synthetic examples always start from measured normal windows, preserving
    # plausible starting temperature, humidity, and small sensor variation.
    normal = features[labels == 0].reshape(-1, WINDOW_SAMPLES, 2)
    profiles = ("gradual_rise", "late_rise", "two_stage_rise")
    synthetic, synthetic_labels, metadata = [], [], []

    for label, name in ((1, "ELEVATED_THERMAL_RISK"), (2, "HIGH_THERMAL_RISK")):
        for index in range(SYNTHETIC_WINDOWS_PER_RISK_CLASS):
            base = normal[rng.integers(len(normal))].copy()

            # Rotate through three explainable shapes instead of adding random
            # noise with no physical story.
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
    """Convert three raw output scores into probabilities that sum to one."""

    # Subtracting the largest score avoids overflow inside exp(). It does not
    # change the final probability ratios.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def main():
    # Remove incomplete sensor rows before constructing fixed-size windows.
    print(f"Loading sensor data from {SOURCE}...", flush=True)
    frame = pd.read_csv(SOURCE).dropna()

    # First create windows from the real controlled recording, then add the
    # explicitly marked classroom augmentation examples.
    print("Preparing measured and synthetic training windows...", flush=True)
    measured_X, measured_y = make_training_data(frame)
    rng = np.random.default_rng(RANDOM_SEED)
    X, y, synthetic_metadata = add_synthetic_risks(measured_X, measured_y, rng)
    save_augmented_windows(X, y, len(measured_X), synthetic_metadata)

    # Standardization gives every input position roughly comparable numerical
    # scale: normalized_value = (raw_value - mean) / standard_deviation.
    # The same mean and scale are exported for ESP32 inference.
    mean = X.mean(axis=0)
    scale = X.std(axis=0)

    # A constant feature has zero standard deviation. Using 1 keeps it at zero
    # after centering and prevents division by zero.
    scale[scale == 0] = 1.0
    X = (X - mean) / scale

    # He-style random initialization keeps early ReLU activations from becoming
    # excessively large or small. Biases begin at zero.
    w1 = rng.normal(0, np.sqrt(2 / X.shape[1]), (X.shape[1], HIDDEN_UNITS)).astype(np.float32)
    b1 = np.zeros(HIDDEN_UNITS, dtype=np.float32)
    w2 = rng.normal(0, np.sqrt(2 / HIDDEN_UNITS), (HIDDEN_UNITS, len(LABELS))).astype(np.float32)
    b2 = np.zeros(len(LABELS), dtype=np.float32)

    # Keeping references in lists lets one Adam update loop handle all weights
    # and biases. Adam remembers a moving average and squared average for every
    # trainable number.
    parameters = [w1, b1, w2, b2]
    first_moment = [np.zeros_like(value) for value in parameters]
    second_moment = [np.zeros_like(value) for value in parameters]

    # Normal windows are far more common. Class weights make errors in the two
    # smaller risk classes count more strongly during training.
    class_counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
    class_weights = len(y) / (len(LABELS) * class_counts)
    step = 0

    # Print at 0%, 10%, ..., 100%. flush=True is important when this script is
    # launched inside Jupyter, where buffered text may otherwise appear late.
    progress_interval = max(1, EPOCHS // 10)
    print(f"Training neural network: 0% (0/{EPOCHS} epochs)", flush=True)

    for epoch in range(EPOCHS):
        # Shuffle every epoch so batches do not always see examples in the same
        # order. The fixed random seed still makes this repeatable.
        order = rng.permutation(len(X))
        for start in range(0, len(X), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            xb, yb = X[indices], y[indices]

            # Forward pass: input -> first dense layer -> ReLU -> second dense
            # layer -> Softmax probabilities.
            hidden_linear = xb @ w1 + b1
            hidden = np.maximum(hidden_linear, 0)
            probabilities = softmax(hidden @ w2 + b2)

            # For Softmax plus cross-entropy, probability - expected_one_hot is
            # the gradient at the output. Apply class weights and batch average.
            gradient_logits = probabilities
            gradient_logits[np.arange(len(yb)), yb] -= 1
            gradient_logits *= class_weights[yb, None] / len(yb)

            # Backpropagate from output layer into the hidden layer. ReLU passes
            # gradients only where its original input was positive.
            gradient_w2 = hidden.T @ gradient_logits
            gradient_b2 = gradient_logits.sum(axis=0)
            gradient_hidden = gradient_logits @ w2.T
            gradient_hidden[hidden_linear <= 0] = 0
            gradients = [xb.T @ gradient_hidden, gradient_hidden.sum(axis=0), gradient_w2, gradient_b2]

            # Adam optimizer: update moving averages, correct their early-step
            # bias, then take a scaled step for each parameter array.
            step += 1
            for index, gradient in enumerate(gradients):
                first_moment[index] = 0.9 * first_moment[index] + 0.1 * gradient
                second_moment[index] = 0.999 * second_moment[index] + 0.001 * gradient ** 2
                corrected_first = first_moment[index] / (1 - 0.9 ** step)
                corrected_second = second_moment[index] / (1 - 0.999 ** step)
                parameters[index] -= LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + 1e-8)

        completed_epochs = epoch + 1
        if completed_epochs % progress_interval == 0 or completed_epochs == EPOCHS:
            percent = round(100 * completed_epochs / EPOCHS)
            print(
                f"Training neural network: {percent}% "
                f"({completed_epochs}/{EPOCHS} epochs)",
                flush=True,
            )

    # Evaluate on the same data used for learning. This is useful for checking
    # the code path, but it is not independent validation or safety accuracy.
    print("Evaluating the trained model...", flush=True)
    probabilities = softmax(np.maximum(X @ w1 + b1, 0) @ w2 + b2)
    prediction = probabilities.argmax(axis=1)
    confusion = [[int(((y == actual) & (prediction == predicted)).sum())
                  for predicted in range(len(LABELS))] for actual in range(len(LABELS))]
    # The report contains human-readable counts and training diagnostics.
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
    # The model JSON contains everything needed to reproduce inference:
    # feature order, normalization constants, learned weights, and biases.
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
    # Saving JSON keeps the learned numbers easy for students and the separate
    # INT8 exporter to inspect.
    print(f"Saving model and report in {ARTIFACTS}/...", flush=True)
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "heatgun_dense_nn.json").write_text(json.dumps(model, indent=2))
    (ARTIFACTS / "heatgun_dense_nn_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
