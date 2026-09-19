#!/usr/bin/env python3
"""Make a small CSV of ten-reading windows for students to label.

Example:
    python tools/prepare_labeling_windows.py \
        --input data/supplied_training_data.csv \
        --output data/imports/student_window_labels.csv

The input is never changed. We use at most 100 valid readings and skip the
first complete window as a warm-up, so 100 readings produce 90 label rows.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_SAMPLES = 10
MAX_READINGS = 100
LABELS = ("NORMAL", "ELEVATED_THERMAL_RISK", "HIGH_THERMAL_RISK")
FEATURE_COLUMNS = [
    f"{name}_{index}"
    for index in range(WINDOW_SAMPLES)
    for name in ("temp_c", "humidity_pct")
]


def prepare_windows(frame, start_row=0):
    """Return an editable table; rows are windows, not individual readings."""
    required = ("temp_c", "humidity_pct")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing input column(s): {', '.join(missing)}")
    if start_row < 0:
        raise ValueError("--start-row must be zero or greater")

    # A failed DHT reading cannot be a model feature. Remove it before making
    # windows, and tell the class how many rows were removed.
    clean = frame.copy()
    for name in required:
        clean[name] = pd.to_numeric(clean[name], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
    skipped = len(frame) - len(clean)
    clean = clean.iloc[start_row:start_row + MAX_READINGS].reset_index(drop=True)
    if len(clean) < WINDOW_SAMPLES + 1:
        raise ValueError("Need at least 11 valid readings to make one label row")

    rows = []
    # Start at reading 2 (index 1): readings 1–10 are a demonstration/warm-up
    # window. Starts 2..91 yield 90 ten-reading windows from 100 readings.
    for start in range(1, len(clean) - WINDOW_SAMPLES + 1):
        window = clean.iloc[start:start + WINDOW_SAMPLES]
        first, last = window.iloc[0], window.iloc[-1]
        row = {
            "window_id": len(rows) + 1,
            "start_sequence": first.get("sequence", start + 1),
            "end_sequence": last.get("sequence", start + WINDOW_SAMPLES),
            "start_temp_c": first["temp_c"],
            "end_temp_c": last["temp_c"],
            "temp_change_c": last["temp_c"] - first["temp_c"],
            "start_humidity_pct": first["humidity_pct"],
            "end_humidity_pct": last["humidity_pct"],
            "humidity_change_pct": last["humidity_pct"] - first["humidity_pct"],
        }
        for index, (_, reading) in enumerate(window.iterrows()):
            row[f"temp_c_{index}"] = reading["temp_c"]
            row[f"humidity_pct_{index}"] = reading["humidity_pct"]
        row["label"] = ""  # Student fills only this column.
        rows.append(row)
    return pd.DataFrame(rows), skipped, len(clean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw DHT CSV")
    parser.add_argument("--output", type=Path, required=True, help="Editable label CSV")
    parser.add_argument("--start-row", type=int, default=0,
                        help="First valid input reading (zero-based; default 0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing output CSV, including any labels")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files; keep raw data unchanged")
    if args.output.exists() and not args.overwrite:
        parser.error("Output already exists; choose another path or use --overwrite")

    frame = pd.read_csv(args.input)
    try:
        windows, skipped, readings = prepare_windows(frame, args.start_row)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output, index=False)
    print(f"Used {readings} readings; wrote {len(windows)} windows to {args.output}")
    if skipped:
        print(f"Skipped {skipped} incomplete/invalid sensor reading(s)")
    print("Fill the label column with: " + ", ".join(LABELS))


if __name__ == "__main__":
    main()
