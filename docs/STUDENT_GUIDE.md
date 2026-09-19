# Student guide

## Your mission

Build a small system that watches a recent temperature/humidity pattern and
prints a risk label. You are learning the process, not building a certified
fire alarm.

## Part 1 — Think before coding

Answer with your group:

1. What rule would you write using only one temperature value?
2. What might that rule miss if temperature is rising quickly?
3. What does a 20-second history tell us that one reading does not?

## Part 2 — Wire and read sensor

Wire DHT22 as shown in [setup](SETUP.md). Flash the supplied firmware. At serial
115200, send:

```text
AT+STATUS
AT+LOG,ON
```

You should see a new `LOG` line every two seconds. Send `AT+LOG,OFF` when done.

## Part 3 — Look at data

Open `data/supplied_training_data.csv`. It has three columns:

```text
sequence,temp_c,humidity_pct
```

Find a calm region and a warming region. Ask: “Would one fixed temperature
threshold notice the warming early?”

## Part 4 — Make windows and label them

One sensor reading does not show a trend. From the repository root, make a
separate labeling CSV:

```sh
.venv-ml311/bin/python tools/prepare_labeling_windows.py \
  --input data/supplied_training_data.csv \
  --output data/imports/student_window_labels.csv
```

It takes at most 100 valid readings. A ten-reading sliding window would make
91 windows; we keep the first as a warm-up and give you the remaining 90 to
label. Open the output CSV. Each row has a summary plus the 20 numbers the
model receives. Fill only `label` with `NORMAL`, `ELEVATED_THERMAL_RISK`, or
`HIGH_THERMAL_RISK`. You may leave some blank. Do not edit the raw CSV.

The first 100 readings may all be normal. To choose a later 100-reading region,
add `--start-row 30000` or another zero-based valid-reading position. If the
output file already exists, the tool refuses to overwrite your labels: choose
a new output name, or use `--overwrite` only when you mean to replace it.

## Part 5 — Train a tiny neural network

From the repository root, run:

```sh
.venv-ml311/bin/python tools/train_heatgun_neural_net.py
```

For an optional run using your completed labels as extra examples, use:

```sh
.venv-ml311/bin/python tools/train_heatgun_neural_net.py \
  --student-windows data/imports/student_window_labels.csv
```

Blank label rows are skipped. The supplied teaching data still provides most
examples; 90 overlapping windows are not 90 independent real-world tests.

Read the printed report. It tells you how many measured and synthetic windows
were used. Open `artifacts/heatgun_augmented_windows.csv` and find the `source`
and `profile` columns. Never call synthetic data “real measurements.”

## Part 6 — Export for ESP32

```sh
.venv-ml311/bin/python tools/export_dense_int8_header.py
pio run --target upload
```

The exporter writes `include/thermal_risk_model.h`. The ESP32 firmware reads
this file when it is compiled.

## Part 7 — Ask ESP32 for its prediction

Wait about 20 seconds after boot, then send:

```text
AT+RISK
```

Example:

```text
ACK+RISK,ready=1,label=NORMAL,confidence=1.000,voted_high=0
OK+RISK
```

Use `AT+RISK,ON` for one result every sensor reading. `voted_high=1` requires
two high-risk results from the newest three windows.

## Part 8 — Reflect

Write short answers:

1. Which part learned from data: sensor, model, or buzzer?
2. Why can a high training score be misleading?
3. What real data would you collect before trusting this near a kitchen?
4. What second sensor would make the system safer, and why?
