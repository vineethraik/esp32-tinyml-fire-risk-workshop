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

Open `data/heatgun_box_002.csv`. It has three columns:

```text
sequence,temp_c,humidity_pct
```

Find a calm region and a warming region. Ask: “Would one fixed temperature
threshold notice the warming early?”

## Part 4 — Train a tiny neural network

From the repository root, run:

```sh
.venv-ml311/bin/python tools/train_heatgun_neural_net.py
```

Read the printed report. It tells you how many measured and synthetic windows
were used. Open `artifacts/heatgun_augmented_windows.csv` and find the `source`
and `profile` columns. Never call synthetic data “real measurements.”

## Part 5 — Export for ESP32

```sh
.venv-ml311/bin/python tools/export_dense_int8_header.py
pio run --target upload
```

The exporter writes `include/thermal_risk_model.h`. The ESP32 firmware reads
this file when it is compiled.

## Part 6 — Ask ESP32 for its prediction

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

## Part 7 — Reflect

Write short answers:

1. Which part learned from data: sensor, model, or buzzer?
2. Why can a high training score be misleading?
3. What real data would you collect before trusting this near a kitchen?
4. What second sensor would make the system safer, and why?
