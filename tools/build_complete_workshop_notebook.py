#!/usr/bin/env python3
"""Build the single self-learning workshop notebook from readable cell sources."""

import json
from pathlib import Path


OUTPUT = Path("notebooks/complete_tinyml_fire_risk_workshop.ipynb")


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


cells = [
    markdown(r"""
# ESP32 TinyML thermal-risk workshop

## From the first sensor reading to a neural network running on ESP32

This notebook is a complete beginner-friendly workshop for approximately **5.5–6 hours**. You can study it alone or follow it with an instructor.

By the end, you will understand and demonstrate this complete path:

```text
DHT22 sensor → ESP32 serial data → Python CSV collector
             → labels → 10-reading windows → neural-network training
             → INT8 model weights → ESP32 inference → 2-of-3 risk vote
```

> **Safety:** This is a classroom thermal-pattern demonstration. It is not a certified fire alarm. Never depend on it to protect people or property. Only an instructor should operate a heat gun, with students and flammable material kept away.
"""),
    markdown(r"""
## Workshop schedule

| Time | Part | Result |
| --- | --- | --- |
| 00:00–00:30 | AI and ML basics | Explain data, labels, training, and inference |
| 00:30–01:15 | ESP32 and DHT22 | Read temperature and humidity |
| 01:15–01:50 | Python collector | Save serial readings into CSV |
| 01:50–02:20 | Inspect and label | Create normal/elevated/high examples |
| 02:20–02:35 | Break | — |
| 02:35–03:45 | Build and train model | Train a `20 → 12 → 3` neural network |
| 03:45–04:20 | Test and quantize | Inspect results and export INT8 weights |
| 04:20–05:10 | ESP32 inference | Flash model and read live risk output |
| 05:10–05:45 | Limits and extensions | Discuss safety, validation, MQ-2, and flame sensing |
"""),
    markdown(r"""
## 1. Requirements

### Hardware per team

- NodeMCU ESP32S V1.1 with ESP32-WROOM-32
- DHT22 V182 temperature/humidity sensor
- Breadboard and jumper wires
- USB **data** cable
- Optional 4.7–10 kΩ pull-up resistor for a bare DHT22
- Instructor-only heat gun for a controlled final demonstration

Later extensions may use an MQ-2 gas sensor, flame sensor, LED, and buzzer. They are not required for this core workshop.

### Software

- Git
- Python 3.11 or newer
- JupyterLab
- NumPy and pandas
- pyserial for the simple serial-to-CSV collector
- VS Code with PlatformIO, or PlatformIO Core
- CP2102 USB driver if the board does not appear as a serial port

### DHT22 wiring

| DHT22 V182 | NodeMCU ESP32S V1.1 |
| --- | --- |
| VCC | `3V3` |
| DATA | `P16` / GPIO16 |
| GND | `GND` |

Never pull DHT22 DATA to 5 V. If the module already has a pull-up resistor, an external resistor is normally unnecessary.
"""),
    markdown(r"""
### PC setup

Run these commands once from a terminal:

```sh
git clone git@github.com:vineethraik/esp32-tinyml-fire-risk-workshop.git
cd esp32-tinyml-fire-risk-workshop
python3 -m venv .venv-ml311
source .venv-ml311/bin/activate        # macOS/Linux
pip install -r requirements.txt
jupyter lab
```

On Windows PowerShell, activation is usually:

```powershell
.venv-ml311\Scripts\Activate.ps1
```

Open this notebook and choose the virtual environment as the Python kernel.
"""),
    code(r"""
# Find the repository root even when Jupyter starts inside notebooks/.
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

candidates = [Path.cwd(), Path.cwd().parent]
PROJECT_ROOT = next(
    (path.resolve() for path in candidates if (path / "platformio.ini").exists()),
    None,
)
assert PROJECT_ROOT is not None, "Open this notebook from inside the cloned repository."
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

print("Project root:", PROJECT_ROOT)
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
"""),
    markdown(r"""
## 2. What are AI and machine learning?

**Artificial intelligence (AI)** is a broad name for computers doing tasks that appear to require judgement.

**Machine learning (ML)** is one method inside AI. Instead of writing every rule ourselves, we give the computer many examples with the answers we want. Training adjusts internal numbers so the model can make a useful guess for a similar new example.

| Conventional program | Machine-learning program |
| --- | --- |
| Human writes `if temperature > 50: high risk` | Human supplies sensor histories and labels |
| Rule is visible and exact | Model learns many small internal weights |
| Excellent when rules are simple and known | Useful when trends or combinations matter |

A real safety system normally combines both: hard safety rules **and** a carefully validated model.

Five important words:

- **Data:** recorded sensor values.
- **Label:** the answer attached to an example.
- **Model:** learned numbers that transform input into a prediction.
- **Training:** adjusting model numbers on a PC using labelled examples.
- **Inference:** using the finished model for one prediction, here on ESP32.

The model does not “understand fire.” It compares a new sensor pattern with patterns represented in its training data.
"""),
    markdown(r"""
## 3. Part A — Read DHT22 using ESP32

Start with one small job: read the sensor and print a clean line every two seconds.

```cpp
#include <Arduino.h>
#include <DHT.h>

constexpr uint8_t DHT_PIN = 16;
constexpr uint8_t DHT_TYPE = DHT22;
constexpr unsigned long SAMPLE_MS = 2000;

DHT dht(DHT_PIN, DHT_TYPE);
unsigned long lastSample = 0;

void setup() {
  Serial.begin(115200);
  dht.begin();
}

void loop() {
  if (millis() - lastSample < SAMPLE_MS) return;
  lastSample = millis();

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("ERROR,DHT_READ");
    return;
  }

  Serial.printf("DATA,%.2f,%.2f\n", temperature, humidity);
}
```

Important parts:

- `DHT_PIN = 16` matches the verified board wiring.
- DHT22 is intentionally read every two seconds; reading it too quickly is unreliable.
- `isnan(...)` rejects failed readings instead of silently storing bad data.
- A stable `DATA,temp,humidity` format makes the PC collector simple.

Expected serial output:

```text
DATA,27.80,54.20
DATA,27.90,54.10
```
"""),
    markdown(r"""
## 4. Part B — Python serial-to-CSV collector

The collector is `tools/collect_serial_csv.py`. It uses pyserial, already included in `requirements.txt`. Run:

```sh
python tools/collect_serial_csv.py \
  --port /dev/cu.usbserial-0001 \
  --output data/raw/my_session.csv \
  --count 100
```

Windows ports look like `COM3`. Omit `--count` to collect until `Ctrl+C`.

The important parsing logic is intentionally small:

```python
fields = line.strip().split(",")
if len(fields) == 3 and fields[0] == "DATA":
    sequence, temperature, humidity = "", fields[1], fields[2]
else:
    return None

temperature_value = float(temperature)
humidity_value = float(humidity)
```

The complete script also:

- ignores ESP32 boot messages and malformed lines;
- accepts the final firmware's `LOG,sequence,temp,humidity` format;
- adds a UTC timestamp from the PC;
- flushes each valid row so a sudden stop loses at most the current line;
- refuses to overwrite an existing raw CSV unless `--append` is explicit.

Its raw CSV has no label column:

```csv
timestamp_utc,sequence,temp_c,humidity_pct
2026-09-18T10:30:02+00:00,,27.80,54.20
```

Labels belong in a later working copy, not in the immutable raw capture.
"""),
    markdown(r"""
## 5. Part C — Inspect the supplied real dataset

Students may collect their own calm-room data, but that will usually contain no meaningful high-risk examples. Therefore everyone receives the same captured heat-gun dataset for repeatable model training.

The raw capture remains unchanged. Derived labels and synthetic examples are stored separately.
"""),
    code(r"""
DATA_FILE = PROJECT_ROOT / "data" / "heatgun_box_002.csv"

raw_data = pd.read_csv(DATA_FILE)
data = raw_data.dropna(subset=["temp_c", "humidity_pct"]).copy()

print("Raw rows:", len(raw_data))
print("Usable rows:", len(data))
print("Temperature range:", data.temp_c.min(), "to", data.temp_c.max(), "°C")
print("Humidity range:", data.humidity_pct.min(), "to", data.humidity_pct.max(), "%")
display(data.head(10))
"""),
    markdown(r"""
### See variation without special plotting libraries

This summary divides the capture into consecutive blocks. It helps us find calm periods, heat rises, and cooling periods before assigning labels.
"""),
    code(r"""
block_size = 500
summary = (
    data.assign(block=np.arange(len(data)) // block_size)
        .groupby("block")
        .agg(
            first_sequence=("sequence", "first"),
            last_sequence=("sequence", "last"),
            min_temp_c=("temp_c", "min"),
            max_temp_c=("temp_c", "max"),
            mean_temp_c=("temp_c", "mean"),
            mean_humidity_pct=("humidity_pct", "mean"),
        )
        .round(2)
)

display(summary)
"""),
    markdown(r"""
## 6. Part D — Manual labeling

Make a working copy and add one label per row or selected region:

- `NORMAL`: ordinary room condition.
- `ELEVATED_THERMAL_RISK`: warming trend or unusually warm condition.
- `HIGH_THERMAL_RISK`: controlled high-temperature part of this experiment.

Labels are human decisions, not sensor measurements. Keep raw and labelled files separate.

The next cell creates a small labeling template. Open the saved CSV in a spreadsheet and fill its `label` column. It does not modify the raw capture.
"""),
    code(r"""
label_template = data.loc[:, ["sequence", "temp_c", "humidity_pct"]].head(100).copy()
label_template["label"] = ""

template_path = PROJECT_ROOT / "artifacts" / "student_labeling_template.csv"
label_template.to_csv(template_path, index=False)

print("Created:", template_path)
display(label_template.head())
"""),
    markdown(r"""
### Labels used by the supplied teaching model

For a repeatable class, the supplied training script generates consistent *teaching labels*:

- high when current temperature is at least 50.7 °C;
- elevated when current temperature is at least 35 °C or the known capture reaches 50.7 °C within the next 60 seconds;
- normal otherwise.

It also creates clearly marked synthetic examples using gradual, late, and two-stage rises. This solves a classroom shortage of examples; it does **not** replace real field data.
"""),
    code(r"""
import train_heatgun_neural_net as training

measured_X, measured_y = training.make_training_data(data)
measured_counts = {
    training.LABELS[index]: int((measured_y == index).sum())
    for index in range(len(training.LABELS))
}

print("Window shape:", measured_X.shape)
print("Each window:", training.WINDOW_SAMPLES, "readings × 2 sensors =", measured_X.shape[1], "features")
print("Measured teaching-label counts:", measured_counts)
"""),
    markdown(r"""
## 7. Part E — Convert time-series data into windows

One reading cannot describe a trend. The model receives the newest ten readings:

```text
[temp(t-9), humidity(t-9), ..., temp(t), humidity(t)]
```

Ten readings at two-second intervals represent 20 seconds. `reshape(-1)` converts the `10 × 2` table into 20 input numbers.
"""),
    code(r"""
example_window = data[["temp_c", "humidity_pct"]].iloc[:10].to_numpy(dtype=np.float32)
example_vector = example_window.reshape(-1)

print("Window table shape:", example_window.shape)
print(example_window)
print("\nFlattened model input shape:", example_vector.shape)
print(example_vector)
"""),
    markdown(r"""
## 8. Part F — The tiny neural network

Our model is deliberately small:

```text
20 inputs → 12 hidden ReLU units → 3 Softmax outputs
```

- The 20 inputs are ten temperature/humidity pairs.
- Twelve hidden units combine inputs into learned pattern signals.
- ReLU keeps positive signals and replaces negative values with zero.
- Three output scores represent normal, elevated, and high risk.
- Softmax converts scores into values that add to one.

During training, wrong predictions adjust weights and biases. We do not manually write the final pattern rules.
"""),
    code(r"""
# Full training run. On this project computer it takes roughly 10–20 seconds.
start = time.perf_counter()
training.main()
training_seconds = time.perf_counter() - start

report_path = PROJECT_ROOT / "artifacts" / "heatgun_dense_nn_report.json"
report = json.loads(report_path.read_text())

print(f"Training time: {training_seconds:.2f} seconds")
print(json.dumps(report, indent=2))
"""),
    markdown(r"""
### Read the result honestly

The confusion matrix counts training examples by actual row and predicted column. A high training score means the model fits examples it has already seen. It is **not** real-world fire-detection accuracy.

The current data comes from one controlled run plus synthetic augmentation. A production study must collect separate sessions across rooms, weather, kitchens, devices, distances, sensor ages, and non-fire heat sources. Train and test sessions must be separated before windows are created, or nearly identical neighboring windows can leak into both sets.
"""),
    code(r"""
labels = report["class_counts"].keys()
confusion = pd.DataFrame(
    report["training_confusion_matrix"],
    index=[f"actual_{name}" for name in labels],
    columns=[f"predicted_{name}" for name in labels],
)

print("Training accuracy:", f"{report['training_accuracy'] * 100:.2f}%")
display(confusion)
"""),
    markdown(r"""
## 9. Part G — Run PC inference using the saved model

The next function performs the same important steps as ESP32:

1. flatten the ten readings;
2. normalize each input using training mean and scale;
3. calculate the first dense layer;
4. apply ReLU;
5. calculate output scores;
6. apply Softmax and select the largest score.
"""),
    code(r"""
model_path = PROJECT_ROOT / "artifacts" / "heatgun_dense_nn.json"
model = json.loads(model_path.read_text())

mean = np.asarray(model["scaler_mean"], dtype=np.float32)
scale = np.asarray(model["scaler_scale"], dtype=np.float32)
w1 = np.asarray(model["dense_1_weights"], dtype=np.float32)
b1 = np.asarray(model["dense_1_bias"], dtype=np.float32)
w2 = np.asarray(model["dense_2_weights"], dtype=np.float32)
b2 = np.asarray(model["dense_2_bias"], dtype=np.float32)

def predict_window(window):
    vector = np.asarray(window, dtype=np.float32).reshape(-1)
    normalized = (vector - mean) / scale
    hidden = np.maximum(normalized @ w1 + b1, 0.0)  # ReLU
    logits = hidden @ w2 + b2
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()            # Softmax
    class_index = int(probabilities.argmax())
    return model["labels"][class_index], probabilities

normal_index = int(np.where(measured_y == 0)[0][0])
high_index = int(np.where(measured_y == 2)[0][0])

for name, index in (("normal example", normal_index), ("high example", high_index)):
    label, probabilities = predict_window(measured_X[index].reshape(10, 2))
    print(name, "→", label, dict(zip(model["labels"], probabilities.round(4))))
"""),
    markdown(r"""
## 10. Part H — Quantize model weights for ESP32

INT8 stores each weight as a one-byte integer from -127 to 127. The exporter finds a scale and rounds floating-point weights:

```text
scale = largest absolute weight / 127
int8 value = round(float value / scale)
```

The current teaching firmware stores INT8 weights, then dequantizes them during its small floating-point calculation. This is easy to inspect but is not a full integer TensorFlow Lite Micro pipeline.
"""),
    code(r"""
export_script = PROJECT_ROOT / "tools" / "export_dense_int8_header.py"
subprocess.run([sys.executable, str(export_script)], check=True)

header_path = PROJECT_ROOT / "include" / "thermal_risk_model.h"
print("Generated:", header_path)
print("Header size:", header_path.stat().st_size, "bytes")
print("\nFirst lines:\n")
print("\n".join(header_path.read_text().splitlines()[:14]))
"""),
    markdown(r"""
## 11. Part I — ESP32 inference integration

The repository keeps each ESP32 learning step separately:

| Folder | Lesson |
| --- | --- |
| `firmware/01_dht_serial` | sensor → clean serial lines |
| `firmware/02_flash_storage` | append readings to local flash and export them |
| `firmware/03_tinyml_inference` | ten-reading window → model → vote |
| repository root | final combined ring logger + export + inference |

Build a small stage with `pio run --project-dir firmware/01_dht_serial`. The final combined firmware remains in `src/main.cpp`.

The inference stage:

1. stores the newest ten valid DHT readings in a circular history;
2. puts them back into chronological order;
3. normalizes all 20 values;
4. runs the two dense layers and Softmax;
5. saves the latest class and confidence;
6. performs a 2-of-3 vote for high risk.

The vote prevents one unusual window from immediately becoming a high-risk alert:

```cpp
highRiskVotes[highRiskVoteIndex] = latestRiskClass == 2 ? 1 : 0;

uint8_t highVotes = 0;
for (size_t i = 0; i < highRiskVoteCount; ++i) {
  highVotes += highRiskVotes[i];
}
votedHighRisk = highRiskVoteCount == 3 && highVotes >= 2;
```

Build and flash manually only when your board is connected:

```sh
pio run --project-dir firmware/01_dht_serial --target upload
python tools/collect_serial_csv.py --port /dev/cu.usbserial-0001 \
  --output data/raw/my_session.csv --count 100

pio run --project-dir firmware/03_tinyml_inference --target upload
pio device monitor --baud 115200 --project-dir firmware/03_tinyml_inference

# Final combined system
pio run --target upload
pio device monitor --baud 115200
```

Wait for ten new samples, about 20 seconds, then send:

```text
AT+RISK
AT+RISK,ON
AT+RISK,OFF
```

Example result:

```text
ACK+RISK,ready=1,label=NORMAL,confidence=1.000,voted_high=0
OK+RISK
```
"""),
    code(r"""
# Safe build helper. It only prints the command unless you deliberately enable it.
RUN_PLATFORMIO_BUILD = False

if RUN_PLATFORMIO_BUILD:
    subprocess.run(["pio", "run"], cwd=PROJECT_ROOT, check=True)
else:
    print("Build skipped. Set RUN_PLATFORMIO_BUILD=True only after PlatformIO is installed.")
    print("Command: pio run")
"""),
    markdown(r"""
## 12. Live demonstration

1. Show `AT+RISK` under ordinary room conditions.
2. Send `AT+RISK,ON` to display every new prediction.
3. Instructor warms the sensor gradually from a safe distance.
4. Students watch the ten-reading window respond over time.
5. Point out the difference between class confidence and `voted_high`.
6. Stop heat and allow the sensor to cool.

If the output does not change, inspect actual DHT values with `AT+LOG,ON`. A live demo is also a lesson about sensor delay, placement, different environments, and models encountering conditions unlike their training data.
"""),
    markdown(r"""
## 13. What this workshop proves—and what it does not

### Demonstrated

- DHT22 readings can be collected through ESP32 serial.
- A PC script can create a CSV dataset.
- Time-series readings can be grouped into model input windows.
- A small dense neural network can be trained locally.
- Quantized weights fit easily into ESP32 firmware.
- Offline inference and 2-of-3 voting run on the device.

### Not demonstrated

- reliable early prediction of real fires;
- smoke or gas detection using only DHT22;
- a false-alarm rate below 5% in real kitchens;
- performance across different rooms, devices, seasons, or sensor ages;
- certified alarm behavior.

For a stronger next version, collect independent real sessions, split training and test by session, add a safely connected and burned-in MQ-2, use the flame sensor only as direct confirmation, measure inference time with `micros()`, and compare ML against a simple threshold baseline.
"""),
    markdown(r"""
## 14. Self-check questions

1. What is the difference between training and inference?
2. Why use ten readings instead of one temperature value?
3. Who decides the meaning of each label?
4. Why must raw and labelled data be separate?
5. Does 99% training accuracy prove the device is safe?
6. What does INT8 change?
7. Why use a 2-of-3 vote?
8. What extra real-world data would you collect?

### Suggested answers

1. Training adjusts model weights on a PC; inference uses finished weights for a prediction.
2. Ten readings show a short trend rather than one moment.
3. Humans define labels and the collection procedure.
4. We need traceability and must not confuse measurements with later judgement.
5. No. It only shows fit to training examples, especially with synthetic data.
6. It stores weights as small one-byte integers using scales.
7. It reduces reaction to one unusual prediction.
8. Different rooms, devices, days, kitchens, heat sources, distances, airflow, and safe controlled fire/smoke scenarios.
"""),
    markdown(r"""
## Final conclusion

You built the complete TinyML idea in small pieces:

```text
measure → collect → inspect → label → window → train
        → evaluate → quantize → deploy → infer → vote
```

The most important lesson is not that “AI detects fire.” It is that an embedded ML system is a chain of human choices: sensor selection, data collection, labels, model size, evaluation, deployment, and safety action. A small model can run easily on ESP32; building trustworthy evidence is the difficult part.
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
