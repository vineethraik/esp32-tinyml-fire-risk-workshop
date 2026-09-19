# Fire Risk Detection with ESP32-WROOM (TinyML)

## Objective

Build a low-cost, offline fire early-warning device that detects the gradual
onset of fire—not only the flame itself—using a time-series window of sensor
readings and an on-device ML model.

## Current workshop hardware (per node)

- NodeMCU ESP32S V1.1 development board, using an ESP32-WROOM-32 module
  (MCU + Wi-Fi)
- DHT22 V182 (temperature + humidity): GPIO 16 (`P16`) data; power from 3V3; 10 kΩ
  pull-up from DATA to 3V3 when using a bare sensor
- Breadboard, jumper wires, and 10 kΩ resistor

Later extensions may add MQ-2 on GPIO34, an LM393 flame sensor on GPIO5,
buzzer on GPIO26, and LED on GPIO27. They are not part of the current model or
required workshop build.

> [!IMPORTANT]
> Burn in the MQ-2 for 48 hours before use. Use a voltage divider on its analog
> output because the MQ-2 module operates at 5 V while the ESP32 ADC input does
> not tolerate 5 V.
>
> Verify the V182 module's printed pin order before wiring. Do not pull the
> DHT22 DATA line to 5 V: ESP32 GPIO inputs are 3.3 V logic.

## Current open-source ML pipeline

1. ESP32 reads DHT22 every two seconds and prints serial data.
2. `tools/collect_serial_csv.py` saves raw CSV on the student's PC.
3. `tools/prepare_labeling_windows.py` makes a separate CSV of at most 90
   ten-reading windows from at most 100 valid readings. Students manually label
   these windows; the first complete window is a warm-up example.
4. The workshop uses `data/supplied_training_data.csv` for repeatable training.
5. `tools/train_heatgun_neural_net.py` creates ten-sample windows, teaching
   labels, clearly marked synthetic rises, and a `20 → 12 → 3` dense network.
   It can optionally add completed student-labeled windows; blank labels are
   ignored. The repeatable supplied-data path remains the default.
6. `tools/export_dense_int8_header.py` converts weights to an ESP32 C++ header.
7. ESP32 performs offline inference and a 2-of-3 high-risk vote.

## Input Vector (per inference)

Window of 10 readings × 2 sensors = 20 features (20 seconds at the current
2-second sample interval):

```text
[temp(t-9), hum(t-9),
 temp(t-8), hum(t-8),
 ...
 temp(t),   hum(t)]
```

## Alert Logic

1. Run inference every two seconds after sensor reading and normalization.
2. Use 2-of-3 voting: alert only when at least two of three consecutive windows
   are classified `HIGH_RISK`.
3. Current workshop output is serial only. Buzzer, LED, Wi-Fi, and MQTT are
   extensions, not implemented alarm behavior.

## Teaching labels

| Class | Meaning | Source |
| --- | --- | --- |
| `NORMAL` | Ordinary DHT window | Measured capture |
| `ELEVATED_THERMAL_RISK` | Warming or near-future high-temperature window | Teaching label + synthetic rises |
| `HIGH_THERMAL_RISK` | Controlled high-temperature window | Measured capture + synthetic rises |

## Workshop success criteria

- Student explains data, labels, training, and inference.
- Student collects DHT readings into a raw CSV.
- Student trains the supplied tiny neural network on a PC.
- Student exports the model and runs risk inference on ESP32.
- Student explains why training accuracy is not real-world safety evidence.

Real product goals such as early-fire detection and a false-alarm rate below 5%
remain unvalidated future work.

## Session plan (5.5–6-hour workshop)

| Block | Duration | Work |
| --- | --- | --- |
| 1 | 1.25 h | AI/ML basics + DHT wiring + serial readings |
| 2 | 1 h | Python collection + data inspection + manual labels |
| Break | 15 min | — |
| 3 | 1.5 h | Windowing + neural-network training + evaluation |
| 4 | 1 h | INT8 export + ESP32 inference |
| 5 | 30–45 min | Live demo + limitations + extensions |

## Pre-session Checklist

- [ ] Python, JupyterLab, PlatformIO, and USB driver prepared
- [ ] Supplied training CSV present
- [ ] Complete notebook run once on the instructor PC
- [ ] Full firmware tested on one board
- [ ] Starter sketch (sensor read only) prepared for students

## Core BOM per student

| Item | Approximate cost |
| --- | ---: |
| ESP32-WROOM-32 development board | INR 400–600 |
| DHT22 | INR 120–180 |
| Jumper wires + breadboard | INR 100–150 |

Optional extensions: MQ-2 (INR 250–350), flame sensor (INR 80–120), and
buzzer/LED/resistors (about INR 50).

## Extensions (post-workshop)

- ESP-NOW mesh for multi-node coverage
- ESP32-S3 + camera for vision-based confirmation
- Home Assistant or Blynk dashboard
- Solar-powered deployment for outdoor or forest use
