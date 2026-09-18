# Fire Risk Detection with ESP32-WROOM (TinyML)

## Objective

Build a low-cost, offline fire early-warning device that detects the gradual
onset of fire—not only the flame itself—using a time-series window of sensor
readings and an on-device ML model.

## Hardware (per node)

- NodeMCU ESP32S V1.1 development board, using an ESP32-WROOM-32 module
  (MCU + Wi-Fi)
- DHT22 V182 (temperature + humidity): GPIO 16 (`P16`) data; power from 3V3; 10 kΩ
  pull-up from DATA to 3V3 when using a bare sensor
- MQ-2 gas sensor (smoke/LPG/CO): AO on GPIO 34 (ADC), DO on GPIO 2
- IR flame sensor LM393 (direct IR flame, 760–1100 nm): GPIO 5
- Buzzer: GPIO 26
- LED: GPIO 27
- Breadboard, jumper wires, and 10 kΩ resistor

> [!IMPORTANT]
> Burn in the MQ-2 for 48 hours before use. Use a voltage divider on its analog
> output because the MQ-2 module operates at 5 V while the ESP32 ADC input does
> not tolerate 5 V.
>
> Verify the V182 module's printed pin order before wiring. Do not pull the
> DHT22 DATA line to 5 V: ESP32 GPIO inputs are 3.3 V logic.

## ML Pipeline

1. Data collection: Arduino sketch → serial → CSV (0.5 Hz / 2-second samples,
   10-sample windows)
2. Labeling: manual (`normal`, `pre-fire`, `fire`), 200+ windows per class
3. Training: Edge Impulse free tier, Time Series input block, window = 10
4. Classifier: Dense neural network or 1D CNN
5. Quantization: INT8 (automatic in Edge Impulse)
6. Export: Arduino/PlatformIO library (auto-generates C++ and `model.h`)
7. Inference: TensorFlow Lite Micro on ESP32 (~25–30 ms per window)

## Input Vector (per inference)

Window of 10 readings × 3 sensors = 30 features (20 seconds at the current
2-second sample interval):

```text
[temp(t-9), hum(t-9), gas(t-9),
 temp(t-8), hum(t-8), gas(t-8),
 ...
 temp(t),   hum(t),   gas(t)]
```

## Alert Logic

1. Run inference every two seconds after sensor reading and normalization.
2. Use 2-of-3 voting: alert only when at least two of three consecutive windows
   are classified `HIGH_RISK`.
3. On alert, activate buzzer and LED, with optional Wi-Fi/MQTT publication.

## Data Collection Targets

| Class | How to generate | Duration |
| --- | --- | --- |
| Normal | Room at rest, no sources | 5–10 min |
| Pre-fire | Candle/incense at 30–50 cm, heat gun on low | 5–10 min |
| Fire | Flame at 10–20 cm, incense fully lit | 5 min |

## Success Criteria

- Detect fire risk before visible flame (trend, not only spike)
- Less than 50 ms inference time on ESP32-WROOM
- Less than 50 KB total RAM usage (firmware + model + buffers)
- False-alarm rate below 5% under normal kitchen conditions

## Session Plan (6.5-hour workshop)

| Block | Duration | Work |
| --- | --- | --- |
| 1 | 1.5 h | Wiring + sensor-read sketch |
| 2 | 2 h | Edge Impulse: upload data → train → export → flash |
| Break | 15 min | — |
| 3 | 1.5 h | Alert logic + live fire demo |
| 4 | 30 min | Recap + extensions |

## Pre-session Checklist

- [ ] MQ-2 burned in for 48 hours
- [ ] Training CSV collected and labeled
- [ ] Edge Impulse project trained and tested
- [ ] Full firmware tested on one board
- [ ] Starter sketch (sensor read only) prepared for students

## BOM per Student (approximately INR 1,000–1,500)

| Item | Approximate cost |
| --- | ---: |
| ESP32-WROOM-32 development board | INR 400–600 |
| DHT22 | INR 120–180 |
| MQ-2 | INR 250–350 |
| IR flame sensor (LM393) | INR 80–120 |
| Buzzer + LED + resistors | INR 50 |
| Jumper wires + breadboard | INR 100–150 |

## Extensions (post-workshop)

- ESP-NOW mesh for multi-node coverage
- ESP32-S3 + camera for vision-based confirmation
- Home Assistant or Blynk dashboard
- Solar-powered deployment for outdoor or forest use
