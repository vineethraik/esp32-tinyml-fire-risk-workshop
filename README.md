# ESP32 TinyML Fire-Risk Detector

> Classroom project only. It is not a certified fire alarm and must never be
> the only safety protection for people or property.

## Start here

This repository is a complete, beginner-friendly 5.5–6 hour workshop. Students
learn the idea of AI and machine learning, inspect real sensor data, train a
small neural network on their PC, export it, and run it on an ESP32.

- [One-day workshop plan](docs/WORKSHOP_PLAN.md) — timings, goals, and outcomes.
- [Facilitator guide](docs/FACILITATOR_GUIDE.md) — exact flow and speaking notes.
- [Student guide](docs/STUDENT_GUIDE.md) — learner tasks, no AI background needed.
- [Demo runbook](docs/DEMO_RUNBOOK.md) — full working demo, step by step.
- [Setup and materials](docs/SETUP.md) — hardware, software, pre-session checks.
- [Simple ML concepts](docs/ML_CONCEPTS.md) — plain-language explanations.

The minimum live hardware demo needs only a NodeMCU ESP32S V1.1 and DHT22 V182.
MQ-2 and flame sensing are later extensions, not prerequisites for this lesson.

## Current firmware: DHT22 logger + teaching neural network

Firmware stores one DHT22 temperature/humidity reading every two seconds.
The flame sensor is reserved for an immediate fire indication and is not stored
in the ML time-series data.

The current on-device model uses ten temperature/humidity readings (20 seconds)
as 20 inputs, then `Dense(20 -> 12)`, ReLU, and `Dense(12 -> 3)`. Outputs are
`NORMAL`, `ELEVATED_THERMAL_RISK`, and `HIGH_THERMAL_RISK`. It was trained from
the captured DHT file plus clearly marked synthetic smooth heating examples:
gradual rise, late rise, and two-stage rise. The raw capture is never changed.

The generated header stores both dense layers as INT8 values. The teaching
firmware dequantizes weights for its tiny floating-point calculation; it is not
yet a fully integer TensorFlow Lite Micro model. This makes the model path easy
to teach and inspect. A real deployment needs independent measured validation
and a fully quantized TFLite export.

## Storage

[partitions.csv](partitions.csv) gives the 4 MiB NodeMCU flash one 2.94 MiB
firmware partition and one exact 1 MiB raw data partition. OTA is disabled.

Each record uses 8 bytes:

```text
24-bit sequence, temperature, humidity, CRC
```

Capacity is 131,072 samples, or about 72.8 hours at a two-second interval.
When full, the logger overwrites the oldest flash sector. It validates each
record with CRC8 after restart. The fixed two-second interval and sequence
number are enough for ML windows; uptime is intentionally not stored.

## DHT22 wiring

| DHT22 signal | NodeMCU ESP32S V1.1 pin |
| --- | --- |
| VCC | `3V3` |
| DATA | `P16` / GPIO16 |
| GND | `GND` |

Add 10 kΩ from `DATA` to `3V3` only if the DHT22 board has no onboard pull-up.

## Build and upload

```sh
pio run --target upload
```

Changing partition layout erases existing flash data. Future USB uploads retain
ring data only while this same partition layout is kept.

The first boot of a new ring-record format erases the 1 MiB ring once. This
prevents old unrelated flash bytes from being interpreted as sensor samples.

## Serial protocol

Serial speed: `115200`. Commands end with newline.

```text
AT
OK

AT+STATUS
ACK+STATUS,records=12,capacity=131072,interval_ms=2000,dht_interval_ms=2000,record_bytes=8,complete=0
OK+STATUS

AT+DELETE
ACK+DELETE,confirm_within_ms=30000

AT+DELETE,CONFIRM
ACK+DELETE,erasing
OK+DELETE,records=0

AT+LOG,ON
ACK+LOG,enabled=1
OK+LOG
LOG,12,27.90,54.80

AT+LOG,OFF
ACK+LOG,enabled=0
OK+LOG

AT+RISK
ACK+RISK,ready=1,label=NORMAL,confidence=1.000,voted_high=0
OK+RISK

AT+RISK,ON
OK+RISK,log=1
RISK,17,NORMAL,1.000,voted_high=0

AT+RISK,OFF
OK+RISK,log=0
```

`AT+DELETE` only arms erasure. Send the exact confirm command within 30 seconds
to erase the complete 1 MiB ring. A timeout or a confirm without arming leaves
the data intact.

The collector requests `AT+EXPORT,<offset>,<count>` chunks and validates every
chunk before appending it. `DATA,...` appears only during export. Logging pauses
during export and resumes when it finishes.

Live serial output is off after boot. Use `AT+LOG,ON` to print each newly stored
two-second DHT sample as `LOG,sequence,temp_c,humidity_pct`; use `AT+LOG,OFF`
to stop it. `AT+LOG` reports the current setting.

`AT+RISK` reports the most recent inference. It becomes ready after ten fresh
DHT samples (about 20 seconds after boot). `AT+RISK,ON` emits one `RISK` line
per new sample; `AT+RISK,OFF` stops it. `voted_high=1` means at least two of the
latest three inference windows were `HIGH_THERMAL_RISK`; hardware alerts are not
enabled by this teaching firmware yet.

## Training artifacts

Run the following after importing a new capture:

```sh
.venv-ml311/bin/python tools/train_heatgun_neural_net.py
.venv-ml311/bin/python tools/export_dense_int8_header.py
pio run --target upload
```

The inspectable derived windows are in
`artifacts/heatgun_augmented_windows.csv`; `source=measured` means captured
data and `source=synthetic` means an explicitly generated teaching example.

## Export to CSV

```sh
/Users/vineethraik/.platformio/penv/bin/python tools/collect_export.py \
  --port /dev/cu.usbserial-0001 \
  --output data/dht_export.csv \
  --delete-after-export
```

The collector uses `pyserial`, included in PlatformIO's bundled Python.
It writes a temporary file and only replaces the target CSV when received row
count matches the device's `OK+EXPORT` count. `--delete-after-export` is
optional: after that verified CSV save, it sends the firmware's two-step delete
command. Without the flag, the ESP32 data remains intact.
