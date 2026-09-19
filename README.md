# ESP32 TinyML Thermal-Risk Workshop

A complete beginner-friendly workshop that starts with a DHT22 sensor reading
and ends with a small neural network running offline on an ESP32.

Students build the system in small, understandable stages:

```text
DHT22 sensor
    → ESP32 serial output
    → Python CSV collector
    → data inspection and labels
    → 10-reading time-series windows
    → neural-network training
    → INT8-weight export
    → ESP32 inference
    → 2-of-3 high-risk vote
```

> **Safety warning:** This repository is a classroom thermal-pattern
> demonstration. It is not a certified fire detector or alarm and must never
> be the only protection for people or property.

## What students will learn

By completing the workshop, students should be able to:

- explain AI, machine learning, data, labels, training, and inference;
- wire a DHT22 to a NodeMCU ESP32S V1.1;
- read temperature and humidity in simple ESP32 firmware;
- capture serial readings into a raw CSV using Python;
- explain why raw data and human labels should be stored separately;
- convert readings into 20-second time-series windows;
- train a small `20 → 12 → 3` dense neural network;
- read a training report and confusion matrix;
- explain why training accuracy is not real-world safety accuracy;
- export model weights as INT8 values for ESP32;
- run offline inference and a 2-of-3 high-risk vote on the device.

## Workshop duration

The complete session takes approximately 5.5–6 hours.

| Time | Part | Result |
| --- | --- | --- |
| 00:00–00:30 | AI and ML basics | Understand data, labels, models, training, and inference |
| 00:30–01:15 | ESP32 and DHT22 | Read temperature and humidity |
| 01:15–01:50 | Python collector | Save live readings into CSV |
| 01:50–02:20 | Data and labels | Inspect data and create a labelled copy |
| 02:20–02:35 | Break | — |
| 02:35–03:45 | Neural-network training | Create windows and train the model |
| 03:45–04:20 | Evaluation and export | Inspect results and export INT8 weights |
| 04:20–05:10 | ESP32 inference | Flash the model and inspect predictions |
| 05:10–05:45 | Demo and limitations | Controlled warming, discussion, and recap |

## Required hardware

### Per student or team

| Item | Quantity | Notes |
| --- | ---: | --- |
| NodeMCU ESP32S V1.1 | 1 | Uses an ESP32-WROOM-32 module |
| DHT22 V182 | 1 | Temperature and humidity sensor/module |
| Micro-USB **data cable** | 1 | A charging-only cable will not work |
| Breadboard | 1 | Small breadboard is sufficient |
| Male-to-male jumper wires | 3–5 | For power, ground, and data |
| 10 kΩ resistor | 1 | Only required when the DHT22 has no onboard pull-up |
| Laptop or desktop computer | 1 | macOS, Linux, or Windows |
| USB-C adapter or hub | If needed | Required when the computer has no USB-A port |

### Shared instructor equipment

- one known-good spare ESP32;
- one known-good spare DHT22;
- two spare Micro-USB data cables;
- heat gun for the controlled demonstration;
- heat-resistant working surface;
- multimeter;
- power extension board;
- projector or shared display.

Only the instructor should operate the heat gun. Keep it away from hands,
wires, plastic, paper, and sealed containers.

### Optional later extensions

These parts are not required for the core workshop:

- MQ-2 smoke/gas sensor;
- voltage-divider resistors for the MQ-2 analogue output;
- LM393 infrared flame sensor;
- buzzer;
- LED and 220–330 Ω resistor;
- additional jumper wires.

An MQ-2 module requires approximately 48 hours of initial burn-in. Its analogue
output can reach 5 V and must not be connected directly to an ESP32 GPIO.

## Required software and services

### Software

- Git
- Python 3.11 or newer
- JupyterLab
- VS Code
- PlatformIO IDE extension or PlatformIO Core
- CP2102 USB driver, only if the board is not detected

Python packages are listed in `requirements.txt`:

```text
numpy
pandas
pyserial
jupyterlab
```

### Services

GitHub is used to download and update the repository. Internet access is also
needed initially for Python packages, PlatformIO, the ESP32 toolchain, and the
DHT library.

After setup, the workshop can run completely offline. It does not require:

- Edge Impulse;
- Google Colab;
- TensorFlow cloud services;
- AWS, Azure, or Google Cloud;
- a paid AI/ML account;
- ESP32 Wi-Fi;
- MQTT, Blynk, or Home Assistant.

## DHT22 wiring

| DHT22 V182 signal | NodeMCU ESP32S V1.1 |
| --- | --- |
| VCC | `3V3` |
| DATA | `P16` / GPIO16 |
| GND | `GND` |

If the DHT22 is a bare sensor without an onboard pull-up, connect a 4.7–10 kΩ
resistor between DATA and 3V3. Never pull the DATA pin to 5 V.

Verify the printed pin order on the exact DHT22 module before applying power.

## Repository structure

```text
.
├── firmware/
│   ├── 01_dht_serial/          # Sensor reading and clean serial output
│   ├── 02_flash_storage/       # Simple LittleFS CSV storage lesson
│   └── 03_tinyml_inference/    # Ten-reading model and voting lesson
├── src/main.cpp                # Final combined firmware
├── include/
│   └── thermal_risk_model.h    # Generated INT8 model weights
├── notebooks/
│   └── complete_tinyml_fire_risk_workshop.ipynb
├── data/
│   ├── supplied_training_data.csv
│   └── archive/                # Earlier captures kept for provenance
├── artifacts/
│   ├── heatgun_dense_nn.json
│   └── heatgun_dense_nn_report.json
├── tools/
│   ├── collect_serial_csv.py
│   ├── collect_export.py
│   ├── train_heatgun_neural_net.py
│   ├── export_dense_int8_header.py
│   └── build_complete_workshop_notebook.py
├── platformio.ini
├── partitions.csv
└── requirements.txt
```

The root PlatformIO project exposes all three teaching stages and the final
combined firmware as separate clickable environments. The projects under
`firmware/` remain independently buildable for instructors who want them.

## Quick start

### 1. Clone using SSH

```sh
git clone git@github.com:vineethraik/esp32-tinyml-fire-risk-workshop.git
cd esp32-tinyml-fire-risk-workshop
```

Repository:
[github.com/vineethraik/esp32-tinyml-fire-risk-workshop](https://github.com/vineethraik/esp32-tinyml-fire-risk-workshop)

### 2. Create the Python environment

macOS or Linux:

```sh
python3 -m venv .venv-ml311
source .venv-ml311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv-ml311
.venv-ml311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Open the workshop notebook

```sh
jupyter lab
```

Open:
[`notebooks/complete_tinyml_fire_risk_workshop.ipynb`](notebooks/complete_tinyml_fire_risk_workshop.ipynb)

Select the `.venv-ml311` Python environment as the notebook kernel.

### 4. Check the ESP32 serial port

```sh
pio device list
```

Typical ports:

- macOS: `/dev/cu.usbserial-0001`
- Linux: `/dev/ttyUSB0`
- Windows: `COM3`

The exact name may differ.

### 5. Use the PlatformIO buttons

Open the repository root—not an individual `firmware/` folder—in VS Code.
Select the PlatformIO icon, expand **PROJECT TASKS**, and choose one environment:

| PlatformIO environment | Firmware shown in class |
| --- | --- |
| `01_dht_serial` | Basic DHT22 serial readings |
| `02_flash_storage` | LittleFS storage and AT commands |
| `03_tinyml_inference` | Small neural-network inference lesson |
| `04_final_combined` | Final ring logger, export, inference, and voting |

Inside the selected environment, click:

1. **General → Build** to compile;
2. **General → Upload** to flash the connected ESP32;
3. **General → Monitor** to open serial at 115200 baud.

Only one firmware runs on the board at a time. The root project's default
environment is `04_final_combined`, so the generic bottom-bar Build and Upload
buttons target the final firmware. During lessons, use the named environment's
buttons to avoid flashing the wrong stage.

## Firmware stage 1: DHT22 serial readings

This is the smallest firmware. It reads DHT22 every two seconds and prints:

```text
DATA,26.80,57.00
```

Build and upload:

```sh
pio run -e 01_dht_serial
pio run -e 01_dht_serial --target upload
pio device monitor -e 01_dht_serial
```

Important code ideas:

- GPIO16 is the DHT22 data pin.
- DHT22 is read every two seconds.
- failed `NaN` readings are rejected;
- output follows one stable `DATA,temp,humidity` format.

## Collect live readings with Python

Close PlatformIO Serial Monitor first. Only one program can own the serial port
at a time.

macOS example:

```sh
python tools/collect_serial_csv.py \
  --port /dev/cu.usbserial-0001 \
  --output data/my_session.csv \
  --count 100
```

Windows example:

```powershell
python tools/collect_serial_csv.py `
  --port COM3 `
  --output data/my_session.csv `
  --count 100
```

Omit `--count` to continue until `Ctrl+C`.

Output format:

```csv
timestamp_utc,sequence,temp_c,humidity_pct
2026-09-18T05:44:44.884292+00:00,,26.90,56.90
2026-09-18T05:44:46.884447+00:00,,26.80,56.90
```

The collector:

- ignores boot messages and malformed lines;
- rejects invalid and non-finite values;
- accepts both beginner `DATA,...` and final-firmware `LOG,...` lines;
- adds a UTC timestamp;
- flushes every valid row;
- refuses to overwrite an existing file unless `--append` is used.

Do not add labels directly to the only raw copy. Duplicate the file and label
the working copy.

## Firmware stage 2: flash storage

This lesson appends DHT readings to a CSV file in LittleFS flash.

```sh
pio run -e 02_flash_storage
pio run -e 02_flash_storage --target upload
pio device monitor -e 02_flash_storage
```

Commands:

```text
AT
AT+STATUS
AT+EXPORT
AT+DELETE
AT+DELETE,CONFIRM
```

Deletion requires two commands within 30 seconds.

> **Partition warning:** Stage 2 uses a simple LittleFS partition layout for
> teaching. The final root firmware uses a different raw-ring layout. Export
> wanted device data before changing between these layouts.

## Data and labels

The official workshop dataset is:

[`data/supplied_training_data.csv`](data/supplied_training_data.csv)

It contains 31,347 captured DHT rows with:

```text
sequence,temp_c,humidity_pct
```

The notebook demonstrates a separate manual-labeling copy. The supplied model
then uses consistent teaching labels:

| Label | Teaching meaning |
| --- | --- |
| `NORMAL` | Ordinary temperature/humidity window |
| `ELEVATED_THERMAL_RISK` | Warming or approaching the controlled high boundary |
| `HIGH_THERMAL_RISK` | Controlled high-temperature window |

Because only one controlled heat-gun capture was available, training also adds
clearly marked synthetic examples:

- gradual rise;
- late rise;
- two-stage rise.

Synthetic examples are useful for teaching the pipeline. They are not a
replacement for varied real fire, smoke, kitchen, weather, and device data.

Earlier captures are preserved under [`data/archive/`](data/archive/README.md)
for provenance and are not used by the official workshop.

## Time-series input

The model receives ten DHT22 readings:

```text
[temp(t-9), humidity(t-9),
 temp(t-8), humidity(t-8),
 ...
 temp(t),   humidity(t)]
```

At a two-second interval:

```text
10 readings × 2 values = 20 features covering 20 seconds
```

This allows the model to react to a short pattern rather than only one value.

## Train the neural network

```sh
python tools/train_heatgun_neural_net.py
```

Model structure:

```text
20 inputs → 12 ReLU hidden units → 3 Softmax outputs
```

Generated files:

- `artifacts/heatgun_augmented_windows.csv` — measured and synthetic windows;
- `artifacts/heatgun_dense_nn.json` — scaler and learned model values;
- `artifacts/heatgun_dense_nn_report.json` — counts and training results.

The augmented window CSV is generated locally and intentionally not committed.

Current supplied-data counts:

| Item | Count |
| --- | ---: |
| Measured windows | 31,307 |
| Synthetic windows | 2,400 |
| Total windows | 33,707 |
| Normal | 30,573 |
| Elevated thermal risk | 1,773 |
| High thermal risk | 1,361 |

The observed training accuracy is approximately 99.77%. This is a training-set
result, not independent real-world accuracy. Overlapping windows and synthetic
data can make training performance look unusually strong.

## Export INT8 weights

```sh
python tools/export_dense_int8_header.py
```

Output:

```text
include/thermal_risk_model.h
```

The exporter scales dense-layer weights into signed one-byte integers. The
current teaching firmware stores INT8 weights but dequantizes them for its tiny
floating-point calculation. It is not a fully integer TensorFlow Lite Micro
model.

## Firmware stage 3: TinyML inference

This stage contains only DHT reading, a ten-reading window, neural inference,
and 2-of-3 voting.

```sh
pio run -e 03_tinyml_inference
pio run -e 03_tinyml_inference --target upload
pio device monitor -e 03_tinyml_inference
```

The first nine readings produce warm-up messages. From the tenth reading:

```text
RISK,NORMAL,1.000,voted_high=0,inference_us=...
```

`inference_us` reports model calculation time on the device. Sensor waiting time
is not included.

## Final combined firmware

The root firmware combines:

- DHT22 sampling every two seconds;
- compact 8-byte records;
- 1 MiB raw flash ring;
- CRC validation;
- restart recovery;
- chunked serial export;
- two-step deletion;
- ten-reading neural inference;
- 2-of-3 high-risk voting.

Build and flash:

```sh
pio run -e 04_final_combined
pio run -e 04_final_combined --target upload
pio device monitor -e 04_final_combined
```

The final partition layout is defined by `partitions.csv`:

- 2.94 MiB application partition;
- 1 MiB raw ring partition;
- no OTA partition.

Storage capacity:

```text
1,048,576 bytes / 8 bytes = 131,072 samples
131,072 samples × 2 seconds ≈ 72.8 hours
```

When full, the ring overwrites its oldest flash sector.

## Final firmware serial commands

Serial speed is `115200`. End every command with a newline.

| Command | Purpose |
| --- | --- |
| `AT` | Connection check |
| `AT+STATUS` | Records, capacity, interval, and logging status |
| `AT+LOG` | Show current live-log setting |
| `AT+LOG,ON` | Print every new stored DHT sample |
| `AT+LOG,OFF` | Stop live sample output |
| `AT+RISK` | Show latest model result |
| `AT+RISK,ON` | Print every new risk result |
| `AT+RISK,OFF` | Stop live risk output |
| `AT+EXPORT` | Export first bounded chunk |
| `AT+EXPORT,<offset>,<count>` | Export a selected chronological chunk |
| `AT+DELETE` | Arm ring deletion for 30 seconds |
| `AT+DELETE,CONFIRM` | Confirm armed deletion |

Status example:

```text
ACK+STATUS,records=12,capacity=131072,interval_ms=2000,dht_interval_ms=2000,record_bytes=8,complete=0,live_log=0
OK+STATUS
```

Risk example after ten fresh readings:

```text
ACK+RISK,ready=1,label=NORMAL,confidence=1.000,voted_high=0
OK+RISK
```

`voted_high=1` means at least two of the latest three windows were classified
as `HIGH_THERMAL_RISK`. Hardware buzzer/LED alarm behavior is not enabled.

## Export the final flash ring

Close Serial Monitor before running the exporter:

```sh
python tools/collect_export.py \
  --port /dev/cu.usbserial-0001 \
  --output data/ring_export.csv
```

The exporter requests verified chunks and uses a restartable partial file. It
does not delete device data by default.

To delete only after a verified complete export:

```sh
python tools/collect_export.py \
  --port /dev/cu.usbserial-0001 \
  --output data/ring_export.csv \
  --delete-after-export
```

## Verified project results

The following were tested on the project system and NodeMCU board:

| Check | Result |
| --- | --- |
| Python collector parser | Passed malformed, `NaN`, beginner, and final-log cases |
| Real Python collection | Five valid DHT readings saved and verified |
| Collector overwrite protection | Passed |
| Complete notebook | All 11 executable cells passed |
| Neural training time | Approximately 10–18 seconds on the tested Mac |
| Stage 1 firmware | Compiled and hardware-tested |
| Stage 2 firmware | Compiled |
| Stage 3 firmware | Compiled |
| Final combined firmware | Compiled, flashed, and serial-checked |
| Final firmware RAM | 25,820 bytes of 327,680 bytes |
| Final firmware flash | 286,661 bytes of 3,080,192 bytes |
| Flash ring | Preserved across normal uploads using the same partition layout |

Results on another computer, cable, board, sensor, or environment may differ.

## Live demonstration

1. Flash the final combined firmware.
2. Wait approximately 20 seconds for ten valid DHT readings.
3. Send `AT+RISK` and show the normal result.
4. Send `AT+RISK,ON`.
5. Instructor warms the sensor gradually from a safe distance.
6. Observe class, confidence, and `voted_high`.
7. Stop heating and allow the sensor to cool.
8. Discuss false alarms, sensor delay, placement, and missing smoke data.

If risk stays normal, first use `AT+LOG,ON` and confirm that DHT temperature is
actually changing.

## Troubleshooting

### ESP32 does not appear as a serial port

- Confirm the cable supports data.
- Try another USB port or known-good cable.
- Install the CP2102 driver.
- Run `pio device list` again.

### Serial port is busy

Close PlatformIO Serial Monitor, Arduino Serial Monitor, Python collectors, and
other terminal applications. Only one process can open the port.

macOS/Linux diagnostic:

```sh
lsof /dev/cu.usbserial-0001
```

### DHT returns `NaN` or `ERROR,DHT_READ`

- Verify 3V3, GND, and GPIO16.
- Verify the module's printed pin order.
- Add a 4.7–10 kΩ DATA-to-3V3 pull-up if none is onboard.
- Keep the two-second read interval.
- Replace loose jumper wires.

### Model reports `ready=0`

Wait for ten valid readings after every reset—approximately 20 seconds.

### Model stays normal during warming

- Check actual readings using `AT+LOG,ON`.
- Remember the DHT22 responds slowly.
- Warm gradually and safely.
- Test conditions may differ from the supplied training capture.

### Python cannot import `serial`

Activate the workshop virtual environment and run:

```sh
pip install -r requirements.txt
```

### PlatformIO cannot flash

- Close every serial monitor.
- Confirm the correct port.
- Use a known-good data cable.
- Disconnect unnecessary external circuits during recovery.
- If required, hold BOOT, tap EN/RESET, start upload, then release BOOT when
  connection begins.

## Safety and limitations

- This is not a certified fire alarm.
- DHT22 does not detect smoke, CO, LPG, or flame.
- A heat rise can come from many non-fire sources.
- One controlled capture is not representative of all buildings or kitchens.
- Synthetic data is clearly marked but is not real field evidence.
- High model confidence does not mean certainty.
- Training accuracy does not establish a false-alarm rate.
- Do not use open flame in a crowded classroom.
- Only an instructor should operate the heat gun.
- Never enclose an active heat gun in a sealed container.
- A real product needs independent sessions, held-out validation, sensor fault
  handling, regulatory review, and hard safety rules alongside ML.

## Instructor preparation checklist

- [ ] Test one complete laptop, cable, ESP32, and DHT22 path.
- [ ] Install dependencies before the workshop.
- [ ] Run the complete notebook once.
- [ ] Pre-flash one known-good demonstration board.
- [ ] Keep spare USB data cables and one spare sensor.
- [ ] Confirm the supplied dataset is present.
- [ ] Prepare a PC-only fallback if student hardware fails.
- [ ] Operate the heat gun only on a heat-resistant surface.
- [ ] End the class with limitations and validation discussion.

## Teaching documents

- [Complete self-learning notebook](notebooks/complete_tinyml_fire_risk_workshop.ipynb)
- [One-day workshop plan](docs/WORKSHOP_PLAN.md)
- [Facilitator guide](docs/FACILITATOR_GUIDE.md)
- [Student guide](docs/STUDENT_GUIDE.md)
- [Simple ML concepts](docs/ML_CONCEPTS.md)
- [Setup and materials](docs/SETUP.md)
- [Live demo runbook](docs/DEMO_RUNBOOK.md)
- [Firmware stages](firmware/README.md)
- [Dataset guide](data/README.md)

## Rebuilding generated material

Rebuild the notebook JSON after editing its source builder:

```sh
python tools/build_complete_workshop_notebook.py
```

Rebuild the model and ESP32 header:

```sh
python tools/train_heatgun_neural_net.py
python tools/export_dense_int8_header.py
```

## License

Code and teaching material are released under the [MIT License](LICENSE).
