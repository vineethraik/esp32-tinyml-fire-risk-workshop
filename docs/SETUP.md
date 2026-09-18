# Materials and setup

## Per team

- NodeMCU ESP32S V1.1 / ESP32-WROOM-32 board
- DHT22 V182 module
- Breadboard and jumper wires
- USB data cable
- PC with Python 3.11+ and PlatformIO

One shared, instructor-operated heat gun is optional for the final demo.
Keep it away from plastic, paper, hands, and any sealed box.

## DHT22 wiring

| DHT22 | ESP32 NodeMCU pin |
| --- | --- |
| VCC | `3V3` |
| DATA | `P16` / GPIO16 |
| GND | `GND` |

If using a bare DHT22 rather than a module with an onboard resistor, add a
4.7–10 kOhm pull-up from DATA to 3V3. Never connect the data pin to 5 V.

## PC setup

```sh
git clone <repository-url>
cd esp32-tinyml-fire-risk-workshop
python3 -m venv .venv-ml311
.venv-ml311/bin/pip install -r requirements.txt
pio run --target upload
```

If PlatformIO is not installed, use its official IDE extension or have the
instructor pre-flash boards. Students can still complete the data and model
activities without hardware.

## Before students arrive

- Test one board, cable, DHT22, and serial port from start to finish.
- Pre-install Python packages on lab PCs or provide a prepared virtual
  environment instructions sheet.
- Keep `data/supplied_training_data.csv` unchanged as the common dataset.
- Run the three training/export commands once and pre-flash at least one demo
  board.
- Print the student guide or send its link before the class.

## Later hardware, not required today

MQ-2 needs a full 48-hour burn-in and its analogue 5 V output needs a safe
voltage divider before reaching ESP32 GPIO34. A flame module can provide direct
flame confirmation. Do not pretend either is ready until individually tested.
