# ESP32 firmware learning stages

Each folder is a small, independently buildable PlatformIO project. For
students, the repository-root `platformio.ini` also exposes every stage as a
separate environment in VS Code's PlatformIO **PROJECT TASKS** sidebar.

| Stage | Purpose | Output |
| --- | --- | --- |
| `01_dht_serial` | Read DHT22 and send clean serial data | `DATA,temp,humidity` |
| `02_flash_storage` | Append readings to local flash and export them | `AT+STATUS`, `AT+EXPORT`, dual-confirm delete |
| `03_tinyml_inference` | Keep ten readings and run the neural network | `RISK,label,confidence,voted_high,inference_us` |
| repository root | Final combined firmware | 1 MiB ring logger + export + TinyML inference |

Both inference stages use `include/thermal_risk_inference.h`. Start with the
short `03_tinyml_inference/src/main.cpp`; inspect the local library only when
you want to see normalization, dense layers, Softmax, and voting.

Build one stage from the repository root:

```sh
pio run -e 01_dht_serial
pio run -e 01_dht_serial --target upload
```

Equivalent sidebar flow: expand `01_dht_serial`, then click **General →
Build**, **Upload**, or **Monitor**. The other environment names are
`02_flash_storage`, `03_tinyml_inference`, and `04_final_combined`.

Only one firmware can run on the board at a time. Stage 02 uses a simple
LittleFS CSV partition for teaching storage. The final root firmware uses the
more robust compact 1 MiB raw ring. Changing between their partition layouts
can erase previously stored device data, so export wanted records first.
