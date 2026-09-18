# ESP32 firmware learning stages

Each folder is a small, independently buildable PlatformIO project.

| Stage | Purpose | Output |
| --- | --- | --- |
| `01_dht_serial` | Read DHT22 and send clean serial data | `DATA,temp,humidity` |
| `02_flash_storage` | Append readings to local flash and export them | `AT+STATUS`, `AT+EXPORT`, dual-confirm delete |
| `03_tinyml_inference` | Keep ten readings and run the neural network | `RISK,label,confidence,voted_high,inference_us` |
| repository root | Final combined firmware | 1 MiB ring logger + export + TinyML inference |

Build one stage:

```sh
pio run --project-dir firmware/01_dht_serial
pio run --project-dir firmware/01_dht_serial --target upload
```

Only one firmware can run on the board at a time. Stage 02 uses a simple
LittleFS CSV partition for teaching storage. The final root firmware uses the
more robust compact 1 MiB raw ring. Changing between their partition layouts
can erase previously stored device data, so export wanted records first.
