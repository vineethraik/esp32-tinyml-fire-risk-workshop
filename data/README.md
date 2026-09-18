# Workshop data

## Official supplied dataset

`supplied_training_data.csv` is the only dataset used by the complete workshop
notebook and neural-network training script. It contains 31,347 captured DHT22
rows with sequence, temperature, and humidity columns.

Keep this raw capture unchanged. Training creates labelled windows and clearly
marked synthetic rise examples under `artifacts/`.

## Student captures

The Python serial collector can create a new raw session:

```sh
python tools/collect_serial_csv.py \
  --port /dev/cu.usbserial-0001 \
  --output data/my_session.csv \
  --count 100
```

Use `COM3` or the correct port on Windows. Make a copy before adding labels.

## Archive

`archive/` preserves earlier captures for provenance. They are not required by
the workshop and should not be mixed into training without first understanding
their different formats and collection conditions.
