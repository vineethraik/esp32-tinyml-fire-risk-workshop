#!/usr/bin/env python3
"""Collect simple ESP32 DHT serial lines into an immutable raw CSV.

The script deliberately does one job: receive valid temperature/humidity
records and preserve them as raw evidence. Labels belong in a separate copy.
"""

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import serial


def parse_args():
    """Define the command-line options shown by --help."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Example: /dev/cu.usbserial-0001 or COM3")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--count", type=int, default=0,
                        help="Stop after this many valid rows; 0 means run until Ctrl+C")
    parser.add_argument("--timeout", type=float, default=0,
                        help="Stop after this many seconds; 0 means no time limit")
    parser.add_argument("--append", action="store_true",
                        help="Append to an existing CSV instead of refusing to overwrite it")
    return parser.parse_args()


def parse_sensor_line(line):
    """Accept the beginner DATA format and the final firmware LOG format."""
    fields = line.strip().split(",")

    # Stage 1 has no sequence number; final firmware includes one so exported
    # and live records can be placed in their original order.
    if len(fields) == 3 and fields[0] == "DATA":
        sequence, temperature, humidity = "", fields[1], fields[2]
    elif len(fields) == 4 and fields[0] == "LOG":
        sequence, temperature, humidity = fields[1], fields[2], fields[3]
    else:
        return None

    try:
        # Converting here filters boot text and malformed serial messages before
        # they can enter the raw dataset.
        temperature_value = float(temperature)
        humidity_value = float(humidity)
        if not math.isfinite(temperature_value) or not math.isfinite(humidity_value):
            return None
        if sequence:
            int(sequence)
    except ValueError:
        return None
    return sequence, temperature_value, humidity_value


def main():
    args = parse_args()

    # Raw captures should be immutable. Refuse accidental replacement unless
    # the student explicitly chooses append mode.
    if args.output.exists() and not args.append:
        raise SystemExit(f"Refusing to overwrite {args.output}; choose another name or use --append")

    # Creating the parent makes paths such as data/imports/team1.csv work on a
    # new clone without manual folder setup.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.output.exists() or args.output.stat().st_size == 0
    mode = "a" if args.append else "x"
    collected = 0
    started = time.monotonic()

    # Both the serial port and CSV close automatically when this block exits,
    # including after Ctrl+C or an exception.
    with serial.Serial(args.port, args.baud, timeout=0.5) as device, \
            args.output.open(mode, newline="") as output_file:
        device.dtr = False
        device.rts = False
        # CP2102 boards can reset when serial opens. Let setup() finish.
        time.sleep(3.0)
        device.reset_input_buffer()

        writer = csv.writer(output_file)
        if new_file:
            # Write the header only once when creating a new capture.
            writer.writerow(["timestamp_utc", "sequence", "temp_c", "humidity_pct"])
            output_file.flush()

        print(f"Collecting from {args.port} into {args.output}. Press Ctrl+C to stop.")
        try:
            while True:
                if args.timeout and time.monotonic() - started >= args.timeout:
                    break
                # Invalid UTF-8 bytes are replaced instead of terminating a
                # long capture because serial noise should not crash the tool.
                line = device.readline().decode("utf-8", "replace").strip()
                parsed = parse_sensor_line(line)
                if parsed is None:
                    continue
                sequence, temperature, humidity = parsed
                # The ESP32 stage-1 format has no clock, so the PC attaches a
                # timezone-aware UTC timestamp when it accepts each row.
                timestamp = datetime.now(timezone.utc).isoformat()
                writer.writerow([timestamp, sequence, f"{temperature:.2f}", f"{humidity:.2f}"])
                # Flush every row so an unplugged cable or interrupted class
                # loses at most the in-progress serial line.
                output_file.flush()
                collected += 1
                print(f"{collected}: {temperature:.2f} C, {humidity:.2f} %")
                if args.count and collected >= args.count:
                    break
        except KeyboardInterrupt:
            print("\nStopped by user")

    print(f"Saved {collected} valid rows to {args.output}")


if __name__ == "__main__":
    main()
