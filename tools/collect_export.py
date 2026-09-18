#!/usr/bin/env python3
"""Import an ESP32 ring buffer as verified, restartable serial chunks."""

import argparse
import csv
import pathlib
import sys
import time

import serial


CSV_HEADER = ["sequence", "temp_c", "humidity_pct"]


def reconnect(device):
    """Reopen the USB serial port after an ESP32 export stops responding."""
    device.close()
    time.sleep(2.0)
    device.open()
    device.dtr = False
    device.rts = False
    # NodeMCU boards can reboot when their CP2102 port is reopened.
    time.sleep(3.0)
    device.reset_input_buffer()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Example: /dev/cu.usbserial-0001")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true",
                        help="Continue a verified .partial file after a failed import")
    parser.add_argument("--delete-after-export", action="store_true",
                        help="Erase the ESP32 ring only after the CSV is complete")
    return parser.parse_args()


def delete_device(args):
    """Use the firmware's two-step deletion after a CSV is safely written."""
    with serial.Serial(args.port, args.baud, timeout=0.5) as device:
        device.dtr = False
        device.rts = False
        # A new CP2102 connection can reset the NodeMCU.
        time.sleep(3.0)
        device.reset_input_buffer()
        device.write(b"AT+DELETE\n")
        device.flush()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            line = device.readline().decode("utf-8", "replace").strip()
            if line.startswith("ACK+DELETE,confirm_within_ms="):
                break
            if line.startswith("ERR+"):
                raise RuntimeError(line)
        else:
            raise TimeoutError("Timed out arming device deletion")

        device.write(b"AT+DELETE,CONFIRM\n")
        device.flush()
        while time.monotonic() < deadline:
            line = device.readline().decode("utf-8", "replace").strip()
            if line == "OK+DELETE,records=0":
                print("Deleted ESP32 ring after verified export")
                return
            if line.startswith("ERR+"):
                raise RuntimeError(line)
        raise TimeoutError("Timed out confirming device deletion")


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = args.output.with_name(f"{args.output.name}.partial")
    if not args.resume:
        partial_output.unlink(missing_ok=True)

    with serial.Serial(args.port, args.baud, timeout=0.5) as device:
        device.dtr = False
        device.rts = False
        # Opening CP2102 serial can reset a NodeMCU. Let firmware reach its
        # command loop before transmitting AT+EXPORT.
        time.sleep(1.0)
        device.reset_input_buffer()
        device.write(b"AT+STATUS\n")
        device.flush()

        deadline = time.monotonic() + args.timeout
        expected_records = None
        while time.monotonic() < deadline:
            line = device.readline().decode("utf-8", "replace").strip()
            if line.startswith("ACK+STATUS,"):
                fields = dict(item.split("=", 1) for item in line.split(",")[1:])
                expected_records = int(fields["records"])
            elif line == "OK+STATUS":
                break
            elif line.startswith("ERR+"):
                raise RuntimeError(line)

        if expected_records is None:
            raise TimeoutError("Timed out waiting for AT+STATUS")

        records = 0
        if args.resume and partial_output.exists():
            with partial_output.open(newline="") as existing_file:
                records = sum(1 for _ in csv.DictReader(existing_file))
            if records > expected_records:
                raise RuntimeError("Partial file has more rows than the device")

        print(f"Importing {expected_records} records from {records} in {args.chunk_size}-record chunks")

        file_mode = "a" if records else "w"
        with partial_output.open(file_mode, newline="") as output_file:
            writer = csv.writer(output_file)
            if not records:
                writer.writerow(CSV_HEADER)

            while records < expected_records:
                wanted = min(args.chunk_size, expected_records - records)
                complete_chunk = None
                for attempt in range(3):
                    device.reset_input_buffer()
                    device.write(f"AT+EXPORT,{records},{wanted}\n".encode())
                    device.flush()
                    received_rows = []
                    received_ack = False
                    chunk_deadline = min(deadline, time.monotonic() + 30.0)
                    while time.monotonic() < chunk_deadline:
                        line = device.readline().decode("utf-8", "replace").strip()
                        if not line:
                            continue
                        if line.startswith("ACK+EXPORT,"):
                            received_ack = True
                        elif line.startswith("DATA,"):
                            received_rows.append(line.split(",")[1:])
                        elif line.startswith("OK+EXPORT,"):
                            fields = dict(item.split("=", 1) for item in line.split(",")[1:])
                            sent = int(fields["records"])
                            if received_ack and sent == wanted and len(received_rows) == wanted:
                                complete_chunk = received_rows
                            break
                        elif line.startswith("ERR+"):
                            raise RuntimeError(line)
                    if complete_chunk is not None:
                        break
                    print(f"Retrying chunk at record {records} ({attempt + 1}/3)")
                    reconnect(device)

                if complete_chunk is None:
                    raise RuntimeError(f"Could not import complete chunk at record {records}")
                writer.writerows(complete_chunk)
                records += len(complete_chunk)

    if records != expected_records:
        raise RuntimeError(f"Incomplete import: expected {expected_records}, got {records}")

    partial_output.replace(args.output)
    print(f"Saved {records} records to {args.output}")

    if args.delete_after_export:
        delete_device(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, serial.SerialException, RuntimeError, TimeoutError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
