# One-day workshop plan: TinyML thermal-risk demo

## What learners finish with

Each team can explain, in simple words, how a machine-learning model learns
from examples; wire a DHT22 to an ESP32; inspect sensor data; train a tiny
neural network on a PC; flash the exported model; and watch a live risk label.

This is a teaching demonstration of thermal-risk patterns. It is not a real
fire-safety product.

## Suggested 5 hours 45 minutes

| Time | Block | Learner outcome |
| --- | --- | --- |
| 00:00–00:25 | Welcome and problem | Understand why simple threshold logic can miss a rising trend. |
| 00:25–01:00 | AI and ML basics | Explain data, label, model, training, and inference. |
| 01:00–01:45 | ESP32 and DHT22 | Wire sensor, flash starter firmware, see temperature/humidity. |
| 01:45–02:00 | Break | — |
| 02:00–02:45 | Data story | Find normal/warming sections; turn up to 100 readings into 90 windows and label some. |
| 02:45–03:35 | Train on PC | Run the neural-network script; optionally add student labels; inspect the report. |
| 03:35–04:05 | TinyML export | Explain INT8 weights; generate the ESP32 model header. |
| 04:05–04:50 | ESP32 inference demo | Flash, use `AT+RISK`, then show controlled warming. |
| 04:50–05:25 | Reflection and limits | Discuss false alarms, missing sensors, validation, ethics, safety. |
| 05:25–05:45 | Optional extensions | MQ-2, flame confirmation, alerts, better datasets. |

For a six-hour session, use the extra 15 minutes for teams to present one
finding from the CSV.

## Core story to repeat

1. A normal program follows rules we write.
2. A machine-learning program learns a rule-like pattern from many examples.
3. We still choose the sensor, labels, safety limits, and what the output does.
4. Training happens on a computer. Inference is the small prediction step on
   ESP32.
5. A model is only as trustworthy as its data and testing.

## Instructor decisions before the day

- Use the supplied dataset for every team. Do not promise that it predicts a
  real fire; it demonstrates the workflow.
- Keep the heat gun outside a sealed container. Never create smoke or flame in
  a crowded room.
- Let one instructor board be the live demo. Teams can use PC-only training if
  USB drivers or hardware time are limited.
- Treat `HIGH_THERMAL_RISK` as a screen label only. The current workshop
  firmware does not activate buzzer or LED.

## Evidence of success

- Student can say the difference between training and inference.
- Student can point to a 10-reading input window and explain why 100 readings
  become 90 label rows in this exercise.
- Student runs training and reads the three output labels.
- ESP32 answers `AT+RISK` after ten readings.
- Student names at least two reasons the demo is not production-ready.
