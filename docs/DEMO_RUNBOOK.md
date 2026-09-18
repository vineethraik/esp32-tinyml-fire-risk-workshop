# Live demo runbook

Use this as a one-page instructor sequence. Keep students clear of the heat
source. Do not use flame or smoke for this lesson.

## Before the audience

1. DHT22 wired to 3V3, P16/GPIO16, GND.
2. Firmware already flashed, or run `pio run --target upload`.
3. Wait 20 seconds after boot so ten samples fill the model window.
4. Open a serial terminal at 115200.

## Demo script

| Say | Do | Expected result |
| --- | --- | --- |
| “First, it only sees numbers.” | Send `AT+RISK`. | `ready=1`, normally `NORMAL`. |
| “We can inspect every guess.” | Send `AT+RISK,ON`. | One `RISK` line every two seconds. |
| “Now we change the recent pattern, not just one number.” | Warm the DHT22 gently at a distance with heat gun. | Output may move to `ELEVATED_THERMAL_RISK`, then high. |
| “We avoid reacting to a one-window glitch.” | Point at `voted_high`. | It becomes `1` only after two high labels in three windows. |
| “This is a demo, not a safety alarm.” | Stop heat, allow cooling. | Discuss why more data/sensors/testing are needed. |

## If output stays normal

- Wait for ten fresh readings after any board reset.
- Bring heat closer slowly; DHT22 response is not instant.
- Check `AT+LOG,ON` to confirm temperature actually rises.
- Do not claim a failure proves the model is bad: first check wiring, sensor
  response, input window, and whether test conditions resemble training.

## If output is high during normal room conditions

- Stop the live demo and state it is a false alarm.
- Check actual temperature/humidity with `AT+LOG,ON`.
- Explain that a model needs more varied normal data.
- Do not “fix” it by silently changing labels during class.

## Optional closing question

“Where should a human still make the final decision?”

Good answers: evacuation, emergency call, turning off a heater, and deciding
whether the dataset is representative.
