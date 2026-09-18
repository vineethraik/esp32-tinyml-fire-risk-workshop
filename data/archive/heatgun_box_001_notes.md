# Heat-gun box run 001

Source CSV: `data/archive/heatgun_box_001.csv`

## Setup reported during collection

- Sensor was inside a mostly closed, non-airtight metal box with holes.
- Hot-air gun was aimed through one hole.
- First gun setting: "100 degree", used until the DHT reading reached 50.70 C.
- Second gun setting: "480", then stopped when the DHT reading reached 61.00 C.

The setting units are recorded exactly as reported and have not been independently
verified. Treat this as a controlled thermal-risk teaching run, not a real fire
or flame-confirmed sample.

## Captured data facts

- Sequences in this export: 0 to 3538.
- Maximum captured DHT temperature: 61.80 C at sequence 2394.
- First captured reading at or above 50.70 C: sequence 2298.
- First captured reading at or above 60.00 C: sequence 2378.

This export includes earlier retained baseline data before the heat-gun run.
Do not label sequence 0 as the heat-gun start. Future runs should clear/export
the ring before a controlled experiment, or record explicit start/phase markers.
