# AI and ML: simple teaching language

## Start with this, not jargon

**Artificial intelligence** is a broad name for computers doing tasks that
seem to need judgement: recognising a picture, translating language, or
deciding whether a sensor pattern looks unusual.

**Machine learning** is one way to build such a system. Instead of writing
every rule ourselves, we show the computer examples and the answer we wanted.
It adjusts its internal numbers to make better guesses on similar examples.

Do not say the machine “understands fire.” Say: *it finds patterns similar to
the examples we gave it.*

## A familiar comparison

| Conventional program | Machine-learning program |
| --- | --- |
| We write the rule: `if temperature > 50 then high risk`. | We give example sensor histories marked normal/elevated/high. |
| Good when the rule is known and simple. | Useful when the pattern is a combination or trend. |
| Rule stays exactly as written. | Learned numbers depend on the training examples. |

Both are useful. A real safety device would usually use both: hard safety rules
plus a model.

## Five words learners need

- **Data:** recorded examples. Here: temperature and humidity every two seconds.
- **Label:** the answer attached to an example. Here: normal, elevated thermal
  risk, or high thermal risk.
- **Model:** the small set of learned numbers that turns inputs into a guess.
- **Training:** adjusting those numbers using labelled examples, on a PC.
- **Inference:** using finished numbers to make one prediction, on ESP32.

## What our model sees

The ESP32 keeps the newest ten readings:

```text
temperature, humidity  x 10 readings = 20 input numbers
```

At a two-second interval, this represents the last 20 seconds. The model does
not see “fire”; it sees a short temperature/humidity history.

## Neural network without the heavy math

Think of a neural network as small groups of adjustable dials.

```text
20 sensor numbers -> 12 pattern dials -> 3 possible labels
```

During training, wrong answers nudge the dials. After many examples, some dials
become sensitive to patterns such as “temperature rises late” or “temperature
rises steadily.” This is only an analogy; the dials are numbers, not real
brain cells.

## Why INT8 is mentioned

Large decimal numbers use more memory. INT8 stores a number in one byte, from
-127 to 127. We scale model weights into that small range so the ESP32 stores a
much smaller model. The current firmware uses INT8 stored weights and a simple,
inspectable calculation. Full integer TFLite is a later improvement.

## Honest limitations to teach

- The supplied capture is one controlled heat-gun experiment.
- Some extra rise examples are synthetic, clearly marked in the derived file.
- DHT22 is slow and cannot detect smoke or flame by itself.
- A high training score proves the model remembers its training examples; it
  does not prove real-world safety.
- More varied real data matters more than making this small network bigger.
