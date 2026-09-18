# Facilitator guide

## Opening words (2 minutes)

“Today we are not teaching a computer what fire is. We are teaching a small
computer to compare a recent sensor pattern with examples. We will see both the
power and the limits of that idea.”

## Plain-language teaching flow

### 1. AI is the big umbrella

Say: “AI is a broad word for computers doing tasks that appear to need a little
judgement. Machine learning is one method inside AI.” Avoid debating whether a
machine is intelligent; return to the task it performs.

### 2. Contrast rule writing and learning from examples

Ask a student for a rule to flag heat. Write it down. Then ask what happens
when the room is hot, when sensor placement changes, or when temperature is
rising but still below the threshold. This opens the reason for a short
time-series window.

### 3. Use the data as evidence

Open the CSV before running training. Have learners identify what each column
means and notice that every row is only a measurement. Labels are a human
decision layered on top of measurements.

### 4. Train, then separate training from inference

Say exactly:

> Training is the slower learning step on our laptop. Inference is the small
> prediction step repeated on ESP32.

Run the script. Point out that the model's reported score is a training score,
not a real safety guarantee.

### 5. Make tiny model visible

Draw this on board:

```text
10 readings -> 20 numbers -> 12 hidden pattern units -> 3 labels
```

Do not teach backpropagation, derivatives, or matrix multiplication unless the
group asks. The useful takeaway is that training chooses many internal numbers
so similar input patterns get similar labels.

### 6. Run the board

Follow [demo runbook](DEMO_RUNBOOK.md). Show normal first. Speak while warming
slowly. If the live result is messy, use it as the most valuable lesson: sensor
data and test conditions matter.

## Questions likely to come up

**“Is this better than a threshold?”**  Not automatically. Thresholds are
simple and reliable when well chosen. ML is useful when several readings and
sensors form a pattern.

**“Does it predict a fire?”**  No. This dataset supports only a thermal-risk
demonstration. Prediction needs relevant, varied real data and careful safety
validation.

**“Why not use ChatGPT or internet AI?”**  This model works offline on ESP32.
It is deliberately small, fast, inspectable, and limited to its sensor inputs.

**“Why does model confidence show 1.000?”**  Small training data can make a
model overconfident. Confidence is not certainty.

**“Why use synthetic data?”**  It lets us teach more shapes when collection
time is limited. It must be labelled clearly and never treated as replacement
for real field data.

## Avoid these claims

- “It detects all fires.”
- “High confidence means it is correct.”
- “Synthetic data is real data.”
- “AI replaces safety engineering.”
- “A model gets better by simply adding more neurons.”

## Handoff checklist

- Read `README.md`, `SETUP.md`, and `DEMO_RUNBOOK.md`.
- Test the exact laptop, cable, board, and DHT22.
- Pre-flash one known-good board.
- Have a PC-only fallback using the supplied dataset and model artifacts.
- End with limitations, not only the successful demo.
