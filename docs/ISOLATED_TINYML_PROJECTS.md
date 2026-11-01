# Isolated TinyML student projects

Standalone **build-and-flash** labs. **No DHT22. No fire-risk firmware.**
Colab-only notebooks, threshold rules, and “would ML help?” essays do **not**
count.

Each project must ship this loop on a real board:

1. Capture labeled windows from the sensor.
2. Train a small net on a PC (NumPy, TFLite, or Edge Impulse).
3. Export **INT8 / TFLite Micro** (or an INT8 C header like the fire workshop).
4. Flash ESP32 / ESP32-CAM.
5. **Live demo:** serial class + LED (or buzzer) while a classmate tries to fool it.

Shared rules:

- 2 or 3 classes. Include `OTHER` unless the world is only your keyword.
- Time split (no shuffle). Then one harder split: new speaker, new room, or new motor.
- Target ≲ 1k weights unless the camera forces more. Write the **parameter count**.
- Grade **false activates / hour** on a live device, not training accuracy.

Shared references:

- [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro)
- [tflite-micro Arduino examples](https://github.com/tensorflow/tflite-micro-arduino-examples)
- [MLPerf Tiny](https://github.com/mlcommons/tiny)
- [HarvardX TinyML courseware](https://github.com/tinyMLx/courseware)
- [ARM ML-KWS for MCU (Hello Edge)](https://github.com/ARM-software/ML-KWS-for-MCU)

Optional trainers: [Edge Impulse keyword spotting](https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/keyword-spotting), [Edge Impulse motion](https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/motion-recognition).

---

## Audio (ESP32 + INMP441 I2S mic)

Hardware: ESP32-WROOM, INMP441 (3V3, GND, WS, SCK, SD), LED on a GPIO.
16 kHz 16-bit. **Do not** feed raw PCM into the net.

Bring-up: [ESP32 I2S](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/i2s.html), [atomic14 ESP32 I2S mic](https://github.com/atomic14/esp32-i2s-mic-test), [micro_speech](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech).

### 1. Wake word (activation)

- **Live demo:** say `go` / `hey` / `esp` → LED on. Other speech stays off.
- **Classes:** `WAKE`, `OTHER`.
- **Features:** 300–800 ms; 8–13 log-mel or MFCC frames.
- **Net:** 40 → 16 → 2 Softmax, or official micro_speech, INT8 on device.
- **Vote:** 2-of-3 windows before LED.
- **Data:** ≥ 5:1 `OTHER`:`WAKE`. New speaker is the test set.
- **Links:** [micro_speech](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech), [Arduino micro_speech](https://github.com/tensorflow/tflite-micro-arduino-examples/tree/main/examples/micro_speech), [ESP-Skainet](https://github.com/espressif/esp-skainet) (heavier).

### 2. Clap detector

- **Live demo:** clap → LED. Talk / keyboard must not light it. Model on flash, not `if (rms > x)`.
- **Classes:** `CLAP`, `OTHER`.
- **Features:** 20–40 ms; RMS, peak, ZCR, 3–4 band energies (~8–16 numbers).
- **Net:** 8 → 8 → 2 or 16 → 8 → 2, INT8.
- **Data:** 0.5 m and 2 m claps; `OTHER` = speech, keys, door.
- **Links:** [micro_speech frontend](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech), [tflite-micro Arduino](https://github.com/tensorflow/tflite-micro-arduino-examples).

### 3. Knock vs clap vs silence

- **Live demo:** knock desk, clap air, stay quiet — serial prints the class each time.
- **Classes:** `SILENCE`, `CLAP`, `KNOCK`.
- **Features:** ~250 ms; 8 mel bands × 8 frames = 64 inputs.
- **Net:** 64 → 16 → 3 Softmax, INT8.
- **Data:** AC-on silence; record a second day as hold-out.
- **Links:** [Hello Edge KWS](https://github.com/ARM-software/ML-KWS-for-MCU), [MLPerf Tiny](https://github.com/mlcommons/tiny).

### 4. Yes / no / other

- **Live demo:** `yes` / `no` from two people. Third person is hold-out. LED color or serial label.
- **Classes:** `YES`, `NO`, `OTHER`.
- **Features:** ~1 s log-mel (PCA down to ~40 if needed).
- **Net:** 40 → 16 → 3, INT8.
- **Data:** several speakers; `OTHER` includes `go`, cough, local-language fillers.
- **Links:** [micro_speech yes/no](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech), [ARM ML-KWS](https://github.com/ARM-software/ML-KWS-for-MCU).

### 5. Spoken digits 0 / 1 / 2

- **Live demo:** say zero, one, two at the board; serial shows the digit. Wrong words → `OTHER` if they add it, or a known miss.
- **Classes:** `ZERO`, `ONE`, `TWO` (add `OTHER` if time).
- **Features:** 1 s, 10-frame log-mel.
- **Net:** 40 → 16 → 3, INT8. Same family as Google micro_speech digits.
- **Data:** ≥ 30 clips per class per speaker; new speaker hold-out.
- **Links:** [Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands), [micro_speech](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech), [Hello Edge](https://github.com/ARM-software/ML-KWS-for-MCU).

### 6. One / two / three claps

- **Live demo:** clap once, twice, or thrice in 1 s; LED blinks that many times from the **model class**, not a peak counter.
- **Classes:** `CLAP1`, `CLAP2`, `CLAP3`.
- **Features:** 1 s energy envelope (10 frames) + gaps (~12–20 numbers).
- **Net:** 12 → 8 → 3, INT8. Dense net, not LSTM, unless they show LSTM is required **and still flash it**.
- **Data:** slow and fast clap trains.
- **Links:** [atomic14 ESP32 I2S](https://github.com/atomic14/esp32-i2s-mic-test), [tflite-micro examples](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples).

---

## Motion (ESP32 + MPU-6050)

Hardware: ESP32, MPU-6050 (SDA/SCL 3V3), LED.
Bring-up: [i2cdevlib MPU6050](https://github.com/jrowberg/i2cdevlib), [magic_wand](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/magic_wand), [Arduino magic_wand](https://github.com/tensorflow/tflite-micro-arduino-examples/tree/main/examples/magic_wand), [Edge Impulse motion](https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/motion-recognition).

### 7. Gesture wake

- **Live demo:** punch or wave in air → LED. Walking must stay off.
- **Classes:** `WAVE` (or `PUNCH`), `REST`, `OTHER`.
- **Features:** 20 samples × 3 axes = 60, or 18 summary stats.
- **Net:** 60 → 16 → 3 INT8.
- **Data:** walk and sit in `OTHER`.
- **Links:** [magic_wand](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/magic_wand), [Arduino TFLite tutorials](https://github.com/arduino/ArduinoTensorFlowLiteTutorials).

### 8. Wand letters W / O

- **Live demo:** hold the IMU like a wand, draw W or O in the air; serial prints the letter. Rest = off.
- **Classes:** `W`, `O`, `REST`.
- **Features:** ~1 s gyro/accel as in Magic Wand (or 128-sample downsample).
- **Net:** student-trained tiny conv/dense **or** official Magic Wand INT8, flashed.
- **Data:** 30+ traces per letter, two people.
- **Links:** [magic_wand](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/magic_wand), [Arduino magic_wand](https://github.com/tensorflow/tflite-micro-arduino-examples/tree/main/examples/magic_wand).

### 9. Walk vs sit vs stand

- **Live demo:** board in a pocket or on a belt; serial class updates while the wearer walks, sits, stands. **Not a medical device.**
- **Classes:** `WALK`, `SIT`, `STAND`.
- **Features:** 1 s @ 50 Hz, 3-axis → 20 stats or 60 samples.
- **Net:** 20 → 12 → 3 INT8.
- **Data:** stairs / sit-to-stand as extra hold-out, not extra medical claims.
- **Links:** [magic_wand](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/magic_wand), [Edge Impulse motion](https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/motion-recognition).

### 10. Fan / motor: off vs ok vs bad

- **Live demo:** IMU taped to a small DC motor or PC fan. Tape a coin for `BAD`. Serial class while it runs.
- **Classes:** `OFF`, `OK`, `BAD`.
- **Features:** 1 s accel; 8–16 FFT band powers or 20 time stats.
- **Net:** 20 → 12 → 3 INT8.
- **Data:** train on one motor; demo a second fan as the hard test.
- **Links:** [Edge Impulse anomaly](https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/anomaly-detection), [MLPerf Tiny](https://github.com/mlcommons/tiny).

---

## Vision (ESP32-CAM)

Hardware: ESP32-CAM, USB-TTL for flash, LED. Grayscale, small frames.
Bring-up: [person_detection](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/person_detection), [Arduino person_detection](https://github.com/tensorflow/tflite-micro-arduino-examples/tree/main/examples/person_detection), [EloquentEsp32cam](https://github.com/eloquentarduino/EloquentEsp32cam).

### 11. Person vs empty

- **Live demo:** empty chair vs classmate in frame; LED on only for `PERSON`.
- **Classes:** `PERSON`, `EMPTY`.
- **Input:** 96×96 gray (official) or a student 32×32 model they trained.
- **Net:** TFLite Micro person_detection **or** their own INT8 conv, on the CAM.
- **Data:** backpacks/coats as `EMPTY` hard negatives.
- **Links:** [person_detection](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/person_detection).

### 12. Hand vs no-hand

- **Live demo:** wave a hand in the center of the frame → LED. Face and empty wall stay off.
- **Classes:** `HAND`, `NONE`.
- **Input:** 32×32 or 48×48 gray crop, student-trained tiny conv, INT8.
- **Data:** fist, palm, face (`NONE`), empty wall.
- **Links:** [EloquentEsp32cam](https://github.com/eloquentarduino/EloquentEsp32cam), [person_detection](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/person_detection).

### 13. Three objects on a desk

- **Live demo:** show empty desk, a bottle, or a pen to the CAM; serial prints `EMPTY` / `BOTTLE` / `PEN`.
- **Classes:** `EMPTY`, `BOTTLE`, `PEN` (or two printed cards + empty).
- **Input:** 48×48 gray, student-trained tiny conv, INT8 on CAM. Not a pixel-mean rule.
- **Data:** two lighting conditions; hold-out is a different bottle/pen.
- **Links:** [EloquentEsp32cam](https://github.com/eloquentarduino/EloquentEsp32cam), [esp-dl](https://github.com/espressif/esp-dl).

---

## Other on-device TinyML

### 14. Motor current: idle / load / stall

- **Live demo:** ACS712 (3.3 V-safe) on a small DC motor. Serial class while idle, loaded (pinch shaft briefly), short stall pulse. Do not cook the motor.
- **Classes:** `IDLE`, `LOAD`, `STALL`.
- **Features:** 10 samples of current (+ Δ) ≈ 20 inputs.
- **Net:** 20 → 12 → 3 INT8 header on ESP32 (same shape as the fire workshop net).
- **Data:** live capture from the ESP32 ADC, train on PC, flash back.
- **Links:** [ACS712](https://www.allegromicro.com/en/products/sense/current-sensor-ics), [tflite-micro](https://github.com/tensorflow/tflite-micro).

### 15. Tap vs double-tap vs none (IMU)

- **Live demo:** IMU on a desk or wrist. Single tap / double tap / nothing → three LED patterns from the **classifier**, not an if-gap timer.
- **Classes:** `TAP`, `DOUBLE`, `NONE`.
- **Features:** 400 ms accel window (raw or 12 stats: peaks + gap).
- **Net:** 12 → 8 → 3 or 40 → 12 → 3 INT8 on ESP32.
- **Data:** walking = `NONE`; soft vs hard taps.
- **Links:** [i2cdevlib MPU6050](https://github.com/jrowberg/i2cdevlib), [Edge Impulse motion](https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/motion-recognition), [magic_wand](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/magic_wand).

---

## Suggested first picks

| Track | Project |
| --- | --- |
| Wake / activation | 1 (word) or 7 (gesture) |
| 3-class, workshop-shaped net | 3, 4, 5, or 14 |
| Camera | 11 or 13 |
| IMU | 8 or 10 |

## Pass / fail for every project

- [ ] Sensor data collected by the team (not only a public notebook)
- [ ] Trained model with a written **parameter count**
- [ ] INT8 or TFLite Micro **running on ESP32 / ESP32-CAM**
- [ ] Live demo ≥ 30 s (serial class + LED)
- [ ] Hold-out test (new speaker / room / object / motor)
- [ ] False activates / hour on the device
- [ ] “Not a safety / medical / security product”
