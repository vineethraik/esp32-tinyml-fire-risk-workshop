#include <Arduino.h>
#include <DHT.h>
#include "thermal_risk_model.h"

// Stage 3: the smallest firmware that demonstrates the complete on-device ML
// path. It keeps ten DHT readings, normalizes 20 values, evaluates a 20->12->3
// dense network, and applies a 2-of-3 high-risk vote.

constexpr uint8_t DHT_PIN = 16;
constexpr uint32_t SAMPLE_INTERVAL_MS = 2000;
constexpr size_t WINDOW_SAMPLES = 10;

DHT dht(DHT_PIN, DHT22);

// Circular histories avoid moving all ten samples after every reading.
float temperatureHistory[WINDOW_SAMPLES]{};
float humidityHistory[WINDOW_SAMPLES]{};
size_t historyIndex = 0;
size_t historyCount = 0;

// The newest three classifications are stored as 1 for HIGH, otherwise 0.
uint8_t highVotes[3]{};
size_t voteIndex = 0;
size_t voteCount = 0;
uint32_t lastSampleMs = 0;

const __FlashStringHelper *riskLabel(uint8_t riskClass) {
  // Output index order must match LABELS in the Python training script.
  if (riskClass == 0) return F("NORMAL");
  if (riskClass == 1) return F("ELEVATED_THERMAL_RISK");
  return F("HIGH_THERMAL_RISK");
}

void runInference() {
  // micros() measures only model calculation time, not the slow DHT22 read.
  const uint32_t startedUs = micros();

  // historyIndex points to the oldest item once the circular buffer is full.
  // Rebuild the exact chronological feature order used during training:
  // temp_0, humidity_0, ..., temp_9, humidity_9.
  float input[thermal_risk_model::kInputSize];
  for (size_t sample = 0; sample < WINDOW_SAMPLES; ++sample) {
    const size_t index = (historyIndex + sample) % WINDOW_SAMPLES;
    input[sample * 2] = temperatureHistory[index];
    input[sample * 2 + 1] = humidityHistory[index];
  }
  // Apply the training mean and standard deviation to every input position.
  for (size_t i = 0; i < thermal_risk_model::kInputSize; ++i) {
    input[i] = (input[i] - thermal_risk_model::kInputMean[i]) /
               thermal_risk_model::kInputScale[i];
  }

  // Dense layer 1 followed by ReLU. Stored INT8 values are converted back to
  // an approximate float by multiplying each value by its array scale.
  float hidden[thermal_risk_model::kHiddenSize];
  for (size_t unit = 0; unit < thermal_risk_model::kHiddenSize; ++unit) {
    hidden[unit] = thermal_risk_model::kDense1Bias[unit] *
                   thermal_risk_model::kDense1BiasScale;
    for (size_t i = 0; i < thermal_risk_model::kInputSize; ++i) {
      hidden[unit] += input[i] * thermal_risk_model::kDense1Weights[i][unit] *
                      thermal_risk_model::kDense1WeightScale;
    }
    hidden[unit] = max(hidden[unit], 0.0f);
  }

  // Dense layer 2 produces one raw score (logit) for each risk class.
  float logits[thermal_risk_model::kOutputSize];
  float maximum = -INFINITY;
  for (size_t output = 0; output < thermal_risk_model::kOutputSize; ++output) {
    logits[output] = thermal_risk_model::kDense2Bias[output] *
                     thermal_risk_model::kDense2BiasScale;
    for (size_t unit = 0; unit < thermal_risk_model::kHiddenSize; ++unit) {
      logits[output] += hidden[unit] * thermal_risk_model::kDense2Weights[unit][output] *
                        thermal_risk_model::kDense2WeightScale;
    }
    maximum = max(maximum, logits[output]);
  }

  // Numerically stable Softmax: subtracting the largest logit prevents expf()
  // overflow while keeping the same final probability ratios.
  float total = 0.0f;
  for (float &logit : logits) {
    logit = expf(logit - maximum);
    total += logit;
  }
  // Select the class with the largest probability and call that probability
  // confidence. Confidence is a model score, not certainty or safety proof.
  uint8_t riskClass = 0;
  float confidence = logits[0] / total;
  for (size_t output = 1; output < thermal_risk_model::kOutputSize; ++output) {
    const float probability = logits[output] / total;
    if (probability > confidence) {
      riskClass = output;
      confidence = probability;
    }
  }

  // A high alert requires two HIGH classifications among three complete
  // windows. ELEVATED alone does not set voted_high.
  highVotes[voteIndex] = riskClass == 2 ? 1 : 0;
  voteIndex = (voteIndex + 1) % 3;
  voteCount = min(voteCount + 1, static_cast<size_t>(3));
  uint8_t highCount = 0;
  for (uint8_t vote : highVotes) highCount += vote;
  const bool votedHigh = voteCount == 3 && highCount >= 2;

  Serial.print(F("RISK,"));
  Serial.print(riskLabel(riskClass));
  Serial.print(',');
  Serial.print(confidence, 3);
  Serial.print(F(",voted_high="));
  Serial.print(votedHigh ? 1 : 0);
  Serial.print(F(",inference_us="));
  Serial.println(micros() - startedUs);
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  Serial.println(F("READY,TINYML_INFERENCE"));
}

void loop() {
  // Non-blocking two-second schedule leaves serial and future features free to
  // run between sensor samples.
  if (millis() - lastSampleMs < SAMPLE_INTERVAL_MS) return;
  lastSampleMs = millis();

  const float humidity = dht.readHumidity();
  const float temperatureC = dht.readTemperature();
  if (isnan(temperatureC) || isnan(humidity)) {
    Serial.println(F("ERROR,DHT_READ"));
    return;
  }

  // Insert the newest valid sample, then advance/wrap the circular index.
  temperatureHistory[historyIndex] = temperatureC;
  humidityHistory[historyIndex] = humidity;
  historyIndex = (historyIndex + 1) % WINDOW_SAMPLES;
  historyCount = min(historyCount + 1, WINDOW_SAMPLES);
  if (historyCount < WINDOW_SAMPLES) {
    // Inference is impossible until all 20 input values are available.
    Serial.printf("WARMUP,%u/%u\n", historyCount, WINDOW_SAMPLES);
    return;
  }
  runInference();
}
