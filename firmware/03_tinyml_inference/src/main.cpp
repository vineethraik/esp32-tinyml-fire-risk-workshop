#include <Arduino.h>
#include <DHT.h>
#include "thermal_risk_model.h"

constexpr uint8_t DHT_PIN = 16;
constexpr uint32_t SAMPLE_INTERVAL_MS = 2000;
constexpr size_t WINDOW_SAMPLES = 10;

DHT dht(DHT_PIN, DHT22);
float temperatureHistory[WINDOW_SAMPLES]{};
float humidityHistory[WINDOW_SAMPLES]{};
size_t historyIndex = 0;
size_t historyCount = 0;
uint8_t highVotes[3]{};
size_t voteIndex = 0;
size_t voteCount = 0;
uint32_t lastSampleMs = 0;

const __FlashStringHelper *riskLabel(uint8_t riskClass) {
  if (riskClass == 0) return F("NORMAL");
  if (riskClass == 1) return F("ELEVATED_THERMAL_RISK");
  return F("HIGH_THERMAL_RISK");
}

void runInference() {
  const uint32_t startedUs = micros();
  float input[thermal_risk_model::kInputSize];
  for (size_t sample = 0; sample < WINDOW_SAMPLES; ++sample) {
    const size_t index = (historyIndex + sample) % WINDOW_SAMPLES;
    input[sample * 2] = temperatureHistory[index];
    input[sample * 2 + 1] = humidityHistory[index];
  }
  for (size_t i = 0; i < thermal_risk_model::kInputSize; ++i) {
    input[i] = (input[i] - thermal_risk_model::kInputMean[i]) /
               thermal_risk_model::kInputScale[i];
  }

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

  float total = 0.0f;
  for (float &logit : logits) {
    logit = expf(logit - maximum);
    total += logit;
  }
  uint8_t riskClass = 0;
  float confidence = logits[0] / total;
  for (size_t output = 1; output < thermal_risk_model::kOutputSize; ++output) {
    const float probability = logits[output] / total;
    if (probability > confidence) {
      riskClass = output;
      confidence = probability;
    }
  }

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
  if (millis() - lastSampleMs < SAMPLE_INTERVAL_MS) return;
  lastSampleMs = millis();

  const float humidity = dht.readHumidity();
  const float temperatureC = dht.readTemperature();
  if (isnan(temperatureC) || isnan(humidity)) {
    Serial.println(F("ERROR,DHT_READ"));
    return;
  }

  temperatureHistory[historyIndex] = temperatureC;
  humidityHistory[historyIndex] = humidity;
  historyIndex = (historyIndex + 1) % WINDOW_SAMPLES;
  historyCount = min(historyCount + 1, WINDOW_SAMPLES);
  if (historyCount < WINDOW_SAMPLES) {
    Serial.printf("WARMUP,%u/%u\n", historyCount, WINDOW_SAMPLES);
    return;
  }
  runInference();
}
