#pragma once

#include <Arduino.h>
#include <math.h>
#include "thermal_risk_model.h"

// Small local inference library for both workshop firmwares. Students only
// need addReading(), ready(), and predict(); the model math lives below.
// The generated thermal_risk_model.h contains learned numbers, not algorithms.
struct RiskResult {
  uint8_t riskClass = 0;
  float confidence = 0.0f;
  bool votedHigh = false;
  uint32_t inferenceUs = 0;
};

class ThermalRiskInference {
 public:
  static constexpr size_t kWindowSamples = thermal_risk_model::kInputSize / 2;
  static_assert(thermal_risk_model::kInputSize == 20,
                "This workshop expects ten temperature/humidity pairs");

  // One valid DHT reading every two seconds. Invalid readings should be
  // rejected by the caller, so they do not advance the model's history.
  void addReading(float temperatureC, float humidityPct) {
    temperature_[nextIndex_] = temperatureC;
    humidity_[nextIndex_] = humidityPct;
    nextIndex_ = (nextIndex_ + 1) % kWindowSamples;
    if (count_ < kWindowSamples) ++count_;
    newReading_ = true;
  }

  bool ready() const { return count_ == kWindowSamples; }
  bool hasPrediction() const { return hasPrediction_; }
  size_t readingCount() const { return count_; }
  const RiskResult &latest() const { return latest_; }

  // Call once after each new reading. Repeated calls return the cached answer
  // and do not add extra votes for the same sensor sample.
  const RiskResult &predict() {
    if (!ready() || !newReading_) return latest_;
    const uint32_t startedUs = micros();

    // Circular buffer -> oldest-to-newest input order used during training.
    float input[thermal_risk_model::kInputSize];
    for (size_t sample = 0; sample < kWindowSamples; ++sample) {
      const size_t index = (nextIndex_ + sample) % kWindowSamples;
      input[sample * 2] = temperature_[index];
      input[sample * 2 + 1] = humidity_[index];
    }
    for (size_t i = 0; i < thermal_risk_model::kInputSize; ++i) {
      input[i] = (input[i] - thermal_risk_model::kInputMean[i]) /
                 thermal_risk_model::kInputScale[i];
    }

    // Dense 20->12, then ReLU. Stored INT8 weights are multiplied by their
    // scale to reconstruct an approximate floating-point value.
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

    // Dense 12->3 produces one score for each teaching label.
    float logits[thermal_risk_model::kOutputSize];
    float maximum = -INFINITY;
    for (size_t output = 0; output < thermal_risk_model::kOutputSize; ++output) {
      logits[output] = thermal_risk_model::kDense2Bias[output] *
                       thermal_risk_model::kDense2BiasScale;
      for (size_t unit = 0; unit < thermal_risk_model::kHiddenSize; ++unit) {
        logits[output] += hidden[unit] *
                          thermal_risk_model::kDense2Weights[unit][output] *
                          thermal_risk_model::kDense2WeightScale;
      }
      maximum = max(maximum, logits[output]);
    }

    // Softmax and argmax. Subtracting maximum prevents expf() overflow.
    float total = 0.0f;
    for (float &logit : logits) {
      logit = expf(logit - maximum);
      total += logit;
    }
    latest_.riskClass = 0;
    latest_.confidence = logits[0] / total;
    for (size_t output = 1; output < thermal_risk_model::kOutputSize; ++output) {
      const float probability = logits[output] / total;
      if (probability > latest_.confidence) {
        latest_.riskClass = output;
        latest_.confidence = probability;
      }
    }

    // An alert needs two HIGH results among three consecutive windows.
    highVotes_[voteIndex_] = latest_.riskClass == 2 ? 1 : 0;
    voteIndex_ = (voteIndex_ + 1) % 3;
    if (voteCount_ < 3) ++voteCount_;
    uint8_t highCount = 0;
    for (uint8_t vote : highVotes_) highCount += vote;
    latest_.votedHigh = voteCount_ == 3 && highCount >= 2;
    latest_.inferenceUs = micros() - startedUs;
    hasPrediction_ = true;
    newReading_ = false;
    return latest_;
  }

  static const __FlashStringHelper *label(uint8_t riskClass) {
    if (riskClass == 0) return F("NORMAL");
    if (riskClass == 1) return F("ELEVATED_THERMAL_RISK");
    return F("HIGH_THERMAL_RISK");
  }

 private:
  float temperature_[kWindowSamples]{};
  float humidity_[kWindowSamples]{};
  size_t nextIndex_ = 0;
  size_t count_ = 0;
  uint8_t highVotes_[3]{};
  size_t voteIndex_ = 0;
  size_t voteCount_ = 0;
  bool newReading_ = false;
  bool hasPrediction_ = false;
  RiskResult latest_{};
};
