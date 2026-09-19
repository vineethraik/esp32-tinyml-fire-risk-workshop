#include <cassert>
#include <cmath>
#include "thermal_risk_inference.h"

int main() {
  ThermalRiskInference model;
  assert(!model.ready());
  assert(!model.hasPrediction());

  for (size_t i = 0; i < ThermalRiskInference::kWindowSamples - 1; ++i) {
    model.addReading(27.0f, 55.0f);
    assert(!model.ready());
    model.predict();
    assert(!model.hasPrediction());
  }

  model.addReading(27.0f, 55.0f);
  assert(model.ready());
  const RiskResult first = model.predict();
  assert(model.hasPrediction());
  assert(first.riskClass < thermal_risk_model::kOutputSize);
  assert(std::isfinite(first.confidence));
  assert(first.confidence >= 0.0f && first.confidence <= 1.0f);
  assert(!first.votedHigh);  // One complete window cannot satisfy 2-of-3.

  // Calling predict() again without a new reading must not count a new vote.
  const RiskResult again = model.predict();
  assert(again.riskClass == first.riskClass);
  assert(again.confidence == first.confidence);
  assert(again.votedHigh == first.votedHigh);
  assert(again.inferenceUs == first.inferenceUs);

  model.addReading(28.0f, 54.0f);
  const RiskResult next = model.predict();
  assert(next.riskClass < thermal_risk_model::kOutputSize);
  assert(std::isfinite(next.confidence));
}
