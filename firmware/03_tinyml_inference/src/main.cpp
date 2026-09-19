#include <Arduino.h>
#include <DHT.h>
#include "thermal_risk_inference.h"

// Stage 3 teaches the device workflow. The reusable local library handles
// the window, neural-network math, and 2-of-3 vote.
constexpr uint8_t DHT_PIN = 16;  // NodeMCU P16 / GPIO16
constexpr uint32_t SAMPLE_INTERVAL_MS = 2000;

DHT dht(DHT_PIN, DHT22);
ThermalRiskInference model;
uint32_t lastSampleMs = 0;

void setup() {
  Serial.begin(115200);
  dht.begin();
  Serial.println(F("READY,TINYML_INFERENCE"));
}

void loop() {
  // DHT22 needs about two seconds between readings; avoid delay() so loop
  // stays available for other work.
  if (millis() - lastSampleMs < SAMPLE_INTERVAL_MS) return;
  lastSampleMs = millis();

  const float humidity = dht.readHumidity();
  const float temperatureC = dht.readTemperature();
  if (isnan(temperatureC) || isnan(humidity)) {
    Serial.println(F("ERROR,DHT_READ"));
    return;
  }

  model.addReading(temperatureC, humidity);
  if (!model.ready()) {
    Serial.printf("WARMUP,%u/%u\n",
                  static_cast<unsigned>(model.readingCount()),
                  static_cast<unsigned>(ThermalRiskInference::kWindowSamples));
    return;
  }

  const RiskResult &result = model.predict();
  Serial.print(F("RISK,"));
  Serial.print(ThermalRiskInference::label(result.riskClass));
  Serial.print(',');
  Serial.print(result.confidence, 3);
  Serial.print(F(",voted_high="));
  Serial.print(result.votedHigh ? 1 : 0);
  Serial.print(F(",inference_us="));
  Serial.println(result.inferenceUs);
}
