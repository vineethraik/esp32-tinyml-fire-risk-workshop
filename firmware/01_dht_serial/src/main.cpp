#include <Arduino.h>
#include <DHT.h>

constexpr uint8_t DHT_PIN = 16;  // NodeMCU P16 / GPIO16
constexpr uint32_t SAMPLE_INTERVAL_MS = 2000;

DHT dht(DHT_PIN, DHT22);
uint32_t lastSampleMs = 0;

void setup() {
  Serial.begin(115200);
  dht.begin();
  Serial.println(F("READY,DHT_SERIAL"));
}

void loop() {
  if (millis() - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMs = millis();

  const float humidity = dht.readHumidity();
  const float temperatureC = dht.readTemperature();
  if (isnan(temperatureC) || isnan(humidity)) {
    Serial.println(F("ERROR,DHT_READ"));
    return;
  }

  Serial.printf("DATA,%.2f,%.2f\n", temperatureC, humidity);
}
