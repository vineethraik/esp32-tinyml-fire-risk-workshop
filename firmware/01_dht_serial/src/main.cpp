#include <Arduino.h>
#include <DHT.h>

// Stage 1: the smallest complete sensor program.
//
// Wiring:
//   DHT22 VCC  -> ESP32 3V3
//   DHT22 DATA -> ESP32 P16 / GPIO16
//   DHT22 GND  -> ESP32 GND
//
// Serial format: DATA,<temperature_celsius>,<humidity_percent>
// The Python collector depends on this stable comma-separated format.

constexpr uint8_t DHT_PIN = 16;  // NodeMCU P16 / GPIO16
// DHT22 is a slow sensor; reading every two seconds is deliberate.
constexpr uint32_t SAMPLE_INTERVAL_MS = 2000;

// The library handles the DHT22 timing protocol on GPIO16.
DHT dht(DHT_PIN, DHT22);
// millis() timing keeps loop() non-blocking; there is no delay(2000).
uint32_t lastSampleMs = 0;

void setup() {
  // USB serial is the only communication channel in this teaching stage.
  Serial.begin(115200);
  dht.begin();
  Serial.println(F("READY,DHT_SERIAL"));
}

void loop() {
  // Unsigned subtraction continues to work when millis() eventually wraps.
  if (millis() - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMs = millis();

  const float humidity = dht.readHumidity();
  const float temperatureC = dht.readTemperature();

  // A disconnected or mistimed DHT22 returns NaN. Never write an invalid row
  // that looks like a genuine measurement.
  if (isnan(temperatureC) || isnan(humidity)) {
    Serial.println(F("ERROR,DHT_READ"));
    return;
  }

  // Two decimal places are sufficient for this sensor and easy to parse.
  Serial.printf("DATA,%.2f,%.2f\n", temperatureC, humidity);
}
