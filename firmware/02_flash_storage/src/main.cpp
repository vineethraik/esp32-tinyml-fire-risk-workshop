#include <Arduino.h>
#include <DHT.h>
#include <LittleFS.h>

constexpr uint8_t DHT_PIN = 16;
constexpr uint32_t SAMPLE_INTERVAL_MS = 2000;
constexpr uint32_t DELETE_CONFIRM_MS = 30000;
constexpr char DATA_FILE[] = "/dht.csv";

DHT dht(DHT_PIN, DHT22);
uint32_t lastSampleMs = 0;
uint32_t sequence = 0;
uint32_t storedRows = 0;
bool deleteArmed = false;
uint32_t deleteArmedAtMs = 0;

void ensureFileExists() {
  if (LittleFS.exists(DATA_FILE)) return;
  File file = LittleFS.open(DATA_FILE, FILE_WRITE);
  file.println(F("sequence,temp_c,humidity_pct"));
  file.close();
}

void restoreFileState() {
  File file = LittleFS.open(DATA_FILE, FILE_READ);
  if (!file) return;
  file.readStringUntil('\n');  // Skip CSV header.
  while (file.available()) {
    const String line = file.readStringUntil('\n');
    const int separator = line.indexOf(',');
    if (separator <= 0) continue;
    sequence = static_cast<uint32_t>(line.substring(0, separator).toInt()) + 1;
    ++storedRows;
  }
  file.close();
}

void appendReading(float temperatureC, float humidity) {
  File file = LittleFS.open(DATA_FILE, FILE_APPEND);
  if (!file) {
    Serial.println(F("ERROR,FLASH_OPEN"));
    return;
  }
  file.printf("%u,%.2f,%.2f\n", sequence, temperatureC, humidity);
  file.close();
  ++sequence;
  ++storedRows;
}

void handleCommand(String command) {
  command.trim();
  if (command == "AT") {
    Serial.println(F("OK"));
  } else if (command == "AT+STATUS") {
    Serial.printf("STATUS,rows=%u,used_bytes=%u,total_bytes=%u\n",
                  storedRows, LittleFS.usedBytes(), LittleFS.totalBytes());
  } else if (command == "AT+EXPORT") {
    File file = LittleFS.open(DATA_FILE, FILE_READ);
    Serial.println(F("EXPORT,BEGIN"));
    while (file && file.available()) Serial.write(file.read());
    if (file) file.close();
    Serial.println(F("EXPORT,END"));
  } else if (command == "AT+DELETE") {
    deleteArmed = true;
    deleteArmedAtMs = millis();
    Serial.println(F("DELETE,ARMED,send AT+DELETE,CONFIRM within 30 seconds"));
  } else if (command == "AT+DELETE,CONFIRM") {
    if (!deleteArmed || millis() - deleteArmedAtMs > DELETE_CONFIRM_MS) {
      Serial.println(F("ERROR,DELETE_NOT_ARMED"));
      return;
    }
    LittleFS.remove(DATA_FILE);
    sequence = 0;
    storedRows = 0;
    deleteArmed = false;
    ensureFileExists();
    Serial.println(F("DELETE,OK"));
  } else if (command.length()) {
    Serial.println(F("ERROR,UNKNOWN_COMMAND"));
  }
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  if (!LittleFS.begin(true)) {
    Serial.println(F("ERROR,FLASH_MOUNT"));
    return;
  }
  ensureFileExists();
  restoreFileState();
  Serial.println(F("READY,FLASH_STORAGE"));
}

void loop() {
  if (Serial.available()) handleCommand(Serial.readStringUntil('\n'));
  if (deleteArmed && millis() - deleteArmedAtMs > DELETE_CONFIRM_MS) {
    deleteArmed = false;
  }
  if (millis() - lastSampleMs < SAMPLE_INTERVAL_MS) return;
  lastSampleMs = millis();

  const float humidity = dht.readHumidity();
  const float temperatureC = dht.readTemperature();
  if (isnan(temperatureC) || isnan(humidity)) {
    Serial.println(F("ERROR,DHT_READ"));
    return;
  }
  appendReading(temperatureC, humidity);
}
