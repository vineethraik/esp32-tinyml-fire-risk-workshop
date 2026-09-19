#include <Arduino.h>
#include <DHT.h>
#include <LittleFS.h>

// Stage 2: store DHT22 readings as a human-readable CSV in LittleFS.
// This stage teaches file storage and AT commands. The final firmware uses a
// different, more compact raw ring-buffer format.

constexpr uint8_t DHT_PIN = 16;
constexpr uint32_t SAMPLE_INTERVAL_MS = 2000;
// Deletion requires an arm command followed by confirmation within 30 seconds.
constexpr uint32_t DELETE_CONFIRM_MS = 30000;
// LittleFS paths begin at the filesystem root, not the source-code folder.
constexpr char DATA_FILE[] = "/dht.csv";

DHT dht(DHT_PIN, DHT22);

// State reconstructed from the CSV after every restart.
uint32_t lastSampleMs = 0;
uint32_t sequence = 0;
uint32_t storedRows = 0;

// Two-step deletion state prevents one mistyped command erasing class data.
bool deleteArmed = false;
uint32_t deleteArmedAtMs = 0;

void ensureFileExists() {
  // A header makes the exported file directly readable by spreadsheet and
  // Python tools. Existing data is never replaced here.
  if (LittleFS.exists(DATA_FILE)) return;
  File file = LittleFS.open(DATA_FILE, FILE_WRITE);
  file.println(F("sequence,temp_c,humidity_pct"));
  file.close();
}

void restoreFileState() {
  // Scan the small teaching CSV once at startup. The newest sequence number
  // lets logging continue instead of starting again from zero.
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
  // Open-append-close makes each completed line durable and keeps the example
  // easy to understand. It is not the most flash-efficient production design.
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
  // trim() accepts terminals that send CRLF as well as LF endings.
  command.trim();
  if (command == "AT") {
    Serial.println(F("OK"));
  } else if (command == "AT+STATUS") {
    Serial.printf("STATUS,rows=%u,used_bytes=%u,total_bytes=%u\n",
                  storedRows, LittleFS.usedBytes(), LittleFS.totalBytes());
  } else if (command == "AT+EXPORT") {
    // Export markers allow a PC or student to distinguish CSV data from other
    // serial messages.
    File file = LittleFS.open(DATA_FILE, FILE_READ);
    Serial.println(F("EXPORT,BEGIN"));
    while (file && file.available()) Serial.write(file.read());
    if (file) file.close();
    Serial.println(F("EXPORT,END"));
  } else if (command == "AT+DELETE") {
    // Arming alone does not delete the file.
    deleteArmed = true;
    deleteArmedAtMs = millis();
    Serial.println(F("DELETE,ARMED,send AT+DELETE,CONFIRM within 30 seconds"));
  } else if (command == "AT+DELETE,CONFIRM") {
    // Reject confirmation when no recent arm command exists.
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

  // begin(true) formats LittleFS only when mounting fails. Switching partition
  // layouts can therefore erase this teaching filesystem; export wanted data
  // before flashing a different layout.
  if (!LittleFS.begin(true)) {
    Serial.println(F("ERROR,FLASH_MOUNT"));
    return;
  }
  ensureFileExists();
  restoreFileState();
  Serial.println(F("READY,FLASH_STORAGE"));
}

void loop() {
  // Commands and sampling share one simple loop; neither requires Wi-Fi or an
  // RTOS task.
  if (Serial.available()) handleCommand(Serial.readStringUntil('\n'));

  // Automatically cancel an unconfirmed delete request.
  if (deleteArmed && millis() - deleteArmedAtMs > DELETE_CONFIRM_MS) {
    deleteArmed = false;
  }
  if (millis() - lastSampleMs < SAMPLE_INTERVAL_MS) return;
  lastSampleMs = millis();

  const float humidity = dht.readHumidity();
  const float temperatureC = dht.readTemperature();
  // Skip failed reads instead of storing fake zero values.
  if (isnan(temperatureC) || isnan(humidity)) {
    Serial.println(F("ERROR,DHT_READ"));
    return;
  }
  appendReading(temperatureC, humidity);
}
