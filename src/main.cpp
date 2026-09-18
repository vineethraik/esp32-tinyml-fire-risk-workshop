#include <Arduino.h>
#include <DHT.h>
#include <Preferences.h>
#include <esp_err.h>
#include <esp_partition.h>
#include "thermal_risk_model.h"

// Current hardware: DHT22 only. Flame is a separate immediate fire indicator,
// not part of this stored DHT time series.
constexpr uint8_t DHT_PIN = 16;  // NodeMCU pin P16 / GPIO16
// Normal data-collection setting: one real DHT22 sample every two seconds.
constexpr uint32_t RECORD_INTERVAL_MS = 2000;
constexpr uint32_t DHT_READ_INTERVAL_MS = 2000;
constexpr bool STOP_WHEN_RING_FULL = false;
constexpr uint32_t DELETE_CONFIRM_WINDOW_MS = 30000;

constexpr char RING_LABEL[] = "ring";
constexpr uint8_t RING_SUBTYPE = 0x40;
constexpr size_t RING_SIZE_BYTES = 1024 * 1024;
constexpr size_t FLASH_SECTOR_BYTES = 4096;
// New compact DHT-only record format. A format change intentionally clears the
// old ring once, so bytes from an older layout are never misread as samples.
constexpr uint32_t RING_FORMAT_VERSION = 3;
constexpr uint32_t SEQUENCE_MASK = 0x00FFFFFF;

constexpr int16_t TEMP_UNAVAILABLE = INT16_MIN;
constexpr uint16_t UINT16_UNAVAILABLE = UINT16_MAX;
// 8 bytes x 131,072 records = exactly 1 MiB (about 72.8 hours at 2 seconds).
struct __attribute__((packed)) SampleRecord {
  uint16_t sequenceLow;
  uint8_t sequenceHigh;
  int16_t temperatureCx100;
  uint16_t humidityPctX100;
  uint8_t crc8;
};

static_assert(sizeof(SampleRecord) == 8, "Ring records must remain 8 bytes");

constexpr size_t RECORD_SIZE = sizeof(SampleRecord);
constexpr size_t RECORD_CAPACITY = RING_SIZE_BYTES / RECORD_SIZE;
constexpr size_t RECORDS_PER_SECTOR = FLASH_SECTOR_BYTES / RECORD_SIZE;

DHT dht(DHT_PIN, DHT22);
const esp_partition_t *ringPartition = nullptr;

float latestTemperatureC = NAN;
float latestHumidityPct = NAN;
bool latestDhtValid = false;
uint32_t lastSampleMs = 0;
uint32_t lastDhtReadMs = 0;
uint32_t writeIndex = 0;
uint32_t nextSequence = 0;
uint32_t latestSequence = 0;
size_t latestIndex = 0;
size_t validRecordCount = 0;
bool hasRecords = false;
bool hasWrapped = false;
bool loggingComplete = false;
bool liveLogEnabled = false;
bool riskLogEnabled = false;
bool deleteArmed = false;
uint32_t deleteArmedAtMs = 0;

int16_t riskTemperatureHistory[10]{};
uint16_t riskHumidityHistory[10]{};
size_t riskHistoryIndex = 0;
size_t riskHistoryCount = 0;
uint8_t highRiskVotes[3]{};
size_t highRiskVoteIndex = 0;
size_t highRiskVoteCount = 0;
uint8_t latestRiskClass = 0;
float latestRiskConfidence = 0.0f;
bool riskReady = false;
bool votedHighRisk = false;

char commandBuffer[48];
size_t commandLength = 0;

uint8_t crc8(const uint8_t *data, size_t length) {
  uint8_t crc = 0;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x80) ? static_cast<uint8_t>((crc << 1) ^ 0x07)
                           : static_cast<uint8_t>(crc << 1);
    }
  }
  return crc;
}

bool isRecordValid(const SampleRecord &record) {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&record);
  bool erased = true;
  for (size_t i = 0; i < RECORD_SIZE; ++i) {
    erased = erased && bytes[i] == 0xFF;
  }
  return !erased &&
         crc8(reinterpret_cast<const uint8_t *>(&record), RECORD_SIZE - 1) == record.crc8;
}

uint32_t recordSequence(const SampleRecord &record) {
  return static_cast<uint32_t>(record.sequenceLow) |
         (static_cast<uint32_t>(record.sequenceHigh) << 16);
}

void setRecordSequence(SampleRecord &record, uint32_t sequence) {
  const uint32_t value = sequence & SEQUENCE_MASK;
  record.sequenceLow = static_cast<uint16_t>(value);
  record.sequenceHigh = static_cast<uint8_t>(value >> 16);
}

bool sequenceIsNewer(uint32_t candidate, uint32_t reference) {
  const uint32_t difference = (candidate - reference) & SEQUENCE_MASK;
  return difference != 0 && difference < 0x00800000;
}

bool readRecord(size_t index, SampleRecord &record) {
  return esp_partition_read(ringPartition, index * RECORD_SIZE, &record,
                            RECORD_SIZE) == ESP_OK;
}

bool slotIsErased(size_t index) {
  SampleRecord record{};
  if (!readRecord(index, record)) {
    return false;
  }

  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&record);
  for (size_t i = 0; i < RECORD_SIZE; ++i) {
    if (bytes[i] != 0xFF) {
      return false;
    }
  }
  return true;
}

bool ensureRingIsFormatted() {
  Preferences preferences;
  if (!preferences.begin("ring_meta", false)) {
    Serial.println(F("ERR+FLASH,nvs_open"));
    return false;
  }

  const uint32_t storedVersion = preferences.getUInt("format", 0);
  if (storedVersion == RING_FORMAT_VERSION) {
    preferences.end();
    return true;
  }

  Serial.println(F("INFO+FLASH,formatting_ring"));
  const esp_err_t result =
      esp_partition_erase_range(ringPartition, 0, RING_SIZE_BYTES);
  if (result != ESP_OK) {
    preferences.end();
    Serial.printf("ERR+FLASH,format,%s\n", esp_err_to_name(result));
    return false;
  }

  preferences.putUInt("format", RING_FORMAT_VERSION);
  preferences.end();
  return true;
}

void resetRingState() {
  writeIndex = 0;
  nextSequence = 0;
  latestSequence = 0;
  latestIndex = 0;
  validRecordCount = 0;
  hasRecords = false;
  hasWrapped = false;
  loggingComplete = false;
}

bool deleteRing() {
  Serial.println(F("ACK+DELETE,erasing"));
  const esp_err_t result =
      esp_partition_erase_range(ringPartition, 0, RING_SIZE_BYTES);
  if (result != ESP_OK) {
    Serial.printf("ERR+DELETE,%s\n", esp_err_to_name(result));
    return false;
  }

  resetRingState();
  lastSampleMs = millis();
  Serial.println(F("OK+DELETE,records=0"));
  return true;
}

size_t countValidRecordsInSector(size_t sectorStartIndex) {
  size_t count = 0;
  SampleRecord record{};
  for (size_t i = 0; i < RECORDS_PER_SECTOR; ++i) {
    if (readRecord(sectorStartIndex + i, record) && isRecordValid(record)) {
      ++count;
    }
  }
  return count;
}

bool eraseSectorForIndex(size_t index) {
  const size_t sectorStartIndex =
      (index / RECORDS_PER_SECTOR) * RECORDS_PER_SECTOR;
  const size_t erasedRecords = countValidRecordsInSector(sectorStartIndex);
  const esp_err_t result = esp_partition_erase_range(
      ringPartition, sectorStartIndex * RECORD_SIZE, FLASH_SECTOR_BYTES);
  if (result != ESP_OK) {
    Serial.printf("ERR+FLASH,erase,%s\n", esp_err_to_name(result));
    return false;
  }

  validRecordCount = validRecordCount > erasedRecords
                         ? validRecordCount - erasedRecords
                         : 0;
  return true;
}

bool prepareWritableSlot() {
  for (size_t attempts = 0; attempts < RECORD_CAPACITY; ++attempts) {
    if (writeIndex % RECORDS_PER_SECTOR == 0 && !eraseSectorForIndex(writeIndex)) {
      return false;
    }
    if (slotIsErased(writeIndex)) {
      return true;
    }

    // Power loss can leave one partial record. Skip it and preserve all valid
    // records around it.
    writeIndex = (writeIndex + 1) % RECORD_CAPACITY;
  }

  Serial.println(F("ERR+FLASH,no_writable_slot"));
  return false;
}

void restoreRingState() {
  static SampleRecord sectorRecords[RECORDS_PER_SECTOR];

  for (size_t sector = 0; sector < RECORD_CAPACITY / RECORDS_PER_SECTOR;
       ++sector) {
    if (esp_partition_read(ringPartition, sector * FLASH_SECTOR_BYTES,
                           sectorRecords, sizeof(sectorRecords)) != ESP_OK) {
      Serial.println(F("ERR+FLASH,scan"));
      return;
    }

    for (size_t slot = 0; slot < RECORDS_PER_SECTOR; ++slot) {
      const SampleRecord &record = sectorRecords[slot];
      if (!isRecordValid(record)) {
        continue;
      }

      const size_t index = sector * RECORDS_PER_SECTOR + slot;
      ++validRecordCount;
      const uint32_t sequence = recordSequence(record);
      if (!hasRecords || sequenceIsNewer(sequence, latestSequence)) {
        hasRecords = true;
        latestSequence = sequence;
        latestIndex = index;
      }
    }
  }

  if (hasRecords) {
    nextSequence = (latestSequence + 1) & SEQUENCE_MASK;
    writeIndex = (latestIndex + 1) % RECORD_CAPACITY;
    hasWrapped = validRecordCount == RECORD_CAPACITY;
  }
}

void readDht22() {
  const float humidity = dht.readHumidity();
  const float temperatureC = dht.readTemperature();
  if (!isnan(humidity) && !isnan(temperatureC)) {
    latestHumidityPct = humidity;
    latestTemperatureC = temperatureC;
    latestDhtValid = true;
  }
}

int16_t toTemperatureCx100() {
  if (!latestDhtValid) {
    return TEMP_UNAVAILABLE;
  }
  return static_cast<int16_t>(latestTemperatureC * 100.0f +
                              (latestTemperatureC >= 0.0f ? 0.5f : -0.5f));
}

uint16_t toHumidityPctX100() {
  if (!latestDhtValid) {
    return UINT16_UNAVAILABLE;
  }
  return static_cast<uint16_t>(latestHumidityPct * 100.0f + 0.5f);
}

void printTemperature(int16_t value) {
  if (value == TEMP_UNAVAILABLE) {
    Serial.print(F("nan"));
  } else {
    Serial.print(static_cast<float>(value) / 100.0f, 2);
  }
}

void printHumidity(uint16_t value) {
  if (value == UINT16_UNAVAILABLE) {
    Serial.print(F("nan"));
  } else {
    Serial.print(static_cast<float>(value) / 100.0f, 2);
  }
}

void printRecord(const __FlashStringHelper *prefix, const SampleRecord &record) {
  Serial.print(prefix);
  Serial.print(recordSequence(record));
  Serial.print(',');
  printTemperature(record.temperatureCx100);
  Serial.print(',');
  printHumidity(record.humidityPctX100);
  Serial.println();
}

const __FlashStringHelper *riskLabel(uint8_t riskClass) {
  switch (riskClass) {
    case 0: return F("NORMAL");
    case 1: return F("ELEVATED_THERMAL_RISK");
    default: return F("HIGH_THERMAL_RISK");
  }
}

void printRisk(uint32_t sequence) {
  Serial.print(F("RISK,"));
  Serial.print(sequence);
  Serial.print(',');
  Serial.print(riskLabel(latestRiskClass));
  Serial.print(',');
  Serial.print(latestRiskConfidence, 3);
  Serial.print(F(",voted_high="));
  Serial.println(votedHighRisk ? 1 : 0);
}

void runRiskInference(const SampleRecord &record) {
  if (record.temperatureCx100 == TEMP_UNAVAILABLE ||
      record.humidityPctX100 == UINT16_UNAVAILABLE) {
    return;
  }

  riskTemperatureHistory[riskHistoryIndex] = record.temperatureCx100;
  riskHumidityHistory[riskHistoryIndex] = record.humidityPctX100;
  riskHistoryIndex = (riskHistoryIndex + 1) % 10;
  riskHistoryCount = min(riskHistoryCount + 1, static_cast<size_t>(10));
  if (riskHistoryCount < 10) {
    return;
  }

  float input[thermal_risk_model::kInputSize];
  for (size_t sample = 0; sample < 10; ++sample) {
    const size_t index = (riskHistoryIndex + sample) % 10;
    input[sample * 2] = static_cast<float>(riskTemperatureHistory[index]) / 100.0f;
    input[sample * 2 + 1] = static_cast<float>(riskHumidityHistory[index]) / 100.0f;
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
  latestRiskClass = 0;
  latestRiskConfidence = logits[0] / total;
  for (size_t output = 1; output < thermal_risk_model::kOutputSize; ++output) {
    const float probability = logits[output] / total;
    if (probability > latestRiskConfidence) {
      latestRiskClass = output;
      latestRiskConfidence = probability;
    }
  }
  riskReady = true;
  highRiskVotes[highRiskVoteIndex] = latestRiskClass == 2 ? 1 : 0;
  highRiskVoteIndex = (highRiskVoteIndex + 1) % 3;
  highRiskVoteCount = min(highRiskVoteCount + 1, static_cast<size_t>(3));
  uint8_t highVotes = 0;
  for (uint8_t vote : highRiskVotes) highVotes += vote;
  votedHighRisk = highRiskVoteCount == 3 && highVotes >= 2;
  if (riskLogEnabled) printRisk(recordSequence(record));
}

bool appendSample() {
  if (ringPartition == nullptr || !prepareWritableSlot()) {
    return false;
  }

  SampleRecord record{};
  setRecordSequence(record, nextSequence);
  record.temperatureCx100 = toTemperatureCx100();
  record.humidityPctX100 = toHumidityPctX100();
  record.crc8 = crc8(reinterpret_cast<const uint8_t *>(&record), RECORD_SIZE - 1);

  const esp_err_t result = esp_partition_write(ringPartition,
                                                writeIndex * RECORD_SIZE,
                                                &record, RECORD_SIZE);
  if (result != ESP_OK) {
    Serial.printf("ERR+FLASH,write,%s\n", esp_err_to_name(result));
    return false;
  }

  hasRecords = true;
  latestSequence = recordSequence(record);
  latestIndex = writeIndex;
  nextSequence = (nextSequence + 1) & SEQUENCE_MASK;
  ++validRecordCount;
  hasWrapped = hasWrapped || validRecordCount == RECORD_CAPACITY;
  writeIndex = (writeIndex + 1) % RECORD_CAPACITY;
  if (liveLogEnabled) {
    printRecord(F("LOG,"), record);
  }
  runRiskInference(record);
  if (STOP_WHEN_RING_FULL && validRecordCount == RECORD_CAPACITY) {
    loggingComplete = true;
    Serial.printf("RING_FULL,records=%u\n", static_cast<unsigned>(validRecordCount));
  }
  return true;
}

void printStatus() {
  Serial.printf("ACK+STATUS,records=%u,capacity=%u,interval_ms=%u,dht_interval_ms=%u,record_bytes=%u,complete=%u,live_log=%u\n",
                static_cast<unsigned>(validRecordCount),
                static_cast<unsigned>(RECORD_CAPACITY), RECORD_INTERVAL_MS,
                DHT_READ_INTERVAL_MS, static_cast<unsigned>(RECORD_SIZE),
                loggingComplete ? 1 : 0, liveLogEnabled ? 1 : 0);
  Serial.println(F("OK+STATUS"));
}

// Export a bounded slice of the chronological valid-record stream. The host
// asks for the next slice only after it has checked the preceding one.
void exportRingRange(size_t startRecord, size_t requestedRecords) {
  if (startRecord > validRecordCount) {
    Serial.println(F("ERR+EXPORT,offset"));
    return;
  }

  const size_t availableRecords = validRecordCount - startRecord;
  const size_t recordsToSend = min(requestedRecords, availableRecords);
  Serial.printf("ACK+EXPORT,offset=%u,records=%u\n",
                static_cast<unsigned>(startRecord),
                static_cast<unsigned>(recordsToSend));
  if (!hasRecords) {
    Serial.printf("OK+EXPORT,offset=%u,records=0\n",
                  static_cast<unsigned>(startRecord));
    return;
  }

  const size_t firstIndex = hasWrapped ? writeIndex : 0;
  SampleRecord record{};

  // A full ring has exactly one valid record in every slot, so ordinal-to-slot
  // mapping is direct. This avoids repeatedly scanning 1 MiB for each small
  // acknowledged export packet.
  if (validRecordCount == RECORD_CAPACITY) {
    for (size_t offset = 0; offset < recordsToSend; ++offset) {
      const size_t index = (firstIndex + startRecord + offset) % RECORD_CAPACITY;
      if (!readRecord(index, record) || !isRecordValid(record)) {
        Serial.println(F("ERR+EXPORT,record"));
        return;
      }
      printRecord(F("DATA,"), record);
      Serial.flush();
    }
    Serial.printf("OK+EXPORT,offset=%u,records=%u\n",
                  static_cast<unsigned>(startRecord),
                  static_cast<unsigned>(recordsToSend));
    return;
  }

  size_t validOrdinal = 0;
  size_t exported = 0;

  for (size_t offset = 0; offset < RECORD_CAPACITY; ++offset) {
    const size_t index = (firstIndex + offset) % RECORD_CAPACITY;
    if (readRecord(index, record) && isRecordValid(record)) {
      if (validOrdinal >= startRecord && exported < recordsToSend) {
        printRecord(F("DATA,"), record);
        // One small packet at a time keeps an individual slice recoverable.
        Serial.flush();
        ++exported;
      }
      ++validOrdinal;
      if (exported == recordsToSend) {
        break;
      }
    }
    if ((offset % 32) == 0) {
      yield();
    }
  }

  Serial.printf("OK+EXPORT,offset=%u,records=%u\n",
                static_cast<unsigned>(startRecord),
                static_cast<unsigned>(exported));
}

void handleCommand(const char *command) {
  if (strcmp(command, "AT") == 0) {
    Serial.println(F("OK"));
  } else if (strcmp(command, "AT+STATUS") == 0) {
    printStatus();
  } else if (strcmp(command, "AT+LOG") == 0) {
    Serial.printf("ACK+LOG,enabled=%u\n", liveLogEnabled ? 1 : 0);
    Serial.println(F("OK+LOG"));
  } else if (strcmp(command, "AT+LOG,ON") == 0) {
    liveLogEnabled = true;
    Serial.println(F("ACK+LOG,enabled=1"));
    Serial.println(F("OK+LOG"));
  } else if (strcmp(command, "AT+LOG,OFF") == 0) {
    liveLogEnabled = false;
    Serial.println(F("ACK+LOG,enabled=0"));
    Serial.println(F("OK+LOG"));
  } else if (strcmp(command, "AT+RISK") == 0) {
    Serial.print(F("ACK+RISK,ready="));
    Serial.print(riskReady ? 1 : 0);
    if (riskReady) {
      Serial.print(F(",label="));
      Serial.print(riskLabel(latestRiskClass));
      Serial.print(F(",confidence="));
      Serial.print(latestRiskConfidence, 3);
      Serial.print(F(",voted_high="));
      Serial.print(votedHighRisk ? 1 : 0);
    }
    Serial.println();
    Serial.println(F("OK+RISK"));
  } else if (strcmp(command, "AT+RISK,ON") == 0) {
    riskLogEnabled = true;
    Serial.println(F("OK+RISK,log=1"));
  } else if (strcmp(command, "AT+RISK,OFF") == 0) {
    riskLogEnabled = false;
    Serial.println(F("OK+RISK,log=0"));
  } else if (strcmp(command, "AT+EXPORT") == 0) {
    // Small default export is safe even when typed manually.
    exportRingRange(0, 256);
  } else if (strncmp(command, "AT+EXPORT,", 10) == 0) {
    char *arguments = const_cast<char *>(command + 10);
    char *separator = strchr(arguments, ',');
    if (separator == nullptr) {
      Serial.println(F("ERR+EXPORT,arguments"));
      return;
    }
    *separator = '\0';
    const long startRecord = strtol(arguments, nullptr, 10);
    const long requestedRecords = strtol(separator + 1, nullptr, 10);
    if (startRecord < 0 || requestedRecords <= 0) {
      Serial.println(F("ERR+EXPORT,arguments"));
      return;
    }
    exportRingRange(static_cast<size_t>(startRecord),
                    static_cast<size_t>(requestedRecords));
  } else if (strcmp(command, "AT+DELETE") == 0) {
    deleteArmed = true;
    deleteArmedAtMs = millis();
    Serial.printf("ACK+DELETE,confirm_within_ms=%u\n", DELETE_CONFIRM_WINDOW_MS);
  } else if (strcmp(command, "AT+DELETE,CONFIRM") == 0) {
    const uint32_t elapsedMs = millis() - deleteArmedAtMs;
    if (!deleteArmed || elapsedMs > DELETE_CONFIRM_WINDOW_MS) {
      deleteArmed = false;
      Serial.println(F("ERR+DELETE,not_armed"));
      return;
    }
    deleteArmed = false;
    deleteRing();
  } else {
    Serial.println(F("ERR+COMMAND"));
  }
}

void processSerialCommands() {
  while (Serial.available() > 0) {
    const char received = static_cast<char>(Serial.read());
    if (received == '\r') {
      continue;
    }
    if (received == '\n') {
      commandBuffer[commandLength] = '\0';
      if (commandLength > 0) {
        handleCommand(commandBuffer);
      }
      commandLength = 0;
      continue;
    }
    if (commandLength < sizeof(commandBuffer) - 1) {
      commandBuffer[commandLength++] = received;
    } else {
      commandLength = 0;
      Serial.println(F("ERR+COMMAND,too_long"));
    }
  }
}

void setup() {
  Serial.begin(115200);
  dht.begin();

  ringPartition = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA,
      static_cast<esp_partition_subtype_t>(RING_SUBTYPE), RING_LABEL);
  if (ringPartition == nullptr || ringPartition->size != RING_SIZE_BYTES) {
    Serial.println(F("ERR+FLASH,ring_partition"));
    return;
  }

  if (!ensureRingIsFormatted()) {
    return;
  }

  restoreRingState();
  readDht22();
  if (STOP_WHEN_RING_FULL && hasRecords && validRecordCount == RECORD_CAPACITY) {
    loggingComplete = true;
    Serial.printf("RING_FULL,records=%u\n", static_cast<unsigned>(validRecordCount));
  } else {
    appendSample();
  }
  lastSampleMs = millis();
  lastDhtReadMs = millis();
  Serial.printf("READY,interval_ms=%u,dht_interval_ms=%u,record_bytes=%u,capacity=%u\n",
                RECORD_INTERVAL_MS, DHT_READ_INTERVAL_MS,
                static_cast<unsigned>(RECORD_SIZE),
                static_cast<unsigned>(RECORD_CAPACITY));
}

void loop() {
  processSerialCommands();

  const uint32_t now = millis();
  if (now - lastDhtReadMs >= DHT_READ_INTERVAL_MS) {
    readDht22();
    lastDhtReadMs = now;
  }

  if (!loggingComplete && now - lastSampleMs >= RECORD_INTERVAL_MS) {
    appendSample();
    lastSampleMs = now;
  }
}
