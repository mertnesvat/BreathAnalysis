/*
 * Breath Analysis Firmware
 * BLE Version - For Flutter App Integration
 *
 * Supports:
 *   - Seeed XIAO ESP32S3 (breadboard prototype)
 *   - ESP32 WROOM-32 (PCB prototype)
 *
 * Sensors:
 *   - Thermistor (NTC 10K) - nasal airflow temperature
 *   - MAX30102 on I2C - heart rate & SpO2
 *
 * Communication: BLE (NimBLE) - connects to Flutter meditation app
 * Data rate: 20Hz sensor streaming when recording
 */

#include <Arduino.h>
#include <NimBLEDevice.h>
#include <Wire.h>
#include "MAX30105.h"
#include "driver/rtc_io.h"
#include "esp_sleep.h"
#include <Adafruit_BMP280.h>

// ============== BLE CONFIGURATION ==============
#define DEVICE_NAME "BreathMonitor"
#define FIRMWARE_VERSION "1.0.0"

// BLE UUIDs
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHAR_DEVICE_INFO    "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define CHAR_SENSOR_DATA    "beb5483e-36e1-4688-b7f5-ea07361b26a9"
#define CHAR_CONTROL        "beb5483e-36e1-4688-b7f5-ea07361b26aa"
#define CHAR_STATUS         "beb5483e-36e1-4688-b7f5-ea07361b26ab"

// ============== SAMPLING CONFIGURATION ==============
const int SAMPLE_RATE_HZ = 20;  // 20Hz over BLE (sufficient for breath)
const int SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;
const int STATUS_INTERVAL_MS = 1000;  // Send status every 1 second

// ============== PIN DEFINITIONS ==============
#if defined(CONFIG_IDF_TARGET_ESP32S3)
// Seeed XIAO ESP32S3 pinout
const int PIN_THERMISTOR = 1;   // A0/D0 (GPIO1) - ADC
const int PIN_LED = 3;          // D2 (GPIO3) - direct drive LED
const int PIN_BTN_POWER = 4;   // D3 (GPIO4) - power/sleep button
const int PIN_SDA = 5;         // D4 (GPIO5) - I2C SDA (MAX30102)
const int PIN_SCL = 6;         // D5 (GPIO6) - I2C SCL (MAX30102)
const int PIN_SDA2 = 7;       // D8 (GPIO7) - I2C SDA2 (BME280)
const int PIN_SCL2 = 8;       // D9 (GPIO8) - I2C SCL2 (BME280)
#else
// ESP32 WROOM-32 (HW-394) pinout
const int PIN_THERMISTOR = 32;  // GPIO32 (ADC1_CH4)
const int PIN_LED = 27;         // GPIO27 - LED via NPN transistor
const int PIN_BTN_POWER = 33;   // GPIO33 (RTC_GPIO8) - Power/sleep
const int PIN_SDA = 21;        // Default I2C SDA
const int PIN_SCL = 22;        // Default I2C SCL
#endif

// ============== BLE COMMAND CODES ==============
const uint8_t CMD_START = 0x01;
const uint8_t CMD_STOP = 0x02;
const uint8_t CMD_PAUSE = 0x03;
const uint8_t CMD_RESUME = 0x04;

// ============== GLOBAL OBJECTS ==============
MAX30105 pulseOximeter;
Adafruit_BMP280 bmp280(&Wire1);
NimBLEServer* pServer = nullptr;
NimBLECharacteristic* pSensorDataChar = nullptr;
NimBLECharacteristic* pStatusChar = nullptr;
NimBLECharacteristic* pControlChar = nullptr;
NimBLECharacteristic* pDeviceInfoChar = nullptr;

// ============== STATE VARIABLES ==============
bool deviceConnected = false;
bool oldDeviceConnected = false;
bool isRecording = false;
bool isPaused = false;
unsigned long sessionStartTime = 0;
unsigned long lastSampleTime = 0;
unsigned long lastStatusTime = 0;
unsigned long lastDiagTime = 0;
unsigned long connectTime = 0;  // When BLE connected (for disconnect timing)

// Sensor state
bool sensorsOK = false;
bool max30102OK = false;
bool bme280OK = false;

// BME280 cached values (read at 20Hz for pressure, humidity updates ~1Hz internally)
float cachedTemperature = 0.0;
float cachedHumidity = 0.0;
float cachedPressurePa = 101325.0;
float baselinePressurePa = 0.0;
bool baselineCaptured = false;
unsigned long lastBME280Read = 0;
const unsigned long BME280_INTERVAL_MS = 50;  // 20Hz reads

// LED state
const int LED_FREQ = 200;  // Low freq: easier for caps to filter, still flicker-free
const int LED_RESOLUTION = 8;

// Button state
const unsigned long DEBOUNCE_MS = 50;
const unsigned long LONG_PRESS_MS = 2000;
unsigned long btnPowerPressTime = 0;
bool btnPowerLastState = HIGH;
bool btnPowerPressed = false;

// Breath LED tracking
float thermBaseline = 2000.0;
float thermSmoothed = 2000.0;
const float SMOOTHING = 0.3;
const float SENSITIVITY = 8.0;

// ============== DATA PACKET STRUCTURE ==============
// Binary packet: 20 bytes (fits in BLE MTU, 20-byte payload limit)
#pragma pack(push, 1)
struct SensorPacket {
    uint32_t timestamp_ms;      // 4 bytes
    uint16_t thermistor;        // 2 bytes (ADC 0-4095)
    uint32_t ir_value;          // 4 bytes
    uint32_t red_value;         // 4 bytes
    int16_t  temperature;       // 2 bytes (°C × 100)
    uint16_t humidity;          // 2 bytes (%RH × 100)
    int16_t  pressure_delta;    // 2 bytes (Pa × 100 relative to baseline)
};  // Total: 20 bytes
#pragma pack(pop)

// Status packet: 4 bytes
#pragma pack(push, 1)
struct StatusPacket {
    uint8_t battery_percent;  // 0-100
    uint8_t sensors_ok;       // 1=OK, 0=error
    uint8_t is_recording;     // 1=recording, 0=idle
    uint8_t reserved;         // future use
};
#pragma pack(pop)

// ============== FUNCTION DECLARATIONS ==============
void setupBLE();
void setupSensors();
void setupLED();
void setupButtons();
void updateLED();
void handleButtons();
void updateBME280();
void sendSensorData();
void sendStatus();
void startRecording();
void stopRecording();
void enterDeepSleep();

// Board info for serial output
#if defined(CONFIG_IDF_TARGET_ESP32S3)
#define BOARD_NAME "XIAO ESP32S3"
#else
#define BOARD_NAME "ESP32 WROOM-32"
#endif

// ============== BLE CALLBACKS ==============
class ServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer, ble_gap_conn_desc* desc) {
        deviceConnected = true;
        connectTime = millis();
        Serial.printf("[BLE] Client connected (heap: %u)\n", ESP.getFreeHeap());

        // Request stable connection parameters:
        //   minInterval=12 (15ms), maxInterval=24 (30ms) — fast enough for 20Hz
        //   latency=0 — don't skip any connection events
        //   timeout=400 (4000ms) — tolerate 4s of signal hiccup before disconnect
        pServer->updateConnParams(desc->conn_handle, 12, 24, 0, 400);
        Serial.println("[BLE] Requested conn params: 15-30ms interval, 4s timeout");

        // Visual feedback: quick pulse
        for (int i = 0; i < 2; i++) {
            ledcWrite(PIN_LED, 255);
            delay(100);
            ledcWrite(PIN_LED, 50);
            delay(100);
        }
    }

    void onDisconnect(NimBLEServer* pServer) {
        deviceConnected = false;
        unsigned long connDuration = (millis() - connectTime) / 1000;
        Serial.printf("[BLE] DISCONNECTED after %lus (heap: %u)\n", connDuration, ESP.getFreeHeap());

        // Pause recording on disconnect (not stop)
        if (isRecording) {
            isPaused = true;
            Serial.println("[Session] Paused (BLE disconnected) — waiting for reconnect");
        }

        // Restart advertising immediately
        NimBLEDevice::startAdvertising();
        Serial.println("[BLE] Advertising restarted");
    }
};

class ControlCallbacks : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic* pCharacteristic) {
        std::string value = pCharacteristic->getValue();
        if (value.length() > 0) {
            uint8_t cmd = value[0];
            Serial.printf("[BLE] Command received: 0x%02X\n", cmd);

            switch (cmd) {
                case CMD_START:
                    startRecording();
                    break;
                case CMD_STOP:
                    stopRecording();
                    break;
                case CMD_PAUSE:
                    if (isRecording) {
                        isPaused = true;
                        Serial.println("[Session] Paused");
                    }
                    break;
                case CMD_RESUME:
                    if (isRecording && isPaused) {
                        isPaused = false;
                        Serial.println("[Session] Resumed");
                    }
                    break;
            }
        }
    }
};

// ============== SETUP ==============
void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("\n========================================");
    Serial.println("  Breath Analysis - BLE Edition");
    Serial.println("  Board: " BOARD_NAME);
    Serial.println("  Firmware: " FIRMWARE_VERSION);
    Serial.println("========================================");

    // Log reset reason — crucial for diagnosing battery disconnects
    esp_reset_reason_t reason = esp_reset_reason();
    Serial.printf("[Boot] Reset reason: %d ", reason);
    switch (reason) {
        case ESP_RST_POWERON:  Serial.println("(power on)"); break;
        case ESP_RST_SW:       Serial.println("(software reset)"); break;
        case ESP_RST_PANIC:    Serial.println("(crash/panic)"); break;
        case ESP_RST_INT_WDT:  Serial.println("(interrupt watchdog)"); break;
        case ESP_RST_TASK_WDT: Serial.println("(task watchdog)"); break;
        case ESP_RST_WDT:      Serial.println("(other watchdog)"); break;
        case ESP_RST_DEEPSLEEP:Serial.println("(deep sleep wake)"); break;
        case ESP_RST_BROWNOUT: Serial.println("(BROWNOUT — battery voltage too low!)"); break;
        default:               Serial.println("(unknown)"); break;
    }
    Serial.printf("[Boot] Free heap: %u bytes\n\n", ESP.getFreeHeap());

    Wire.begin(PIN_SDA, PIN_SCL);

    setupLED();
    setupButtons();
    setupSensors();
    setupBLE();

    Serial.println("\n[READY] BLE advertising as: " DEVICE_NAME);
    Serial.println("[READY] Waiting for app connection...\n");
}

// ============== MAIN LOOP ==============
void loop() {
    handleButtons();
    updateLED();

    // Handle reconnection
    if (deviceConnected && !oldDeviceConnected) {
        // Just connected - resume if was paused
        if (isRecording && isPaused) {
            isPaused = false;
            Serial.println("[Session] Resumed (BLE reconnected)");
        }
        oldDeviceConnected = deviceConnected;
    }

    if (!deviceConnected && oldDeviceConnected) {
        oldDeviceConnected = deviceConnected;
    }

    // Send sensor data when recording and connected
    if (isRecording && !isPaused && deviceConnected) {
        unsigned long currentTime = millis();
        if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
            lastSampleTime = currentTime;
            sendSensorData();
        }
    }

    // Send status periodically when connected
    if (deviceConnected) {
        unsigned long currentTime = millis();
        if (currentTime - lastStatusTime >= STATUS_INTERVAL_MS) {
            lastStatusTime = currentTime;
            sendStatus();
        }
    }

    // Diagnostics every 60 seconds — track heap to catch memory leaks
    {
        unsigned long currentTime = millis();
        if (currentTime - lastDiagTime >= 60000) {
            lastDiagTime = currentTime;
            unsigned long uptime = currentTime / 1000;
            Serial.printf("[Diag] uptime=%lus heap=%u conn=%s rec=%s\n",
                uptime, ESP.getFreeHeap(),
                deviceConnected ? "yes" : "no",
                isRecording ? (isPaused ? "paused" : "active") : "idle");
        }
    }
}

// ============== BLE SETUP ==============
void setupBLE() {
    Serial.println("[BLE] Initializing NimBLE...");

    NimBLEDevice::init(DEVICE_NAME);
    NimBLEDevice::setPower(ESP_PWR_LVL_P9);  // Max power for range

    // Create server
    pServer = NimBLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    // Create service
    NimBLEService* pService = pServer->createService(SERVICE_UUID);

    // Device Info characteristic (read-only)
    pDeviceInfoChar = pService->createCharacteristic(
        CHAR_DEVICE_INFO,
        NIMBLE_PROPERTY::READ
    );
    String deviceInfo = String(DEVICE_NAME) + "|" + FIRMWARE_VERSION + "|" + String(SAMPLE_RATE_HZ);
    pDeviceInfoChar->setValue(deviceInfo.c_str());

    // Sensor Data characteristic (notify)
    pSensorDataChar = pService->createCharacteristic(
        CHAR_SENSOR_DATA,
        NIMBLE_PROPERTY::NOTIFY
    );

    // Control characteristic (write)
    pControlChar = pService->createCharacteristic(
        CHAR_CONTROL,
        NIMBLE_PROPERTY::WRITE
    );
    pControlChar->setCallbacks(new ControlCallbacks());

    // Status characteristic (notify)
    pStatusChar = pService->createCharacteristic(
        CHAR_STATUS,
        NIMBLE_PROPERTY::NOTIFY | NIMBLE_PROPERTY::READ
    );

    // Start service
    pService->start();

    // Start advertising
    NimBLEAdvertising* pAdvertising = NimBLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);
    pAdvertising->start();

    Serial.println("[BLE] Service started, advertising...");
}

// ============== LED SETUP ==============
void setupLED() {
    ledcAttach(PIN_LED, LED_FREQ, LED_RESOLUTION);
    ledcWrite(PIN_LED, 0);
    Serial.println("[LED] Breath-reactive LED initialized");
}

// ============== BUTTON SETUP ==============
void setupButtons() {
    pinMode(PIN_BTN_POWER, INPUT_PULLUP);

#if defined(CONFIG_IDF_TARGET_ESP32S3)
    // ESP32-S3: use ext1 wakeup (ext0 not available)
    esp_sleep_enable_ext1_wakeup(1ULL << PIN_BTN_POWER, ESP_EXT1_WAKEUP_ANY_LOW);
    Serial.printf("[Buttons] Power (GPIO%d) ready\n", PIN_BTN_POWER);
#else
    esp_sleep_enable_ext0_wakeup((gpio_num_t)PIN_BTN_POWER, LOW);
    Serial.println("[Buttons] Power (GPIO33) ready");
#endif
}

// ============== SENSOR SETUP ==============
void setupSensors() {
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);

    int thermRaw = analogRead(PIN_THERMISTOR);
    Serial.printf("[Sensor] Thermistor: %d\n", thermRaw);
    bool thermistorOK = (thermRaw > 100 && thermRaw < 4000);

    Serial.print("[Sensor] MAX30102... ");
    if (pulseOximeter.begin(Wire, I2C_SPEED_STANDARD)) {
        Serial.println("OK");
        max30102OK = true;
        // powerLevel=60, sampleAvg=4, ledMode=2(Red+IR), sampleRate=400,
        // pulseWidth=411(18-bit), adcRange=16384(max range, prevents saturation)
        pulseOximeter.setup(60, 4, 2, 400, 411, 16384);
        pulseOximeter.enableDIETEMPRDY();
    } else {
        Serial.println("FAILED");
        max30102OK = false;
    }

    // Initialize BME280 on second I2C bus
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    Wire1.begin(PIN_SDA2, PIN_SCL2);

    // Scan I2C bus to find what's connected
    Serial.println("[I2C] Scanning Wire1 (D8/D9)...");
    int wire1Count = 0;
    for (byte addr = 1; addr < 127; addr++) {
        Wire1.beginTransmission(addr);
        if (Wire1.endTransmission() == 0) {
            wire1Count++;
            Serial.printf("  Found device at 0x%02X\n", addr);
            // Read chip ID register (0xD0) for Bosch sensors
            if (addr == 0x76 || addr == 0x77) {
                Wire1.beginTransmission(addr);
                Wire1.write(0xD0);
                Wire1.endTransmission();
                Wire1.requestFrom(addr, (uint8_t)1);
                if (Wire1.available()) {
                    uint8_t chipId = Wire1.read();
                    Serial.printf("  Chip ID: 0x%02X", chipId);
                    if (chipId == 0x60) Serial.println(" (BME280)");
                    else if (chipId == 0x58) Serial.println(" (BMP280 — no humidity!)");
                    else Serial.printf(" (unknown)\n");
                }
            }
        }
    }
    if (wire1Count == 0) {
        Serial.println("  No devices found on Wire1!");
        // Full scan of primary bus to check if BMP280 ended up there
        Serial.println("[I2C] Full scan of Wire (D4/D5)...");
        for (byte addr = 1; addr < 127; addr++) {
            Wire.beginTransmission(addr);
            if (Wire.endTransmission() == 0) {
                Serial.printf("  Wire: found 0x%02X", addr);
                if (addr == 0x57) Serial.println(" (MAX30102)");
                else if (addr == 0x76 || addr == 0x77) Serial.println(" (BMP280 — wrong bus!)");
                else Serial.printf("\n");
            }
        }
    }

    Serial.print("[Sensor] BMP280... ");
    if (bmp280.begin(0x76) || bmp280.begin(0x77)) {
        Serial.println("OK");
        bme280OK = true;

        // Configure for high-frequency pressure reading
        // Pressure 16x oversample (~0.18 Pa resolution), temp 2x
        bmp280.setSampling(
            Adafruit_BMP280::MODE_NORMAL,
            Adafruit_BMP280::SAMPLING_X2,   // temperature
            Adafruit_BMP280::SAMPLING_X16,  // pressure (max resolution)
            Adafruit_BMP280::FILTER_X4,     // IIR filter for pressure noise
            Adafruit_BMP280::STANDBY_MS_1   // 1ms standby (fastest)
        );

        cachedTemperature = bmp280.readTemperature();
        cachedPressurePa = bmp280.readPressure();

        Serial.printf("  Temp: %.2f°C, P: %.2f hPa\n",
            cachedTemperature, cachedPressurePa / 100.0);
    } else {
        Serial.println("FAILED (check wiring: CSB→VCC, SDO→GND)");
        bme280OK = false;
    }
#endif

    sensorsOK = thermistorOK && max30102OK;
    Serial.printf("[Sensor] Status: %s (BMP280: %s)\n",
        sensorsOK ? "ALL OK" : "ERROR",
        bme280OK ? "OK" : "N/A");
}

// ============== BUTTON HANDLER ==============
void handleButtons() {
    unsigned long now = millis();
    bool powerState = digitalRead(PIN_BTN_POWER);

    // Power button (long press = sleep)
    if (powerState == LOW && btnPowerLastState == HIGH) {
        btnPowerPressTime = now;
        btnPowerPressed = true;
    } else if (powerState == LOW && btnPowerPressed) {
        if (now - btnPowerPressTime >= LONG_PRESS_MS) {
            if (isRecording) {
                Serial.println("[Button] Power long-press IGNORED — recording active");
                // Flash LED to indicate "can't sleep while recording"
                for (int i = 0; i < 4; i++) {
                    ledcWrite(PIN_LED, 255);
                    delay(50);
                    ledcWrite(PIN_LED, 0);
                    delay(50);
                }
                btnPowerPressed = false;  // Reset so it doesn't keep triggering
            } else {
                Serial.println("[Button] Power long-press - sleeping");
                enterDeepSleep();
            }
        }
    } else if (powerState == HIGH && btnPowerLastState == LOW) {
        btnPowerPressed = false;
    }
    btnPowerLastState = powerState;
}

// ============== LED UPDATE ==============
void updateLED() {
    if (!sensorsOK) {
        ledcWrite(PIN_LED, 0);
        return;
    }

    // Not connected: slow pulse to show it's alive
    if (!deviceConnected) {
        float pulse = (sin(millis() * 0.002) + 1.0) * 0.5;  // 0-1, ~0.3Hz
        ledcWrite(PIN_LED, (int)(pulse * 100));  // Max 100 when idle
        return;
    }

    // Connected: breath-reactive
    int thermRaw = analogRead(PIN_THERMISTOR);
    thermSmoothed = (SMOOTHING * thermRaw) + ((1.0 - SMOOTHING) * thermSmoothed);
    thermBaseline = (0.005 * thermSmoothed) + (0.995 * thermBaseline);

    float deviation = thermSmoothed - thermBaseline;
    float amplified = deviation * SENSITIVITY;
    float normalized = 0.5 + (amplified / 100.0);
    normalized = constrain(normalized, 0.0, 1.0);

    float corrected = pow(normalized, 1.8);
    int brightness = (int)(corrected * 255);
    if (brightness < 10) brightness = 10;

    // Recording: add pulse overlay
    if (isRecording && !isPaused) {
        float pulse = sin(millis() * 0.0125) * 30;
        brightness = constrain(brightness + (int)pulse, 10, 255);
    }

    // Paused: fast blink
    if (isPaused) {
        brightness = (millis() % 500 < 250) ? brightness : 20;
    }

    ledcWrite(PIN_LED, brightness);
}

// ============== START RECORDING ==============
void startRecording() {
    if (!sensorsOK) {
        Serial.println("[Session] Cannot start - sensors not OK");
        return;
    }

    isRecording = true;
    isPaused = false;
    sessionStartTime = millis();
    lastSampleTime = sessionStartTime;
    Serial.println("[Session] STARTED");

    // Visual feedback
    for (int b = 0; b <= 255; b += 10) {
        ledcWrite(PIN_LED, b);
        delay(5);
    }
}

// ============== STOP RECORDING ==============
void stopRecording() {
    if (!isRecording) return;

    unsigned long duration = millis() - sessionStartTime;
    isRecording = false;
    isPaused = false;
    Serial.printf("[Session] STOPPED (duration: %lu ms)\n", duration);

    // Visual feedback
    for (int i = 0; i < 3; i++) {
        ledcWrite(PIN_LED, 255);
        delay(100);
        ledcWrite(PIN_LED, 0);
        delay(100);
    }
}

// ============== UPDATE BME280 CACHE ==============
void updateBME280() {
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    if (!bme280OK) return;

    unsigned long now = millis();
    if (now - lastBME280Read < BME280_INTERVAL_MS) return;

    lastBME280Read = now;
    cachedTemperature = bmp280.readTemperature();
    cachedPressurePa = bmp280.readPressure();
    // BMP280 has no humidity — cachedHumidity stays 0

    // Capture baseline pressure from first second of readings
    if (!baselineCaptured && now > 1000) {
        baselinePressurePa = cachedPressurePa;
        baselineCaptured = true;
        Serial.printf("[BMP280] Baseline pressure: %.2f hPa\n", baselinePressurePa / 100.0);
    }
#endif
}

// ============== SEND SENSOR DATA ==============
void sendSensorData() {
    if (!pSensorDataChar) return;

    updateBME280();

    SensorPacket packet;
    packet.timestamp_ms = millis() - sessionStartTime;
    packet.thermistor = (uint16_t)analogRead(PIN_THERMISTOR);
    packet.ir_value = pulseOximeter.getIR();
    packet.red_value = pulseOximeter.getRed();
    packet.temperature = (int16_t)(cachedTemperature * 100.0f);
    packet.humidity = (uint16_t)(cachedHumidity * 100.0f);

    // Pressure delta: (current - baseline) × 100, clamped to int16 range
    float deltaPA = cachedPressurePa - baselinePressurePa;
    float deltaScaled = deltaPA * 100.0f;
    if (deltaScaled > 32767.0f) deltaScaled = 32767.0f;
    if (deltaScaled < -32768.0f) deltaScaled = -32768.0f;
    packet.pressure_delta = (int16_t)deltaScaled;

    pSensorDataChar->setValue((uint8_t*)&packet, sizeof(SensorPacket));
    pSensorDataChar->notify();
}

// ============== SEND STATUS ==============
void sendStatus() {
    if (!pStatusChar) return;

    StatusPacket status;
    status.battery_percent = 100;  // TODO: Read actual battery
    status.sensors_ok = sensorsOK ? 1 : 0;
    status.is_recording = isRecording ? (isPaused ? 2 : 1) : 0;
    status.reserved = 0;

    pStatusChar->setValue((uint8_t*)&status, sizeof(StatusPacket));
    pStatusChar->notify();
}

// ============== DEEP SLEEP ==============
void enterDeepSleep() {
    if (isRecording) {
        stopRecording();
    }

    Serial.println("[Power] Entering deep sleep...");

    // Fade out LED
    for (int b = 255; b >= 0; b -= 5) {
        ledcWrite(PIN_LED, b);
        delay(20);
    }

    // Stop BLE
    NimBLEDevice::deinit(true);

    delay(100);

    // Configure RTC GPIO for wake
    rtc_gpio_pullup_en((gpio_num_t)PIN_BTN_POWER);
    rtc_gpio_pulldown_dis((gpio_num_t)PIN_BTN_POWER);

#if defined(CONFIG_IDF_TARGET_ESP32S3)
    esp_sleep_enable_ext1_wakeup(1ULL << PIN_BTN_POWER, ESP_EXT1_WAKEUP_ANY_LOW);
#else
    esp_sleep_enable_ext0_wakeup((gpio_num_t)PIN_BTN_POWER, LOW);
#endif

    Serial.println("[Power] Sleeping. Press power button to wake.");
    Serial.flush();
    esp_deep_sleep_start();
}
