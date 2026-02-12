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
#include <SensirionI2cSht4x.h>

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
const int PIN_SDA2 = 7;       // D8 (GPIO7) - I2C SDA2 (SHT40)
const int PIN_SCL2 = 8;       // D9 (GPIO8) - I2C SCL2 (SHT40)
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
SensirionI2cSht4x sht40;
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

// Sensor state
bool sensorsOK = false;
bool max30102OK = false;
bool sht40OK = false;

// SHT40 cached values (read at 1Hz, sent in every 20Hz packet)
float cachedTemperature = 0.0;
float cachedHumidity = 0.0;
unsigned long lastSHT40Read = 0;
const unsigned long SHT40_INTERVAL_MS = 1000;

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
// Binary packet: 18 bytes (fits in BLE MTU, 20-byte payload limit)
#pragma pack(push, 1)
struct SensorPacket {
    uint32_t timestamp_ms;  // 4 bytes
    uint16_t thermistor;    // 2 bytes (ADC 0-4095)
    uint32_t ir_value;      // 4 bytes
    uint32_t red_value;     // 4 bytes
    int16_t  temperature;   // 2 bytes (°C × 100, e.g. 2350 = 23.50°C)
    uint16_t humidity;      // 2 bytes (%RH × 100, e.g. 4520 = 45.20%)
};  // Total: 18 bytes
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
void updateSHT40();
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
    void onConnect(NimBLEServer* pServer) {
        deviceConnected = true;
        Serial.println("[BLE] Client connected");

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
        Serial.println("[BLE] Client disconnected");

        // Pause recording on disconnect (not stop)
        if (isRecording) {
            isPaused = true;
            Serial.println("[Session] Paused (BLE disconnected)");
        }

        // Restart advertising
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
    Serial.println("========================================\n");

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

    // Initialize SHT40 on second I2C bus
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    Serial.print("[Sensor] SHT40... ");
    Wire1.begin(PIN_SDA2, PIN_SCL2);
    sht40.begin(Wire1, SHT40_I2C_ADDR_44);

    float testTemp, testHum;
    uint16_t sht40Error = sht40.measureHighPrecision(testTemp, testHum);
    if (sht40Error == 0) {
        Serial.println("OK");
        sht40OK = true;
        cachedTemperature = testTemp;
        cachedHumidity = testHum;
        Serial.printf("  Temp: %.2f°C, RH: %.2f%%\n", testTemp, testHum);
    } else {
        Serial.printf("FAILED (error: %d)\n", sht40Error);
        sht40OK = false;
    }
#endif

    sensorsOK = thermistorOK && max30102OK;
    Serial.printf("[Sensor] Status: %s (SHT40: %s)\n",
        sensorsOK ? "ALL OK" : "ERROR",
        sht40OK ? "OK" : "N/A");
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
            Serial.println("[Button] Power long-press - sleeping");
            enterDeepSleep();
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

// ============== UPDATE SHT40 CACHE ==============
void updateSHT40() {
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    if (!sht40OK) return;

    unsigned long now = millis();
    if (now - lastSHT40Read < SHT40_INTERVAL_MS) return;

    lastSHT40Read = now;
    float temp, hum;
    uint16_t error = sht40.measureHighPrecision(temp, hum);
    if (error == 0) {
        cachedTemperature = temp;
        cachedHumidity = hum;
    }
#endif
}

// ============== SEND SENSOR DATA ==============
void sendSensorData() {
    if (!pSensorDataChar) return;

    updateSHT40();

    SensorPacket packet;
    packet.timestamp_ms = millis() - sessionStartTime;
    packet.thermistor = (uint16_t)analogRead(PIN_THERMISTOR);
    packet.ir_value = pulseOximeter.getIR();
    packet.red_value = pulseOximeter.getRed();
    packet.temperature = (int16_t)(cachedTemperature * 100.0f);
    packet.humidity = (uint16_t)(cachedHumidity * 100.0f);

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
