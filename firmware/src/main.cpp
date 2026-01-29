/*
 * Breath Analysis Firmware for ESP32 WROOM-32 (HW-394)
 *
 * Prototype 2: Simplified for meditation (no piezo)
 *
 * Sensors:
 *   - Thermistor (NTC 10K) on GPIO32 - nasal airflow temperature
 *   - MAX30102 on I2C (SDA=GPIO21, SCL=GPIO22) - heart rate & SpO2
 *
 * Data is streamed via WebSocket to a Python receiver for recording.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsServer.h>
#include <Wire.h>
#include "MAX30105.h"
#include <ArduinoJson.h>
#include "driver/rtc_io.h"  // For RTC GPIO pull-up during deep sleep

// ============== CONFIGURATION ==============
// WiFi credentials
const char* WIFI_SSID = "NESS_iOt";
const char* WIFI_PASS = "mugemert2024";

// Sampling configuration
const int SAMPLE_RATE_HZ = 50;  // 50 samples per second
const int SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;

// Pin definitions for ESP32 WROOM-32
const int PIN_THERMISTOR = 32;  // GPIO32 (ADC1_CH4) - works with WiFi
const int PIN_LED = 27;         // GPIO27 - status LED (3V COB LED strip via transistor)
const int PIN_BTN_POWER = 33;   // GPIO33 (RTC_GPIO8) - Power/sleep button (long press 2s)
const int PIN_BTN_RECORD = 18;  // GPIO18 - Record toggle button
// I2C uses default pins: SDA=GPIO21, SCL=GPIO22

// Thermistor parameters (for 10K NTC with 10K voltage divider)
const float THERMISTOR_NOMINAL = 10000.0;  // 10K at 25°C
const float TEMP_NOMINAL = 25.0;
const float B_COEFFICIENT = 3950.0;  // Check your thermistor datasheet
const float SERIES_RESISTOR = 10000.0;  // 10K pullup resistor

// ============== GLOBAL OBJECTS ==============
WebSocketsServer webSocket = WebSocketsServer(81);
MAX30105 pulseOximeter;

// Session state
bool isRecording = false;
unsigned long sessionStartTime = 0;
unsigned long lastSampleTime = 0;
int connectedClients = 0;

// Sensor & LED state
bool sensorsOK = false;
bool max30102OK = false;
unsigned long lastLedToggle = 0;
bool ledState = false;

// Breath-reactive LED (PWM)
const int LED_FREQ = 5000;      // PWM frequency
const int LED_RESOLUTION = 8;   // 8-bit (0-255)

// Button state
const unsigned long DEBOUNCE_MS = 50;        // Button debounce time
const unsigned long LONG_PRESS_MS = 2000;    // Long press for power off
unsigned long btnPowerPressTime = 0;
unsigned long btnRecordPressTime = 0;
bool btnPowerLastState = HIGH;               // Pull-up, so HIGH = not pressed
bool btnRecordLastState = HIGH;
bool btnPowerPressed = false;
bool btnRecordPressed = false;

// Thermistor baseline tracking for breath detection
float thermBaseline = 2000.0;   // Running baseline
float thermMin = 2000.0;        // Track min (inhale)
float thermMax = 2000.0;        // Track max (exhale)
float thermSmoothed = 2000.0;   // Smoothed reading
const float SMOOTHING = 0.3;    // Higher = more responsive (0.1-0.5)
const float SENSITIVITY = 8.0;  // Amplify small changes

// ============== FUNCTION DECLARATIONS ==============
void setupWiFi();
void setupSensors();
void setupLED();
void setupButtons();
void updateLED();
void handleButtons();
void toggleRecording();
void enterDeepSleep();
void webSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length);
void sendSensorData();
float readThermistorTemp();

// ============== SETUP ==============
void setup() {
    Serial.begin(115200);
    delay(1000);  // Wait for serial to stabilize

    Serial.println("\n========================================");
    Serial.println("  Breath Analysis - Meditation Monitor");
    Serial.println("  Board: ESP32 WROOM-32 (HW-394)");
    Serial.println("========================================\n");

    // Initialize I2C with default pins (SDA=21, SCL=22)
    Wire.begin();

    setupLED();
    setupButtons();
    setupWiFi();
    setupSensors();

    // Start WebSocket server
    webSocket.begin();
    webSocket.onEvent(webSocketEvent);

    Serial.println("\n[READY] Connect to WebSocket at ws://" + WiFi.localIP().toString() + ":81");
    Serial.println("[READY] Send 'start' to begin recording, 'stop' to end\n");
}

// ============== MAIN LOOP ==============
void loop() {
    webSocket.loop();
    handleButtons();
    updateLED();

    // Only sample when recording and at the correct interval
    if (isRecording) {
        unsigned long currentTime = millis();
        if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
            lastSampleTime = currentTime;
            sendSensorData();
        }
    }
}

// ============== WIFI SETUP ==============
void setupWiFi() {
    Serial.print("[WiFi] Connecting to ");
    Serial.println(WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 60) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Connected!");
        Serial.print("[WiFi] IP Address: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\n[WiFi] Connection FAILED!");
        Serial.println("[WiFi] Check credentials and restart");
    }
}

// ============== LED SETUP ==============
void setupLED() {
    // Setup PWM for breath-reactive LED (ESP32 Arduino 3.x API)
    ledcAttach(PIN_LED, LED_FREQ, LED_RESOLUTION);
    ledcWrite(PIN_LED, 0);
    Serial.println("[LED] Breath-reactive LED on GPIO27 (PWM)");
}

// ============== BUTTON SETUP ==============
void setupButtons() {
    // Use internal pull-ups - buttons connect to GND when pressed
    pinMode(PIN_BTN_POWER, INPUT_PULLUP);
    pinMode(PIN_BTN_RECORD, INPUT_PULLUP);

    // Configure wake-up from deep sleep on power button (GPIO5)
    esp_sleep_enable_ext0_wakeup((gpio_num_t)PIN_BTN_POWER, LOW);

    Serial.println("[Buttons] Power (GPIO5), Record (GPIO18) initialized");
    Serial.println("[Buttons] Power: long press 2s = sleep | Record: press = toggle");
}

// ============== BUTTON HANDLER ==============
void handleButtons() {
    unsigned long now = millis();

    // Read current button states (LOW = pressed due to pull-up)
    bool powerState = digitalRead(PIN_BTN_POWER);
    bool recordState = digitalRead(PIN_BTN_RECORD);

    // ---- POWER BUTTON (long press detection) ----
    if (powerState == LOW && btnPowerLastState == HIGH) {
        // Button just pressed - start timing
        btnPowerPressTime = now;
        btnPowerPressed = true;
    } else if (powerState == LOW && btnPowerPressed) {
        // Button still held - check for long press
        if (now - btnPowerPressTime >= LONG_PRESS_MS) {
            Serial.println("[Button] Power long-press detected - entering deep sleep");
            enterDeepSleep();
        }
    } else if (powerState == HIGH && btnPowerLastState == LOW) {
        // Button released
        btnPowerPressed = false;
    }
    btnPowerLastState = powerState;

    // ---- RECORD BUTTON (single press with debounce) ----
    if (recordState == LOW && btnRecordLastState == HIGH) {
        // Button just pressed - start debounce
        btnRecordPressTime = now;
    } else if (recordState == HIGH && btnRecordLastState == LOW) {
        // Button released - check if it was a valid press (debounced)
        if (now - btnRecordPressTime >= DEBOUNCE_MS) {
            Serial.println("[Button] Record button pressed");
            toggleRecording();
        }
    }
    btnRecordLastState = recordState;
}

// ============== TOGGLE RECORDING ==============
void toggleRecording() {
    if (isRecording) {
        // Stop recording
        isRecording = false;
        unsigned long duration = millis() - sessionStartTime;
        Serial.printf("[Session] Recording STOPPED (duration: %lu ms)\n", duration);

        // Notify WebSocket clients
        JsonDocument doc;
        doc["type"] = "session_end";
        doc["duration_ms"] = duration;
        doc["source"] = "button";
        String json;
        serializeJson(doc, json);
        webSocket.broadcastTXT(json);

        // Visual feedback: quick blink
        for (int i = 0; i < 3; i++) {
            ledcWrite(PIN_LED, 255);
            delay(100);
            ledcWrite(PIN_LED, 0);
            delay(100);
        }
    } else {
        // Start recording
        isRecording = true;
        sessionStartTime = millis();
        lastSampleTime = sessionStartTime;
        Serial.println("[Session] Recording STARTED (via button)");

        // Notify WebSocket clients
        JsonDocument doc;
        doc["type"] = "session_start";
        doc["timestamp"] = sessionStartTime;
        doc["source"] = "button";
        String json;
        serializeJson(doc, json);
        webSocket.broadcastTXT(json);

        // Visual feedback: fade up
        for (int b = 0; b <= 255; b += 5) {
            ledcWrite(PIN_LED, b);
            delay(10);
        }
    }
}

// ============== DEEP SLEEP ==============
void enterDeepSleep() {
    // Stop any recording
    if (isRecording) {
        toggleRecording();
    }

    // Visual feedback: fade out
    Serial.println("[Power] Entering deep sleep...");
    for (int b = 255; b >= 0; b -= 5) {
        ledcWrite(PIN_LED, b);
        delay(20);
    }
    ledcWrite(PIN_LED, 0);

    // Disconnect WiFi cleanly
    webSocket.close();
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);

    delay(100);

    // Configure RTC GPIO for wake-up with pull-up enabled
    rtc_gpio_pullup_en((gpio_num_t)PIN_BTN_POWER);
    rtc_gpio_pulldown_dis((gpio_num_t)PIN_BTN_POWER);

    // Enter deep sleep - will wake on GPIO5 (power button) going LOW
    Serial.println("[Power] Sleeping now. Press power button to wake.");
    esp_deep_sleep_start();
}

// ============== LED UPDATE ==============
void updateLED() {
    if (!sensorsOK) {
        // Sensors not OK: LED OFF
        ledcWrite(PIN_LED, 0);
        return;
    }

    // Read thermistor and smooth it
    int thermRaw = analogRead(PIN_THERMISTOR);
    thermSmoothed = (SMOOTHING * thermRaw) + ((1.0 - SMOOTHING) * thermSmoothed);

    // Slow-moving baseline (adapts over ~10 seconds)
    thermBaseline = (0.005 * thermSmoothed) + (0.995 * thermBaseline);

    // Calculate deviation from baseline
    float deviation = thermSmoothed - thermBaseline;

    // Amplify the deviation for sensitivity
    // Breath typically causes ±10-30 ADC units change
    // We want this to map to full LED range
    float amplified = deviation * SENSITIVITY;

    // Map to 0-1 range centered at 0.5
    // Positive deviation (exhale/warm) = brighter
    // Negative deviation (inhale/cool) = dimmer
    float normalized = 0.5 + (amplified / 100.0);
    normalized = constrain(normalized, 0.0, 1.0);

    // Apply gamma correction for more natural brightness
    float gamma = 1.8;  // Slightly less aggressive gamma
    float corrected = pow(normalized, gamma);

    int brightness = (int)(corrected * 255);

    // Minimum brightness so LED doesn't turn fully off
    if (brightness < 10) brightness = 10;

    // Recording indicator: subtle pulse overlay (2Hz sine wave, ±30 brightness)
    if (isRecording) {
        float pulse = sin(millis() * 0.0125) * 30;  // 2Hz oscillation
        brightness = constrain(brightness + (int)pulse, 10, 255);
    }

    ledcWrite(PIN_LED, brightness);
}

// ============== SENSOR SETUP ==============
void setupSensors() {
    // Configure ADC
    analogReadResolution(12);  // 12-bit ADC (0-4095)
    analogSetAttenuation(ADC_11db);  // Full 0-3.3V range

    // Test thermistor reading
    int thermRaw = analogRead(PIN_THERMISTOR);
    Serial.printf("[Sensor] Thermistor raw: %d (expected ~2000 at room temp)\n", thermRaw);

    // Check if thermistor is connected (should be between 100-4000)
    bool thermistorOK = (thermRaw > 100 && thermRaw < 4000);

    // Initialize MAX30102
    Serial.print("[Sensor] Initializing MAX30102... ");
    if (pulseOximeter.begin(Wire, I2C_SPEED_STANDARD)) {
        Serial.println("OK");
        max30102OK = true;

        // Configure for pulse oximetry
        pulseOximeter.setup(
            60,    // LED brightness (0-255)
            4,     // Sample average (1, 2, 4, 8, 16, 32)
            2,     // LED mode (1=Red, 2=Red+IR, 3=Red+IR+Green)
            400,   // Sample rate (50-3200)
            411,   // Pulse width (69, 118, 215, 411)
            4096   // ADC range (2048, 4096, 8192, 16384)
        );

        pulseOximeter.enableDIETEMPRDY();  // Enable die temperature reading
    } else {
        Serial.println("FAILED - check wiring!");
        max30102OK = false;
    }

    // Set overall sensor status
    sensorsOK = thermistorOK && max30102OK;
    Serial.printf("[Sensor] Status: %s\n", sensorsOK ? "ALL OK - LED ON" : "ISSUES - LED OFF");
}

// ============== WEBSOCKET EVENT HANDLER ==============
void webSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {
        case WStype_DISCONNECTED:
            Serial.printf("[WS] Client #%u disconnected\n", num);
            connectedClients--;
            if (connectedClients == 0 && isRecording) {
                isRecording = false;
                Serial.println("[Session] Auto-stopped (no clients)");
            }
            break;

        case WStype_CONNECTED:
            {
                IPAddress ip = webSocket.remoteIP(num);
                Serial.printf("[WS] Client #%u connected from %s\n", num, ip.toString().c_str());
                connectedClients++;

                // Send welcome message with device info
                JsonDocument doc;
                doc["type"] = "info";
                doc["device"] = "ESP32_WROOM32";
                doc["sample_rate"] = SAMPLE_RATE_HZ;
                doc["sensors"] = "thermistor,max30102";

                String json;
                serializeJson(doc, json);
                webSocket.sendTXT(num, json);
            }
            break;

        case WStype_TEXT:
            {
                String msg = String((char*)payload);
                msg.trim();
                msg.toLowerCase();

                if (msg == "start") {
                    isRecording = true;
                    sessionStartTime = millis();
                    lastSampleTime = sessionStartTime;
                    Serial.println("[Session] Recording STARTED");

                    // Send start confirmation
                    JsonDocument doc;
                    doc["type"] = "session_start";
                    doc["timestamp"] = sessionStartTime;
                    String json;
                    serializeJson(doc, json);
                    webSocket.broadcastTXT(json);

                } else if (msg == "stop") {
                    isRecording = false;
                    unsigned long duration = millis() - sessionStartTime;
                    Serial.printf("[Session] Recording STOPPED (duration: %lu ms)\n", duration);

                    // Send stop confirmation
                    JsonDocument doc;
                    doc["type"] = "session_end";
                    doc["duration_ms"] = duration;
                    String json;
                    serializeJson(doc, json);
                    webSocket.broadcastTXT(json);

                } else if (msg == "ping") {
                    webSocket.sendTXT(num, "{\"type\":\"pong\"}");
                }
            }
            break;
    }
}

// ============== SEND SENSOR DATA ==============
void sendSensorData() {
    // Calculate timestamp relative to session start
    unsigned long timestamp = millis() - sessionStartTime;

    // Read thermistor
    int thermistorRaw = analogRead(PIN_THERMISTOR);

    // Read MAX30102
    uint32_t irValue = pulseOximeter.getIR();
    uint32_t redValue = pulseOximeter.getRed();

    // Build JSON message
    JsonDocument doc;
    doc["t"] = timestamp;           // timestamp in ms
    doc["th"] = thermistorRaw;      // thermistor raw ADC (0-4095)
    doc["ir"] = irValue;            // MAX30102 IR value
    doc["rd"] = redValue;           // MAX30102 Red value

    String json;
    serializeJson(doc, json);
    webSocket.broadcastTXT(json);
}

// ============== THERMISTOR TEMPERATURE CALCULATION ==============
float readThermistorTemp() {
    int raw = analogRead(PIN_THERMISTOR);

    // Convert ADC value to resistance
    float resistance = SERIES_RESISTOR / ((4095.0 / raw) - 1.0);

    // Steinhart-Hart equation (simplified B-parameter version)
    float steinhart = resistance / THERMISTOR_NOMINAL;     // (R/Ro)
    steinhart = log(steinhart);                            // ln(R/Ro)
    steinhart /= B_COEFFICIENT;                            // 1/B * ln(R/Ro)
    steinhart += 1.0 / (TEMP_NOMINAL + 273.15);           // + (1/To)
    steinhart = 1.0 / steinhart;                          // Invert
    steinhart -= 273.15;                                   // Convert to Celsius

    return steinhart;
}
