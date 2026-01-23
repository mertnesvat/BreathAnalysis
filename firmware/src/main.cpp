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

// ============== CONFIGURATION ==============
// WiFi credentials
const char* WIFI_SSID = "NESS_iOt";
const char* WIFI_PASS = "mugemert2024";

// Sampling configuration
const int SAMPLE_RATE_HZ = 50;  // 50 samples per second
const int SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;

// Pin definitions for ESP32 WROOM-32
const int PIN_THERMISTOR = 32;  // GPIO32 (ADC1_CH4) - works with WiFi
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

// ============== FUNCTION DECLARATIONS ==============
void setupWiFi();
void setupSensors();
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

// ============== SENSOR SETUP ==============
void setupSensors() {
    // Configure ADC
    analogReadResolution(12);  // 12-bit ADC (0-4095)
    analogSetAttenuation(ADC_11db);  // Full 0-3.3V range

    // Test thermistor reading
    int thermRaw = analogRead(PIN_THERMISTOR);
    Serial.printf("[Sensor] Thermistor raw: %d (expected ~2000 at room temp)\n", thermRaw);

    // Initialize MAX30102
    Serial.print("[Sensor] Initializing MAX30102... ");
    if (pulseOximeter.begin(Wire, I2C_SPEED_STANDARD)) {
        Serial.println("OK");

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
    }
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
