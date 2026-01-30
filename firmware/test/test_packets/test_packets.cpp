/*
 * Unit Tests for Breath Analysis Firmware
 * Tests packet structures and configuration constants
 * Runs on both device (ESP32) and native (computer)
 */

#ifdef ARDUINO
#include <Arduino.h>
#else
#include <stdint.h>
#include <cmath>
#endif
#include <unity.h>

// ============== PACKET STRUCTURES (must match main.cpp) ==============
#pragma pack(push, 1)
struct SensorPacket {
    uint32_t timestamp_ms;
    uint16_t thermistor;
    uint32_t ir_value;
    uint32_t red_value;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct StatusPacket {
    uint8_t battery_percent;
    uint8_t sensors_ok;
    uint8_t is_recording;
    uint8_t reserved;
};
#pragma pack(pop)

// ============== CONFIGURATION CONSTANTS ==============
const int SAMPLE_RATE_HZ = 20;
const int SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;
const int STATUS_INTERVAL_MS = 1000;

const uint8_t CMD_START = 0x01;
const uint8_t CMD_STOP = 0x02;
const uint8_t CMD_PAUSE = 0x03;
const uint8_t CMD_RESUME = 0x04;

// ============== PACKET SIZE TESTS ==============
void test_sensor_packet_size() {
    // SensorPacket must be exactly 14 bytes for BLE MTU compatibility
    TEST_ASSERT_EQUAL(14, sizeof(SensorPacket));
}

void test_status_packet_size() {
    // StatusPacket must be exactly 4 bytes
    TEST_ASSERT_EQUAL(4, sizeof(StatusPacket));
}

void test_sensor_packet_field_offsets() {
    // Verify packed structure has no padding
    SensorPacket packet;
    uint8_t* base = (uint8_t*)&packet;

    TEST_ASSERT_EQUAL(0, (uint8_t*)&packet.timestamp_ms - base);
    TEST_ASSERT_EQUAL(4, (uint8_t*)&packet.thermistor - base);
    TEST_ASSERT_EQUAL(6, (uint8_t*)&packet.ir_value - base);
    TEST_ASSERT_EQUAL(10, (uint8_t*)&packet.red_value - base);
}

void test_status_packet_field_offsets() {
    StatusPacket status;
    uint8_t* base = (uint8_t*)&status;

    TEST_ASSERT_EQUAL(0, (uint8_t*)&status.battery_percent - base);
    TEST_ASSERT_EQUAL(1, (uint8_t*)&status.sensors_ok - base);
    TEST_ASSERT_EQUAL(2, (uint8_t*)&status.is_recording - base);
    TEST_ASSERT_EQUAL(3, (uint8_t*)&status.reserved - base);
}

// ============== CONFIGURATION TESTS ==============
void test_sample_rate_valid() {
    TEST_ASSERT_EQUAL(20, SAMPLE_RATE_HZ);
    TEST_ASSERT_EQUAL(50, SAMPLE_INTERVAL_MS);  // 1000/20 = 50ms
}

void test_status_interval() {
    TEST_ASSERT_EQUAL(1000, STATUS_INTERVAL_MS);
}

void test_command_codes_unique() {
    TEST_ASSERT_NOT_EQUAL(CMD_START, CMD_STOP);
    TEST_ASSERT_NOT_EQUAL(CMD_START, CMD_PAUSE);
    TEST_ASSERT_NOT_EQUAL(CMD_START, CMD_RESUME);
    TEST_ASSERT_NOT_EQUAL(CMD_STOP, CMD_PAUSE);
    TEST_ASSERT_NOT_EQUAL(CMD_STOP, CMD_RESUME);
    TEST_ASSERT_NOT_EQUAL(CMD_PAUSE, CMD_RESUME);
}

void test_command_codes_values() {
    TEST_ASSERT_EQUAL(0x01, CMD_START);
    TEST_ASSERT_EQUAL(0x02, CMD_STOP);
    TEST_ASSERT_EQUAL(0x03, CMD_PAUSE);
    TEST_ASSERT_EQUAL(0x04, CMD_RESUME);
}

// ============== PACKET SERIALIZATION TESTS ==============
void test_sensor_packet_serialization() {
    SensorPacket packet;
    packet.timestamp_ms = 12345;
    packet.thermistor = 2048;
    packet.ir_value = 100000;
    packet.red_value = 80000;

    uint8_t* bytes = (uint8_t*)&packet;

    // Verify little-endian serialization (ESP32 is little-endian)
    // timestamp_ms = 12345 = 0x00003039
    TEST_ASSERT_EQUAL(0x39, bytes[0]);
    TEST_ASSERT_EQUAL(0x30, bytes[1]);
    TEST_ASSERT_EQUAL(0x00, bytes[2]);
    TEST_ASSERT_EQUAL(0x00, bytes[3]);

    // thermistor = 2048 = 0x0800
    TEST_ASSERT_EQUAL(0x00, bytes[4]);
    TEST_ASSERT_EQUAL(0x08, bytes[5]);
}

void test_status_packet_values() {
    StatusPacket status;
    status.battery_percent = 75;
    status.sensors_ok = 1;
    status.is_recording = 2;  // paused
    status.reserved = 0;

    TEST_ASSERT_EQUAL(75, status.battery_percent);
    TEST_ASSERT_EQUAL(1, status.sensors_ok);
    TEST_ASSERT_EQUAL(2, status.is_recording);
}

// ============== THERMISTOR VALIDATION TESTS ==============
void test_thermistor_range_valid() {
    // Valid thermistor readings: 101-3999 (12-bit ADC, excluding boundaries)
    int validReadings[] = {101, 500, 1000, 2000, 3000, 3999};
    for (int i = 0; i < 6; i++) {
        bool isValid = (validReadings[i] > 100 && validReadings[i] < 4000);
        TEST_ASSERT_TRUE(isValid);
    }
}

void test_thermistor_range_invalid() {
    // Invalid readings (sensor disconnected or shorted)
    int invalidReadings[] = {0, 50, 100, 4000, 4095};
    for (int i = 0; i < 5; i++) {
        bool isValid = (invalidReadings[i] > 100 && invalidReadings[i] < 4000);
        TEST_ASSERT_FALSE(isValid);
    }
}

// ============== TEST RUNNER ==============
void run_all_tests() {
    UNITY_BEGIN();

    // Packet structure tests
    RUN_TEST(test_sensor_packet_size);
    RUN_TEST(test_status_packet_size);
    RUN_TEST(test_sensor_packet_field_offsets);
    RUN_TEST(test_status_packet_field_offsets);

    // Configuration tests
    RUN_TEST(test_sample_rate_valid);
    RUN_TEST(test_status_interval);
    RUN_TEST(test_command_codes_unique);
    RUN_TEST(test_command_codes_values);

    // Serialization tests
    RUN_TEST(test_sensor_packet_serialization);
    RUN_TEST(test_status_packet_values);

    // Validation tests
    RUN_TEST(test_thermistor_range_valid);
    RUN_TEST(test_thermistor_range_invalid);

    UNITY_END();
}

#ifdef ARDUINO
// Device mode: Arduino setup/loop
void setup() {
    delay(2000);
    run_all_tests();
}

void loop() {}
#else
// Native mode: standard main()
int main() {
    run_all_tests();
    return 0;
}
#endif
