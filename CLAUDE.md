# Breath Analysis Project - Claude Context

## Project Overview

A meditation quality measurement system using ESP32 and physiological sensors. Collects breath patterns and HRV data during Vipassana meditation sessions to provide objective feedback on practice quality.

**Core hypothesis:** Breath patterns (rate, variability, depth, transitions) contain valuable information about meditation quality.

## Hardware Setup

### Microcontroller
- **ESP32 WROOM-32** (HW-394 DevKit)
- Dual-core, WiFi/BLE capable
- 12-bit ADC

### Sensors (Current)
| Sensor | Pin | Purpose |
|--------|-----|---------|
| Thermistor (NTC 10K) | GPIO32 (ADC1_CH4) | Nasal airflow temperature (breath detection) |
| MAX30102 | I2C (SDA=21, SCL=22) | Heart rate, SpO2, HRV |
| Status LED | GPIO27 (via NPN transistor) | Breath-reactive feedback |
| Power Button | GPIO33 (RTC_GPIO8) | Sleep/wake control |
| Record Button | GPIO18 | Manual recording toggle |

### Power System
- **18650 battery** with casing
- **TP4056** charging module
- Battery voltage: 3.7-4.2V
- ESP32 onboard regulator provides 3.3V

**Known issue:** MAX30102 may fail to initialize on battery power if voltage is marginal. USB power (5V) works reliably.

## Firmware (`firmware/`)

### Build System
- **PlatformIO** with Arduino framework
- Board: `esp32dev`

### Key Commands
```bash
# Build firmware
pio run

# Upload to device
pio run -t upload

# Run tests on computer (fast, ~1s)
pio test -e native

# Run tests on device (slower, ~30s)
pio test -e esp32dev

# Serial monitor
pio device monitor -b 115200
```

### Dependencies
- `h2zero/NimBLE-Arduino@^1.4.3` - BLE stack
- `sparkfun/SparkFun MAX3010x Pulse and Proximity Sensor Library@^1.1.2` - Pulse oximeter

### BLE Protocol

**Device Name:** `BreathMonitor`

**Service UUID:** `4fafc201-1fb5-459e-8fcc-c5c9c331914b`

| Characteristic | UUID | Properties | Description |
|---------------|------|------------|-------------|
| Device Info | `...26a8` | Read | `name|version|sampleRate` |
| Sensor Data | `...26a9` | Notify | 14-byte binary packet @ 20Hz |
| Control | `...26aa` | Write | Commands: 0x01=start, 0x02=stop, 0x03=pause, 0x04=resume |
| Status | `...26ab` | Notify, Read | 4-byte status packet @ 1Hz |

**Sensor Packet (14 bytes):**
```c
struct SensorPacket {
    uint32_t timestamp_ms;  // 4 bytes - session time
    uint16_t thermistor;    // 2 bytes - ADC 0-4095
    uint32_t ir_value;      // 4 bytes - MAX30102 IR
    uint32_t red_value;     // 4 bytes - MAX30102 Red
};
```

**Status Packet (4 bytes):**
```c
struct StatusPacket {
    uint8_t battery_percent;  // 0-100
    uint8_t sensors_ok;       // 1=OK, 0=error
    uint8_t is_recording;     // 0=idle, 1=recording, 2=paused
    uint8_t reserved;
};
```

### State Machine
- **Idle** → Start command → **Recording**
- **Recording** → Pause/disconnect → **Paused**
- **Paused** → Resume/reconnect → **Recording**
- **Recording/Paused** → Stop command → **Idle**
- Long-press power button (2s) → **Deep Sleep**
- Button press in deep sleep → Wake

### LED Behavior
| State | LED Pattern |
|-------|-------------|
| Sensors error | Off |
| Not connected | Slow breathing pulse (~0.3Hz) |
| Connected, idle | Breath-reactive brightness |
| Recording | Breath-reactive + pulse overlay |
| Paused | Fast blink |

### Deep Sleep
- Wake source: GPIO33 (power button) on LOW
- RTC GPIO pullup enabled for wake detection
- BLE deinit before sleep

## Testing

### Test Structure
```
firmware/test/
├── test_packets/         # Packet structure & config tests (12 tests)
└── test_state_machine/   # Recording state logic tests (15 tests)
```

### Test Strategy
- **Native tests** (`pio test -e native`): Run on computer, fast (~1s), use for development
- **Device tests** (`pio test -e esp32dev`): Run on ESP32, slower (~30s), use before release

Tests cover:
- Packet sizes (critical for BLE MTU compatibility)
- Field offsets (verify no padding)
- Command codes
- State transitions
- Thermistor validation ranges

## Project Structure
```
BreathAnalysis/
├── firmware/           # ESP32 PlatformIO project
│   ├── src/main.cpp   # Main firmware
│   ├── test/          # Unit tests
│   └── platformio.ini
├── hardware/          # Schematics, wiring diagrams
├── processing/        # Python signal processing scripts
├── analysis/          # Jupyter notebooks, visualization
├── data/              # Raw session data
└── docs/              # Additional documentation
```

## Sensor Validation

### Thermistor
- Valid range: ADC 101-3999 (12-bit)
- < 100: likely disconnected/shorted to GND
- > 4000: likely disconnected/shorted to VCC

### MAX30102
- Requires stable power (3.3V-5V depending on breakout board)
- I2C address: 0x57
- If init fails: check I2C wiring, try I2C bus recovery

## Common Issues

1. **MAX30102 fails on battery, works on USB**
   - Battery voltage too low or noisy
   - Add 100µF capacitor near MAX30102
   - Ensure using ESP32's 3.3V regulated output

2. **BLE won't connect**
   - Check if device is advertising (LED breathing pulse)
   - Ensure previous connection is fully disconnected
   - Reset device

3. **Deep sleep won't wake**
   - Verify button wired between GPIO33 and GND
   - Check RTC GPIO pullup is enabled

## Integration with Flutter App

The firmware is designed to work with a Flutter meditation app. The app:
- Scans for `BreathMonitor` BLE device
- Connects and subscribes to sensor/status notifications
- Sends control commands for session management
- Receives and processes 20Hz sensor data stream
