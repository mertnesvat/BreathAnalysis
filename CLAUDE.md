# Breath Analysis Project - Claude Context

## Project Overview

A meditation quality measurement system using ESP32 and physiological sensors. Collects breath patterns and HRV data during Vipassana meditation sessions to provide objective feedback on practice quality.

**Core hypothesis:** Breath patterns (rate, variability, depth, transitions) contain valuable information about meditation quality.

## Hardware Setup

### Microcontroller
- **ESP32-WROOM-32E-N4** (SMD module for custom PCB)
- Dual-core, WiFi/BLE capable
- 12-bit ADC

### Sensors (Current)
| Sensor | Pin | Purpose |
|--------|-----|---------|
| Thermistor (NTC 10K) | GPIO32 (ADC1_CH4) | Nasal airflow temperature (breath detection) |
| MAX30102 | I2C (SDA=21, SCL=22) | Heart rate, SpO2, HRV |
| Status LED | GPIO27 (via NPN transistor) | Breath-reactive feedback |

### Control Buttons
| Button | Pin | Function |
|--------|-----|----------|
| RST | EN | Hardware reset (pulls EN low) |
| PWR | GPIO33 (RTC_GPIO8) | Sleep/wake control |
| BOOT | GPIO0 | Bootloader mode (hold during reset to flash) |

### Power System
- **18650 battery** (BH-18650-A1AJ006 holder)
- **TP4056** Li-ion charger IC (1A charge current)
- **AMS1117-3.3** voltage regulator (SOT-89)
- **USB-C** connector (TYPE-C-31-M-12) for charging
- Battery voltage: 3.7-4.2V (VBAT)
- Regulated output: 3.3V

### PCB Power Circuit
```
USB-C (5V) ──┬── TP4056 ── 18650 Battery (3.7-4.2V)
             │                    │
             └── C6 (10µF)        ├── C7 (10µF)
                                  │
                            AMS1117-3.3
                                  │
                            3.3V Rail ── C5 (10µF)
                                  │
                               ESP32
```

### USB-C Configuration
- CC1 & CC2 pins: 5.1kΩ pull-down resistors (R7, R8)
- Identifies device as USB-C sink for 5V power
- VBUS pins (A4B9, B4A9) → TP4056 VCC
- GND pins (A1B12, B1A12) → Common ground

### TP4056 Charging Circuit
| Pin | Connection | Purpose |
|-----|------------|---------|
| VCC (4, 8) | USB VBUS | 5V input |
| BAT (5) | Battery+ | Charge output |
| GND (1, 3, 9) | GND | Ground + thermal pad |
| PROG (2) | R6 (1.2kΩ) to GND | Sets charge current (~1A) |
| CHRG# (7) | LED3 (Red) via R9 | Charging indicator |
| STDBY# (6) | LED5 (Green) via R10 | Charge complete indicator |

### ESP32 Support Circuits
| Component | Value | Connection | Purpose |
|-----------|-------|------------|---------|
| R1 | 10kΩ | EN → 3.3V | EN pull-up |
| R2 | 10kΩ | GPIO33 → 3.3V | PWR button pull-up |
| R3 | 10kΩ | GPIO0 → 3.3V | BOOT button pull-up |
| C1 | 100nF | EN → GND | EN noise filter |

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
- Wake source: GPIO33 (PWR button) on LOW
- External 10kΩ pull-up (R2) keeps GPIO33 HIGH when not pressed
- RTC GPIO configured for wake detection
- BLE deinit before sleep

### Bootloader Mode
- Hold BOOT button → Press RST → Release RST → Release BOOT
- GPIO0 LOW during reset enters download mode for firmware flashing
- Normal boot: GPIO0 HIGH (via R3 pull-up)

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
   - Verify PWR button wired between GPIO33 and GND
   - Check RTC GPIO pullup is enabled (R2 provides external pull-up)

4. **Can't enter bootloader mode**
   - Hold BOOT button, press RST, release RST, then release BOOT
   - GPIO0 must be LOW during EN rising edge

5. **USB-C not recognized as power source**
   - Verify CC1/CC2 have 5.1kΩ pull-down resistors (R7, R8)
   - Check VBUS continuity from USB-C to TP4056

## Integration with Flutter App

The firmware is designed to work with a Flutter meditation app. The app:
- Scans for `BreathMonitor` BLE device
- Connects and subscribes to sensor/status notifications
- Sends control commands for session management
- Receives and processes 20Hz sensor data stream
