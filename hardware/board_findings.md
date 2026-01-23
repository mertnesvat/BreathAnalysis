# Board Findings & Compatibility

## Tested Boards

### 1. Seeed XIAO ESP32S3 ✓ RECOMMENDED
- **Status**: Working reliably
- **WiFi**: Stable on both 5GHz (NESS) and 2.4GHz
- **Form Factor**: Tiny (21x17.5mm) - great for wearables
- **USB**: Native USB CDC (no separate chip)
- **Pinout**:
  - Thermistor: A0 (GPIO1)
  - I2C SDA: D4 (GPIO5)
  - I2C SCL: D5 (GPIO6)
- **Notes**: Used for Prototype 1 (glasses wearable). Reliable WiFi.

### 2. ESP32-S3 N16R8 DevKit ✗ NOT RECOMMENDED
- **Status**: Unreliable WiFi
- **WiFi**: Intermittent connection failures (~50% success rate)
- **Issue**: Small PCB antenna, possibly defective batches
- **Pinout Used**:
  - Thermistor: GPIO4
  - I2C SDA: GPIO8
  - I2C SCL: GPIO9
- **Notes**: Tried 2 boards, both had WiFi issues. PSRAM also not detected properly.

### 3. ESP32 WROOM-32 (HW-394) ✓ WORKING
- **Status**: Working reliably - used for Prototype 2
- **WiFi**: Reliable! Connected on first try (large PCB antenna, 2dB gain)
- **USB**: CH340/CP2102 USB-to-Serial
- **Chip**: Original ESP32 (dual-core, not S3)
- **Pinout**:
  - Thermistor: GPIO32 (ADC1_CH4) - works with WiFi
  - I2C SDA: GPIO21 (default)
  - I2C SCL: GPIO22 (default)
  - 3.3V and GND available

## Wiring Diagram for ESP32 WROOM-32

```
                    ESP32 WROOM-32 (HW-394)
                    ═══════════════════════

    LEFT SIDE                              RIGHT SIDE
    ─────────                              ──────────
    [EN]                                   [MOSI/GPIO23]
    [GPIO36/VP] ADC1_CH0                   [GPIO22/SCL] ←── MAX30102 SCL
    [GPIO39/VN] ADC1_CH3                   [GPIO1/TX0]
    [GPIO34] ADC1_CH6                      [GPIO3/RX0]
    [GPIO35] ADC1_CH7                      [GPIO21/SDA] ←── MAX30102 SDA
    [GPIO32] ADC1_CH4 ←── THERMISTOR       [GPIO19/MISO]
    [GPIO33] ADC1_CH5                      [GPIO18/SCK]
    [GPIO25]                               [GPIO5]
    [GPIO26]                               [GPIO17]
    [GPIO27]                               [GPIO16]
    [GPIO14]                               [GPIO4]
    [GPIO12]                               [GPIO2]
    [GPIO13]                               [GPIO15]
    [GND] ←── GND rail                     [GND]
    [VIN]                                  [3V3] ←── 3.3V rail
```

## Thermistor Voltage Divider (same for all boards)

```
        3.3V
         │
    ┌────┴────┐
    │   10K   │  Fixed resistor
    └────┬────┘
         │
         ├──────────→ ADC Pin (GPIO32 on WROOM)
         │
    ┌────┴────┐
    │   NTC   │  Thermistor 10K
    │  10K    │
    └────┬────┘
         │
        GND
```

## WiFi Configuration
- **Network**: NESS_iOt (2.4GHz) - ESP32 only supports 2.4GHz
- **Password**: mugemert2024
- **Note**: 5GHz networks (like NESS) won't work with any ESP32

## Key Learnings
1. ESP32-S3 boards with small antennas have unreliable WiFi
2. Original ESP32 WROOM-32 with large antenna is more reliable
3. XIAO ESP32S3 is exception - good WiFi despite small size (quality antenna design)
4. Always use ADC1 pins when WiFi is active (ADC2 conflicts with WiFi)
