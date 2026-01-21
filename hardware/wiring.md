# Hardware Wiring Guide

## Board: Seeed XIAO ESP32S3

The XIAO ESP32S3 is a compact but powerful board with USB-C, WiFi, and sufficient GPIO for this project.

![XIAO ESP32S3 Pinout](https://files.seeedstudio.com/wiki/SeeedStudio-XIAO-ESP32S3/img/2.png)

## Pin Assignments

### Analog Sensors (ADC)
| Sensor | XIAO Pin | GPIO | Notes |
|--------|----------|------|-------|
| Thermistor | A0 (D0) | GPIO1 | Voltage divider with 10K resistor |
| Piezoelectric | A1 (D1) | GPIO2 | May need signal conditioning |

### I2C Sensors
| Sensor | SDA | SCL | Address |
|--------|-----|-----|---------|
| MAX30102 | D4 (GPIO5) | D5 (GPIO6) | 0x57 |

### Power
| Connection | XIAO Pin |
|------------|----------|
| 3.3V | 3V3 |
| Ground | GND |
| USB Power | 5V (via USB-C) |

## Wiring Diagrams

### Thermistor Circuit (Voltage Divider)
```
    3.3V (XIAO 3V3 pin)
     │
    [10K Fixed Resistor]
     │
     ├──────► A0 / D0 / GPIO1 (ADC input)
     │
    [10K NTC Thermistor]
     │
    GND
```

At room temp (~25°C), you'll read ~1.65V (mid-scale, ~2048 ADC).
Breath will cause small but detectable temperature changes.

### Piezoelectric Sensor Circuit
```
    Piezo Output (+)
         │
         ├──────► A1 / D1 / GPIO2 (ADC input)
         │
        [1M Resistor] (load resistor)
         │
    Piezo Output (-) ──► GND
```

Optional: Add a voltage clamp (two diodes to 3.3V and GND) to protect ESP32.
Optional: Add RC low-pass filter to reduce noise.

### MAX30102 (I2C)
```
    MAX30102        XIAO ESP32S3
    ─────────       ────────────
    VIN ──────────► 3V3
    GND ──────────► GND
    SDA ──────────► D4 (GPIO5)
    SCL ──────────► D5 (GPIO6)
    INT ──────────► (optional)
```

## Breadboard Layout (Visual Reference)
```
XIAO ESP32S3 (mounted on breadboard)
┌─────────────────────────────────────────────────────┐
│                                                     │
│  [3V3]──────┬──────[10K R]──────┬──[Thermistor]──[GND]
│             │                   │
│             │           [A0/D0]─┘
│                                                     │
│  [GND]──────┬──────[1M R]───────┬──[Piezo +]
│             │                   │   [Piezo -]──[GND]
│             │           [A1/D1]─┘
│                                                     │
│  [3V3]──────────────────────────[MAX30102 VIN]
│  [GND]──────────────────────────[MAX30102 GND]
│  [D4]───────────────────────────[MAX30102 SDA]
│  [D5]───────────────────────────[MAX30102 SCL]
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Component List

### Required
- 1x Seeed XIAO ESP32S3
- 1x NTC 10K Thermistor (3950 or similar B-value)
- 1x 10K resistor (1% tolerance preferred)
- 1x Piezoelectric disc/sensor
- 1x 1M resistor
- 1x MAX30102 pulse oximeter module
- Jumper wires
- Breadboard (for prototyping)

### Optional
- MicroSD card module (for standalone logging)
- 3.7V LiPo battery + charging module
- 3D printed enclosure
- Elastic bands/straps for sensor placement

## Sensor Placement Suggestions

### Thermistor
- **Option A**: Small tube directing airflow, thermistor inside
- **Option B**: Mounted on upper lip area (mustache position)
- **Option C**: Inside nostril clip (most sensitive but intrusive)

### Piezoelectric
- **Option A**: Elastic chest strap (chest expansion)
- **Option B**: Belly band (diaphragmatic breathing)
- **Option C**: Adhesive mount on sternum

### Pulse Oximeter
- **Option A**: Finger clip (most accurate)
- **Option B**: Earlobe clip (less intrusive)
- **Option C**: Wrist mount (least accurate but most convenient)

## Calibration Notes

1. **Thermistor**: Record baseline at rest, then calibrate during normal breathing
2. **Piezoelectric**: Sensitivity varies greatly, adjust gain/threshold per individual
3. **MAX30102**: Built-in calibration, but finger pressure affects readings
