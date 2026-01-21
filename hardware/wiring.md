# Hardware Wiring Guide

## ESP32 Pin Assignments

### Analog Sensors (ADC)
| Sensor | ESP32 Pin | Notes |
|--------|-----------|-------|
| Thermistor | GPIO34 (ADC1_CH6) | Use voltage divider with 10K resistor |
| Piezoelectric | GPIO35 (ADC1_CH7) | May need signal conditioning circuit |

### I2C Sensors
| Sensor | SDA | SCL | Address |
|--------|-----|-----|---------|
| MAX30102 | GPIO21 | GPIO22 | 0x57 |

### Power
| Connection | Pin |
|------------|-----|
| 3.3V | 3V3 |
| Ground | GND |

## Wiring Diagrams

### Thermistor Circuit (Voltage Divider)
```
    3.3V
     │
    [10K Fixed Resistor]
     │
     ├──────► GPIO34 (ADC input)
     │
    [10K NTC Thermistor]
     │
    GND
```

At room temp (~25°C), you'll read ~1.65V (mid-scale).
Breath will cause small but detectable temperature changes.

### Piezoelectric Sensor Circuit
```
    Piezo Output (+)
         │
         ├──────► GPIO35 (ADC input)
         │
        [1M Resistor] (load resistor)
         │
    Piezo Output (-) ──► GND
```

Optional: Add a voltage clamp (two diodes to 3.3V and GND) to protect ESP32.
Optional: Add RC low-pass filter to reduce noise.

### MAX30102 (I2C)
```
    MAX30102        ESP32
    ─────────       ─────
    VIN ──────────► 3.3V
    GND ──────────► GND
    SDA ──────────► GPIO21
    SCL ──────────► GPIO22
    INT ──────────► (optional, GPIO interrupt)
```

## Component List

### Required
- 1x ESP32 DevKit (any variant with ADC pins)
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
