# Breath Analysis for Meditation Quality

A hardware + software project to objectively measure and analyze breath patterns during Vipassana meditation sessions, enabling practitioners to understand and improve their practice.

## Purpose

In meditation practice, particularly Vipassana, there is no objective feedback mechanism to understand:
- Whether a session went well or poorly
- What factors contributed to the quality of the session
- How to systematically improve over time

**"You can't improve what you can't measure."**

This project aims to bridge that gap by collecting physiological data during meditation and developing analysis tools to extract meaningful insights.

## Approach

### Core Hypothesis
Breath patterns contain valuable information about meditation quality:
- **Breath rate** - tends to slow during deeper states
- **Breath variability** - rhythm changes may indicate mental wandering vs. settled attention
- **Breath depth** - shallow vs. deep breathing patterns
- **Transitions** - how breath changes throughout a session

### Important Nuance
Not all "irregularities" indicate poor meditation. For example:
- Returning attention to breath (after wandering) is a **positive** act of mindfulness
- Some breath changes indicate **grounding**, not distraction
- The goal is pattern recognition, not simple metrics

## Hardware

### Microcontroller
- **ESP32** - Dual-core, WiFi/BLE capable, sufficient ADC channels, low power

### Sensors (Phase 1)

| Sensor | Purpose | Placement | Notes |
|--------|---------|-----------|-------|
| **Thermistor (NTC 10K)** | Breath detection via nasal airflow temperature | Near nostrils (headband/nose clip) | Inhale=cool, exhale=warm |
| **Piezoelectric sensor** | Chest/abdominal expansion | Chest strap or belly band | Breath depth & rhythm |
| **MAX30102 (Pulse Oximeter)** | Heart rate variability (HRV) | Fingertip or earlobe | Parasympathetic activation |

### Optional Sensors (Future Phases)

| Sensor | Purpose | Notes |
|--------|---------|-------|
| **Humidity sensor (DHT22/SHT31)** | Secondary breath detection | Slower response than thermistor |
| **GSR (Skin Conductance)** | Stress/relaxation response | Fingertip electrodes |
| **MPU6050 (Accelerometer)** | Movement/stillness detection | Body stability indicator |
| **CO2 sensor (MH-Z19)** | Breath CO2 levels | Metabolic indicator |

## Project Phases

### Phase 1: Data Collection (Current)
- [ ] Hardware assembly and sensor calibration
- [ ] ESP32 firmware for multi-sensor data logging
- [ ] SD card or WiFi data transmission
- [ ] Collect 10+ minute meditation sessions
- [ ] Raw data storage format (CSV/binary)

### Phase 2: Signal Processing
- [ ] Noise filtering and signal conditioning
- [ ] Breath cycle detection algorithm
- [ ] Feature extraction:
  - Breaths per minute (BPM)
  - Breath-to-breath interval variability
  - Inhale/exhale ratio
  - Breath depth estimation
  - HRV metrics (if using pulse oximeter)

### Phase 3: Analysis & Visualization
- [ ] Session visualization dashboard
- [ ] Pattern identification
- [ ] Session comparison tools
- [ ] Correlation with subjective session ratings

### Phase 4: Insights & Feedback (Future)
- [ ] Machine learning for pattern classification
- [ ] Post-session feedback generation
- [ ] Longitudinal progress tracking
- [ ] Real-time subtle feedback (optional, may interfere with practice)

## Data Collection Strategy

### Session Protocol
1. Start recording ~1 minute before meditation begins
2. Record continuously for full session (10-30 minutes)
3. Stop recording ~1 minute after session ends
4. Log subjective session rating (1-10) and notes

### Sampling Rates
- Thermistor: 50-100 Hz (breath detection)
- Piezoelectric: 50-100 Hz (breath depth)
- Pulse oximeter: 25-50 Hz (HRV requires this resolution)

### Data Format
```
timestamp_ms, thermistor_raw, piezo_raw, hr_bpm, spo2, ir_raw, red_raw
```

## Directory Structure

```
BreathAnalysis/
├── README.md                 # This file
├── firmware/                 # ESP32 Arduino/PlatformIO code
│   ├── src/
│   └── platformio.ini
├── hardware/                 # Schematics, PCB designs, wiring diagrams
├── data/                     # Raw meditation session data
│   └── sessions/
├── processing/               # Signal processing scripts (Python)
│   ├── filters.py
│   ├── breath_detection.py
│   └── feature_extraction.py
├── analysis/                 # Analysis and visualization
│   ├── notebooks/           # Jupyter notebooks for exploration
│   └── dashboard/           # Web dashboard (future)
└── docs/                     # Additional documentation
    └── sensor_calibration.md
```

## Technical Considerations

### Challenges
1. **Sensor placement** - Must be unobtrusive to not disturb meditation
2. **Motion artifacts** - Filtering out non-breath signals
3. **Individual variation** - Breath patterns vary greatly between people
4. **Interpretation complexity** - Same pattern may mean different things

### Signal Processing Approach
1. Low-pass filter to remove high-frequency noise
2. Bandpass filter for breath frequency range (0.1-0.5 Hz typical)
3. Peak detection for breath cycle identification
4. Adaptive thresholding for individual calibration

## Resources & References

- Vipassana breath observation technique
- Heart Rate Variability (HRV) and meditation studies
- Respiratory Sinus Arrhythmia (RSA) research
- ESP32 ADC calibration and best practices

## Getting Started

1. Clone this repository
2. See `hardware/` for wiring diagrams
3. Flash ESP32 with firmware from `firmware/`
4. Run calibration procedure
5. Start collecting meditation session data

---

*"The goal is not to control the breath, but to observe it. This project extends that observation beyond the session."*
