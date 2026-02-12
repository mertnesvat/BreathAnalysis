# Current State - Breath Analysis System

*Last updated: 2026-02-11*

## What Works
- **Hardware**: XIAO ESP32S3 soldered prototype with new casing
- **Firmware**: BLE advertising as `BreathMonitor`, 20Hz sensor streaming, deep sleep — builds and uploads clean
- **MAX30102**: ADC range fixed (4096 → 16384), reads ~53K-76K IR with finger, heartbeat visible in raw data
- **Thermistor**: Reads ~1800-2014 at boot (valid range), breath patterns visible in raw data
- **App → Firebase pipeline**: Raw samples uploaded as gzip, Cloud Function triggered
- **BLE deduplication**: FIXED — App-side dedup in `device_provider.dart` + Cloud Function safety net
- **Breath count**: FIXED — 15-second moving average baseline subtraction
- **HRV signal processing**: FIXED — 2.5s moving average DC removal, prominence threshold 0.3

## Previous Issues (All Resolved)
- ~~7x BLE sample duplication~~ — Fixed with dedup
- ~~Breath count wildly wrong (120 vs ~12)~~ — Fixed with baseline subtraction
- ~~HRV metrics all 0~~ — Fixed with proper signal processing
- ~~MAX30102 ADC saturation~~ — Fixed with adcRange=16384
- ~~LED optical crosstalk destroying HRV~~ — Root cause identified (see docs/PowerNoiseAnalysis.md)

## Current Focus: Hardware Casing & Sensor Stability

### New Casing
- Built a new physical casing for the device
- Goal: keep thermistor stable near nostril for full session duration
- First test (1-min) had thermistor wiring issue (ADC reading 0-18) — voltage divider connection was loose
- After fixing connection, firmware boots with `Thermistor: 2014` and `ALL OK`
- **Currently recording a 10-minute test session** with the new casing

### Thermistor Signal Degradation Problem
Analyzed a 15-minute meditation session — key finding: **thermistor signal degrades 10x over 15 minutes**.

| Time Window | Signal Strength (std) | Quality |
|-------------|----------------------|---------|
| 0-3 min     | 20.9                 | Excellent |
| 3-6 min     | 12.2                 | Good |
| 6-9 min     | 9.2                  | Moderate |
| 9-12 min    | 5.3                  | Weak |
| 12-15 min   | 2.5                  | Nearly flat — unreliable |

**Root cause is partially physical sensor drift** (not just thermal):
- Baseline drop paused at minutes 4-7 (inconsistent with pure thermal drift)
- HR spiked during that period (user noticed sensor moving, got anxious)
- Breath amplitude dropped faster than thermal models predict
- See analysis plots in `~/Downloads/15min_*.png`

### Thermistor Type Research
Detailed findings in `docs/ThermistorReplacementFindings.md`. Summary:
- **Glass bead** (τ=0.5-1.5s): Best for breath, but fragile — **keep using this**
- **SMD 0402 chip** (τ≈3s): 2.5x weaker signal but robust — viable backup
- **Metal case / epoxy bead** (τ=5-15s): Too slow for breath detection
- **Recommendation**: Protect glass bead with strain relief (silicone wire + hot glue anchor)

## 15-Minute Session Analysis Results

**Data quality**: Perfect — 17,995 samples, zero gaps, 100% finger contact, no ADC saturation.

### Breath Metrics
- 192 breaths detected, 12.8 br/min average
- Rate progression: 9.0 → 13.7 → 14.0 → 14.3 → 13.0 br/min (5-min windows)
- Regularity: 0.636 (moderate)
- ⚠️ Late-session detection unreliable due to weak signal — overcounting from noise

### HRV Metrics
- 906 heartbeat peaks, 863 valid RR intervals (90.5% coverage)
- Avg HR: 62.9 BPM, stable across session (62-66 BPM in 3-min windows)
- SDNN: 126.8ms, RMSSD: 176.3ms — **suspiciously high**, likely inflated by noise
- 21% of consecutive RR intervals had >300ms jumps (noise contamination)
- True RMSSD probably closer to 40-80ms after proper outlier rejection

### HR Timeline (shows panic event)
- Minutes 0-3: HR drops 67 → 58 BPM (relaxation)
- Minutes 3-5: HR jumps to 65 BPM (noticed sensor moving, anxiety)
- Minutes 5-9: Settles back to ~60 BPM
- Minute 12: Another spike to 73 BPM (noticed sensor again?)

## Next Steps

1. **Analyze 10-min new-casing recording** — verify sensor stability improvement
2. **Build Flutter debug/visualization screen** — real-time display of raw sensor values, thermistor waveform, IR waveform, and BLE connection state, so we can see exactly what's being sent and whether sensors are reacting correctly
3. **Improve HRV outlier rejection** — clean up false peaks inflating RMSSD/SDNN
4. **Add adaptive breath threshold** — prevent overcounting when signal weakens

## Key Files

| File | Location | Role |
|------|----------|------|
| Firmware | `firmware/src/main.cpp` | ESP32 sensor reading + BLE |
| BLE Service | App `lib/services/ble_service.dart` | Receives BLE notifications |
| Device Provider | App `lib/providers/device_provider.dart` | Buffers samples with dedup |
| Storage Service | App `lib/services/breath_storage_service.dart` | Gzip + upload to Firebase |
| Cloud Function | App `functions/src/index.ts` | Breath + HRV analysis |
| Analysis Plots | `~/Downloads/15min_*.png` | Session visualization (4 figures) |
| Thermistor Research | `docs/ThermistorReplacementFindings.md` | Sensor replacement options |
| Noise Analysis | `docs/PowerNoiseAnalysis.md` | LED optical crosstalk investigation |

## Hardware Setup (XIAO ESP32S3)

| Sensor/Component | Pin | Notes |
|-----------------|-----|-------|
| Thermistor (NTC 10K glass bead) | GPIO1 (A0) | 10K voltage divider to 3.3V |
| LED | GPIO3 (D2) | Direct drive, no NPN |
| Power button | GPIO4 (D3) | Internal pullup |
| I2C SDA (MAX30102) | GPIO5 (D4) | |
| I2C SCL (MAX30102) | GPIO6 (D5) | |
| Upload port | `/dev/cu.usbmodem1101` or `…101` | Check with `ls /dev/cu.*` |
