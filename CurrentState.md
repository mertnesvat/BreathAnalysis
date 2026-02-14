# Current State - Breath Analysis System

*Last updated: 2026-02-14*

## What Works
- **Hardware**: XIAO ESP32S3 soldered prototype with thermistor + MAX30102 + BMP280
- **Firmware**: BLE advertising as `BreathMonitor`, 20Hz sensor streaming, deep sleep, BLE disconnect diagnostics
- **BMP280**: Pressure + temperature on Wire1 (D8/D9), reads ~996 hPa, 0.18 Pa resolution — **confirmed as primary breath sensor**
- **MAX30102**: ADC range fixed (4096 → 16384), reads ~53K-76K IR with finger
- **Thermistor**: Reads ~1600-1900, breath patterns visible but **degrades to 31% by minute 14** — secondary/backup only
- **BLE packet**: 20 bytes — [ts 4B, therm 2B, IR 4B, Red 4B, temp 2B, hum 2B, pressure_delta 2B]
- **Flutter debug screen**: Real-time waveforms for all signals (thermistor, IR, pressure, humidity, temp)
- **App → Firebase pipeline**: Raw samples uploaded as gzip (v3 format, 7 fields per sample)
- **BLE stability**: Firmware now requests 15-30ms connection interval + 4s supervision timeout, blocks sleep during recording

## Sensor Configuration

| Sensor | I2C Bus | Address | Pin | Signals |
|--------|---------|---------|-----|---------|
| Thermistor NTC 10K | ADC | — | GPIO1 (A0) | Nasal airflow temperature |
| MAX30102 | Wire | 0x57 | GPIO5/6 (D4/D5) | IR + Red (HRV, SpO2) |
| BMP280 | Wire1 | 0x76 | GPIO7/8 (D8/D9) | Pressure + Temperature |

**Note**: The BMP280 breakout was sold as "BME280" but chip ID is 0x58 (BMP280), not 0x60 (BME280). No humidity sensor. Uses `Adafruit BMP280 Library`. CSB must be tied to VCC for I2C mode, SDO to GND for address 0x76.

## Decision: BMP280 Pressure Is the Primary Breath Sensor

Validated across **three recording sessions** totaling 17 minutes of continuous data (7 min + 5 min + 5 min):

### 7-minute all-sensors session (`allSensors7Min.gz`)

| Metric | Thermistor | Pressure (BMP280) | IR (MAX30102) |
|--------|-----------|-------------------|---------------|
| Peaks detected | 73 | 127 | 4 (unusable) |
| Breath rate | 10.4 br/min | 18.0 br/min | — |
| Signal at min 0 | 38.8 ADC | 3.7 Pa | — |
| Signal at min 6 | 12.3 ADC | 2.0 Pa | — |
| **Retained at end** | **32%** | **55%** | — |

- Pressure rate (18.0 br/min = 3.34s/breath) matches expected 3.3s normal breathing
- Threshold sweep (0.10x-0.50x std) gives 124-129 peaks — signal is clean and robust
- 67% of pressure peaks match a thermistor peak (±2s), but pressure catches 40+ extra breaths thermistor misses
- Breath waveform shape consistent from start to end — no sensor degradation
- Temperature drift: +8.94°C over 7 min (23.6→32.5°C) — explains thermistor failure but doesn't affect pressure

### Extended sessions: minutes 5-15 (`5-10MinSnapshot.gz` + `10-15MinSnapshot`)

| Minute | Thermistor Signal | Pressure Signal |
|--------|------------------|----------------|
| 5 (baseline) | 100% | 100% |
| 8 | 34% | 123% |
| 10 | 66% | 118% |
| 12 | 33% | 101% |
| 14 | 31% | 87% |

- Pressure maintains **87% of signal at minute 14** while thermistor drops to **31%**
- Pressure variation (64-170%) reflects actual breathing pattern changes, not sensor degradation
- Sneeze event clearly captured in the data (visible gap in breath pattern)
- Breath hold experiment at minute 12: not a perfectly clean gap due to BMP280 noise floor (~1 Pa), but rolling amplitude visibly dips

### Why pressure works and thermistor fails

- **Thermistor** relies on ΔT between inhaled (cool) and exhaled (warm) air. As the sensor area warms to body temperature (+9°C in 7 min), ΔT vanishes.
- **Pressure** relies on airflow dynamics (Bernoulli effect). Air movement creates pressure changes regardless of temperature. Signal amplitude is physics-based, not temperature-dependent.

### Pressure signal characteristics
- Typical breath amplitude: 2-4 Pa (AC component after baseline removal)
- Noise floor: ~1 Pa (sensor + environmental)
- SNR: 5-8x during normal breathing
- Breath hold detection: possible via rolling amplitude metric, not individual peak detection
- Response time: milliseconds (same as thermistor)

Analysis graphs: `analysis/all_sensors_7min_analysis.png`, `analysis/pressure_deep_dive.png`, `analysis/extended_sessions_5_15min.png`

## BLE Disconnect Investigation

Session dropped at **10 minutes 47 seconds** with `FlutterBluePlusException | fbp-code: 6 | device is not connected`.

Firmware updated with diagnostics:
- **Reset reason logging** at boot — distinguishes brown-out (`ESP_RST_BROWNOUT`) from normal operation
- **Disconnect duration + heap** logged — tracks how long connection survived
- **60-second diagnostic prints** — heap monitoring catches memory leaks
- **Connection parameters**: 15-30ms interval, 0 latency, 4s supervision timeout (was default ~2s)
- **Sleep blocked during recording** — power button long-press ignored while session active

Most likely cause: BLE supervision timeout (signal hiccup > default 2s limit) or phone OS power management. The 4s timeout + faster connection interval should improve stability. Next disconnect will show the exact cause in serial log.

## SHT40 Humidity — Replaced

Previously tested SHT40 humidity sensor (2 sessions analyzed):
- **5-min session** (`5minsHumidity.gz`): Humidity range 59-74%, detected 18 breath peaks (vs thermistor 29)
- Humidity response time τ63 ≈ 4-8s — acts as low-pass filter, merges adjacent breaths
- At 7 br/min (8.5s cycle), SHT40 only captures **22% of actual signal amplitude**
- **Verdict**: Too slow for primary breath detection. Replaced with BMP280 pressure.

## HRV Status — Needs Work

From the 7-min and 15-min sessions:
- IR signal only detected 4 breath-rate peaks in 7 min — unusable for both breath and HRV
- 99.5% finger contact in earlier 15-min session, but only 5% beat detection rate
- IR AC amplitude ~700 counts on ~60K DC level (1.2% AC/DC ratio)
- **Root cause**: Simple peak detection insufficient. Need proper bandpass filter (0.5-5 Hz for HRV, different from breath rate).

## Next Steps

1. **Design enclosure/casing** — better sensor placement for nose proximity, secure fit for long sessions
2. **Record 1-hour session** — validate pressure stability beyond 15 minutes, test BLE reconnection with new firmware
3. **Record 24-hour ambient session** — all-day breathing patterns, understand baseline variability
4. **Improve HRV peak detection** — implement bandpass filter for IR signal
5. **Add real-time feedback UI** — breath detection confidence, signal quality indicators
6. **Tune breath hold detection** — use rolling amplitude (3s window std) instead of individual peak detection

## Previous Issues (All Resolved)
- ~~7x BLE sample duplication~~ — Fixed with dedup in device_provider.dart
- ~~Breath count wildly wrong~~ — Fixed with 15s moving average baseline
- ~~HRV metrics all 0~~ — Fixed with 2.5s DC removal + prominence threshold
- ~~MAX30102 ADC saturation~~ — Fixed with adcRange=16384
- ~~LED optical crosstalk~~ — Root cause identified (see docs/PowerNoiseAnalysis.md)
- ~~SHT40 too slow for breath~~ — Replaced with BMP280 pressure
- ~~BLE disconnect during recording~~ — Added 4s supervision timeout, sleep block, diagnostics
- ~~NimBLE 1.4.3 callback crash~~ — Fixed: use `ble_gap_conn_desc*` not `NimBLEConnInfo&`

## Key Files

| File | Location | Role |
|------|----------|------|
| Firmware | `firmware/src/main.cpp` | ESP32 sensor reading + BLE (20-byte packet) + diagnostics |
| Platform config | `firmware/platformio.ini` | BMP280 + NimBLE + MAX30102 libs |
| Breath data model | App `lib/models/breath_data.dart` | Parses 14/18/20-byte BLE packets |
| Debug screen | App `lib/screens/device_debug_screen.dart` | Real-time waveforms + signal quality |
| Signal metrics | App `lib/utils/signal_metrics.dart` | SignalType enum with pressure support |
| Waveform painter | App `lib/widgets/signal_waveform_painter.dart` | Canvas drawing for all signal types |
| Storage service | App `lib/services/breath_storage_service.dart` | Gzip upload (v3 format, 7 fields) |
| 7-min analysis | `analysis/plot_all_sensors_7min.py` | All sensors comparison (thermistor vs pressure vs IR) |
| Pressure deep dive | `analysis/pressure_deep_dive.py` | Breath intervals, SNR, waveform consistency |
| Extended sessions | `analysis/plot_extended_sessions.py` | Minutes 5-15, breath hold experiment |
| BMP280 analysis | `analysis/plot_bmp280_session.py` | Original pressure vs thermistor comparison |
| Long session analysis | `analysis/analyze_long_session.py` | Minute-by-minute degradation analysis |
| Noise Analysis | `docs/PowerNoiseAnalysis.md` | LED optical crosstalk investigation |
| Thermistor Research | `docs/ThermistorReplacementFindings.md` | Sensor replacement options |

## Hardware Setup (XIAO ESP32S3)

| Sensor/Component | Pin | Notes |
|-----------------|-----|-------|
| Thermistor (NTC 10K glass bead) | GPIO1 (A0) | 10K voltage divider to 3.3V |
| LED | GPIO3 (D2) | Direct drive, no NPN |
| Power button | GPIO4 (D3) | Internal pullup |
| I2C SDA (MAX30102) | GPIO5 (D4) | Wire (default bus) |
| I2C SCL (MAX30102) | GPIO6 (D5) | Wire (default bus) |
| I2C SDA2 (BMP280) | GPIO7 (D8) | Wire1 (second bus) |
| I2C SCL2 (BMP280) | GPIO8 (D9) | Wire1 (second bus) |
| BMP280 CSB | → VCC (3.3V) | Required for I2C mode |
| BMP280 SDO | → GND | Sets address to 0x76 |
| Upload port | `/dev/cu.usbmodem1101` or `…101` | Check with `ls /dev/cu.*` |
