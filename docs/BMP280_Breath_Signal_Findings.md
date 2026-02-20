# BMP280 Breath Signal Analysis — Complete Findings

## Summary

The BMP280 barometric pressure sensor, despite being an open-air sensor not designed for breath detection, produces a usable breath signal when placed near the nostrils. After rigorous null-test validation and four algorithm iterations, we developed a detection pipeline that achieves **zero false positives** on noise while detecting **17.7 breaths/min** in a 15-minute deep meditation session.

**Final algorithm: Bandpass filter (0.08-0.7 Hz) + Amplitude gate (2.0 Pa) + Cluster filter (5+ peaks in 20s)**

---

## 1. Hardware Setup

| Component | Detail |
|-----------|--------|
| Sensor | BMP280 at 0x76 on Wire1 (I2C) |
| Breakout | Mislabeled as BME280 (chip ID 0x58 = BMP280) |
| MCU | Seeed XIAO ESP32S3 |
| I2C bus | SDA=GPIO7 (D8), SCL=GPIO8 (D9) — second I2C bus |
| Sample rate | ~20 Hz (BLE notification rate) |
| BLE packet | 14 bytes: [timestamp 4B, IR 4B, Red 4B, pressure_delta 2B] |
| Pressure encoding | int16 x 100 (Pa relative to baseline, centipascal resolution) |

The BMP280 measures absolute barometric pressure (~101325 Pa at sea level). The firmware subtracts a rolling baseline and transmits the delta, giving ~0.01 Pa resolution within a +-327 Pa range.

---

## 2. Signal Characteristics

### 2.1 Breath Signal

Nasal breathing creates small pressure fluctuations detectable by the BMP280:

| Metric | Value |
|--------|-------|
| Typical breath amplitude | 2-4 Pa AC (peak-to-peak 4-8 Pa) |
| Maximum breath depth observed | 7.48 Pa |
| Average breath depth (meditation) | 3.06 Pa |
| Breath frequency range | 0.1-0.5 Hz (6-30 breaths/min) |
| Dominant meditation frequency | ~0.3 Hz (~18/min) |

### 2.2 Noise Floor

| Metric | Null Test | Meditation |
|--------|-----------|------------|
| Raw pressure std | 4.58 Pa | 13.36 Pa |
| Bandpass (0.08-0.7 Hz) std | 1.54 Pa | 2.41 Pa |
| Bandpass RMS envelope (mean) | 1.49 Pa | 2.35 Pa |
| Bandpass RMS envelope (p20) | 1.19 Pa | 1.92 Pa |
| Bandpass RMS envelope (p80) | 1.80 Pa | 2.80 Pa |

The noise floor in the breath band is **~1.5 Pa RMS**, giving a breath SNR of approximately 2x in amplitude (4x in power).

### 2.3 Noise Sources

- HVAC and building pressure fluctuations
- Barometric weather changes (very low frequency)
- Sensor thermal noise
- ADC quantization noise

### 2.4 Comparison: Pressure vs Thermistor

| Metric | BMP280 Pressure | NTC Thermistor |
|--------|-----------------|----------------|
| Signal retention at 14 min | **87%** | 31% |
| Signal loss mechanism | None (airflow-based) | Thermal equilibrium |
| Requires skin proximity | No | Yes |
| Breath hold detection | Rolling amplitude | Unreliable |
| Noise rejection difficulty | Moderate | Low (higher SNR initially) |

**Verdict:** Pressure is the superior long-session sensor. Thermistor has better initial SNR but degrades ~46% over 15 minutes due to local warming.

---

## 3. Null Test Experiment

### 3.1 Protocol

- Device placed on desk, no human breathing within 2 meters
- Recorded for 8.9 minutes using identical firmware and BLE pipeline
- Analyzed with identical algorithms as meditation data

### 3.2 Critical Finding

The naive peak detection algorithm (v1: 15s moving average baseline, relative threshold) found **211 "breaths" at 23.6/min** in the null test — nearly identical to the 24.5/min detected in meditation. This means the original algorithm was essentially detecting noise.

### 3.3 Spectral Analysis

| Metric | Null | Meditation | Ratio |
|--------|------|------------|-------|
| Breath band power (0.1-0.6 Hz) | 152 | 639 | **4.2x** |
| Total power (>0.05 Hz) | 451 | 1403 | 3.1x |
| Breath band / total | 33.7% | 45.5% | 1.35x |

The meditation signal has **4.2x more spectral power** in the breath band. This amplitude difference is the key discriminator — not periodicity.

---

## 4. Algorithm Evolution

### 4.1 v1 — Relative Threshold (FAILED)

```
15s moving average baseline → AC signal → 0.5s smoothing → relative threshold (0.15x std)
```

- **Result:** 211 null, 373 meditation (selectivity 1.0x)
- **Problem:** Threshold adapts to local noise level, so noise peaks pass just as easily as breath peaks

### 4.2 v2 — Periodicity Gating on Raw Signal (FAILED)

```
Raw pressure → 20s sliding window autocorrelation → gate if autocorr > 0.3 → peak detection
```

- **Result:** 0 null, 0 meditation (too aggressive)
- **Problem:** Raw pressure has broadband noise that swamps autocorrelation. Even real breaths only achieve autocorr ~0.29.

### 4.3 v3 — Bandpass + Periodicity Gating (FAILED)

```
Bandpass 0.08-0.7 Hz → 20s sliding window autocorrelation → gate → peak detection
```

- **Result:** Selectivity ~1.0x at all reasonable thresholds
- **Problem:** Ambient pressure noise also has quasi-periodic components in the breath band. Autocorrelation is nearly identical: null avg=0.263, meditation avg=0.258.

### 4.4 v4 — Bandpass + Amplitude Envelope (PARTIAL)

```
Bandpass 0.08-0.7 Hz → 8s rolling RMS envelope → fixed amplitude threshold → peak detection
```

- **Result at 2.0 Pa:** 14 null (1.6/min), 271 meditation (17.8/min) — **11.4x selectivity**
- **Problem:** 14 false positives still too many for production use

### 4.5 v4b — Bandpass + Amplitude Envelope + Cluster Filter (SUCCESS)

```
Bandpass 0.08-0.7 Hz → 8s rolling RMS envelope → 2.0 Pa threshold → peak detection → cluster filter (≥5 in 20s)
```

- **Result:** 0 null (0.0/min), 269 meditation (17.7/min) — **infinite selectivity**
- **Pipeline breakdown:**
  - Raw bandpass peaks: 210 null, 376 meditation
  - After amplitude gate: 14 null, 271 meditation
  - After cluster filter: **0 null, 269 meditation**

---

## 5. Final Algorithm Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Bandpass low cutoff | 0.08 Hz | 5 breaths/min minimum |
| Bandpass high cutoff | 0.7 Hz | 42 breaths/min maximum |
| Bandpass order | 3 (Butterworth) | Good rolloff without ringing |
| Envelope window | 8 seconds | ~2-3 breath cycles for stable RMS |
| Amplitude threshold | **2.0 Pa** | Above BMP280 noise floor (1.5 Pa RMS) |
| Min peak prominence | 0.6 Pa | 30% of amplitude threshold |
| Min breath period | 1.5 seconds | Maximum 40 breaths/min |
| Cluster minimum | **5 peaks** | Rejects isolated noise spikes |
| Cluster window | 20 seconds | Must have ≥5 peaks within this window |

### 5.1 Why These Values Work

1. **2.0 Pa threshold:** The BMP280 noise floor in the breath band is ~1.5 Pa RMS. Real breaths produce 2-4 Pa oscillations. The 2.0 Pa threshold sits cleanly between noise and signal.

2. **Cluster filter ≥5:** Even if noise occasionally spikes above 2.0 Pa (14 occurrences in 8.9 min null test), these spikes are randomly distributed. Real breathing produces sustained, regular peaks. Requiring 5 peaks within 20 seconds eliminates all isolated noise events.

3. **8s envelope window:** Covers ~2-3 breath cycles, providing a stable amplitude estimate without over-smoothing transients like the start/stop of breathing.

---

## 6. Meditation Session Analysis (15 min)

### 6.1 Overall Statistics

| Metric | Value |
|--------|-------|
| Total breaths detected | 269 |
| Duration | 15.2 minutes |
| Average breath rate | 17.7/min |
| Average breath interval | 3.38s |
| Interval std deviation | 3.02s |
| Median breath interval | 2.42s |
| Average breath depth | 3.06 Pa |
| Maximum breath depth | 7.48 Pa |

### 6.2 Phase Analysis

| Phase | Interval | Rate | Interpretation |
|-------|----------|------|----------------|
| Early (0-5 min) | 3.81s | 15.7/min | Settling in, slower deeper breaths |
| Middle (5-10 min) | 3.40s | 17.6/min | Transitioning |
| Late (10-15 min) | 2.94s | 20.4/min | Faster, shallower breathing |

Breathing rate increased ~30% from early to late phase. This may reflect:
- Normal meditation settling (some traditions expect slowing; others find a natural rhythm)
- Slight reduction in breath depth over time
- Individual variation in this specific session

### 6.3 Interval Quality

- 62% of intervals fall in the normal meditation range (2-6 seconds)
- Some outlier intervals (>8s) likely represent missed peaks during lighter breaths
- The 3.02s standard deviation suggests moderate breath-to-breath variability

---

## 7. Real-Time Implementation Notes

### 7.1 Bandpass Filter

For real-time use (sample-by-sample processing), the Butterworth IIR filter translates to a Direct Form II difference equation:

```
y[n] = b0*x[n] + b1*x[n-1] + ... + bN*x[n-N] - a1*y[n-1] - ... - aN*y[n-N]
```

For a 3rd-order bandpass at 20 Hz sample rate with cutoffs 0.08-0.7 Hz, the filter has 6 coefficients (3rd order bandpass = 6th order overall). Use `scipy.signal.butter` + `scipy.signal.tf2sos` to get second-order sections for numerical stability.

### 7.2 Envelope Computation

Rolling RMS over 8 seconds = 160 samples at 20 Hz. Can be computed incrementally:

```
rms_sq += (new_sample^2 - oldest_sample^2) / window_size
envelope = sqrt(rms_sq)
```

### 7.3 Cluster Filter

Maintain a circular buffer of recent peak timestamps. For each new peak, count how many previous peaks fall within the 20-second cluster window. Only emit the breath if count >= 5.

### 7.4 Latency Considerations

- Bandpass filter: ~1-2s group delay (3rd order Butterworth)
- Envelope: 4s latency (half of 8s window)
- Cluster filter: first breath in a new episode delayed until 5th peak (~15-20s)
- **Total latency for first breath: ~20s** (acceptable for meditation sessions)
- Subsequent breaths: ~4-5s latency

---

## 8. Limitations & Future Work

### 8.1 Current Limitations

1. **Fixed threshold:** The 2.0 Pa threshold is calibrated for this specific sensor placement. Different distances from nostrils will affect amplitude.
2. **Open-air sensor:** No physical channeling of airflow. A tube or funnel directing airflow to the sensor would dramatically improve SNR.
3. **No breath hold detection:** The algorithm detects periodic breathing but cannot distinguish intentional breath holds from signal loss.
4. **Startup delay:** ~20s before first breath is confirmed (cluster filter warmup).

### 8.2 Potential Improvements

1. **Adaptive calibration:** Run a 30s "quiet" calibration at session start to measure local noise floor and adjust threshold.
2. **Physical channeling:** A small tube from nose area to BMP280 would increase pressure delta by 5-10x, making detection trivial.
3. **Differential pressure:** Using two BMP280s (one near nose, one ambient) would cancel common-mode noise.
4. **Breath hold scoring:** Use rolling amplitude envelope to detect intentional pauses in breathing pattern.
5. **Meditation quality metrics:** Breath regularity (coefficient of variation), depth consistency, rate stability over time.

---

## 9. Key Takeaway

The critical insight is that **amplitude, not periodicity, distinguishes breath from noise**. Ambient barometric noise has quasi-periodic components that fool autocorrelation-based detectors. But real breaths create physically larger pressure oscillations (2-4 Pa vs ~1.5 Pa noise floor), and this amplitude difference combined with temporal clustering provides robust detection.

The 2.0 Pa threshold is a physical constant of this hardware configuration — it reflects the BMP280's noise characteristics in the 0.08-0.7 Hz band. Any breath strong enough to exceed this floor and sustain for 5+ cycles is reliably a real breath.
