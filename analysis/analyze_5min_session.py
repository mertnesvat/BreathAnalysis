#!/usr/bin/env python3
"""
Comprehensive analysis of 5-minute breath session with SHT40 humidity + HRV.
Analyzes: humidity breath detection, thermistor breath detection, HRV from IR signal.
"""

import gzip
import json
import numpy as np
import sys

# ── Load data ────────────────────────────────────────────────────────────────

def load_session(path):
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    print(f"Format version: {data.get('version', '?')}")
    print(f"Sample rate: {data.get('sampleRate', '?')} Hz")
    print(f"Start: {data.get('startTime', '?')}")
    print(f"End:   {data.get('endTime', '?')}")
    return data

# ── Peak detection (no scipy needed) ─────────────────────────────────────────

def find_peaks(signal, min_distance=10, prominence_threshold=0.0):
    """Simple peak detection: local maxima with minimum distance and prominence."""
    peaks = []
    n = len(signal)
    for i in range(1, n - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            # Check prominence: how much does peak stand above neighbors?
            left_min = min(signal[max(0, i-min_distance):i]) if i > 0 else signal[i]
            right_min = min(signal[i+1:min(n, i+min_distance+1)]) if i < n-1 else signal[i]
            prominence = signal[i] - max(left_min, right_min)
            if prominence >= prominence_threshold:
                peaks.append(i)

    # Enforce minimum distance between peaks
    if len(peaks) < 2:
        return peaks

    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_distance:
            filtered.append(p)
        elif signal[p] > signal[filtered[-1]]:
            filtered[-1] = p  # Keep the taller peak
    return filtered

def moving_average(signal, window):
    """Causal moving average for baseline removal."""
    if len(signal) < window:
        return np.full_like(signal, np.mean(signal))
    kernel = np.ones(window) / window
    # Pad start to avoid edge effects
    padded = np.concatenate([np.full(window-1, signal[0]), signal])
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed

# ── Main analysis ────────────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/Users/mert/Downloads/5minsHumidity.gz'
    data = load_session(path)

    samples = data['samples']
    n = len(samples)
    print(f"\nTotal samples: {n}")

    # Parse arrays: [ts, therm, ir, red, temp, humidity]
    ts = np.array([s[0] for s in samples], dtype=np.float64)
    therm = np.array([s[1] for s in samples], dtype=np.float64)
    ir = np.array([s[2] for s in samples], dtype=np.float64)
    red = np.array([s[3] for s in samples], dtype=np.float64)

    # SHT40 fields (may be null for old firmware)
    has_sht40 = len(samples[0]) >= 6 and samples[0][4] is not None
    if has_sht40:
        temp = np.array([s[4] if s[4] is not None else np.nan for s in samples], dtype=np.float64)
        hum = np.array([s[5] if s[5] is not None else np.nan for s in samples], dtype=np.float64)

    duration_s = (ts[-1] - ts[0]) / 1000.0
    sample_rate = (n - 1) / duration_s

    print(f"Duration: {duration_s:.1f}s ({duration_s/60:.1f} min)")
    print(f"Effective sample rate: {sample_rate:.1f} Hz")

    # Check for gaps
    diffs = np.diff(ts)
    gaps = np.sum(diffs > 100)  # >100ms gaps
    print(f"Timestamp gaps (>100ms): {gaps}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: THERMISTOR BREATH ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("THERMISTOR BREATH ANALYSIS")
    print("="*70)

    print(f"Range: {therm.min():.0f} - {therm.max():.0f} (Δ{therm.max()-therm.min():.0f})")
    print(f"Mean: {therm.mean():.1f}, Std: {therm.std():.1f}")

    # Detrend with 15s moving average (matches firmware algorithm)
    window_15s = int(15 * sample_rate)
    therm_baseline = moving_average(therm, window_15s)
    therm_detrended = therm - therm_baseline

    print(f"Detrended std: {therm_detrended.std():.2f}")

    # Find breath peaks in detrended thermistor
    # Smooth first with 0.5s window
    smooth_window = int(0.5 * sample_rate)
    therm_smooth = moving_average(therm_detrended, max(smooth_window, 3))

    # Peaks should be ~2-8s apart (7.5-30 breaths/min)
    min_peak_dist = int(2.0 * sample_rate)
    therm_std = therm_smooth.std()
    therm_peaks = find_peaks(therm_smooth, min_distance=min_peak_dist,
                             prominence_threshold=therm_std * 0.3)

    if len(therm_peaks) >= 2:
        peak_times = ts[therm_peaks]
        intervals = np.diff(peak_times) / 1000.0  # seconds
        breath_rate = 60.0 / intervals.mean() if intervals.mean() > 0 else 0
        print(f"\nBreath peaks detected: {len(therm_peaks)}")
        print(f"Mean breath interval: {intervals.mean():.2f}s")
        print(f"Breath rate: {breath_rate:.1f} breaths/min")
        print(f"Interval std: {intervals.std():.2f}s (variability)")
        print(f"Interval range: {intervals.min():.2f}s - {intervals.max():.2f}s")
    else:
        print(f"\nInsufficient peaks detected: {len(therm_peaks)}")

    # 30-second windowed analysis
    print("\n--- 30s Windowed Breath Rate (Thermistor) ---")
    window_30s_samples = int(30 * sample_rate)
    for start_idx in range(0, n - window_30s_samples, window_30s_samples):
        end_idx = start_idx + window_30s_samples
        t_start = (ts[start_idx] - ts[0]) / 1000.0
        t_end = (ts[end_idx] - ts[0]) / 1000.0

        window_peaks = [p for p in therm_peaks if start_idx <= p < end_idx]
        if len(window_peaks) >= 2:
            w_intervals = np.diff(ts[window_peaks]) / 1000.0
            w_rate = 60.0 / w_intervals.mean()
            print(f"  {t_start:5.0f}-{t_end:5.0f}s: {w_rate:5.1f} br/min ({len(window_peaks)} peaks)")
        else:
            print(f"  {t_start:5.0f}-{t_end:5.0f}s: -- (insufficient peaks: {len(window_peaks)})")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: HUMIDITY BREATH ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    if has_sht40:
        print("\n" + "="*70)
        print("HUMIDITY (SHT40) BREATH ANALYSIS")
        print("="*70)

        valid_hum = hum[~np.isnan(hum)]
        print(f"Range: {valid_hum.min():.2f}% - {valid_hum.max():.2f}% (Δ{valid_hum.max()-valid_hum.min():.2f}%)")
        print(f"Mean: {valid_hum.mean():.2f}%, Std: {valid_hum.std():.4f}%")

        # Detrend humidity with 15s moving average
        hum_baseline = moving_average(hum, window_15s)
        hum_detrended = hum - hum_baseline
        hum_detrended_std = np.nanstd(hum_detrended)
        print(f"Detrended std: {hum_detrended_std:.4f}%")

        # Smooth with 1s window (humidity is already slow, ~1Hz effective update)
        smooth_1s = int(1.0 * sample_rate)
        hum_smooth = moving_average(hum_detrended, max(smooth_1s, 3))

        # Find peaks in humidity (breath = humidity rise from exhale)
        hum_smooth_std = np.nanstd(hum_smooth)
        min_hum_peak_dist = int(2.0 * sample_rate)
        hum_peaks = find_peaks(hum_smooth, min_distance=min_hum_peak_dist,
                               prominence_threshold=hum_smooth_std * 0.2)

        if len(hum_peaks) >= 2:
            hum_peak_times = ts[hum_peaks]
            hum_intervals = np.diff(hum_peak_times) / 1000.0
            hum_breath_rate = 60.0 / hum_intervals.mean() if hum_intervals.mean() > 0 else 0
            print(f"\nHumidity breath peaks: {len(hum_peaks)}")
            print(f"Mean interval: {hum_intervals.mean():.2f}s")
            print(f"Breath rate: {hum_breath_rate:.1f} breaths/min")
            print(f"Interval std: {hum_intervals.std():.2f}s")
        else:
            print(f"\nInsufficient humidity peaks: {len(hum_peaks)}")
            print("  (Sensor may be too far from nostrils for breath detection)")

        # 30s windowed humidity analysis
        print("\n--- 30s Windowed Breath Rate (Humidity) ---")
        for start_idx in range(0, n - window_30s_samples, window_30s_samples):
            end_idx = start_idx + window_30s_samples
            t_start = (ts[start_idx] - ts[0]) / 1000.0
            t_end = (ts[end_idx] - ts[0]) / 1000.0

            window_peaks = [p for p in hum_peaks if start_idx <= p < end_idx]
            if len(window_peaks) >= 2:
                w_intervals = np.diff(ts[window_peaks]) / 1000.0
                w_rate = 60.0 / w_intervals.mean()
                print(f"  {t_start:5.0f}-{t_end:5.0f}s: {w_rate:5.1f} br/min ({len(window_peaks)} peaks)")
            else:
                print(f"  {t_start:5.0f}-{t_end:5.0f}s: -- (insufficient peaks: {len(window_peaks)})")

        # Correlation between thermistor and humidity breath detection
        if len(therm_peaks) >= 2 and len(hum_peaks) >= 2:
            print(f"\n--- Thermistor vs Humidity Comparison ---")
            therm_br = 60.0 / (np.diff(ts[therm_peaks]) / 1000.0).mean()
            hum_br = 60.0 / (np.diff(ts[hum_peaks]) / 1000.0).mean()
            print(f"Thermistor breath rate: {therm_br:.1f} br/min")
            print(f"Humidity breath rate:   {hum_br:.1f} br/min")
            print(f"Difference: {abs(therm_br - hum_br):.1f} br/min")

        # Temperature analysis
        print("\n" + "="*70)
        print("TEMPERATURE (SHT40) ANALYSIS")
        print("="*70)
        valid_temp = temp[~np.isnan(temp)]
        print(f"Range: {valid_temp.min():.2f}°C - {valid_temp.max():.2f}°C (Δ{valid_temp.max()-valid_temp.min():.2f}°C)")
        print(f"Mean: {valid_temp.mean():.2f}°C, Std: {valid_temp.std():.4f}°C")

        # Check if temperature shows any breath-related signal
        temp_baseline = moving_average(temp, window_15s)
        temp_detrended = temp - temp_baseline
        print(f"Detrended std: {np.nanstd(temp_detrended):.4f}°C")
        print(f"(Temperature response too slow for breath detection at 4-8s lag)")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: HRV ANALYSIS (from IR signal)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("HRV ANALYSIS (MAX30102 IR Signal)")
    print("="*70)

    # Check finger contact
    finger_present = ir > 50000
    contact_pct = np.sum(finger_present) / len(ir) * 100
    print(f"IR range: {ir.min():.0f} - {ir.max():.0f}")
    print(f"Finger contact: {contact_pct:.1f}% of samples (IR > 50000)")

    if contact_pct < 30:
        print("⚠ Insufficient finger contact for HRV analysis")
    else:
        # Use only samples with finger contact
        # Find longest continuous segment with finger on
        segments = []
        seg_start = None
        for i in range(len(finger_present)):
            if finger_present[i] and seg_start is None:
                seg_start = i
            elif not finger_present[i] and seg_start is not None:
                segments.append((seg_start, i))
                seg_start = None
        if seg_start is not None:
            segments.append((seg_start, len(finger_present)))

        # Use longest segment
        if segments:
            longest = max(segments, key=lambda s: s[1] - s[0])
            seg_len = longest[1] - longest[0]
            seg_dur = (ts[longest[1]-1] - ts[longest[0]]) / 1000.0
            print(f"Longest continuous contact: {seg_dur:.1f}s ({seg_len} samples)")

            ir_seg = ir[longest[0]:longest[1]]
            ts_seg = ts[longest[0]:longest[1]]

            # DC removal with 2.5s moving average (matches known good algorithm)
            dc_window = int(2.5 * sample_rate)
            ir_baseline = moving_average(ir_seg, dc_window)
            ir_ac = ir_seg - ir_baseline

            print(f"IR DC level: {ir_baseline.mean():.0f}")
            print(f"IR AC amplitude (std): {ir_ac.std():.1f}")
            print(f"IR AC range: {ir_ac.min():.0f} to {ir_ac.max():.0f}")

            # Smooth AC signal slightly (0.05s) to reduce noise
            smooth_ir = moving_average(ir_ac, max(int(0.05 * sample_rate), 2))

            # Detect heartbeat peaks
            # Heart rate 40-200 BPM = 0.3-1.5s between beats
            min_beat_dist = int(0.35 * sample_rate)  # ~170 BPM max
            ir_ac_std = smooth_ir.std()

            # Use prominence threshold of 0.3 * std (from memory: known good)
            beat_peaks = find_peaks(smooth_ir, min_distance=min_beat_dist,
                                   prominence_threshold=ir_ac_std * 0.3)

            # Also try inverted signal (some sensors have inverted polarity)
            beat_peaks_inv = find_peaks(-smooth_ir, min_distance=min_beat_dist,
                                        prominence_threshold=ir_ac_std * 0.3)

            # Use whichever found more peaks
            if len(beat_peaks_inv) > len(beat_peaks):
                print("(Using inverted IR signal - sensor polarity inverted)")
                beat_peaks = beat_peaks_inv

            print(f"\nHeartbeat peaks detected: {len(beat_peaks)}")

            if len(beat_peaks) >= 3:
                # Calculate RR intervals
                rr_intervals_ms = np.diff(ts_seg[beat_peaks])

                # Filter physiologically plausible RR intervals (300-1500ms = 40-200 BPM)
                valid_rr = rr_intervals_ms[(rr_intervals_ms >= 300) & (rr_intervals_ms <= 1500)]

                print(f"Valid RR intervals: {len(valid_rr)} / {len(rr_intervals_ms)}")

                if len(valid_rr) >= 3:
                    # Heart rate
                    avg_rr = valid_rr.mean()
                    avg_hr = 60000.0 / avg_rr
                    hr_min = 60000.0 / valid_rr.max()
                    hr_max = 60000.0 / valid_rr.min()

                    print(f"\n--- Heart Rate ---")
                    print(f"Average HR: {avg_hr:.1f} BPM")
                    print(f"HR range: {hr_min:.0f} - {hr_max:.0f} BPM")
                    print(f"Mean RR interval: {avg_rr:.1f} ms")

                    # SDNN (standard deviation of NN intervals)
                    sdnn = valid_rr.std()

                    # RMSSD (root mean square of successive differences)
                    successive_diffs = np.diff(valid_rr)
                    rmssd = np.sqrt(np.mean(successive_diffs ** 2))

                    # pNN50 (percentage of successive differences > 50ms)
                    pnn50 = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100

                    print(f"\n--- HRV Metrics ---")
                    print(f"SDNN:  {sdnn:.1f} ms")
                    print(f"RMSSD: {rmssd:.1f} ms")
                    print(f"pNN50: {pnn50:.1f}%")

                    # Interpret HRV
                    print(f"\n--- Interpretation ---")
                    if sdnn > 100:
                        print(f"SDNN {sdnn:.0f}ms: High variability (good parasympathetic tone)")
                    elif sdnn > 50:
                        print(f"SDNN {sdnn:.0f}ms: Normal variability")
                    else:
                        print(f"SDNN {sdnn:.0f}ms: Low variability (may indicate stress or artifact)")

                    if rmssd > 40:
                        print(f"RMSSD {rmssd:.0f}ms: Good vagal tone (typical for meditation)")
                    elif rmssd > 20:
                        print(f"RMSSD {rmssd:.0f}ms: Moderate vagal activity")
                    else:
                        print(f"RMSSD {rmssd:.0f}ms: Low vagal tone (check signal quality)")

                    # RMSSD trend (first half vs second half)
                    half = len(valid_rr) // 2
                    if half >= 3:
                        diffs_1 = np.diff(valid_rr[:half])
                        diffs_2 = np.diff(valid_rr[half:])
                        rmssd_1 = np.sqrt(np.mean(diffs_1 ** 2)) if len(diffs_1) > 0 else 0
                        rmssd_2 = np.sqrt(np.mean(diffs_2 ** 2)) if len(diffs_2) > 0 else 0
                        print(f"\nRMSSD trend: {rmssd_1:.1f}ms → {rmssd_2:.1f}ms", end="")
                        if rmssd_2 > rmssd_1 * 1.1:
                            print(" (↑ improving - relaxation deepening)")
                        elif rmssd_2 < rmssd_1 * 0.9:
                            print(" (↓ decreasing)")
                        else:
                            print(" (→ stable)")

                    # 60s windowed HR
                    print(f"\n--- 60s Windowed Heart Rate ---")
                    for start_idx in range(0, len(ir_seg) - int(60*sample_rate), int(60*sample_rate)):
                        end_idx = start_idx + int(60 * sample_rate)
                        t_start = (ts_seg[start_idx] - ts_seg[0]) / 1000.0
                        t_end = (ts_seg[end_idx] - ts_seg[0]) / 1000.0

                        w_peaks = [p for p in beat_peaks if start_idx <= p < end_idx]
                        if len(w_peaks) >= 3:
                            w_rr = np.diff(ts_seg[w_peaks])
                            w_valid = w_rr[(w_rr >= 300) & (w_rr <= 1500)]
                            if len(w_valid) >= 2:
                                w_hr = 60000.0 / w_valid.mean()
                                w_rmssd = np.sqrt(np.mean(np.diff(w_valid)**2)) if len(w_valid) >= 3 else 0
                                print(f"  {t_start:5.0f}-{t_end:5.0f}s: HR {w_hr:5.1f} BPM, RMSSD {w_rmssd:5.1f}ms ({len(w_peaks)} beats)")
                            else:
                                print(f"  {t_start:5.0f}-{t_end:5.0f}s: -- (no valid RR intervals)")
                        else:
                            print(f"  {t_start:5.0f}-{t_end:5.0f}s: -- (insufficient beats: {len(w_peaks)})")

                else:
                    print("⚠ Too few valid RR intervals for HRV calculation")
            else:
                print("⚠ Too few heartbeat peaks for HRV analysis")
                print(f"   IR AC std: {ir_ac_std:.1f} — may need different threshold")
                print(f"   Try: Is LED causing optical crosstalk? (check memory notes)")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: SIGNAL QUALITY SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SIGNAL QUALITY SUMMARY")
    print("="*70)

    checks = []

    # Timestamp quality
    mono = np.all(np.diff(ts) >= 0)
    checks.append(("Timestamps monotonic", mono))
    checks.append(("No data gaps (>100ms)", gaps == 0))
    checks.append((f"Sample rate stable ({sample_rate:.1f} Hz)", abs(sample_rate - 20) < 2))

    # Thermistor quality
    therm_valid = np.all((therm >= 101) & (therm <= 3999))
    checks.append(("Thermistor in valid range", therm_valid))
    checks.append((f"Thermistor breath signal (std>{10})", therm.std() > 10))

    # Finger contact
    checks.append((f"Finger contact >{50}%", contact_pct > 50))

    # SHT40
    if has_sht40:
        checks.append(("SHT40 data present", True))
        hum_range = valid_hum.max() - valid_hum.min()
        checks.append((f"Humidity range >{1}%", hum_range > 1))

    for label, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {label}")

    passed_count = sum(1 for _, p in checks if p)
    print(f"\n  Score: {passed_count}/{len(checks)} checks passed")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
