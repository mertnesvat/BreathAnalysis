#!/usr/bin/env python3
"""
Long session analysis — focus on thermistor degradation over time,
humidity signal comparison, and HRV analysis.
"""

import gzip
import json
import numpy as np
import sys

# ── Utilities ────────────────────────────────────────────────────────────────

def moving_average(signal, window):
    if len(signal) < window:
        return np.full_like(signal, np.mean(signal))
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window-1, signal[0]), signal])
    return np.convolve(padded, kernel, mode='valid')

def find_peaks(signal, min_distance=10, prominence_threshold=0.0):
    peaks = []
    n = len(signal)
    for i in range(1, n - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            left_min = min(signal[max(0, i-min_distance):i]) if i > 0 else signal[i]
            right_min = min(signal[i+1:min(n, i+min_distance+1)]) if i < n-1 else signal[i]
            prominence = signal[i] - max(left_min, right_min)
            if prominence >= prominence_threshold:
                peaks.append(i)
    if len(peaks) < 2:
        return peaks
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_distance:
            filtered.append(p)
        elif signal[p] > signal[filtered[-1]]:
            filtered[-1] = p
    return filtered

# ── Load ─────────────────────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/Users/mert/Downloads/superLongSessionWithSHT40.gz'
    with gzip.open(path, 'rt') as f:
        data = json.load(f)

    samples = data['samples']
    n = len(samples)

    ts = np.array([s[0] for s in samples], dtype=np.float64)
    therm = np.array([s[1] for s in samples], dtype=np.float64)
    ir = np.array([s[2] for s in samples], dtype=np.float64)
    red = np.array([s[3] for s in samples], dtype=np.float64)
    temp = np.array([s[4] if s[4] is not None else np.nan for s in samples], dtype=np.float64)
    hum = np.array([s[5] if s[5] is not None else np.nan for s in samples], dtype=np.float64)

    duration_s = (ts[-1] - ts[0]) / 1000.0
    sr = (n - 1) / duration_s

    print("=" * 70)
    print(f"LONG SESSION ANALYSIS — {duration_s/60:.1f} minutes")
    print("=" * 70)
    print(f"Samples: {n}, Duration: {duration_s:.1f}s, Rate: {sr:.1f} Hz")
    print(f"Gaps (>100ms): {np.sum(np.diff(ts) > 100)}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: THERMISTOR DEGRADATION OVER TIME
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("THERMISTOR — SIGNAL DEGRADATION ANALYSIS")
    print("=" * 70)
    print(f"Overall range: {therm.min():.0f} - {therm.max():.0f} (Δ{therm.max()-therm.min():.0f})")
    print(f"Overall mean: {therm.mean():.1f}, std: {therm.std():.1f}")

    # 15s baseline removal
    window_15s = int(15 * sr)
    therm_baseline = moving_average(therm, window_15s)
    therm_ac = therm - therm_baseline

    # Breath detection
    smooth_w = int(0.5 * sr)
    therm_smooth = moving_average(therm_ac, max(smooth_w, 3))
    min_peak_dist = int(2.0 * sr)

    # ── Minute-by-minute thermistor analysis ──
    print(f"\n{'Min':>4} {'Mean':>7} {'Std':>7} {'AC Std':>7} {'Peaks':>6} {'BR/min':>7} {'Signal':>10}")
    print("-" * 58)

    minute_samples = int(60 * sr)
    minute_therm_stds = []
    minute_therm_ac_stds = []
    minute_breath_rates = []

    for m in range(int(duration_s // 60)):
        s_start = m * minute_samples
        s_end = min((m + 1) * minute_samples, n)
        if s_end - s_start < minute_samples * 0.5:
            break

        seg_therm = therm[s_start:s_end]
        seg_ac = therm_ac[s_start:s_end]
        seg_smooth = therm_smooth[s_start:s_end]
        seg_ts = ts[s_start:s_end]

        raw_std = seg_therm.std()
        ac_std = seg_ac.std()
        minute_therm_stds.append(raw_std)
        minute_therm_ac_stds.append(ac_std)

        # Find peaks in this minute
        seg_peaks = find_peaks(seg_smooth, min_distance=min_peak_dist,
                               prominence_threshold=ac_std * 0.3)
        n_peaks = len(seg_peaks)

        if n_peaks >= 2:
            intervals = np.diff(seg_ts[seg_peaks]) / 1000.0
            br = 60.0 / intervals.mean()
            minute_breath_rates.append(br)
        else:
            br = 0
            minute_breath_rates.append(0)

        # Signal quality bar
        bar_len = int(ac_std / 5)  # Scale: 5 ADC units per bar char
        bar = "█" * min(bar_len, 20)
        if not bar:
            bar = "▁"

        status = f"{m:4d} {seg_therm.mean():7.0f} {raw_std:7.1f} {ac_std:7.1f} {n_peaks:6d} {br:7.1f} {bar}"
        print(status)

    # Degradation summary
    if len(minute_therm_ac_stds) >= 4:
        first_3 = np.mean(minute_therm_ac_stds[:3])
        last_3 = np.mean(minute_therm_ac_stds[-3:])
        degradation = (1 - last_3 / first_3) * 100 if first_3 > 0 else 0
        print(f"\nThermistor AC signal strength:")
        print(f"  First 3 min avg std: {first_3:.1f}")
        print(f"  Last 3 min avg std:  {last_3:.1f}")
        print(f"  Signal degradation:  {degradation:.0f}%")
        if degradation > 30:
            print(f"  ⚠ SIGNIFICANT DEGRADATION — thermistor losing sensitivity")
        elif degradation > 10:
            print(f"  ⚡ Moderate degradation — still usable but weakening")
        else:
            print(f"  ✓ Minimal degradation")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: HUMIDITY SIGNAL OVER TIME
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("HUMIDITY (SHT40) — SIGNAL STABILITY ANALYSIS")
    print("=" * 70)
    print(f"Overall range: {np.nanmin(hum):.2f}% - {np.nanmax(hum):.2f}% (Δ{np.nanmax(hum)-np.nanmin(hum):.2f}%)")
    print(f"Overall mean: {np.nanmean(hum):.2f}%, std: {np.nanstd(hum):.4f}%")

    hum_baseline = moving_average(hum, window_15s)
    hum_ac = hum - hum_baseline
    hum_smooth = moving_average(hum_ac, max(int(1.0 * sr), 3))

    print(f"\n{'Min':>4} {'Mean%':>7} {'Std%':>8} {'AC Std%':>8} {'Peaks':>6} {'BR/min':>7} {'Signal':>10}")
    print("-" * 62)

    minute_hum_ac_stds = []
    minute_hum_rates = []

    for m in range(int(duration_s // 60)):
        s_start = m * minute_samples
        s_end = min((m + 1) * minute_samples, n)
        if s_end - s_start < minute_samples * 0.5:
            break

        seg_hum = hum[s_start:s_end]
        seg_hum_ac = hum_ac[s_start:s_end]
        seg_hum_smooth = hum_smooth[s_start:s_end]
        seg_ts = ts[s_start:s_end]

        raw_std = np.nanstd(seg_hum)
        ac_std = np.nanstd(seg_hum_ac)
        minute_hum_ac_stds.append(ac_std)

        seg_peaks = find_peaks(seg_hum_smooth, min_distance=min_peak_dist,
                               prominence_threshold=ac_std * 0.2)
        n_peaks = len(seg_peaks)

        if n_peaks >= 2:
            intervals = np.diff(seg_ts[seg_peaks]) / 1000.0
            br = 60.0 / intervals.mean()
            minute_hum_rates.append(br)
        else:
            br = 0
            minute_hum_rates.append(0)

        bar_len = int(ac_std * 10)  # Scale for humidity
        bar = "█" * min(bar_len, 20)
        if not bar:
            bar = "▁"

        print(f"{m:4d} {np.nanmean(seg_hum):7.2f} {raw_std:8.4f} {ac_std:8.4f} {n_peaks:6d} {br:7.1f} {bar}")

    # Humidity stability
    if len(minute_hum_ac_stds) >= 4:
        first_3 = np.mean(minute_hum_ac_stds[:3])
        last_3 = np.mean(minute_hum_ac_stds[-3:])
        change = ((last_3 / first_3) - 1) * 100 if first_3 > 0 else 0
        print(f"\nHumidity AC signal strength:")
        print(f"  First 3 min avg std: {first_3:.4f}%")
        print(f"  Last 3 min avg std:  {last_3:.4f}%")
        print(f"  Change: {change:+.0f}%")
        if abs(change) < 20:
            print(f"  ✓ Signal stable over time (no equilibrium problem)")
        else:
            print(f"  Signal {'strengthened' if change > 0 else 'weakened'} over session")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: THERMISTOR vs HUMIDITY COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("THERMISTOR vs HUMIDITY — MINUTE-BY-MINUTE COMPARISON")
    print("=" * 70)
    print(f"{'Min':>4} {'Therm BR':>9} {'Hum BR':>9} {'Therm AC':>9} {'Hum AC':>9} {'Winner':>10}")
    print("-" * 56)

    for m in range(min(len(minute_breath_rates), len(minute_hum_rates))):
        t_br = minute_breath_rates[m]
        h_br = minute_hum_rates[m]
        t_ac = minute_therm_ac_stds[m]
        h_ac = minute_hum_ac_stds[m]

        # Determine which sensor is more reliable this minute
        # (has more detected peaks / nonzero breath rate)
        if t_br > 0 and h_br > 0:
            winner = "Both"
        elif t_br > 0:
            winner = "Therm"
        elif h_br > 0:
            winner = "Humid"
        else:
            winner = "Neither"

        print(f"{m:4d} {t_br:9.1f} {h_br:9.1f} {t_ac:9.1f} {h_ac:9.4f} {winner:>10}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: TEMPERATURE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEMPERATURE (SHT40)")
    print("=" * 70)
    print(f"Range: {np.nanmin(temp):.2f}°C - {np.nanmax(temp):.2f}°C (Δ{np.nanmax(temp)-np.nanmin(temp):.2f}°C)")
    print(f"Start: {np.nanmean(temp[:int(30*sr)]):.2f}°C, End: {np.nanmean(temp[-int(30*sr):]):.2f}°C")
    print(f"Drift: {np.nanmean(temp[-int(30*sr):]) - np.nanmean(temp[:int(30*sr)]):.2f}°C over {duration_s/60:.0f} min")
    print("(Confirms local warming that kills thermistor signal)")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: HRV ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("HRV ANALYSIS (MAX30102 IR)")
    print("=" * 70)

    finger_present = ir > 50000
    contact_pct = np.sum(finger_present) / len(ir) * 100
    print(f"IR range: {ir.min():.0f} - {ir.max():.0f}")
    print(f"Finger contact: {contact_pct:.1f}%")

    if contact_pct < 10:
        print("⚠ Insufficient finger contact for HRV")
    else:
        # Find all continuous segments with finger contact
        segments = []
        seg_start = None
        for i in range(len(finger_present)):
            if finger_present[i] and seg_start is None:
                seg_start = i
            elif not finger_present[i] and seg_start is not None:
                if i - seg_start > int(5 * sr):  # Min 5s segments
                    segments.append((seg_start, i))
                seg_start = None
        if seg_start is not None and len(finger_present) - seg_start > int(5 * sr):
            segments.append((seg_start, len(finger_present)))

        print(f"Continuous contact segments (>5s): {len(segments)}")
        for i, (s, e) in enumerate(segments):
            dur = (ts[e-1] - ts[s]) / 1000.0
            print(f"  Segment {i}: {(ts[s]-ts[0])/1000:.0f}s - {(ts[e-1]-ts[0])/1000:.0f}s ({dur:.0f}s)")

        # Analyze each segment for HRV
        all_rr = []
        all_rr_times = []  # timestamps of each RR interval for trend analysis
        all_beat_count = 0

        for seg_idx, (seg_s, seg_e) in enumerate(segments):
            ir_seg = ir[seg_s:seg_e]
            ts_seg = ts[seg_s:seg_e]
            seg_dur = (ts_seg[-1] - ts_seg[0]) / 1000.0

            # DC removal with 2.5s moving average
            dc_window = int(2.5 * sr)
            ir_baseline = moving_average(ir_seg, dc_window)
            ir_ac = ir_seg - ir_baseline

            # Light smoothing
            ir_smooth = moving_average(ir_ac, max(int(0.05 * sr), 2))
            ir_std = ir_smooth.std()

            if ir_std < 50:
                print(f"  Segment {seg_idx}: AC std too low ({ir_std:.0f}), skipping")
                continue

            # Detect beats (try both polarities)
            min_beat_dist = int(0.35 * sr)
            peaks_pos = find_peaks(ir_smooth, min_distance=min_beat_dist,
                                   prominence_threshold=ir_std * 0.3)
            peaks_neg = find_peaks(-ir_smooth, min_distance=min_beat_dist,
                                   prominence_threshold=ir_std * 0.3)

            beats = peaks_pos if len(peaks_pos) >= len(peaks_neg) else peaks_neg
            polarity = "normal" if len(peaks_pos) >= len(peaks_neg) else "inverted"

            if len(beats) < 3:
                print(f"  Segment {seg_idx}: Only {len(beats)} beats, skipping")
                continue

            # RR intervals
            rr = np.diff(ts_seg[beats])
            valid_mask = (rr >= 300) & (rr <= 1500)
            valid_rr = rr[valid_mask]

            all_beat_count += len(beats)
            if len(valid_rr) >= 2:
                all_rr.extend(valid_rr.tolist())
                # Store midpoint time of each RR interval for trend analysis
                for j in range(len(beats) - 1):
                    if valid_mask[j]:
                        mid_t = (ts_seg[beats[j]] + ts_seg[beats[j+1]]) / 2.0
                        all_rr_times.append(mid_t)

            expected_beats = int(seg_dur / 0.85)  # ~70 BPM expected
            detection_rate = len(beats) / expected_beats * 100 if expected_beats > 0 else 0

            print(f"  Segment {seg_idx}: {len(beats)} beats ({polarity}), "
                  f"{len(valid_rr)} valid RR, "
                  f"detection rate: {detection_rate:.0f}% of expected")

        # Overall HRV metrics
        all_rr = np.array(all_rr)
        all_rr_times = np.array(all_rr_times)

        print(f"\nTotal beats: {all_beat_count}")
        print(f"Total valid RR intervals: {len(all_rr)}")

        if len(all_rr) >= 5:
            avg_rr = all_rr.mean()
            avg_hr = 60000.0 / avg_rr

            print(f"\n--- Heart Rate ---")
            print(f"Average HR: {avg_hr:.1f} BPM")
            print(f"HR range: {60000/all_rr.max():.0f} - {60000/all_rr.min():.0f} BPM")
            print(f"Mean RR: {avg_rr:.0f} ms, Std RR: {all_rr.std():.0f} ms")

            # Time-domain HRV
            sdnn = all_rr.std()
            diffs = np.diff(all_rr)
            rmssd = np.sqrt(np.mean(diffs ** 2))
            pnn50 = np.sum(np.abs(diffs) > 50) / len(diffs) * 100

            print(f"\n--- HRV Metrics (Time Domain) ---")
            print(f"SDNN:  {sdnn:.1f} ms")
            print(f"RMSSD: {rmssd:.1f} ms")
            print(f"pNN50: {pnn50:.1f}%")

            # Interpretation
            if sdnn > 100:
                print(f"  → High variability (good)")
            elif sdnn > 50:
                print(f"  → Normal variability")
            else:
                print(f"  → Low variability")

            if rmssd > 40:
                print(f"  → Good vagal tone")
            elif rmssd > 20:
                print(f"  → Moderate vagal tone")
            else:
                print(f"  → Low vagal tone")

            # 3-minute windowed HRV trend
            print(f"\n--- 3-Minute Windowed HRV Trend ---")
            print(f"{'Window':>12} {'HR':>7} {'SDNN':>7} {'RMSSD':>7} {'N_RR':>6}")
            print("-" * 46)

            window_3min_ms = 3 * 60 * 1000
            t0 = ts[0]
            for w in range(int(duration_s // 180)):
                w_start = t0 + w * window_3min_ms
                w_end = w_start + window_3min_ms

                mask = (all_rr_times >= w_start) & (all_rr_times < w_end)
                w_rr = all_rr[mask]

                t_label = f"{w*3}-{(w+1)*3} min"

                if len(w_rr) >= 5:
                    w_hr = 60000.0 / w_rr.mean()
                    w_sdnn = w_rr.std()
                    w_diffs = np.diff(w_rr)
                    w_rmssd = np.sqrt(np.mean(w_diffs ** 2)) if len(w_diffs) > 0 else 0
                    print(f"{t_label:>12} {w_hr:7.1f} {w_sdnn:7.1f} {w_rmssd:7.1f} {len(w_rr):6d}")
                else:
                    print(f"{t_label:>12}    --      --      --  {len(w_rr):6d}")

        elif len(all_rr) > 0:
            print(f"Mean HR estimate: {60000/all_rr.mean():.0f} BPM (too few intervals for HRV)")
        else:
            print("⚠ No valid RR intervals found")
            print("  Check: LED optical crosstalk? Finger stability?")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: OVERALL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Thermistor verdict
    if len(minute_therm_ac_stds) >= 4:
        first_3 = np.mean(minute_therm_ac_stds[:3])
        last_3 = np.mean(minute_therm_ac_stds[-3:])
        deg = (1 - last_3 / first_3) * 100 if first_3 > 0 else 0
        print(f"Thermistor: {deg:.0f}% signal degradation over {duration_s/60:.0f} min")

    # Humidity verdict
    if len(minute_hum_ac_stds) >= 4:
        first_3h = np.mean(minute_hum_ac_stds[:3])
        last_3h = np.mean(minute_hum_ac_stds[-3:])
        hchange = ((last_3h / first_3h) - 1) * 100 if first_3h > 0 else 0
        print(f"Humidity:   {hchange:+.0f}% signal change ({('stable' if abs(hchange) < 30 else 'degraded')})")

    # Temperature drift
    t_start_val = np.nanmean(temp[:int(30*sr)])
    t_end_val = np.nanmean(temp[-int(30*sr):])
    print(f"Temperature: {t_start_val:.1f}°C → {t_end_val:.1f}°C ({t_end_val-t_start_val:+.1f}°C drift)")

    # HRV verdict
    if len(all_rr) >= 5:
        print(f"HRV: {len(all_rr)} valid RR intervals, HR {60000/all_rr.mean():.0f} BPM, RMSSD {rmssd:.0f} ms")
    else:
        print(f"HRV: Only {len(all_rr)} valid RR intervals — insufficient")

    # Breath detection minutes
    therm_working_mins = sum(1 for br in minute_breath_rates if br > 2)
    hum_working_mins = sum(1 for br in minute_hum_rates if br > 2)
    total_mins = len(minute_breath_rates)
    print(f"\nBreath detection coverage:")
    print(f"  Thermistor: {therm_working_mins}/{total_mins} minutes with detected breaths")
    print(f"  Humidity:   {hum_working_mins}/{total_mins} minutes with detected breaths")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == '__main__':
    main()
