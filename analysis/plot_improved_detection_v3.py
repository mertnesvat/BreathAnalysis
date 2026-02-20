#!/usr/bin/env python3
"""
Improved breath detection v3 — Bandpass + adaptive periodicity.

Problem with v2: raw pressure has too much broadband noise, so
autocorrelation is weak even for real breaths.

Fix: bandpass filter to breath band (0.1-0.6 Hz) BEFORE checking
periodicity. This boosts SNR dramatically.
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# ═══════════════════════════════════════════════════
# BREATH DETECTION V3
# ═══════════════════════════════════════════════════

def detect_breaths_v3(ts, pressure, sample_rate,
                      bandpass=(0.08, 0.7),    # Hz — covers 5-42 breaths/min
                      gate_window_s=20,
                      gate_step_s=5,
                      min_autocorr=0.25,
                      min_prominence_pa=0.5,    # after bandpass, amplitudes are smaller
                      min_breath_period_s=1.5,
                      max_breath_period_s=10.0):
    """
    v3: Bandpass → autocorrelation gate → peak detection.
    """
    n = len(ts)
    nyq = sample_rate / 2

    # Step 1: Bandpass filter to breath frequency range
    b, a = butter(3, [bandpass[0]/nyq, bandpass[1]/nyq], btype='band')
    bp_signal = filtfilt(b, a, pressure)

    # Step 2: Periodicity gate on bandpass-filtered signal
    gate_win = int(gate_window_s * sample_rate)
    gate_step = int(gate_step_s * sample_rate)
    gate_mask = np.zeros(n, dtype=bool)
    gate_info = []

    min_lag = int(min_breath_period_s * sample_rate)
    max_lag = int(max_breath_period_s * sample_rate)

    for start in range(0, n - gate_win, gate_step):
        end = start + gate_win
        segment = bp_signal[start:end]
        seg_centered = segment - np.mean(segment)

        # Autocorrelation
        autocorr = np.correlate(seg_centered, seg_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Find strongest peak in breath lag range
        search = autocorr[min_lag:min(max_lag, len(autocorr))]
        if len(search) > 10:
            ac_peaks, ac_props = find_peaks(search, height=0.05, prominence=0.03)
            if len(ac_peaks) > 0:
                best_idx = ac_peaks[np.argmax(ac_props['peak_heights'])]
                best_val = search[best_idx]
                period_s = (best_idx + min_lag) / sample_rate
            else:
                best_val = 0
                period_s = 0
        else:
            best_val = 0
            period_s = 0

        t_center = (ts[start] + ts[min(end-1, n-1)]) / 2
        gate_info.append((t_center, best_val, period_s))

        if best_val >= min_autocorr:
            gate_mask[start:end] = True

    # Step 3: Peak detection on bandpass signal
    all_peaks, all_props = find_peaks(
        bp_signal,
        prominence=min_prominence_pa,
        distance=int(min_breath_period_s * sample_rate),
    )

    # Step 4: Keep only gated peaks
    gated_peaks = all_peaks[gate_mask[all_peaks]]

    return bp_signal, gated_peaks, all_peaks, gate_mask, gate_info


# ═══════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════

with gzip.open('/Users/mert/Downloads/NullTest.gz', 'rt') as f:
    null_data = json.load(f)
with gzip.open('/Users/mert/Downloads/15MinsLatestChanges.gz', 'rt') as f:
    med_data = json.load(f)

null_samples = np.array(null_data['samples'])
null_ts = null_samples[:, 0] / 1000.0
null_pressure = null_samples[:, 3]
null_rate = (len(null_ts) - 1) / (null_ts[-1] - null_ts[0])
null_duration = (null_ts[-1] - null_ts[0]) / 60

med_samples = np.array(med_data['samples'])
med_ts = med_samples[:, 0] / 1000.0
med_pressure = med_samples[:, 3]
med_rate = (len(med_ts) - 1) / (med_ts[-1] - med_ts[0])
med_duration = (med_ts[-1] - med_ts[0]) / 60

# ═══════════════════════════════════════════════════
# RUN DETECTION
# ═══════════════════════════════════════════════════

print("="*70)
print("BREATH DETECTION V3 — Bandpass + Periodicity Gate")
print("="*70)

# Threshold sweep first to find optimal
print(f"\n{'Threshold Sweep':─<70}")
print(f"{'min_autocorr':<14} {'Null breaths':<14} {'Null/min':<10} {'Med breaths':<14} {'Med/min':<10} {'Ratio'}")
best_thresh = 0.25
best_ratio = 0
for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
    _, ng, _, _, _ = detect_breaths_v3(null_ts, null_pressure, null_rate, min_autocorr=thresh)
    _, mg, _, _, _ = detect_breaths_v3(med_ts, med_pressure, med_rate, min_autocorr=thresh)
    nr = len(ng) / null_duration
    mr = len(mg) / med_duration
    ratio = mr / nr if nr > 0 else (999 if mr > 0 else 0)
    marker = ""
    if ratio > best_ratio and mr > 5:  # at least some breaths detected
        best_ratio = ratio
        best_thresh = thresh
        marker = " ***"
    print(f"  {thresh:<12.2f} {len(ng):<14} {nr:<10.1f} {len(mg):<14} {mr:<10.1f} {ratio:.1f}x{marker}")

print(f"\nBest threshold: {best_thresh} (selectivity: {best_ratio:.1f}x)")

# Run with best threshold
null_bp, null_gated, null_all, null_gate, null_ginfo = \
    detect_breaths_v3(null_ts, null_pressure, null_rate, min_autocorr=best_thresh)
med_bp, med_gated, med_all, med_gate, med_ginfo = \
    detect_breaths_v3(med_ts, med_pressure, med_rate, min_autocorr=best_thresh)

print(f"\n{'Results with threshold={best_thresh}':─<70}")
print(f"  Null: {len(null_gated)} breaths ({len(null_gated)/null_duration:.1f}/min)")
print(f"  Med:  {len(med_gated)} breaths ({len(med_gated)/med_duration:.1f}/min)")
print(f"  Gate coverage — Null: {np.sum(null_gate)/len(null_gate)*100:.1f}%, Med: {np.sum(med_gate)/len(med_gate)*100:.1f}%")

if len(med_gated) > 1:
    intervals = np.diff(med_ts[med_gated])
    print(f"  Med intervals: {np.mean(intervals):.2f}s ± {np.std(intervals):.2f}s ({60/np.mean(intervals):.1f}/min)")

# Autocorrelation comparison
null_ac_vals = [g[1] for g in null_ginfo]
med_ac_vals = [g[1] for g in med_ginfo]
print(f"\n  Autocorrelation — Null: avg={np.mean(null_ac_vals):.3f}, max={np.max(null_ac_vals):.3f}")
print(f"  Autocorrelation — Med:  avg={np.mean(med_ac_vals):.3f}, max={np.max(med_ac_vals):.3f}")

# ═══════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════

fig, axes = plt.subplots(5, 2, figsize=(18, 22))
fig.suptitle(f'Breath Detection v3 — Bandpass Filter + Periodicity Gate (threshold={best_thresh})\n'
             f'Left: Null Test (no breathing) | Right: Deep Meditation',
             fontsize=14, fontweight='bold')

for col, (ts_arr, bp_arr, gated, all_p, gate, ginfo, pres_raw, label, dur, rate_val) in enumerate([
    (null_ts, null_bp, null_gated, null_all, null_gate, null_ginfo, null_pressure,
     f'NULL — {len(null_gated)} breaths ({len(null_gated)/null_duration:.1f}/min)', null_duration, null_rate),
    (med_ts, med_bp, med_gated, med_all, med_gate, med_ginfo, med_pressure,
     f'MEDITATION — {len(med_gated)} breaths ({len(med_gated)/med_duration:.1f}/min)', med_duration, med_rate),
]):
    t_min = ts_arr / 60
    color = '#78909C' if col == 0 else '#4527A0'

    # Row 1: Raw pressure
    ax = axes[0, col]
    ax.plot(t_min, pres_raw, color=color, linewidth=0.3, alpha=0.7)
    ax.set_title(f'Raw Pressure — {label}', fontweight='bold',
                 color='#F44336' if col == 0 else '#4527A0')
    ax.set_ylabel('Pressure (Pa)')
    ax.set_xlim(0, dur)
    ax.grid(alpha=0.3)

    # Row 2: Bandpass filtered + peaks + gate
    ax = axes[1, col]
    ax.fill_between(t_min, bp_arr.min() * 1.1, bp_arr.max() * 1.1,
                    where=gate, alpha=0.15, color='#4CAF50', label='Periodic')
    ax.plot(t_min, bp_arr, color=color, linewidth=0.8)
    rejected = np.setdiff1d(all_p, gated)
    if len(rejected) > 0:
        ax.plot(t_min[rejected], bp_arr[rejected], 'x', color='#BDBDBD', markersize=4,
                label=f'Rejected ({len(rejected)})')
    if len(gated) > 0:
        ax.plot(t_min[gated], bp_arr[gated], 'v', color='#E91E63', markersize=5,
                label=f'Breaths ({len(gated)})')
    ax.set_title('Bandpass (0.08-0.7 Hz) + Gate')
    ax.set_ylabel('Filtered (Pa)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, dur)
    ax.grid(alpha=0.3)

    # Row 3: 30s zoom of bandpass
    ax = axes[2, col]
    mid = len(ts_arr) // 2
    half = int(15 * rate_val)
    s, e = max(0, mid - half), min(len(ts_arr), mid + half)
    t_off = ts_arr[s]

    ax.fill_between(ts_arr[s:e] - t_off, bp_arr[s:e].min() * 1.1, bp_arr[s:e].max() * 1.1,
                    where=gate[s:e], alpha=0.15, color='#4CAF50')
    ax.plot(ts_arr[s:e] - t_off, bp_arr[s:e], color=color, linewidth=1.5)

    zoom_rej = rejected[(rejected >= s) & (rejected < e)]
    if len(zoom_rej) > 0:
        ax.plot(ts_arr[zoom_rej] - t_off, bp_arr[zoom_rej], 'x', color='#BDBDBD', markersize=6)
    zoom_good = gated[(gated >= s) & (gated < e)]
    if len(zoom_good) > 0:
        ax.plot(ts_arr[zoom_good] - t_off, bp_arr[zoom_good], 'v', color='#E91E63', markersize=8)

    ax.set_title('30s Zoom (midpoint)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Filtered (Pa)')
    ax.grid(alpha=0.3)

    # Row 4: Autocorrelation per window
    ax = axes[3, col]
    gi_times = [g[0] / 60 for g in ginfo]
    gi_acvals = [g[1] for g in ginfo]
    ax.bar(gi_times, gi_acvals,
           width=5/60,
           color=[('#4CAF50' if v >= best_thresh else '#EF9A9A') for v in gi_acvals], alpha=0.8)
    ax.axhline(y=best_thresh, color='#F44336', linestyle='--', linewidth=1.5,
               label=f'Threshold ({best_thresh})')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Periodicity Gate (per 20s window)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, dur)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Row 5: Breath rate in passing windows
    ax = axes[4, col]
    passing = [(t, p) for t, v, p in ginfo if v >= best_thresh and p > 0]
    if passing:
        pt, pp = zip(*passing)
        rates = [60/p for p in pp]
        ax.scatter([t/60 for t in pt], rates, color='#E91E63', s=40, zorder=3)
        ax.plot([t/60 for t in pt], rates, color='#E91E63', linewidth=1, alpha=0.5)
        ax.set_ylim(0, max(rates) * 1.3)
    else:
        ax.text(0.5, 0.5, 'No periodic windows detected',
                transform=ax.transAxes, ha='center', va='center', fontsize=12, color='grey')
    ax.set_ylabel('Breath Rate (/min)')
    ax.set_xlabel('Time (min)')
    ax.set_title('Breath Rate (periodic windows only)')
    ax.set_xlim(0, dur)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/improved_detection_v3.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to analysis/improved_detection_v3.png")
