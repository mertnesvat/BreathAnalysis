#!/usr/bin/env python3
"""
Improved breath detection with noise rejection.
Uses periodicity gating: only counts peaks in windows where
autocorrelation confirms a periodic signal in the breath band.
Tests on both null (no breathing) and meditation recordings.
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ═══════════════════════════════════════════════════
# IMPROVED BREATH DETECTION ALGORITHM
# ═══════════════════════════════════════════════════

def detect_breaths_v2(ts, pressure, sample_rate,
                      baseline_window_s=15,
                      smooth_window_s=0.5,
                      gate_window_s=30,
                      gate_step_s=10,
                      min_autocorr=0.3,
                      min_prominence_pa=1.5,
                      min_breath_period_s=1.5,
                      max_breath_period_s=8.0):
    """
    Robust breath detection with periodicity gating.

    1. Baseline subtract (slow drift removal)
    2. Smooth signal
    3. Sliding window periodicity gate via autocorrelation
    4. Peak detection only in windows that pass the gate
    5. Absolute prominence threshold to reject noise spikes

    Returns: (ac_signal, smoothed, peaks, gate_mask, gate_info)
    """
    n = len(ts)

    # Step 1: Baseline subtraction (15s moving average)
    win = int(baseline_window_s * sample_rate)
    baseline = np.convolve(pressure, np.ones(win)/win, mode='same')
    ac = pressure - baseline

    # Step 2: Smooth for peak detection
    sw = int(smooth_window_s * sample_rate)
    smoothed = np.convolve(ac, np.ones(sw)/sw, mode='same')

    # Step 3: Periodicity gate — sliding window autocorrelation
    gate_win = int(gate_window_s * sample_rate)
    gate_step = int(gate_step_s * sample_rate)
    gate_mask = np.zeros(n, dtype=bool)  # True = periodic breathing detected
    gate_info = []  # (t_center, autocorr_peak, period_s)

    min_lag = int(min_breath_period_s * sample_rate)
    max_lag = int(max_breath_period_s * sample_rate)

    for start in range(0, n - gate_win, gate_step):
        end = start + gate_win
        segment = smoothed[start:end]
        seg_centered = segment - np.mean(segment)

        # Autocorrelation
        autocorr = np.correlate(seg_centered, seg_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # positive lags
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]  # normalize

        # Find strongest peak in breath lag range
        search_region = autocorr[min_lag:min(max_lag, len(autocorr))]
        if len(search_region) > 10:
            ac_peaks, ac_props = find_peaks(search_region, height=0.1, prominence=0.05)
            if len(ac_peaks) > 0:
                best_idx = ac_peaks[np.argmax(ac_props['peak_heights'])]
                best_val = search_region[best_idx]
                period_s = (best_idx + min_lag) / sample_rate
            else:
                best_val = 0
                period_s = 0
        else:
            best_val = 0
            period_s = 0

        t_center = (ts[start] + ts[min(end-1, n-1)]) / 2
        gate_info.append((t_center, best_val, period_s))

        # Gate: allow peaks if autocorrelation is strong enough
        if best_val >= min_autocorr:
            gate_mask[start:end] = True

    # Step 4: Peak detection with absolute prominence threshold
    all_peaks, all_props = find_peaks(
        smoothed,
        prominence=min_prominence_pa,
        distance=int(min_breath_period_s * sample_rate),
    )

    # Step 5: Filter peaks to only gated (periodic) regions
    gated_peaks = all_peaks[gate_mask[all_peaks]]

    return ac, smoothed, gated_peaks, all_peaks, gate_mask, gate_info


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

print("Running improved detection (v2)...")
print(f"Parameters: min_autocorr=0.3, min_prominence=1.5 Pa, gate_window=30s\n")

null_ac, null_smooth, null_gated, null_all, null_gate, null_ginfo = \
    detect_breaths_v2(null_ts, null_pressure, null_rate)
med_ac, med_smooth, med_gated, med_all, med_gate, med_ginfo = \
    detect_breaths_v2(med_ts, med_pressure, med_rate)

# Also run old algorithm for comparison
def detect_breaths_v1(ts, pressure, rate):
    window = int(15 * rate)
    baseline = np.convolve(pressure, np.ones(window)/window, mode='same')
    ac = pressure - baseline
    sw = int(0.5 * rate)
    smooth = np.convolve(ac, np.ones(sw)/sw, mode='same')
    std = np.std(smooth)
    peaks, _ = find_peaks(smooth, height=std * 0.15, distance=int(1.5 * rate), prominence=std * 0.1)
    return peaks

null_v1_peaks = detect_breaths_v1(null_ts, null_pressure, null_rate)
med_v1_peaks = detect_breaths_v1(med_ts, med_pressure, med_rate)

# ═══════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════

print(f"{'':─<70}")
print(f"{'Metric':<35} {'Null Test':<18} {'Meditation':<18}")
print(f"{'':─<70}")
print(f"{'OLD v1 — detected breaths':<35} {len(null_v1_peaks):<18} {len(med_v1_peaks):<18}")
print(f"{'OLD v1 — rate (/min)':<35} {len(null_v1_peaks)/null_duration:<18.1f} {len(med_v1_peaks)/med_duration:<18.1f}")
print(f"{'':─<70}")
print(f"{'NEW v2 — all peaks (pre-gate)':<35} {len(null_all):<18} {len(med_all):<18}")
print(f"{'NEW v2 — gated breaths':<35} {len(null_gated):<18} {len(med_gated):<18}")
print(f"{'NEW v2 — rate (/min)':<35} {len(null_gated)/null_duration:<18.1f} {len(med_gated)/med_duration:<18.1f}")
print(f"{'':─<70}")

# Gate coverage
null_gate_pct = np.sum(null_gate) / len(null_gate) * 100
med_gate_pct = np.sum(med_gate) / len(med_gate) * 100
print(f"{'Periodicity gate coverage':<35} {null_gate_pct:<18.1f}% {med_gate_pct:<18.1f}%")

# Autocorrelation strength
null_ac_vals = [g[1] for g in null_ginfo]
med_ac_vals = [g[1] for g in med_ginfo]
print(f"{'Avg autocorrelation':<35} {np.mean(null_ac_vals):<18.3f} {np.mean(med_ac_vals):<18.3f}")
print(f"{'Max autocorrelation':<35} {np.max(null_ac_vals):<18.3f} {np.max(med_ac_vals):<18.3f}")

if len(med_gated) > 1:
    intervals = np.diff(med_ts[med_gated])
    print(f"\nMeditation breath intervals: {np.mean(intervals):.2f}s ± {np.std(intervals):.2f}s")
    print(f"  ({60/np.mean(intervals):.1f} breaths/min from intervals)")

# Sweep min_autocorr to find the sweet spot
print(f"\n{'':─<70}")
print("THRESHOLD SWEEP — min_autocorr values")
print(f"{'':─<70}")
print(f"{'Threshold':<12} {'Null breaths':<15} {'Null rate':<12} {'Med breaths':<15} {'Med rate':<12} {'Selectivity'}")
for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    _, _, ng, _, _, _ = detect_breaths_v2(null_ts, null_pressure, null_rate, min_autocorr=thresh)
    _, _, mg, _, _, _ = detect_breaths_v2(med_ts, med_pressure, med_rate, min_autocorr=thresh)
    nr = len(ng) / null_duration
    mr = len(mg) / med_duration
    sel = mr / nr if nr > 0 else float('inf')
    marker = " <-- chosen" if thresh == 0.30 else ""
    print(f"  {thresh:<10.2f} {len(ng):<15} {nr:<12.1f} {len(mg):<15} {mr:<12.1f} {sel:.1f}x{marker}")

# ═══════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════

fig, axes = plt.subplots(4, 2, figsize=(18, 18))
fig.suptitle('Improved Breath Detection v2 — Periodicity Gating\n'
             'Left: Null Test (no breathing) | Right: Meditation',
             fontsize=14, fontweight='bold')

for col, (ts_arr, ac_arr, smooth_arr, gated_peaks, all_peaks, gate, ginfo, label, duration) in enumerate([
    (null_ts, null_ac, null_smooth, null_gated, null_all, null_gate, null_ginfo,
     f'NULL TEST — {len(null_gated)} breaths ({len(null_gated)/null_duration:.1f}/min)', null_duration),
    (med_ts, med_ac, med_smooth, med_gated, med_all, med_gate, med_ginfo,
     f'MEDITATION — {len(med_gated)} breaths ({len(med_gated)/med_duration:.1f}/min)', med_duration),
]):
    t_min = ts_arr / 60
    color = '#78909C' if col == 0 else '#4527A0'
    rate_val = null_rate if col == 0 else med_rate

    # Row 1: Full waveform with gate overlay
    ax = axes[0, col]
    ax.fill_between(t_min, smooth_arr.min(), smooth_arr.max(),
                    where=gate, alpha=0.15, color='#4CAF50', label='Periodic (gated)')
    ax.plot(t_min, ac_arr, color=color, linewidth=0.3, alpha=0.5)
    ax.plot(t_min, smooth_arr, color=color, linewidth=1.0)
    # Show rejected peaks
    rejected = np.setdiff1d(all_peaks, gated_peaks)
    if len(rejected) > 0:
        ax.plot(t_min[rejected], smooth_arr[rejected], 'x', color='#BDBDBD', markersize=4, label=f'Rejected ({len(rejected)})')
    if len(gated_peaks) > 0:
        ax.plot(t_min[gated_peaks], smooth_arr[gated_peaks], 'v', color='#E91E63', markersize=6, label=f'Breaths ({len(gated_peaks)})')
    ax.set_title(label, color='#F44336' if col == 0 else '#4527A0', fontweight='bold')
    ax.set_ylabel('Pressure AC (Pa)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, duration)
    ax.grid(alpha=0.3)

    # Row 2: 30s zoom
    ax = axes[1, col]
    mid = len(ts_arr) // 2
    half = int(15 * rate_val)
    s, e = max(0, mid - half), min(len(ts_arr), mid + half)
    t_off = ts_arr[s]

    ax.fill_between(ts_arr[s:e] - t_off, smooth_arr[s:e].min(), smooth_arr[s:e].max(),
                    where=gate[s:e], alpha=0.15, color='#4CAF50')
    ax.plot(ts_arr[s:e] - t_off, ac_arr[s:e], color=color, linewidth=0.4, alpha=0.4)
    ax.plot(ts_arr[s:e] - t_off, smooth_arr[s:e], color=color, linewidth=1.5)

    zoom_rej = rejected[(rejected >= s) & (rejected < e)]
    if len(zoom_rej) > 0:
        ax.plot(ts_arr[zoom_rej] - t_off, smooth_arr[zoom_rej], 'x', color='#BDBDBD', markersize=6)
    zoom_good = gated_peaks[(gated_peaks >= s) & (gated_peaks < e)]
    if len(zoom_good) > 0:
        ax.plot(ts_arr[zoom_good] - t_off, smooth_arr[zoom_good], 'v', color='#E91E63', markersize=8)

    ax.set_title('30s Zoom (midpoint)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure AC (Pa)')
    ax.grid(alpha=0.3)

    # Row 3: Autocorrelation strength over time
    ax = axes[2, col]
    gi_times = [g[0] / 60 for g in ginfo]
    gi_acvals = [g[1] for g in ginfo]
    gi_periods = [g[2] for g in ginfo]

    ax.bar(gi_times, gi_acvals, width=10/60, color=[('#4CAF50' if v >= 0.3 else '#EF9A9A') for v in gi_acvals], alpha=0.8)
    ax.axhline(y=0.3, color='#F44336', linestyle='--', linewidth=1.5, label='Threshold (0.3)')
    ax.set_ylabel('Autocorrelation Peak')
    ax.set_title('Periodicity Gate — Per Window')
    ax.legend(fontsize=9)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Row 4: Detected breath period per window
    ax = axes[3, col]
    passing = [(t, p) for t, v, p in ginfo if v >= 0.3 and p > 0]
    if passing:
        pt, pp = zip(*passing)
        ax.scatter(pt, [60/p for p in pp], color='#E91E63', s=30, zorder=3)
        ax.plot(pt, [60/p for p in pp], color='#E91E63', linewidth=1, alpha=0.5)
    ax.set_ylabel('Breath Rate (/min)')
    ax.set_xlabel('Time (min)')
    ax.set_title('Detected Breath Rate (periodic windows only)')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 40)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/improved_detection.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPlot saved to analysis/improved_detection.png")

# Final verdict
print(f"\n{'='*60}")
print("FINAL VERDICT")
print(f"{'='*60}")
null_rate_v2 = len(null_gated) / null_duration
med_rate_v2 = len(med_gated) / med_duration
print(f"\n  Old algorithm (v1):")
print(f"    Null: {len(null_v1_peaks)} breaths ({len(null_v1_peaks)/null_duration:.1f}/min)")
print(f"    Med:  {len(med_v1_peaks)} breaths ({len(med_v1_peaks)/med_duration:.1f}/min)")
print(f"    Selectivity: {(len(med_v1_peaks)/med_duration) / (len(null_v1_peaks)/null_duration):.1f}x")
print(f"\n  New algorithm (v2):")
print(f"    Null: {len(null_gated)} breaths ({null_rate_v2:.1f}/min)")
print(f"    Med:  {len(med_gated)} breaths ({med_rate_v2:.1f}/min)")
if null_rate_v2 > 0:
    print(f"    Selectivity: {med_rate_v2/null_rate_v2:.1f}x")
else:
    print(f"    Selectivity: ∞ (perfect noise rejection!)")
