#!/usr/bin/env python3
"""
Breath detection v4b — Fixed amplitude threshold + cluster filtering.

v4 found that fixed threshold of ~2.0 Pa gives 11.4x selectivity.
v4b adds cluster filtering: isolated peaks in quiet regions are rejected,
only sustained breathing (≥3 peaks in 20s) counts.
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt


def detect_breaths_v4b(ts, pressure, sample_rate,
                       bandpass=(0.08, 0.7),
                       amplitude_threshold_pa=2.0,
                       envelope_window_s=8,
                       min_cluster_breaths=3,
                       cluster_window_s=20,
                       min_breath_period_s=1.5):
    """
    v4b: Bandpass → amplitude gate → peak detection → cluster filter.

    Returns only peaks that are part of sustained breathing episodes
    (≥min_cluster_breaths peaks within cluster_window_s).
    """
    n = len(ts)
    nyq = sample_rate / 2

    # Step 1: Bandpass
    b, a = butter(3, [bandpass[0]/nyq, bandpass[1]/nyq], btype='band')
    bp_signal = filtfilt(b, a, pressure)

    # Step 2: Rolling RMS envelope
    env_win = int(envelope_window_s * sample_rate)
    padded = np.pad(bp_signal**2, (env_win//2, env_win//2), mode='edge')
    rolling_mean_sq = np.convolve(padded, np.ones(env_win)/env_win, mode='valid')[:n]
    envelope = np.sqrt(np.maximum(rolling_mean_sq, 0))

    # Step 3: Amplitude gate
    gate_mask = envelope > amplitude_threshold_pa

    # Step 4: Peak detection
    min_prom = amplitude_threshold_pa * 0.3
    all_peaks, _ = find_peaks(
        bp_signal,
        prominence=min_prom,
        distance=int(min_breath_period_s * sample_rate),
    )

    # Step 5: Keep gated peaks
    gated_peaks = all_peaks[gate_mask[all_peaks]]

    # Step 6: Cluster filter — keep only peaks in groups of ≥min_cluster_breaths
    if len(gated_peaks) < min_cluster_breaths:
        clustered_peaks = np.array([], dtype=int)
    else:
        clustered_peaks = []
        for i, pk in enumerate(gated_peaks):
            # Count how many other gated peaks are within cluster_window_s
            nearby = np.abs(ts[gated_peaks] - ts[pk]) < cluster_window_s
            if np.sum(nearby) >= min_cluster_breaths:
                clustered_peaks.append(pk)
        clustered_peaks = np.array(clustered_peaks, dtype=int)

    return bp_signal, envelope, clustered_peaks, gated_peaks, all_peaks, gate_mask


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
# COMPREHENSIVE SWEEP
# ═══════════════════════════════════════════════════

print("="*70)
print("BREATH DETECTION V4b — Amplitude Gate + Cluster Filter")
print("="*70)

# Sweep fixed threshold with cluster filter
print(f"\n{'Threshold + Cluster Sweep':─<70}")
print(f"{'thresh':<8} {'cluster':<8} {'Null':<8} {'N/min':<8} "
      f"{'Med':<8} {'M/min':<8} {'Select.':<10} {'Note'}")

best_config = (2.0, 3)
best_score = 0  # score = selectivity * min(med_rate, 20) / 20

for thresh in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
    for min_clust in [1, 2, 3, 4, 5]:
        _, _, nc, _, _, _ = detect_breaths_v4b(
            null_ts, null_pressure, null_rate,
            amplitude_threshold_pa=thresh, min_cluster_breaths=min_clust)
        _, _, mc, _, _, _ = detect_breaths_v4b(
            med_ts, med_pressure, med_rate,
            amplitude_threshold_pa=thresh, min_cluster_breaths=min_clust)

        nr = len(nc) / null_duration
        mr = len(mc) / med_duration
        sel = mr / nr if nr > 0 else (999 if mr > 0 else 0)

        # Score: we want high selectivity AND reasonable med rate (12-20/min)
        med_quality = min(mr, 20) / 20 if mr >= 8 else 0  # at least 8/min
        null_quality = max(0, 1 - nr/3)  # penalize >3/min null rate
        score = sel * med_quality * null_quality if sel < 500 else med_quality * null_quality * 100

        marker = ""
        if score > best_score:
            best_score = score
            best_config = (thresh, min_clust)
            marker = " <-- BEST"

        if min_clust in [1, 3, 5] or marker:
            print(f"  {thresh:<6.1f} {min_clust:<8} {len(nc):<8} {nr:<8.1f} "
                  f"{len(mc):<8} {mr:<8.1f} {sel:<10.1f} {marker}")

thresh_opt, clust_opt = best_config
print(f"\nOptimal: threshold={thresh_opt} Pa, min_cluster={clust_opt}")
print(f"Score: {best_score:.1f}")

# Run with optimal
null_bp, null_env, null_final, null_gated, null_all, null_gate = \
    detect_breaths_v4b(null_ts, null_pressure, null_rate,
                       amplitude_threshold_pa=thresh_opt, min_cluster_breaths=clust_opt)
med_bp, med_env, med_final, med_gated, med_all, med_gate = \
    detect_breaths_v4b(med_ts, med_pressure, med_rate,
                       amplitude_threshold_pa=thresh_opt, min_cluster_breaths=clust_opt)

print(f"\n{'Final Results':─<70}")
print(f"  Null: {len(null_final)} breaths ({len(null_final)/null_duration:.1f}/min)")
print(f"  Med:  {len(med_final)} breaths ({len(med_final)/med_duration:.1f}/min)")
selectivity = (len(med_final)/med_duration) / (len(null_final)/null_duration) \
    if len(null_final) > 0 else float('inf')
print(f"  Selectivity: {selectivity:.1f}x")

if len(med_final) > 1:
    intervals = np.diff(med_ts[med_final])
    print(f"  Med breath intervals: {np.mean(intervals):.2f}s +/- {np.std(intervals):.2f}s "
          f"({60/np.mean(intervals):.1f}/min)")
    # Expected range for meditation: 3-5s per breath (12-20/min)
    normal_range = np.sum((intervals >= 2.0) & (intervals <= 6.0))
    print(f"  Intervals in normal range (2-6s): {normal_range}/{len(intervals)} "
          f"({normal_range/len(intervals)*100:.0f}%)")


# ═══════════════════════════════════════════════════
# PIPELINE BREAKDOWN — how many peaks survive each stage
# ═══════════════════════════════════════════════════

print(f"\n{'Pipeline Breakdown':─<70}")
for label, all_p, gated_p, final_p, dur in [
    ("Null", null_all, null_gated, null_final, null_duration),
    ("Med", med_all, med_gated, med_final, med_duration),
]:
    print(f"  {label}:")
    print(f"    Raw peaks (bandpass):     {len(all_p):>6} ({len(all_p)/dur:.1f}/min)")
    print(f"    After amplitude gate:     {len(gated_p):>6} ({len(gated_p)/dur:.1f}/min)")
    print(f"    After cluster filter:     {len(final_p):>6} ({len(final_p)/dur:.1f}/min)")


# ═══════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════

fig, axes = plt.subplots(5, 2, figsize=(18, 22))
fig.suptitle(f'Breath Detection v4b — Amplitude Gate ({thresh_opt} Pa) + Cluster Filter (≥{clust_opt})\n'
             f'Left: Null Test (no breathing) | Right: Deep Meditation',
             fontsize=14, fontweight='bold')

for col, (ts_arr, bp_arr, env_arr, final_p, gated_p, all_p, gate, pres_raw,
          label, dur, rate_val) in enumerate([
    (null_ts, null_bp, null_env, null_final, null_gated, null_all, null_gate,
     null_pressure,
     f'NULL — {len(null_final)} breaths ({len(null_final)/null_duration:.1f}/min)',
     null_duration, null_rate),
    (med_ts, med_bp, med_env, med_final, med_gated, med_all, med_gate,
     med_pressure,
     f'MEDITATION — {len(med_final)} breaths ({len(med_final)/med_duration:.1f}/min)',
     med_duration, med_rate),
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

    # Row 2: Bandpass + envelope + detected breaths
    ax = axes[1, col]
    ax.fill_between(t_min, bp_arr.min() * 1.2, bp_arr.max() * 1.2,
                    where=gate, alpha=0.12, color='#4CAF50', label='Above threshold')
    ax.plot(t_min, bp_arr, color=color, linewidth=0.6, alpha=0.7)
    ax.plot(t_min, env_arr, color='#FF9800', linewidth=1.5, label='RMS envelope')
    ax.plot(t_min, -env_arr, color='#FF9800', linewidth=1.5, alpha=0.4)
    ax.axhline(y=thresh_opt, color='#F44336', linestyle='--', linewidth=1,
               label=f'Threshold ({thresh_opt} Pa)')

    # Show different peak types
    cluster_rejected = np.setdiff1d(gated_p, final_p)
    amplitude_rejected = np.setdiff1d(all_p, gated_p)

    if len(amplitude_rejected) > 0:
        ax.plot(t_min[amplitude_rejected], bp_arr[amplitude_rejected], 'x',
                color='#E0E0E0', markersize=3, label=f'Low amplitude ({len(amplitude_rejected)})')
    if len(cluster_rejected) > 0:
        ax.plot(t_min[cluster_rejected], bp_arr[cluster_rejected], 'x',
                color='#FFAB91', markersize=4, label=f'Isolated ({len(cluster_rejected)})')
    if len(final_p) > 0:
        ax.plot(t_min[final_p], bp_arr[final_p], 'v', color='#E91E63', markersize=6,
                label=f'Breaths ({len(final_p)})')

    ax.set_title('Bandpass + Amplitude Gate + Cluster Filter')
    ax.set_ylabel('Filtered (Pa)')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, dur)
    ax.grid(alpha=0.3)

    # Row 3: 30s zoom
    ax = axes[2, col]
    mid = len(ts_arr) // 2
    half = int(15 * rate_val)
    s, e = max(0, mid - half), min(len(ts_arr), mid + half)
    t_off = ts_arr[s]

    ax.fill_between(ts_arr[s:e] - t_off, bp_arr[s:e].min() * 1.2, bp_arr[s:e].max() * 1.2,
                    where=gate[s:e], alpha=0.12, color='#4CAF50')
    ax.plot(ts_arr[s:e] - t_off, bp_arr[s:e], color=color, linewidth=1.5)
    ax.plot(ts_arr[s:e] - t_off, env_arr[s:e], color='#FF9800', linewidth=2)
    ax.axhline(y=thresh_opt, color='#F44336', linestyle='--', linewidth=1)

    zoom_amp_rej = amplitude_rejected[(amplitude_rejected >= s) & (amplitude_rejected < e)]
    zoom_clust_rej = cluster_rejected[(cluster_rejected >= s) & (cluster_rejected < e)]
    zoom_good = final_p[(final_p >= s) & (final_p < e)]

    if len(zoom_amp_rej) > 0:
        ax.plot(ts_arr[zoom_amp_rej] - t_off, bp_arr[zoom_amp_rej], 'x',
                color='#E0E0E0', markersize=5)
    if len(zoom_clust_rej) > 0:
        ax.plot(ts_arr[zoom_clust_rej] - t_off, bp_arr[zoom_clust_rej], 'x',
                color='#FFAB91', markersize=6)
    if len(zoom_good) > 0:
        ax.plot(ts_arr[zoom_good] - t_off, bp_arr[zoom_good], 'v',
                color='#E91E63', markersize=8)

    ax.set_title('30s Zoom (midpoint)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Filtered (Pa)')
    ax.grid(alpha=0.3)

    # Row 4: Envelope timeline with threshold
    ax = axes[3, col]
    ax.plot(t_min, env_arr, color='#FF9800', linewidth=1.5)
    ax.axhline(y=thresh_opt, color='#F44336', linestyle='--', linewidth=1.5,
               label=f'Threshold ({thresh_opt} Pa)')
    ax.fill_between(t_min, 0, env_arr,
                    where=env_arr > thresh_opt, alpha=0.3, color='#4CAF50', label='Breathing')
    ax.fill_between(t_min, 0, env_arr,
                    where=env_arr <= thresh_opt, alpha=0.3, color='#EF9A9A', label='Noise/quiet')
    ax.set_ylabel('RMS Envelope (Pa)')
    ax.set_title('Amplitude Envelope')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0, dur)
    ax.grid(alpha=0.3)

    # Row 5: Breath rate over time (rolling 60s window)
    ax = axes[4, col]
    if len(final_p) >= 3:
        peak_times = ts_arr[final_p]
        # Rolling rate: count peaks in 60s windows
        rate_times = []
        rate_vals = []
        for t_center in np.arange(peak_times[0] + 30, peak_times[-1] - 30, 10):
            in_window = np.sum(np.abs(peak_times - t_center) < 30)
            rate_times.append(t_center / 60)
            rate_vals.append(in_window)  # peaks per 60s

        ax.plot(rate_times, rate_vals, color='#E91E63', linewidth=2)
        ax.fill_between(rate_times, 0, rate_vals, alpha=0.2, color='#E91E63')
        ax.set_ylabel('Breath Rate (/min)')
        ax.set_ylim(0, max(rate_vals) * 1.3 if rate_vals else 30)
    else:
        ax.text(0.5, 0.5, f'Only {len(final_p)} breaths detected — insufficient for rate',
                transform=ax.transAxes, ha='center', va='center', fontsize=12, color='grey')
        ax.set_ylabel('Breath Rate (/min)')

    ax.set_xlabel('Time (min)')
    ax.set_title('Breath Rate (rolling 60s window)')
    ax.set_xlim(0, dur)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/improved_detection_v4b.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to analysis/improved_detection_v4b.png")
