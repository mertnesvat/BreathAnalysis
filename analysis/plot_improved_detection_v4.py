#!/usr/bin/env python3
"""
Improved breath detection v4 — Bandpass + amplitude envelope gating.

Problem with v3: periodicity (autocorrelation) is similar for null and
meditation because ambient pressure noise also has quasi-periodic components.

Key insight: the AMPLITUDE difference is the real discriminator.
- Meditation bandpass RMS: ~2-4 Pa (real breath oscillations)
- Null bandpass RMS: ~0.5-1 Pa (ambient noise)
- That's a 2-4x amplitude difference we can exploit.

Fix: use rolling RMS envelope of bandpass signal as gate, calibrated
against a noise floor estimate.
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt


# ═══════════════════════════════════════════════════
# BREATH DETECTION V4
# ═══════════════════════════════════════════════════

def detect_breaths_v4(ts, pressure, sample_rate,
                      bandpass=(0.08, 0.7),       # Hz — covers 5-42 breaths/min
                      envelope_window_s=8,         # rolling RMS window
                      amplitude_threshold_pa=None, # auto-calibrate if None
                      noise_floor_multiplier=2.0,  # threshold = multiplier * quietest 20%
                      min_prominence_frac=0.3,     # prominence as fraction of local envelope
                      min_breath_period_s=1.5,
                      max_breath_period_s=10.0):
    """
    v4: Bandpass → amplitude envelope gate → adaptive peak detection.

    Instead of checking periodicity (which ambient noise also has),
    we check amplitude: real breaths create larger oscillations than noise.
    """
    n = len(ts)
    nyq = sample_rate / 2

    # Step 1: Bandpass filter to breath frequency range
    b, a = butter(3, [bandpass[0]/nyq, bandpass[1]/nyq], btype='band')
    bp_signal = filtfilt(b, a, pressure)

    # Step 2: Compute rolling RMS envelope
    env_win = int(envelope_window_s * sample_rate)
    # Pad to handle edges
    padded = np.pad(bp_signal**2, (env_win//2, env_win//2), mode='edge')
    rolling_mean_sq = np.convolve(padded, np.ones(env_win)/env_win, mode='valid')[:n]
    envelope = np.sqrt(np.maximum(rolling_mean_sq, 0))

    # Step 3: Auto-calibrate threshold from quietest portion of signal
    if amplitude_threshold_pa is None:
        # Use the 20th percentile of envelope as "noise floor"
        noise_floor = np.percentile(envelope, 20)
        amplitude_threshold_pa = noise_floor * noise_floor_multiplier

    # Step 4: Gate — only keep regions above amplitude threshold
    gate_mask = envelope > amplitude_threshold_pa

    # Step 5: Peak detection on bandpass signal with adaptive prominence
    # Prominence scales with local envelope — this prevents detecting
    # tiny noise peaks that happen to be in a "gated" region
    min_prom = amplitude_threshold_pa * min_prominence_frac
    all_peaks, all_props = find_peaks(
        bp_signal,
        prominence=min_prom,
        distance=int(min_breath_period_s * sample_rate),
    )

    # Step 6: Keep only gated peaks
    gated_peaks = all_peaks[gate_mask[all_peaks]]

    return bp_signal, envelope, gated_peaks, all_peaks, gate_mask, amplitude_threshold_pa


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
# THRESHOLD SWEEP
# ═══════════════════════════════════════════════════

print("="*70)
print("BREATH DETECTION V4 — Bandpass + Amplitude Envelope Gate")
print("="*70)

# First compute bandpass signals to characterize amplitude
nyq_n = null_rate / 2
b, a = butter(3, [0.08/nyq_n, 0.7/nyq_n], btype='band')
null_bp = filtfilt(b, a, null_pressure)

nyq_m = med_rate / 2
b, a = butter(3, [0.08/nyq_m, 0.7/nyq_m], btype='band')
med_bp = filtfilt(b, a, med_pressure)

print(f"\nBandpass signal characteristics:")
print(f"  Null — std: {np.std(null_bp):.3f} Pa, RMS: {np.sqrt(np.mean(null_bp**2)):.3f} Pa")
print(f"  Med  — std: {np.std(med_bp):.3f} Pa, RMS: {np.sqrt(np.mean(med_bp**2)):.3f} Pa")
print(f"  Amplitude ratio: {np.std(med_bp)/np.std(null_bp):.1f}x")

# Sweep noise_floor_multiplier
print(f"\n{'Multiplier Sweep':─<70}")
print(f"{'multiplier':<12} {'Threshold':<12} {'Null breaths':<14} {'Null/min':<10} "
      f"{'Med breaths':<14} {'Med/min':<10} {'Select.'}")
best_mult = 2.0
best_selectivity = 0
for mult in [1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    _, _, ng, _, _, nt = detect_breaths_v4(null_ts, null_pressure, null_rate,
                                            noise_floor_multiplier=mult)
    _, _, mg, _, _, mt = detect_breaths_v4(med_ts, med_pressure, med_rate,
                                            noise_floor_multiplier=mult)
    nr = len(ng) / null_duration
    mr = len(mg) / med_duration
    sel = mr / nr if nr > 0 else (999 if mr > 0 else 0)
    marker = ""
    if sel > best_selectivity and mr > 5:
        best_selectivity = sel
        best_mult = mult
        marker = " ***"
    print(f"  {mult:<10.1f} {nt:<12.3f} {len(ng):<14} {nr:<10.1f} "
          f"{len(mg):<14} {mr:<10.1f} {sel:.1f}x{marker}")

print(f"\nBest multiplier: {best_mult} (selectivity: {best_selectivity:.1f}x)")

# Also try fixed thresholds
print(f"\n{'Fixed Threshold Sweep':─<70}")
print(f"{'threshold Pa':<14} {'Null breaths':<14} {'Null/min':<10} "
      f"{'Med breaths':<14} {'Med/min':<10} {'Select.'}")
for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]:
    _, _, ng, _, _, _ = detect_breaths_v4(null_ts, null_pressure, null_rate,
                                           amplitude_threshold_pa=thresh)
    _, _, mg, _, _, _ = detect_breaths_v4(med_ts, med_pressure, med_rate,
                                           amplitude_threshold_pa=thresh)
    nr = len(ng) / null_duration
    mr = len(mg) / med_duration
    sel = mr / nr if nr > 0 else (999 if mr > 0 else 0)
    print(f"  {thresh:<12.2f} {len(ng):<14} {nr:<10.1f} "
          f"{len(mg):<14} {mr:<10.1f} {sel:.1f}x")


# ═══════════════════════════════════════════════════
# RUN WITH BEST SETTINGS
# ═══════════════════════════════════════════════════

null_bp, null_env, null_gated, null_all, null_gate, null_thresh = \
    detect_breaths_v4(null_ts, null_pressure, null_rate, noise_floor_multiplier=best_mult)
med_bp, med_env, med_gated, med_all, med_gate, med_thresh = \
    detect_breaths_v4(med_ts, med_pressure, med_rate, noise_floor_multiplier=best_mult)

print(f"\n{'Results':─<70}")
print(f"  Null: {len(null_gated)} breaths ({len(null_gated)/null_duration:.1f}/min), "
      f"threshold={null_thresh:.3f} Pa, gate coverage={np.mean(null_gate)*100:.1f}%")
print(f"  Med:  {len(med_gated)} breaths ({len(med_gated)/med_duration:.1f}/min), "
      f"threshold={med_thresh:.3f} Pa, gate coverage={np.mean(med_gate)*100:.1f}%")

if len(med_gated) > 1:
    intervals = np.diff(med_ts[med_gated])
    print(f"  Med intervals: {np.mean(intervals):.2f}s +/- {np.std(intervals):.2f}s "
          f"({60/np.mean(intervals):.1f}/min)")

# Envelope statistics
print(f"\n  Envelope — Null: mean={np.mean(null_env):.3f}, "
      f"p20={np.percentile(null_env, 20):.3f}, p80={np.percentile(null_env, 80):.3f} Pa")
print(f"  Envelope — Med:  mean={np.mean(med_env):.3f}, "
      f"p20={np.percentile(med_env, 20):.3f}, p80={np.percentile(med_env, 80):.3f} Pa")


# ═══════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════

fig, axes = plt.subplots(5, 2, figsize=(18, 22))
fig.suptitle(f'Breath Detection v4 — Bandpass + Amplitude Envelope Gate\n'
             f'Left: Null Test (no breathing) | Right: Deep Meditation',
             fontsize=14, fontweight='bold')

for col, (ts_arr, bp_arr, env_arr, gated, all_p, gate, thresh_val, pres_raw,
          label, dur, rate_val) in enumerate([
    (null_ts, null_bp, null_env, null_gated, null_all, null_gate, null_thresh,
     null_pressure,
     f'NULL — {len(null_gated)} breaths ({len(null_gated)/null_duration:.1f}/min)',
     null_duration, null_rate),
    (med_ts, med_bp, med_env, med_gated, med_all, med_gate, med_thresh,
     med_pressure,
     f'MEDITATION — {len(med_gated)} breaths ({len(med_gated)/med_duration:.1f}/min)',
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

    # Row 2: Bandpass + envelope + gate
    ax = axes[1, col]
    ax.fill_between(t_min, bp_arr.min() * 1.2, bp_arr.max() * 1.2,
                    where=gate, alpha=0.12, color='#4CAF50', label='Above threshold')
    ax.plot(t_min, bp_arr, color=color, linewidth=0.6, alpha=0.8)
    ax.plot(t_min, env_arr, color='#FF9800', linewidth=1.5, label='RMS envelope')
    ax.plot(t_min, -env_arr, color='#FF9800', linewidth=1.5, alpha=0.5)
    ax.axhline(y=thresh_val, color='#F44336', linestyle='--', linewidth=1,
               label=f'Threshold ({thresh_val:.2f} Pa)')
    ax.axhline(y=-thresh_val, color='#F44336', linestyle='--', linewidth=1, alpha=0.5)

    rejected = np.setdiff1d(all_p, gated)
    if len(rejected) > 0:
        ax.plot(t_min[rejected], bp_arr[rejected], 'x', color='#BDBDBD', markersize=4,
                label=f'Rejected ({len(rejected)})')
    if len(gated) > 0:
        ax.plot(t_min[gated], bp_arr[gated], 'v', color='#E91E63', markersize=5,
                label=f'Breaths ({len(gated)})')

    ax.set_title('Bandpass (0.08-0.7 Hz) + Amplitude Gate')
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
    ax.plot(ts_arr[s:e] - t_off, env_arr[s:e], color='#FF9800', linewidth=2,
            label='RMS envelope')
    ax.axhline(y=thresh_val, color='#F44336', linestyle='--', linewidth=1)

    zoom_rej = rejected[(rejected >= s) & (rejected < e)]
    if len(zoom_rej) > 0:
        ax.plot(ts_arr[zoom_rej] - t_off, bp_arr[zoom_rej], 'x', color='#BDBDBD', markersize=6)
    zoom_good = gated[(gated >= s) & (gated < e)]
    if len(zoom_good) > 0:
        ax.plot(ts_arr[zoom_good] - t_off, bp_arr[zoom_good], 'v', color='#E91E63', markersize=8)

    ax.set_title('30s Zoom (midpoint)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Filtered (Pa)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 4: Envelope over time
    ax = axes[3, col]
    ax.plot(t_min, env_arr, color='#FF9800', linewidth=1.5)
    ax.axhline(y=thresh_val, color='#F44336', linestyle='--', linewidth=1.5,
               label=f'Threshold ({thresh_val:.3f} Pa)')
    ax.fill_between(t_min, 0, env_arr,
                    where=env_arr > thresh_val, alpha=0.3, color='#4CAF50', label='Breathing')
    ax.fill_between(t_min, 0, env_arr,
                    where=env_arr <= thresh_val, alpha=0.3, color='#EF9A9A', label='Noise/quiet')
    ax.set_ylabel('RMS Envelope (Pa)')
    ax.set_title('Amplitude Envelope — Breath vs Noise')
    ax.legend(fontsize=9)
    ax.set_xlim(0, dur)
    ax.grid(alpha=0.3)

    # Row 5: Histogram of envelope values
    ax = axes[4, col]
    env_max = max(np.max(null_env), np.max(med_env)) * 1.1
    bins = np.linspace(0, env_max, 50)
    ax.hist(env_arr, bins=bins, color=color, alpha=0.7, density=True, edgecolor='white')
    ax.axvline(x=thresh_val, color='#F44336', linestyle='--', linewidth=2,
               label=f'Threshold ({thresh_val:.3f} Pa)')
    pct_above = np.mean(gate) * 100
    ax.set_title(f'Envelope Distribution — {pct_above:.0f}% above threshold')
    ax.set_xlabel('RMS Envelope (Pa)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/improved_detection_v4.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to analysis/improved_detection_v4.png")
