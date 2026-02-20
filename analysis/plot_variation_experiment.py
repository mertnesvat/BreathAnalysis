#!/usr/bin/env python3
"""
Variation Breathing Experiment Analysis
========================================
Null test (10 min, no breathing) vs Variation breathing (10 min):
  Phase 1: 0:00 - 5:30  — Normal breathing
  Phase 2: 5:30 - 6:30  — Fast breathing
  Phase 3: 6:30 - 7:30  — Very shallow / breath hold
  Phase 4: 7:30 - 8:30  — Deep, slow, full lungs
  Phase 5: 8:30 - 10:00 — Back to normal

Algorithm: v4b (bandpass + amplitude gate + cluster filter)
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, welch
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec


# ═══════════════════════════════════════════════════
# v4b ALGORITHM (identical to production)
# ═══════════════════════════════════════════════════

def detect_breaths_v4b(ts, pressure, sample_rate,
                       bandpass=(0.08, 0.7),
                       amplitude_threshold_pa=2.0,
                       envelope_window_s=8,
                       min_cluster_breaths=5,
                       cluster_window_s=20,
                       min_breath_period_s=1.5):
    """
    v4b: Bandpass -> amplitude gate -> peak detection -> cluster filter.
    Returns only peaks that are part of sustained breathing episodes.
    """
    n = len(ts)
    nyq = sample_rate / 2

    # Step 1: Bandpass filter (0.08-0.7 Hz = 5-42 breaths/min)
    b, a = butter(3, [bandpass[0] / nyq, bandpass[1] / nyq], btype='band')
    bp_signal = filtfilt(b, a, pressure)

    # Step 2: Rolling RMS envelope (8s window)
    env_win = int(envelope_window_s * sample_rate)
    padded = np.pad(bp_signal ** 2, (env_win // 2, env_win // 2), mode='edge')
    rolling_mean_sq = np.convolve(padded, np.ones(env_win) / env_win, mode='valid')[:n]
    envelope = np.sqrt(np.maximum(rolling_mean_sq, 0))

    # Step 3: Amplitude gate (2.0 Pa threshold)
    gate_mask = envelope > amplitude_threshold_pa

    # Step 4: Peak detection
    min_prom = amplitude_threshold_pa * 0.3
    all_peaks, all_props = find_peaks(
        bp_signal, prominence=min_prom,
        distance=int(min_breath_period_s * sample_rate))

    # Step 5: Keep only gated peaks
    gated_peaks = all_peaks[gate_mask[all_peaks]]

    # Step 6: Cluster filter (>=5 peaks within 20s window)
    if len(gated_peaks) < min_cluster_breaths:
        clustered_peaks = np.array([], dtype=int)
    else:
        clustered_peaks = []
        for pk in gated_peaks:
            nearby = np.sum(np.abs(ts[gated_peaks] - ts[pk]) < cluster_window_s)
            if nearby >= min_cluster_breaths:
                clustered_peaks.append(pk)
        clustered_peaks = np.array(clustered_peaks, dtype=int)

    return bp_signal, envelope, clustered_peaks, gated_peaks, all_peaks, gate_mask


# ═══════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════

print("Loading data...")
with gzip.open('/Users/mert/Downloads/10minsNullTest2.gz', 'rt') as f:
    null_data = json.load(f)
with gzip.open('/Users/mert/Downloads/10mins_variation_breath.gz', 'rt') as f:
    breath_data = json.load(f)

null_samples = np.array(null_data['samples'])
null_ts = null_samples[:, 0] / 1000.0
null_pressure = null_samples[:, 3]
null_rate = (len(null_ts) - 1) / (null_ts[-1] - null_ts[0])
null_duration = (null_ts[-1] - null_ts[0]) / 60

breath_samples = np.array(breath_data['samples'])
breath_ts = breath_samples[:, 0] / 1000.0
breath_pressure = breath_samples[:, 3]
breath_rate = (len(breath_ts) - 1) / (breath_ts[-1] - breath_ts[0])
breath_duration = (breath_ts[-1] - breath_ts[0]) / 60

print(f"Null test:  {null_duration:.1f} min, {len(null_ts)} samples, {null_rate:.1f} Hz")
print(f"Breathing:  {breath_duration:.1f} min, {len(breath_ts)} samples, {breath_rate:.1f} Hz")


# ═══════════════════════════════════════════════════
# RUN v4b ON BOTH
# ═══════════════════════════════════════════════════

print("\nRunning v4b algorithm...")
null_bp, null_env, null_final, null_gated, null_all, null_gate = \
    detect_breaths_v4b(null_ts, null_pressure, null_rate)
breath_bp, breath_env, breath_final, breath_gated, breath_all, breath_gate = \
    detect_breaths_v4b(breath_ts, breath_pressure, breath_rate)


# ═══════════════════════════════════════════════════
# PHASE DEFINITIONS (in seconds from start)
# ═══════════════════════════════════════════════════

phases = [
    (0,     330,  'Normal Breath',       '#4CAF50'),  # 0:00 - 5:30
    (330,   390,  'Fast Breath',         '#FF9800'),  # 5:30 - 6:30
    (390,   450,  'Shallow/Hold',        '#F44336'),  # 6:30 - 7:30
    (450,   510,  'Deep Slow Breath',    '#2196F3'),  # 7:30 - 8:30
    (510,   600,  'Return to Normal',    '#9C27B0'),  # 8:30 - 10:00
]


# ═══════════════════════════════════════════════════
# CONSOLE OUTPUT — DETAILED STATISTICS
# ═══════════════════════════════════════════════════

print("\n" + "=" * 80)
print("VARIATION BREATHING EXPERIMENT — v4b ANALYSIS")
print("=" * 80)

# Pipeline breakdown
print(f"\n{'Pipeline Breakdown':─<80}")
for label, all_p, gated_p, final_p, dur in [
    ("Null Test", null_all, null_gated, null_final, null_duration),
    ("Breathing", breath_all, breath_gated, breath_final, breath_duration),
]:
    print(f"\n  {label} ({dur:.1f} min):")
    print(f"    Raw bandpass peaks:     {len(all_p):>6} ({len(all_p) / dur:.1f}/min)")
    print(f"    After amplitude gate:   {len(gated_p):>6} ({len(gated_p) / dur:.1f}/min)")
    print(f"    After cluster filter:   {len(final_p):>6} ({len(final_p) / dur:.1f}/min)")

selectivity = (len(breath_final) / breath_duration) / (len(null_final) / null_duration) \
    if len(null_final) > 0 else float('inf')
print(f"\n  Selectivity: {'∞' if selectivity == float('inf') else f'{selectivity:.1f}x'}")

# Phase-by-phase analysis
print(f"\n{'Phase-by-Phase Analysis':─<80}")
print(f"  {'Phase':<22} {'Time':<12} {'Breaths':<10} {'Rate/min':<10} "
      f"{'Avg Depth':<12} {'Avg Interval':<14} {'Envelope RMS'}")
print(f"  {'─' * 22} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 14} {'─' * 12}")

phase_stats = []
for t_start, t_end, name, color in phases:
    # Find peaks in this phase
    if len(breath_final) > 0:
        phase_mask = (breath_ts[breath_final] >= t_start) & (breath_ts[breath_final] < t_end)
        phase_peaks = breath_final[phase_mask]
    else:
        phase_peaks = np.array([], dtype=int)

    phase_dur = (t_end - t_start) / 60
    n_breaths = len(phase_peaks)
    bpm = n_breaths / phase_dur if phase_dur > 0 else 0

    # Average depth (amplitude of detected peaks)
    if n_breaths > 0:
        depths = np.abs(breath_bp[phase_peaks])
        avg_depth = np.mean(depths)
    else:
        avg_depth = 0

    # Average interval between breaths
    if n_breaths > 1:
        intervals = np.diff(breath_ts[phase_peaks])
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
    else:
        avg_interval = 0
        std_interval = 0

    # Envelope RMS in this phase
    env_mask = (breath_ts >= t_start) & (breath_ts < t_end)
    env_rms = np.mean(breath_env[env_mask]) if np.any(env_mask) else 0

    phase_stats.append({
        'name': name, 'color': color, 't_start': t_start, 't_end': t_end,
        'n_breaths': n_breaths, 'bpm': bpm, 'avg_depth': avg_depth,
        'avg_interval': avg_interval, 'std_interval': std_interval,
        'env_rms': env_rms, 'phase_peaks': phase_peaks,
    })

    time_str = f"{t_start // 60:.0f}:{t_start % 60:02.0f}-{t_end // 60:.0f}:{t_end % 60:02.0f}"
    interval_str = f"{avg_interval:.2f}s ±{std_interval:.2f}" if avg_interval > 0 else "N/A"
    depth_str = f"{avg_depth:.2f} Pa" if avg_depth > 0 else "N/A"

    print(f"  {name:<22} {time_str:<12} {n_breaths:<10} {bpm:<10.1f} "
          f"{depth_str:<12} {interval_str:<14} {env_rms:.2f} Pa")

# Null test signal stats
print(f"\n{'Null Test Signal Stats':─<80}")
print(f"  Pressure raw std:       {np.std(null_pressure):.2f} Pa")
print(f"  Bandpass signal std:    {np.std(null_bp):.2f} Pa")
print(f"  Envelope mean:          {np.mean(null_env):.2f} Pa")
print(f"  Envelope max:           {np.max(null_env):.2f} Pa")
print(f"  Envelope > 2.0 Pa:     {np.sum(null_env > 2.0) / len(null_env) * 100:.1f}% of time")

# Breathing signal stats
print(f"\n{'Breathing Signal Stats':─<80}")
print(f"  Pressure raw std:       {np.std(breath_pressure):.2f} Pa")
print(f"  Bandpass signal std:    {np.std(breath_bp):.2f} Pa")
print(f"  Envelope mean:          {np.mean(breath_env):.2f} Pa")
print(f"  Envelope max:           {np.max(breath_env):.2f} Pa")
print(f"  Envelope > 2.0 Pa:     {np.sum(breath_env > 2.0) / len(breath_env) * 100:.1f}% of time")


# ═══════════════════════════════════════════════════
# FIGURE 1: Side-by-side comparison (Null vs Breathing)
# ═══════════════════════════════════════════════════

fig1 = plt.figure(figsize=(20, 24))
gs = gridspec.GridSpec(6, 2, figure=fig1, hspace=0.35, wspace=0.25,
                       height_ratios=[1.2, 1, 1, 0.8, 0.8, 0.7])
fig1.suptitle('Variation Breathing Experiment — v4b Algorithm\n'
              'Left: 10-min Null Test (no breathing) │ Right: 10-min Breathing with Variations',
              fontsize=15, fontweight='bold', y=0.98)


# ── Row 1: Raw pressure ──
for col, (ts_arr, pres, label, dur) in enumerate([
    (null_ts, null_pressure, 'NULL TEST — Raw Pressure', null_duration),
    (breath_ts, breath_pressure, 'BREATHING — Raw Pressure', breath_duration),
]):
    ax = fig1.add_subplot(gs[0, col])
    t_min = ts_arr / 60
    color = '#78909C' if col == 0 else '#4527A0'
    ax.plot(t_min, pres, color=color, linewidth=0.3, alpha=0.7)
    ax.set_title(label, fontweight='bold',
                 color='#D32F2F' if col == 0 else '#311B92', fontsize=12)
    ax.set_ylabel('Pressure δ (Pa)')
    ax.set_xlim(0, dur)
    ax.grid(alpha=0.2)

    # Add phase bands for breathing column
    if col == 1:
        for t_s, t_e, name, c in phases:
            ax.axvspan(t_s / 60, t_e / 60, alpha=0.08, color=c)
            ax.text((t_s + t_e) / 2 / 60, ax.get_ylim()[1] * 0.95, name,
                    ha='center', va='top', fontsize=7, color=c, fontweight='bold')


# ── Row 2: Bandpass + envelope + detected peaks ──
for col, (ts_arr, bp_arr, env_arr, final_p, gated_p, all_p, gate, dur, rate_val) in enumerate([
    (null_ts, null_bp, null_env, null_final, null_gated, null_all, null_gate,
     null_duration, null_rate),
    (breath_ts, breath_bp, breath_env, breath_final, breath_gated, breath_all, breath_gate,
     breath_duration, breath_rate),
]):
    ax = fig1.add_subplot(gs[1, col])
    t_min = ts_arr / 60
    color = '#78909C' if col == 0 else '#4527A0'

    ax.fill_between(t_min, bp_arr.min() * 1.3, bp_arr.max() * 1.3,
                    where=gate, alpha=0.10, color='#4CAF50')
    ax.plot(t_min, bp_arr, color=color, linewidth=0.5, alpha=0.6)
    ax.plot(t_min, env_arr, color='#FF9800', linewidth=1.8, label='RMS envelope', zorder=3)
    ax.plot(t_min, -env_arr, color='#FF9800', linewidth=1.8, alpha=0.4, zorder=3)
    ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1.2,
               label='Threshold (2.0 Pa)', zorder=4)
    ax.axhline(y=-2.0, color='#F44336', linestyle='--', linewidth=1.2, alpha=0.4, zorder=4)

    # Show rejected vs accepted peaks
    cluster_rejected = np.setdiff1d(gated_p, final_p)
    if len(cluster_rejected) > 0:
        ax.plot(t_min[cluster_rejected], bp_arr[cluster_rejected], 'x',
                color='#FFAB91', markersize=4, label=f'Isolated ({len(cluster_rejected)})')
    if len(final_p) > 0:
        ax.plot(t_min[final_p], bp_arr[final_p], 'v', color='#E91E63', markersize=5,
                label=f'Breaths ({len(final_p)})', zorder=5)

    n_breaths = len(final_p)
    bpm = n_breaths / dur
    title = f'{"NULL" if col == 0 else "BREATHING"} — {n_breaths} breaths ({bpm:.1f}/min)'
    ax.set_title(title, fontweight='bold',
                 color='#D32F2F' if col == 0 else '#311B92')
    ax.set_ylabel('Bandpass Signal (Pa)')
    ax.set_xlim(0, dur)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.2)

    if col == 1:
        for t_s, t_e, _, c in phases:
            ax.axvspan(t_s / 60, t_e / 60, alpha=0.06, color=c)


# ── Row 3: Envelope comparison ──
for col, (ts_arr, env_arr, final_p, dur, label) in enumerate([
    (null_ts, null_env, null_final, null_duration, 'Null'),
    (breath_ts, breath_env, breath_final, breath_duration, 'Breathing'),
]):
    ax = fig1.add_subplot(gs[2, col])
    t_min = ts_arr / 60

    ax.plot(t_min, env_arr, color='#FF9800', linewidth=1.5)
    ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1.5)
    ax.fill_between(t_min, 0, env_arr,
                    where=env_arr > 2.0, alpha=0.3, color='#4CAF50', label='Above threshold')
    ax.fill_between(t_min, 0, env_arr,
                    where=env_arr <= 2.0, alpha=0.3, color='#EF9A9A', label='Below threshold')

    ax.set_ylabel('RMS Envelope (Pa)')
    ax.set_title(f'{label} — Amplitude Envelope', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, dur)
    ax.grid(alpha=0.2)

    if col == 1:
        for t_s, t_e, name, c in phases:
            ax.axvspan(t_s / 60, t_e / 60, alpha=0.06, color=c)
            ax.text((t_s + t_e) / 2 / 60, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 5,
                    name, ha='center', va='top', fontsize=7, color=c, fontweight='bold')


# ── Row 4: 30s zoom — null midpoint vs breathing midpoint ──
for col, (ts_arr, bp_arr, env_arr, final_p, gate, rate_val, color, title) in enumerate([
    (null_ts, null_bp, null_env, null_final, null_gate, null_rate, '#78909C',
     'Null — 30s Zoom (midpoint)'),
    (breath_ts, breath_bp, breath_env, breath_final, breath_gate, breath_rate, '#4527A0',
     'Breathing — 30s Zoom (normal phase ~2:30)'),
]):
    ax = fig1.add_subplot(gs[3, col])

    if col == 0:
        mid = len(ts_arr) // 2
    else:
        # Zoom into normal breathing phase at ~2:30
        mid = np.searchsorted(ts_arr, 150)

    half = int(15 * rate_val)
    s, e = max(0, mid - half), min(len(ts_arr), mid + half)
    t_sec = ts_arr[s:e] - ts_arr[s]

    ax.fill_between(t_sec, bp_arr[s:e].min() * 1.3, bp_arr[s:e].max() * 1.3,
                    where=gate[s:e], alpha=0.10, color='#4CAF50')
    ax.plot(t_sec, bp_arr[s:e], color=color, linewidth=1.2)
    ax.plot(t_sec, env_arr[s:e], color='#FF9800', linewidth=2.5, zorder=3)
    ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1.2, zorder=4)

    zoom_peaks = final_p[(final_p >= s) & (final_p < e)]
    if len(zoom_peaks) > 0:
        ax.plot(ts_arr[zoom_peaks] - ts_arr[s], bp_arr[zoom_peaks], 'v',
                color='#E91E63', markersize=10, zorder=5)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Filtered (Pa)')
    ax.grid(alpha=0.2)


# ── Row 5: PSD comparison ──
ax = fig1.add_subplot(gs[4, 0])

# Use Welch PSD for cleaner estimate
null_f, null_pxx = welch(null_bp, fs=null_rate, nperseg=min(1024, len(null_bp)))
breath_f, breath_pxx = welch(breath_bp, fs=breath_rate, nperseg=min(1024, len(breath_bp)))

ax.semilogy(null_f, null_pxx, color='#78909C', linewidth=1.5, label='Null test')
ax.semilogy(breath_f, breath_pxx, color='#4527A0', linewidth=1.5, label='Breathing')
ax.axvspan(0.08, 0.7, alpha=0.12, color='#E91E63', label='Breath band')
ax.set_xlim(0, 2)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (Pa²/Hz)')
ax.set_title('Power Spectral Density', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)


# ── Row 5 right: Envelope histogram ──
ax = fig1.add_subplot(gs[4, 1])
bins = np.linspace(0, 8, 80)
ax.hist(null_env, bins=bins, color='#78909C', alpha=0.6, density=True,
        label=f'Null (mean={np.mean(null_env):.2f} Pa)', edgecolor='white', linewidth=0.3)
ax.hist(breath_env, bins=bins, color='#4527A0', alpha=0.6, density=True,
        label=f'Breathing (mean={np.mean(breath_env):.2f} Pa)', edgecolor='white', linewidth=0.3)
ax.axvline(x=2.0, color='#F44336', linestyle='--', linewidth=2.5, label='Threshold (2.0 Pa)')
ax.set_xlabel('RMS Envelope (Pa)')
ax.set_ylabel('Density')
ax.set_title('Envelope Distribution', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)


# ── Row 6: Summary table ──
ax = fig1.add_subplot(gs[5, :])
ax.axis('off')

# Build table data
table_data = [
    ['Metric', 'Null Test', 'Breathing', 'Verdict'],
    ['Duration', f'{null_duration:.1f} min', f'{breath_duration:.1f} min', ''],
    ['Raw bandpass peaks', str(len(null_all)), str(len(breath_all)), ''],
    ['After amplitude gate', str(len(null_gated)), str(len(breath_gated)), ''],
    ['After cluster filter', str(len(null_final)), str(len(breath_final)),
     '0 false positives' if len(null_final) == 0 else f'{len(null_final)} false positives'],
    ['Breath rate', f'{len(null_final) / null_duration:.1f}/min',
     f'{len(breath_final) / breath_duration:.1f}/min', ''],
    ['Selectivity', '', '',
     '∞' if len(null_final) == 0 else f'{selectivity:.1f}x'],
    ['Envelope mean', f'{np.mean(null_env):.2f} Pa', f'{np.mean(breath_env):.2f} Pa', ''],
    ['Envelope max', f'{np.max(null_env):.2f} Pa', f'{np.max(breath_env):.2f} Pa', ''],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='center', colWidths=[0.25, 0.2, 0.2, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Style
for j in range(4):
    table[0, j].set_facecolor('#E8EAF6')
    table[0, j].set_text_props(fontweight='bold')
for i in [3, 4, 5]:
    for j in range(4):
        table[i, j].set_facecolor('#E8F5E9')
if len(null_final) == 0:
    table[4, 3].set_text_props(fontweight='bold', color='#2E7D32')

plt.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/variation_experiment_overview.png',
            dpi=150, bbox_inches='tight')
plt.close(fig1)
print("\nSaved: variation_experiment_overview.png")


# ═══════════════════════════════════════════════════
# FIGURE 2: Detailed phase analysis of breathing session
# ═══════════════════════════════════════════════════

fig2 = plt.figure(figsize=(20, 28))
gs2 = gridspec.GridSpec(7, 1, figure=fig2, hspace=0.35,
                        height_ratios=[1.4, 1.2, 1.0, 1.0, 1.0, 1.0, 0.8])
fig2.suptitle('Breathing Variation Session — Detailed Phase Analysis\n'
              f'{len(breath_final)} breaths over {breath_duration:.1f} min',
              fontsize=15, fontweight='bold', y=0.99)


# ── Panel 1: Full session with phase annotations ──
ax = fig2.add_subplot(gs2[0])
t_min = breath_ts / 60

# Phase background bands
for t_s, t_e, name, c in phases:
    ax.axvspan(t_s / 60, t_e / 60, alpha=0.12, color=c)
    y_top = max(np.max(breath_bp) * 1.1, 8)
    ax.text((t_s + t_e) / 2 / 60, y_top, name, ha='center', va='top',
            fontsize=9, color=c, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=c))

ax.plot(t_min, breath_bp, color='#7E57C2', linewidth=0.5, alpha=0.5)
ax.plot(t_min, breath_env, color='#FF9800', linewidth=2, zorder=3, label='RMS envelope')
ax.plot(t_min, -breath_env, color='#FF9800', linewidth=2, alpha=0.4, zorder=3)
ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1.2, zorder=4, label='Threshold')

if len(breath_final) > 0:
    # Color peaks by their phase
    for ps in phase_stats:
        if ps['n_breaths'] > 0:
            pp = ps['phase_peaks']
            ax.plot(t_min[pp], breath_bp[pp], 'v', color=ps['color'],
                    markersize=6, zorder=5, markeredgecolor='black', markeredgewidth=0.3)

ax.set_ylabel('Bandpass Signal (Pa)')
ax.set_title('Full Session — All Phases Annotated', fontweight='bold', fontsize=13)
ax.set_xlim(0, breath_duration)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.2)


# ── Panel 2: Envelope + breath rate rolling ──
ax = fig2.add_subplot(gs2[1])
ax2 = ax.twinx()

# Envelope
ax.plot(t_min, breath_env, color='#FF9800', linewidth=2, label='RMS Envelope')
ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1, alpha=0.5)
ax.fill_between(t_min, 0, breath_env,
                where=breath_env > 2.0, alpha=0.2, color='#4CAF50')
ax.set_ylabel('RMS Envelope (Pa)', color='#FF9800')
ax.tick_params(axis='y', labelcolor='#FF9800')

# Rolling breath rate (30s window, 5s step)
if len(breath_final) >= 3:
    peak_ts_sec = breath_ts[breath_final]
    rate_t, rate_v = [], []
    for t_c in np.arange(15, breath_ts[-1] - 15, 5):
        in_win = np.sum((peak_ts_sec >= t_c - 15) & (peak_ts_sec < t_c + 15))
        rate_t.append(t_c / 60)
        rate_v.append(in_win * 2)  # scale 30s window to per-minute

    ax2.plot(rate_t, rate_v, color='#E91E63', linewidth=2.5, label='Breath Rate')
    ax2.set_ylabel('Breath Rate (/min)', color='#E91E63')
    ax2.tick_params(axis='y', labelcolor='#E91E63')
    ax2.set_ylim(0, max(rate_v) * 1.3 if rate_v else 40)

for t_s, t_e, name, c in phases:
    ax.axvspan(t_s / 60, t_e / 60, alpha=0.06, color=c)

ax.set_xlabel('Time (min)')
ax.set_title('Envelope & Instantaneous Breath Rate', fontweight='bold')
ax.set_xlim(0, breath_duration)
ax.grid(alpha=0.2)


# ── Panels 3-7: Individual phase 30s zooms ──
for i, ps in enumerate(phase_stats):
    ax = fig2.add_subplot(gs2[2 + i])

    t_s, t_e = ps['t_start'], ps['t_end']
    phase_mask = (breath_ts >= t_s) & (breath_ts < t_e)
    t_sec = breath_ts[phase_mask] - t_s

    ax.plot(t_sec, breath_bp[phase_mask], color=ps['color'], linewidth=1.0, alpha=0.7)
    ax.plot(t_sec, breath_env[phase_mask], color='#FF9800', linewidth=2.5, zorder=3)
    ax.plot(t_sec, -breath_env[phase_mask], color='#FF9800', linewidth=2.5, alpha=0.4, zorder=3)
    ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1.2, zorder=4)
    ax.axhline(y=-2.0, color='#F44336', linestyle='--', linewidth=1.2, alpha=0.4, zorder=4)

    # Plot detected breaths
    if ps['n_breaths'] > 0:
        pp = ps['phase_peaks']
        ax.plot(breath_ts[pp] - t_s, breath_bp[pp], 'v', color='#E91E63',
                markersize=8, zorder=5, markeredgecolor='black', markeredgewidth=0.5)

    # Stats box
    dur_s = t_e - t_s
    stats_text = (f"Breaths: {ps['n_breaths']}  │  Rate: {ps['bpm']:.1f}/min  │  "
                  f"Avg Depth: {ps['avg_depth']:.2f} Pa  │  "
                  f"Avg Interval: {ps['avg_interval']:.2f}s" if ps['avg_interval'] > 0
                  else f"Breaths: {ps['n_breaths']}  │  Rate: {ps['bpm']:.1f}/min  │  "
                       f"Avg Depth: {ps['avg_depth']:.2f} Pa")

    ax.text(0.5, 0.97, stats_text, transform=ax.transAxes, ha='center', va='top',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                      edgecolor=ps['color'], linewidth=2))

    time_label = f"{t_s // 60:.0f}:{t_s % 60:02.0f} - {t_e // 60:.0f}:{t_e % 60:02.0f}"
    ax.set_title(f'Phase {i + 1}: {ps["name"]} ({time_label})',
                 fontweight='bold', color=ps['color'], fontsize=12)
    ax.set_xlabel('Time within phase (s)')
    ax.set_ylabel('Pressure (Pa)')
    ax.set_xlim(0, dur_s)
    ax.grid(alpha=0.2)


plt.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/variation_experiment_phases.png',
            dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved: variation_experiment_phases.png")


# ═══════════════════════════════════════════════════
# FIGURE 3: Breath-by-breath interval + depth
# ═══════════════════════════════════════════════════

if len(breath_final) > 1:
    fig3, axes3 = plt.subplots(3, 1, figsize=(18, 14),
                               gridspec_kw={'height_ratios': [1, 1, 0.8]})
    fig3.suptitle('Breath-by-Breath Analysis — Intervals, Depth, and Phase Transitions',
                  fontsize=14, fontweight='bold')

    intervals = np.diff(breath_ts[breath_final])
    interval_times = breath_ts[breath_final][1:] / 60
    peak_depths = np.abs(breath_bp[breath_final])
    peak_times = breath_ts[breath_final] / 60

    # ── Panel 1: Intervals ──
    ax = axes3[0]
    for ps in phase_stats:
        ax.axvspan(ps['t_start'] / 60, ps['t_end'] / 60, alpha=0.10, color=ps['color'])

    ax.plot(interval_times, intervals, 'o-', color='#5C6BC0',
            markersize=4, linewidth=0.8, alpha=0.7)

    if len(intervals) >= 8:
        rolling = np.convolve(intervals, np.ones(8) / 8, mode='valid')
        rolling_t = interval_times[3:3 + len(rolling)]
        ax.plot(rolling_t, rolling, color='#E91E63', linewidth=3, zorder=4,
                label='8-breath rolling avg')

    ax.axhspan(2.5, 5.0, alpha=0.06, color='#4CAF50', label='Normal range')
    ax.set_ylabel('Breath Interval (s)')
    ax.set_title('Breath-to-Breath Intervals', fontweight='bold')
    ax.set_xlim(0, breath_duration)
    ax.set_ylim(0, min(np.percentile(intervals, 98) * 1.5, 15))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Phase labels
    for ps in phase_stats:
        ax.text((ps['t_start'] + ps['t_end']) / 2 / 60, ax.get_ylim()[1] * 0.95,
                ps['name'], ha='center', va='top', fontsize=8, color=ps['color'],
                fontweight='bold')

    # ── Panel 2: Breath depth over time ──
    ax = axes3[1]
    for ps in phase_stats:
        ax.axvspan(ps['t_start'] / 60, ps['t_end'] / 60, alpha=0.10, color=ps['color'])

    ax.bar(peak_times, peak_depths, width=0.02, color='#4527A0', alpha=0.6)

    if len(peak_depths) >= 8:
        rolling_d = np.convolve(peak_depths, np.ones(8) / 8, mode='valid')
        rolling_dt = peak_times[3:3 + len(rolling_d)]
        ax.plot(rolling_dt, rolling_d, color='#FF9800', linewidth=3, zorder=4,
                label='8-breath rolling avg')

    ax.set_ylabel('Breath Depth (Pa)')
    ax.set_title('Breath Depth (Peak Amplitude)', fontweight='bold')
    ax.set_xlim(0, breath_duration)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # ── Panel 3: Phase summary bar chart ──
    ax = axes3[2]
    x = np.arange(len(phase_stats))
    width = 0.25

    rates = [ps['bpm'] for ps in phase_stats]
    depths = [ps['avg_depth'] for ps in phase_stats]
    envs = [ps['env_rms'] for ps in phase_stats]
    colors = [ps['color'] for ps in phase_stats]
    names = [ps['name'] for ps in phase_stats]

    bars1 = ax.bar(x - width, rates, width, label='Rate (/min)', color=colors, alpha=0.8)
    bars2 = ax.bar(x, [d * 3 for d in depths], width, label='Depth (Pa × 3)',
                   color=colors, alpha=0.5, hatch='//')
    bars3 = ax.bar(x + width, [e * 3 for e in envs], width, label='Envelope (Pa × 3)',
                   color=colors, alpha=0.3, hatch='\\\\')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Value')
    ax.set_title('Phase Comparison', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis='y')

    # Value labels on rate bars
    for bar, val in zip(bars1, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig3.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/variation_experiment_intervals.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("Saved: variation_experiment_intervals.png")


# ═══════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════

print("\n" + "=" * 80)
print("EXPERIMENT VERDICT")
print("=" * 80)

print(f"\n  NULL TEST:  {len(null_final)} false positives in {null_duration:.1f} min "
      f"→ {'PASS (zero FP)' if len(null_final) == 0 else 'FAIL'}")
print(f"  BREATHING:  {len(breath_final)} breaths in {breath_duration:.1f} min "
      f"({len(breath_final) / breath_duration:.1f}/min)")
print(f"  SELECTIVITY: {'∞' if len(null_final) == 0 else f'{selectivity:.1f}x'}")

print(f"\n  Phase Detection Results:")
for ps in phase_stats:
    expected = ""
    if "Normal" in ps['name']:
        expected = "expect ~14-20/min"
    elif "Fast" in ps['name']:
        expected = "expect >20/min"
    elif "Shallow" in ps['name'] or "Hold" in ps['name']:
        expected = "expect low rate or gaps"
    elif "Deep" in ps['name']:
        expected = "expect <12/min, high depth"

    status = ""
    if ps['bpm'] > 0:
        status = "DETECTED"
    else:
        status = "NO SIGNAL"

    print(f"    {ps['name']:<22} → {ps['bpm']:>5.1f}/min, depth {ps['avg_depth']:.2f} Pa  "
          f"({expected}) [{status}]")

print("\n" + "=" * 80)
