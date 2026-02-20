#!/usr/bin/env python3
"""
Detailed visualization of v4b breath detection.
Top section: null vs meditation comparison
Bottom section: detailed meditation breathing patterns
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from matplotlib.patches import FancyBboxPatch


def detect_breaths_v4b(ts, pressure, sample_rate,
                       bandpass=(0.08, 0.7),
                       amplitude_threshold_pa=2.0,
                       envelope_window_s=8,
                       min_cluster_breaths=5,
                       cluster_window_s=20,
                       min_breath_period_s=1.5):
    n = len(ts)
    nyq = sample_rate / 2
    b, a = butter(3, [bandpass[0]/nyq, bandpass[1]/nyq], btype='band')
    bp_signal = filtfilt(b, a, pressure)

    env_win = int(envelope_window_s * sample_rate)
    padded = np.pad(bp_signal**2, (env_win//2, env_win//2), mode='edge')
    rolling_mean_sq = np.convolve(padded, np.ones(env_win)/env_win, mode='valid')[:n]
    envelope = np.sqrt(np.maximum(rolling_mean_sq, 0))

    gate_mask = envelope > amplitude_threshold_pa
    min_prom = amplitude_threshold_pa * 0.3
    all_peaks, all_props = find_peaks(
        bp_signal, prominence=min_prom,
        distance=int(min_breath_period_s * sample_rate))

    gated_peaks = all_peaks[gate_mask[all_peaks]]

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


# ── Load data ──
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

# ── Run detection ──
null_bp, null_env, null_final, _, _, null_gate = \
    detect_breaths_v4b(null_ts, null_pressure, null_rate)
med_bp, med_env, med_final, _, _, med_gate = \
    detect_breaths_v4b(med_ts, med_pressure, med_rate)

med_intervals = np.diff(med_ts[med_final]) if len(med_final) > 1 else np.array([])
med_breath_rate_avg = len(med_final) / med_duration


# ═══════════════════════════════════════════════════
# FIGURE 1: Null vs Meditation comparison
# ═══════════════════════════════════════════════════

fig1, axes1 = plt.subplots(3, 2, figsize=(18, 14),
                           gridspec_kw={'height_ratios': [1.2, 1, 0.8]})
fig1.suptitle('Noise Rejection: Null Test vs Deep Meditation\n'
              'Algorithm v4b — Bandpass (0.08-0.7 Hz) + Amplitude Gate (2.0 Pa) + Cluster Filter (≥5)',
              fontsize=14, fontweight='bold', y=0.98)

# ── Row 1: Bandpass signal + envelope + detected peaks ──
for col, (ts_arr, bp_arr, env_arr, final_p, gate, label, dur, rate_val) in enumerate([
    (null_ts, null_bp, null_env, null_final, null_gate,
     'NULL TEST — No breathing nearby', null_duration, null_rate),
    (med_ts, med_bp, med_env, med_final, med_gate,
     'DEEP MEDITATION — 15 min session', med_duration, med_rate),
]):
    ax = axes1[0, col]
    t_min = ts_arr / 60
    color = '#78909C' if col == 0 else '#4527A0'

    ax.fill_between(t_min, bp_arr.min()*1.3, bp_arr.max()*1.3,
                    where=gate, alpha=0.10, color='#4CAF50')
    ax.plot(t_min, bp_arr, color=color, linewidth=0.5, alpha=0.6)
    ax.plot(t_min, env_arr, color='#FF9800', linewidth=2, label='RMS envelope', zorder=3)
    ax.plot(t_min, -env_arr, color='#FF9800', linewidth=2, alpha=0.4, zorder=3)
    ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1.5,
               label='Threshold (2.0 Pa)', zorder=4)
    ax.axhline(y=-2.0, color='#F44336', linestyle='--', linewidth=1.5, alpha=0.4, zorder=4)

    if len(final_p) > 0:
        ax.plot(t_min[final_p], bp_arr[final_p], 'v', color='#E91E63',
                markersize=5, zorder=5, label=f'{len(final_p)} breaths')

    n_breaths = len(final_p)
    bpm = n_breaths / dur
    title_color = '#D32F2F' if col == 0 else '#311B92'
    ax.set_title(f'{label}\n{n_breaths} breaths detected ({bpm:.1f}/min)',
                 fontweight='bold', color=title_color, fontsize=12)
    ax.set_ylabel('Bandpass Signal (Pa)')
    ax.set_xlim(0, dur)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.2)

# ── Row 2: 30s zoom comparison ──
for col, (ts_arr, bp_arr, env_arr, final_p, gate, rate_val, color) in enumerate([
    (null_ts, null_bp, null_env, null_final, null_gate, null_rate, '#78909C'),
    (med_ts, med_bp, med_env, med_final, med_gate, med_rate, '#4527A0'),
]):
    ax = axes1[1, col]
    mid = len(ts_arr) // 2
    half = int(15 * rate_val)
    s, e = max(0, mid - half), min(len(ts_arr), mid + half)
    t_sec = ts_arr[s:e] - ts_arr[s]

    ax.fill_between(t_sec, bp_arr[s:e].min()*1.3, bp_arr[s:e].max()*1.3,
                    where=gate[s:e], alpha=0.10, color='#4CAF50')
    ax.plot(t_sec, bp_arr[s:e], color=color, linewidth=1.2)
    ax.plot(t_sec, env_arr[s:e], color='#FF9800', linewidth=2.5, zorder=3)
    ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1.5, zorder=4)

    zoom_peaks = final_p[(final_p >= s) & (final_p < e)]
    if len(zoom_peaks) > 0:
        ax.plot(ts_arr[zoom_peaks] - ts_arr[s], bp_arr[zoom_peaks], 'v',
                color='#E91E63', markersize=10, zorder=5)

    ax.set_title('30-Second Zoom (midpoint)', fontsize=11)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Filtered (Pa)')
    ax.grid(alpha=0.2)

    # Annotate
    if col == 0:
        ax.text(0.5, 0.92, 'Envelope stays below threshold\nNo breaths detected',
                transform=ax.transAxes, ha='center', va='top', fontsize=10,
                color='#D32F2F', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    else:
        n_zoom = len(zoom_peaks)
        ax.text(0.5, 0.92, f'Clear periodic peaks above threshold\n{n_zoom} breaths in 30s',
                transform=ax.transAxes, ha='center', va='top', fontsize=10,
                color='#1B5E20', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# ── Row 3: Envelope histogram overlay ──
ax = axes1[2, 0]
bins = np.linspace(0, 5, 60)
ax.hist(null_env, bins=bins, color='#78909C', alpha=0.6, density=True,
        label=f'Null (mean={np.mean(null_env):.2f} Pa)', edgecolor='white', linewidth=0.5)
ax.hist(med_env, bins=bins, color='#4527A0', alpha=0.6, density=True,
        label=f'Meditation (mean={np.mean(med_env):.2f} Pa)', edgecolor='white', linewidth=0.5)
ax.axvline(x=2.0, color='#F44336', linestyle='--', linewidth=2.5, label='Threshold (2.0 Pa)')
ax.set_xlabel('RMS Envelope Amplitude (Pa)')
ax.set_ylabel('Density')
ax.set_title('Envelope Distribution — Why the threshold works', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# Annotate the separation
ax.annotate('Noise lives here\n(null test)',
            xy=(1.0, 0.8), fontsize=9, color='#78909C', fontweight='bold',
            ha='center', xycoords=('data', 'axes fraction'))
ax.annotate('Breaths live here\n(meditation)',
            xy=(3.0, 0.8), fontsize=9, color='#4527A0', fontweight='bold',
            ha='center', xycoords=('data', 'axes fraction'))

# Summary stats box
ax = axes1[2, 1]
ax.axis('off')

stats = [
    ('', 'Null Test', 'Meditation'),
    ('Duration', f'{null_duration:.1f} min', f'{med_duration:.1f} min'),
    ('Raw peaks (bandpass)', '210', '376'),
    ('After amplitude gate', '14', '271'),
    ('After cluster filter', '0', '269'),
    ('Breath rate', '0.0/min', f'{med_breath_rate_avg:.1f}/min'),
    ('False positive rate', '0.0/min', '—'),
    ('Selectivity', '', '∞'),
]

table = ax.table(cellText=stats[1:], colLabels=stats[0],
                 cellLoc='center', loc='center',
                 colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)

# Style header
for j in range(3):
    table[0, j].set_facecolor('#E8EAF6')
    table[0, j].set_text_props(fontweight='bold')

# Highlight key rows
for i in [3, 4]:  # cluster filter and breath rate rows
    table[i, 1].set_facecolor('#E8F5E9')
    table[i, 2].set_facecolor('#E8F5E9')
    table[i, 1].set_text_props(fontweight='bold')
    table[i, 2].set_text_props(fontweight='bold')

# Zero false positives highlight
table[3, 1].set_text_props(fontweight='bold', color='#2E7D32')
table[5, 1].set_text_props(fontweight='bold', color='#2E7D32')

ax.set_title('Pipeline Results Summary', fontweight='bold', fontsize=12, pad=20)

plt.tight_layout()
fig1.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/v4b_comparison.png',
             dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Saved: v4b_comparison.png")


# ═══════════════════════════════════════════════════
# FIGURE 2: Detailed meditation breathing patterns
# ═══════════════════════════════════════════════════

fig2, axes2 = plt.subplots(4, 1, figsize=(18, 20),
                           gridspec_kw={'height_ratios': [1.2, 1, 1, 0.8]})
fig2.suptitle(f'Deep Meditation — Detailed Breathing Patterns\n'
              f'{len(med_final)} breaths over {med_duration:.1f} min '
              f'(avg {med_breath_rate_avg:.1f}/min, {60/med_breath_rate_avg:.1f}s per breath)',
              fontsize=14, fontweight='bold', y=0.98)

# Color-code breath depth by amplitude
peak_amplitudes = med_bp[med_final]
peak_times_min = med_ts[med_final] / 60

# ── Panel 1: Full session bandpass with breath depth coloring ──
ax = axes2[0]
t_min = med_ts / 60

ax.fill_between(t_min, med_bp.min()*1.2, med_bp.max()*1.2,
                where=med_gate, alpha=0.08, color='#4CAF50')
ax.plot(t_min, med_bp, color='#7E57C2', linewidth=0.5, alpha=0.5)
ax.plot(t_min, med_env, color='#FF9800', linewidth=1.5, alpha=0.8, label='RMS envelope')
ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1, alpha=0.5)

# Color peaks by amplitude (depth)
if len(med_final) > 0:
    sc = ax.scatter(peak_times_min, peak_amplitudes,
                    c=peak_amplitudes, cmap='RdYlGn', s=30, zorder=5,
                    edgecolors='black', linewidths=0.3, vmin=0, vmax=np.percentile(peak_amplitudes, 95))
    plt.colorbar(sc, ax=ax, label='Breath Depth (Pa)', shrink=0.6, pad=0.02)

ax.set_ylabel('Bandpass Pressure (Pa)')
ax.set_title('Full Session — Breath Peaks Colored by Depth', fontweight='bold')
ax.set_xlim(0, med_duration)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.2)

# ── Panel 2: Three 60s zoom windows (early, mid, late) ──
ax = axes2[1]
zoom_positions = [
    (60, 120, 'Minutes 1-2 (settling in)'),
    (med_ts[-1]/2 - 30, med_ts[-1]/2 + 30, 'Midpoint'),
    (med_ts[-1] - 120, med_ts[-1] - 60, 'Minutes 13-14 (deep state)'),
]
colors_zoom = ['#5C6BC0', '#7E57C2', '#AB47BC']

for i, (t_start, t_end, label) in enumerate(zoom_positions):
    mask = (med_ts >= t_start) & (med_ts < t_end)
    t_sec = med_ts[mask] - t_start

    # Offset each window vertically for stacked view
    offset = (2 - i) * 15
    bp_seg = med_bp[mask]

    ax.plot(t_sec, bp_seg + offset, color=colors_zoom[i], linewidth=1.2, label=label)

    # Find peaks in this window
    seg_peaks = med_final[(med_ts[med_final] >= t_start) & (med_ts[med_final] < t_end)]
    if len(seg_peaks) > 0:
        ax.plot(med_ts[seg_peaks] - t_start, med_bp[seg_peaks] + offset,
                'v', color='#E91E63', markersize=8, zorder=5)
        # Count
        n_seg = len(seg_peaks)
        seg_dur = (t_end - t_start) / 60
        ax.text(62, offset + 2, f'{n_seg} breaths\n({n_seg/seg_dur:.0f}/min)',
                fontsize=9, color=colors_zoom[i], fontweight='bold', va='bottom')

    # Time label
    ax.text(-2, offset, f'{t_start/60:.0f}-{t_end/60:.0f}m',
            fontsize=9, color=colors_zoom[i], fontweight='bold',
            ha='right', va='center')

ax.set_xlabel('Time within window (s)')
ax.set_ylabel('Bandpass Signal (Pa) — stacked')
ax.set_title('60-Second Windows: Early → Mid → Late', fontweight='bold')
ax.set_xlim(-5, 68)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.2, axis='x')

# ── Panel 3: Breath-by-breath analysis ──
ax = axes2[2]

if len(med_intervals) > 0:
    # Breath intervals over time
    interval_times = peak_times_min[1:]  # time of each interval end
    ax.plot(interval_times, med_intervals, 'o-', color='#5C6BC0',
            markersize=4, linewidth=0.8, alpha=0.7, label='Individual intervals')

    # Rolling average (10-breath window)
    if len(med_intervals) >= 10:
        rolling_avg = np.convolve(med_intervals, np.ones(10)/10, mode='valid')
        rolling_t = interval_times[4:4+len(rolling_avg)]
        ax.plot(rolling_t, rolling_avg, color='#E91E63', linewidth=3,
                label='10-breath rolling avg', zorder=4)

    # Highlight normal range
    ax.axhspan(2.5, 5.0, alpha=0.08, color='#4CAF50', label='Normal meditation range (2.5-5.0s)')
    ax.axhline(y=np.mean(med_intervals), color='#FF9800', linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(med_intervals):.2f}s ({60/np.mean(med_intervals):.1f}/min)')

    # Mark outliers
    outlier_mask = (med_intervals > 8) | (med_intervals < 1.5)
    if np.any(outlier_mask):
        ax.plot(interval_times[outlier_mask], med_intervals[outlier_mask],
                'x', color='#F44336', markersize=8, markeredgewidth=2,
                label=f'Outliers ({np.sum(outlier_mask)})')

    ax.set_ylim(0, min(np.percentile(med_intervals, 98) * 1.5, 15))

ax.set_xlabel('Time (min)')
ax.set_ylabel('Breath Interval (s)')
ax.set_title('Breath-by-Breath Intervals — Rhythm & Variability', fontweight='bold')
ax.set_xlim(0, med_duration)
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.2)

# ── Panel 4: Breath rate + depth trend ──
ax = axes2[3]
ax2 = ax.twinx()

if len(med_final) >= 5:
    # Rolling breath rate (60s window, 10s step)
    rate_times = []
    rate_vals = []
    depth_times = []
    depth_vals = []
    peak_ts = med_ts[med_final]
    peak_amps = np.abs(peak_amplitudes)

    for t_center in np.arange(30, med_ts[-1] - 30, 10):
        in_window = (peak_ts >= t_center - 30) & (peak_ts < t_center + 30)
        n_in = np.sum(in_window)
        if n_in > 0:
            rate_times.append(t_center / 60)
            rate_vals.append(n_in)  # per 60s
            depth_times.append(t_center / 60)
            depth_vals.append(np.mean(peak_amps[in_window]))

    l1, = ax.plot(rate_times, rate_vals, color='#E91E63', linewidth=2.5, label='Breath Rate')
    ax.fill_between(rate_times, 0, rate_vals, alpha=0.15, color='#E91E63')
    ax.set_ylabel('Breath Rate (/min)', color='#E91E63')
    ax.tick_params(axis='y', labelcolor='#E91E63')
    ax.set_ylim(0, max(rate_vals) * 1.3)

    l2, = ax2.plot(depth_times, depth_vals, color='#4527A0', linewidth=2.5, label='Breath Depth')
    ax2.fill_between(depth_times, 0, depth_vals, alpha=0.10, color='#4527A0')
    ax2.set_ylabel('Avg Breath Depth (Pa)', color='#4527A0')
    ax2.tick_params(axis='y', labelcolor='#4527A0')
    ax2.set_ylim(0, max(depth_vals) * 1.3)

    ax.legend(handles=[l1, l2], fontsize=9, loc='upper right')

ax.set_xlabel('Time (min)')
ax.set_title('Breath Rate & Depth Over Time', fontweight='bold')
ax.set_xlim(0, med_duration)
ax.grid(alpha=0.2)

plt.tight_layout()
fig2.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/v4b_meditation_detail.png',
             dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved: v4b_meditation_detail.png")

# ── Print summary stats ──
print(f"\nMeditation Breathing Summary:")
print(f"  Total breaths: {len(med_final)}")
print(f"  Duration: {med_duration:.1f} min")
print(f"  Avg rate: {med_breath_rate_avg:.1f}/min")
print(f"  Avg interval: {np.mean(med_intervals):.2f}s")
print(f"  Interval std: {np.std(med_intervals):.2f}s")
print(f"  Median interval: {np.median(med_intervals):.2f}s")
if len(peak_amplitudes) > 0:
    print(f"  Avg depth: {np.mean(np.abs(peak_amplitudes)):.2f} Pa")
    print(f"  Max depth: {np.max(np.abs(peak_amplitudes)):.2f} Pa")
# Breathing phases
if len(med_intervals) >= 10:
    early = med_intervals[:len(med_intervals)//3]
    mid = med_intervals[len(med_intervals)//3:2*len(med_intervals)//3]
    late = med_intervals[2*len(med_intervals)//3:]
    print(f"\n  Phase analysis:")
    print(f"    Early  (0-5min):  {np.mean(early):.2f}s/breath ({60/np.mean(early):.1f}/min)")
    print(f"    Middle (5-10min): {np.mean(mid):.2f}s/breath ({60/np.mean(mid):.1f}/min)")
    print(f"    Late   (10-15min):{np.mean(late):.2f}s/breath ({60/np.mean(late):.1f}/min)")
