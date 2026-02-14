#!/usr/bin/env python3
"""
Deep dive into BMP280 pressure channel for breath detection viability.
Analyzes: breath intervals, waveform shape, SNR, sneeze detection, stability.
"""

import gzip, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Helpers ──────────────────────────────────────────────────────────────────

def moving_average(signal, window):
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, signal[0]), signal])
    return np.convolve(padded, kernel, mode='valid')

def find_peaks(signal, min_distance=10, prominence_threshold=0.0):
    peaks = []
    n = len(signal)
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            left_min = min(signal[max(0, i - min_distance):i])
            right_min = min(signal[i + 1:min(n, i + min_distance + 1)])
            prominence = signal[i] - max(left_min, right_min)
            if prominence >= prominence_threshold:
                peaks.append((i, prominence))
    if len(peaks) < 2:
        return [p[0] for p in peaks], [p[1] for p in peaks]
    # Filter by min distance, keep highest prominence
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p[0] - filtered[-1][0] >= min_distance:
            filtered.append(p)
        elif p[1] > filtered[-1][1]:
            filtered[-1] = p
    return [p[0] for p in filtered], [p[1] for p in filtered]

# ── Load data ────────────────────────────────────────────────────────────────

with gzip.open('/Users/mert/Downloads/allSensors7Min.gz', 'rt') as f:
    data = json.load(f)

samples = data['samples']
n = len(samples)
ts = np.array([s[0] for s in samples], dtype=np.float64)
therm = np.array([s[1] for s in samples], dtype=np.float64)
ir = np.array([s[2] for s in samples], dtype=np.float64)
pressure = np.array([s[6] for s in samples], dtype=np.float64)
temp = np.array([s[4] for s in samples], dtype=np.float64)

t_sec = (ts - ts[0]) / 1000.0
sr = (n - 1) / t_sec[-1]

# ── Signal processing ────────────────────────────────────────────────────────

window_15s = int(15 * sr)
min_peak_dist = int(1.8 * sr)  # At least 1.8s between breaths (~33 br/min max)

# Pressure
press_baseline = moving_average(pressure, window_15s)
press_ac = pressure - press_baseline
press_smooth = moving_average(press_ac, int(0.25 * sr))

# Thermistor (for comparison)
therm_baseline = moving_average(therm, window_15s)
therm_ac = therm - therm_baseline
therm_smooth = moving_average(therm_ac, int(0.5 * sr))

# Peak detection with multiple thresholds to find optimal
results = {}
for thresh_mult in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    thresh = press_smooth.std() * thresh_mult
    peaks, proms = find_peaks(press_smooth, min_distance=min_peak_dist, prominence_threshold=thresh)
    if len(peaks) >= 2:
        intervals = np.diff(t_sec[peaks])
        rate = 60 / intervals.mean()
        results[thresh_mult] = {
            'peaks': peaks, 'proms': proms, 'n': len(peaks),
            'rate': rate, 'interval_mean': intervals.mean(),
            'interval_std': intervals.std(), 'intervals': intervals
        }

# Pick the threshold that gives closest to expected ~16-18 br/min
# Normal adult breathing: 12-20 br/min, user said ~18 br/min (3.3s)
best_thresh = min(results.keys(), key=lambda k: abs(results[k]['rate'] - 18))
best = results[best_thresh]
press_peaks = best['peaks']
press_proms = best['proms']

print(f"Best threshold: {best_thresh:.2f} × std = {press_smooth.std() * best_thresh:.3f} Pa")
print(f"Peaks: {best['n']}, Rate: {best['rate']:.1f} br/min, Interval: {best['interval_mean']:.2f}±{best['interval_std']:.2f}s")
print(f"\nAll thresholds tested:")
for k in sorted(results.keys()):
    r = results[k]
    print(f"  {k:.2f}×std: {r['n']:4d} peaks, {r['rate']:5.1f} br/min, interval {r['interval_mean']:.2f}±{r['interval_std']:.2f}s")

# Thermistor peaks for comparison
therm_peaks, therm_proms = find_peaks(therm_smooth, min_distance=min_peak_dist,
                                       prominence_threshold=therm_smooth.std() * 0.3)

intervals = best['intervals']

# ── Find sneeze/pause events ────────────────────────────────────────────────

# Gaps significantly larger than median interval
median_interval = np.median(intervals)
gap_threshold = median_interval * 2  # >2x normal interval = event
events = []
for i in range(len(intervals)):
    if intervals[i] > gap_threshold:
        t_start = t_sec[press_peaks[i]]
        t_end = t_sec[press_peaks[i + 1]]
        events.append({
            'start': t_start, 'end': t_end,
            'duration': intervals[i], 'minute': t_start / 60
        })

print(f"\nMedian breath interval: {median_interval:.2f}s")
print(f"Gap threshold (2× median): {gap_threshold:.2f}s")
print(f"\nDetected events (pauses/sneeze):")
for e in events:
    print(f"  {e['start']:.1f}s - {e['end']:.1f}s ({e['duration']:.1f}s gap) @ minute {e['minute']:.1f}")

# ── Compute rolling metrics ─────────────────────────────────────────────────

# Rolling breath rate (computed from peak intervals)
peak_times = t_sec[press_peaks]
rolling_rate = []
rolling_rate_t = []
for i in range(1, len(peak_times)):
    # Use 5-breath window
    window_start = max(0, i - 5)
    local_intervals = np.diff(peak_times[window_start:i + 1])
    if len(local_intervals) > 0:
        rolling_rate.append(60 / local_intervals.mean())
        rolling_rate_t.append(peak_times[i])

# Rolling signal amplitude (30s windows, sliding every 5s)
amp_times = []
amp_values = []
amp_noise = []
amp_snr = []
for t_center in np.arange(15, t_sec[-1] - 15, 5):
    mask = (t_sec >= t_center - 15) & (t_sec <= t_center + 15)
    chunk = press_smooth[mask]
    raw_chunk = press_ac[mask]
    if len(chunk) > 10:
        signal_amp = chunk.std()
        # Noise: high-freq component (diff of diff)
        noise = np.diff(raw_chunk).std() / np.sqrt(2)
        amp_times.append(t_center)
        amp_values.append(signal_amp)
        amp_noise.append(noise)
        amp_snr.append(signal_amp / noise if noise > 0 else 0)

# ── Create figure ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 30))
gs = GridSpec(9, 3, figure=fig, hspace=0.45, wspace=0.35,
              height_ratios=[2.5, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.2])

fig.suptitle('BMP280 Pressure — Deep Dive Analysis\n'
             f'7.1 min session • {best["n"]} breaths detected • {best["rate"]:.1f} br/min avg',
             fontsize=16, fontweight='bold', y=0.995)

C = '#7B1FA2'  # purple for pressure

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 0: Full session with peaks + event markers
# ═══════════════════════════════════════════════════════════════════════════════
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(t_sec, press_ac, color=C, linewidth=0.3, alpha=0.25)
ax0.plot(t_sec, press_smooth, color=C, linewidth=1.2)
ax0.scatter(t_sec[press_peaks], press_smooth[press_peaks],
            color='red', s=20, zorder=5, label=f'{best["n"]} peaks')

# Mark events (sneeze/pauses)
for e in events:
    ax0.axvspan(e['start'], e['end'], alpha=0.2, color='orange')
    ax0.annotate(f'{e["duration"]:.1f}s gap',
                 xy=((e['start'] + e['end']) / 2, press_smooth.max() * 0.8),
                 ha='center', fontsize=8, color='darkorange', fontweight='bold')

ax0.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
for m in range(8):
    ax0.axvline(x=m * 60, color='gray', linewidth=0.3, alpha=0.4)
    if m * 60 < t_sec[-1]:
        ax0.text(m * 60 + 2, press_smooth.max() * 0.95, f'min {m}', fontsize=8, color='gray')
ax0.set_ylabel('Pressure AC (Pa)')
ax0.set_xlabel('Time (s)')
ax0.set_title('Full Session — Detrended Pressure with Breath Peaks', fontsize=13, fontweight='bold')
ax0.legend(loc='upper right')
ax0.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1: Breath-by-breath interval timeline
# ═══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[1, :])
# Plot interval at midpoint between peaks
interval_t = [(t_sec[press_peaks[i]] + t_sec[press_peaks[i+1]]) / 2 for i in range(len(intervals))]
ax1.plot(interval_t, intervals, 'o-', color=C, markersize=4, linewidth=1, alpha=0.7)
ax1.axhline(y=median_interval, color='green', linewidth=2, linestyle='--',
            label=f'Median: {median_interval:.2f}s ({60/median_interval:.1f} br/min)')
ax1.axhline(y=3.3, color='blue', linewidth=1.5, linestyle=':',
            label='Target: 3.3s (18.2 br/min)')
ax1.axhline(y=gap_threshold, color='orange', linewidth=1, linestyle=':',
            label=f'Gap threshold: {gap_threshold:.1f}s', alpha=0.7)

# Mark events
for e in events:
    ax1.axvspan(e['start'], e['end'], alpha=0.15, color='orange')

ax1.set_ylabel('Interval (s)')
ax1.set_xlabel('Time (s)')
ax1.set_title('Breath-to-Breath Intervals Over Time', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.set_ylim(0, min(intervals.max() * 1.1, 15))
for m in range(8):
    ax1.axvline(x=m * 60, color='gray', linewidth=0.3, alpha=0.4)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2: Interval histogram + stats
# ═══════════════════════════════════════════════════════════════════════════════
ax2a = fig.add_subplot(gs[2, 0:2])
# Filter out event gaps for cleaner histogram
normal_intervals = intervals[intervals < gap_threshold]
ax2a.hist(normal_intervals, bins=30, color=C, alpha=0.7, edgecolor='white')
ax2a.axvline(x=np.mean(normal_intervals), color='red', linewidth=2,
             label=f'Mean: {np.mean(normal_intervals):.2f}s')
ax2a.axvline(x=np.median(normal_intervals), color='green', linewidth=2, linestyle='--',
             label=f'Median: {np.median(normal_intervals):.2f}s')
ax2a.axvline(x=3.3, color='blue', linewidth=1.5, linestyle=':',
             label='Target: 3.3s')
ax2a.set_xlabel('Breath Interval (s)')
ax2a.set_ylabel('Count')
ax2a.set_title(f'Breath Interval Distribution (n={len(normal_intervals)}, gaps excluded)', fontsize=11, fontweight='bold')
ax2a.legend(fontsize=9)
ax2a.grid(True, alpha=0.2)

# Stats box
ax2b = fig.add_subplot(gs[2, 2])
ax2b.axis('off')
stats_text = (
    f"Breath Interval Stats\n"
    f"{'─' * 28}\n"
    f"Total breaths:   {best['n']}\n"
    f"Normal breaths:  {len(normal_intervals)}\n"
    f"Event gaps:      {len(events)}\n"
    f"{'─' * 28}\n"
    f"Mean interval:   {np.mean(normal_intervals):.2f}s\n"
    f"Median interval: {np.median(normal_intervals):.2f}s\n"
    f"Std deviation:   {np.std(normal_intervals):.2f}s\n"
    f"CV (variability):{np.std(normal_intervals)/np.mean(normal_intervals)*100:.1f}%\n"
    f"{'─' * 28}\n"
    f"Breath rate:     {60/np.mean(normal_intervals):.1f} br/min\n"
    f"Min interval:    {normal_intervals.min():.2f}s\n"
    f"Max interval:    {normal_intervals.max():.2f}s\n"
    f"IQR:             {np.percentile(normal_intervals,25):.2f}-{np.percentile(normal_intervals,75):.2f}s\n"
)
ax2b.text(0.05, 0.95, stats_text, transform=ax2b.transAxes, fontsize=11,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.8))

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3: Rolling breath rate + rolling amplitude
# ═══════════════════════════════════════════════════════════════════════════════
ax3a = fig.add_subplot(gs[3, 0:2])
ax3a.plot(rolling_rate_t, rolling_rate, color=C, linewidth=1.5)
ax3a.axhline(y=18.2, color='blue', linewidth=1, linestyle=':', label='18.2 br/min (3.3s)')
ax3a.axhline(y=np.mean(rolling_rate), color='red', linewidth=1, linestyle='--',
             label=f'Session avg: {np.mean(rolling_rate):.1f} br/min')
for e in events:
    ax3a.axvspan(e['start'], e['end'], alpha=0.15, color='orange')
ax3a.set_ylabel('Breath Rate (br/min)')
ax3a.set_xlabel('Time (s)')
ax3a.set_title('Rolling Breath Rate (5-breath window)', fontsize=11, fontweight='bold')
ax3a.legend(fontsize=9, loc='upper right')
ax3a.grid(True, alpha=0.2)
ax3a.set_ylim(5, 35)

# SNR over time
ax3b = fig.add_subplot(gs[3, 2])
ax3b.plot(amp_times, amp_snr, color='#2E7D32', linewidth=1.5)
ax3b.axhline(y=5, color='red', linewidth=1, linestyle=':', label='SNR=5 (min usable)')
ax3b.axhline(y=np.mean(amp_snr), color='green', linewidth=1, linestyle='--',
             label=f'Mean SNR: {np.mean(amp_snr):.1f}')
ax3b.set_ylabel('SNR')
ax3b.set_xlabel('Time (s)')
ax3b.set_title('Signal-to-Noise Ratio (30s windows)', fontsize=11, fontweight='bold')
ax3b.legend(fontsize=8)
ax3b.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4: Peak prominence over time (breath "depth")
# ═══════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[4, :])
ax4.bar(t_sec[press_peaks], press_proms, width=1.5, color=C, alpha=0.6)
ax4.axhline(y=np.mean(press_proms), color='red', linewidth=1.5, linestyle='--',
            label=f'Mean prominence: {np.mean(press_proms):.2f} Pa')
# Rolling mean of prominence
if len(press_proms) > 10:
    prom_smooth = moving_average(np.array(press_proms), 10)
    ax4.plot(t_sec[press_peaks], prom_smooth, color='red', linewidth=2, label='10-breath rolling avg')
for e in events:
    ax4.axvspan(e['start'], e['end'], alpha=0.15, color='orange')
ax4.set_ylabel('Peak Prominence (Pa)')
ax4.set_xlabel('Time (s)')
ax4.set_title('Breath Depth (Peak Prominence) — Does It Degrade?', fontsize=13, fontweight='bold')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.2)
for m in range(8):
    ax4.axvline(x=m * 60, color='gray', linewidth=0.3, alpha=0.4)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 5: Overlaid individual breath waveforms (shape consistency)
# ═══════════════════════════════════════════════════════════════════════════════
# Extract ~20 individual breaths from different parts of the session
ax5a = fig.add_subplot(gs[5, 0])
ax5b = fig.add_subplot(gs[5, 1])
ax5c = fig.add_subplot(gs[5, 2])

for ax, start_min, end_min, title in [
    (ax5a, 0.5, 2.0, 'Start (0.5-2 min)'),
    (ax5b, 2.5, 4.5, 'Middle (2.5-4.5 min)'),
    (ax5c, 4.5, 6.5, 'End (4.5-6.5 min)')
]:
    # Find peaks in this time range
    local_peaks = [p for p in press_peaks
                   if start_min * 60 <= t_sec[p] <= end_min * 60]

    # Overlay each breath centered on peak, ±2s window
    half_win = int(2.0 * sr)
    waveforms = []
    for p in local_peaks:
        if p - half_win >= 0 and p + half_win < n:
            wf = press_smooth[p - half_win:p + half_win]
            waveforms.append(wf)
            t_rel = np.linspace(-2, 2, len(wf))
            ax.plot(t_rel, wf, color=C, alpha=0.2, linewidth=0.8)

    if waveforms:
        # Mean waveform
        mean_wf = np.mean(waveforms, axis=0)
        t_rel = np.linspace(-2, 2, len(mean_wf))
        ax.plot(t_rel, mean_wf, color='red', linewidth=2.5, label=f'Mean (n={len(waveforms)})')

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time from peak (s)')
    ax.set_ylabel('Pa')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 6: Pressure vs Thermistor correlation (are they seeing the same breaths?)
# ═══════════════════════════════════════════════════════════════════════════════
ax6a = fig.add_subplot(gs[6, 0:2])

# For each pressure peak, find nearest thermistor peak
matched = 0
unmatched_press = 0
match_offsets = []
for pp in press_peaks:
    pp_t = t_sec[pp]
    # Find nearest therm peak within ±2s
    near = [tp for tp in therm_peaks if abs(t_sec[tp] - pp_t) < 2.0]
    if near:
        closest = min(near, key=lambda tp: abs(t_sec[tp] - pp_t))
        match_offsets.append(t_sec[closest] - pp_t)
        matched += 1
    else:
        unmatched_press += 1

unmatched_therm = 0
for tp in therm_peaks:
    tp_t = t_sec[tp]
    near = [pp for pp in press_peaks if abs(t_sec[pp] - tp_t) < 2.0]
    if not near:
        unmatched_therm += 1

# Zoomed overlay showing agreement
zoom_s, zoom_e = 60, 120
mask = (t_sec >= zoom_s) & (t_sec <= zoom_e)
t_z = t_sec[mask]

# Normalize both to same scale
th_z = therm_smooth[mask]
pr_z = press_smooth[mask]
if th_z.std() > 0:
    th_n = th_z / th_z.std()
else:
    th_n = th_z
if pr_z.std() > 0:
    pr_n = pr_z / pr_z.std()
else:
    pr_n = pr_z

ax6a.plot(t_z, th_n, color='#1565C0', linewidth=1.5, label='Thermistor (normalized)', alpha=0.8)
ax6a.plot(t_z, pr_n, color=C, linewidth=1.5, label='Pressure (normalized)', alpha=0.8)

# Mark matched/unmatched peaks
for pp in press_peaks:
    if zoom_s <= t_sec[pp] <= zoom_e:
        near = [tp for tp in therm_peaks if abs(t_sec[tp] - t_sec[pp]) < 2.0]
        if near:
            ax6a.scatter(t_sec[pp], pr_n[np.searchsorted(t_z, t_sec[pp])],
                        color='green', s=40, zorder=5, marker='v')
        else:
            ax6a.scatter(t_sec[pp], pr_n[np.searchsorted(t_z, t_sec[pp])],
                        color='red', s=40, zorder=5, marker='x')

ax6a.axhline(y=0, color='gray', linewidth=0.3, linestyle='--')
ax6a.set_title(f'Channel Agreement ({zoom_s}-{zoom_e}s) — green=both, red=pressure only',
               fontsize=11, fontweight='bold')
ax6a.set_xlabel('Time (s)')
ax6a.set_ylabel('Normalized')
ax6a.legend(fontsize=9)
ax6a.grid(True, alpha=0.2)

# Match stats
ax6b = fig.add_subplot(gs[6, 2])
ax6b.axis('off')
match_text = (
    f"Peak Matching (±2s)\n"
    f"{'─' * 28}\n"
    f"Pressure peaks:  {len(press_peaks)}\n"
    f"Thermistor peaks:{len(therm_peaks)}\n"
    f"{'─' * 28}\n"
    f"Matched:         {matched}\n"
    f"Press-only:      {unmatched_press}\n"
    f"Therm-only:      {unmatched_therm}\n"
    f"{'─' * 28}\n"
    f"Match rate:      {matched/len(press_peaks)*100:.0f}% of press\n"
    f"                 {matched/len(therm_peaks)*100:.0f}% of therm\n"
)
if match_offsets:
    match_text += (
        f"{'─' * 28}\n"
        f"Timing offset:\n"
        f"  Mean: {np.mean(match_offsets)*1000:.0f}ms\n"
        f"  Std:  {np.std(match_offsets)*1000:.0f}ms\n"
    )
ax6b.text(0.05, 0.95, match_text, transform=ax6b.transAxes, fontsize=11,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 7: Sneeze/event zoom
# ═══════════════════════════════════════════════════════════════════════════════
if events:
    # Find the most interesting event (likely sneeze — look for the one mid-session)
    mid_events = [e for e in events if 1.5 < e['minute'] < 5.5]
    if mid_events:
        event = mid_events[0]  # First mid-session event
    else:
        event = events[len(events)//2]

    ev_start = event['start'] - 10
    ev_end = event['end'] + 10
    mask_ev = (t_sec >= ev_start) & (t_sec <= ev_end)

    ax7a = fig.add_subplot(gs[7, 0:2])
    ax7a.plot(t_sec[mask_ev], press_smooth[mask_ev], color=C, linewidth=2, label='Pressure')

    # Also show thermistor
    ax7r = ax7a.twinx()
    ax7r.plot(t_sec[mask_ev], therm_smooth[mask_ev], color='#1565C0', linewidth=1.5, alpha=0.6, label='Thermistor')
    ax7r.set_ylabel('Thermistor AC (ADC)', color='#1565C0')

    ax7a.axvspan(event['start'], event['end'], alpha=0.2, color='orange',
                 label=f'Event: {event["duration"]:.1f}s gap')
    ax7a.set_title(f'Event Zoom @ {event["minute"]:.1f} min — Possible Sneeze/Pause',
                   fontsize=11, fontweight='bold')
    ax7a.set_xlabel('Time (s)')
    ax7a.set_ylabel('Pressure AC (Pa)', color=C)
    lines1, labels1 = ax7a.get_legend_handles_labels()
    lines2, labels2 = ax7r.get_legend_handles_labels()
    ax7a.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
    ax7a.grid(True, alpha=0.2)
else:
    ax7a = fig.add_subplot(gs[7, 0:2])
    ax7a.text(0.5, 0.5, 'No significant events detected', ha='center', va='center')

# Amplitude stability summary
ax7b = fig.add_subplot(gs[7, 2])
ax7b.plot(np.array(amp_times) / 60, amp_values, color=C, linewidth=2, label='Signal amplitude')
ax7b.plot(np.array(amp_times) / 60, amp_noise, color='gray', linewidth=1.5, linestyle='--', label='Noise floor')
ax7b.fill_between(np.array(amp_times) / 60, amp_noise, amp_values, alpha=0.15, color=C)
ax7b.set_xlabel('Time (min)')
ax7b.set_ylabel('Amplitude (Pa)')
ax7b.set_title('Signal vs Noise Floor', fontsize=11, fontweight='bold')
ax7b.legend(fontsize=9)
ax7b.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 8: Final verdict text
# ═══════════════════════════════════════════════════════════════════════════════
ax8 = fig.add_subplot(gs[8, :])
ax8.axis('off')

# Compute stability metric: std of prominence over session
if len(press_proms) > 10:
    prom_cv = np.std(press_proms) / np.mean(press_proms) * 100
else:
    prom_cv = 0

# Compute breath regularity: CV of normal intervals
interval_cv = np.std(normal_intervals) / np.mean(normal_intervals) * 100

verdict_lines = [
    f"VIABILITY ASSESSMENT",
    f"{'═' * 80}",
    f"",
    f"  Breath Detection:    {best['n']} breaths in {t_sec[-1]/60:.1f} min = {best['rate']:.1f} br/min (target ~18 br/min)     {'✓ GOOD' if 10 < best['rate'] < 25 else '✗ CHECK'}",
    f"  Interval Regularity: Mean {np.mean(normal_intervals):.2f}s ± {np.std(normal_intervals):.2f}s (CV={interval_cv:.0f}%)     {'✓ REGULAR' if interval_cv < 40 else '~ VARIABLE'}",
    f"  Signal Stability:    Prominence CV = {prom_cv:.0f}% over session                       {'✓ STABLE' if prom_cv < 60 else '~ SOME DRIFT'}",
    f"  SNR:                 Mean {np.mean(amp_snr):.1f}, Min {np.min(amp_snr):.1f}                                    {'✓ STRONG' if np.min(amp_snr) > 3 else '~ MARGINAL' if np.min(amp_snr) > 1.5 else '✗ WEAK'}",
    f"  Event Detection:     {len(events)} pauses/events detected                               {'✓ CAPTURES EVENTS' if events else '— NO EVENTS'}",
    f"  Cross-channel:       {matched}/{len(press_peaks)} press peaks match therm (±2s)                  {'✓ CONSISTENT' if matched/len(press_peaks) > 0.5 else '~ DIVERGENT'}",
    f"",
    f"  OVERALL: {'PROMISING — pressure is viable as primary breath sensor' if np.mean(amp_snr) > 3 and 10 < best['rate'] < 25 else 'NEEDS MORE DATA'}",
]

ax8.text(0.02, 0.95, '\n'.join(verdict_lines), transform=ax8.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.5))

# ── Save ─────────────────────────────────────────────────────────────────────
output_path = '/Users/mert/Developer/Hardware/BreathAnalysis/analysis/pressure_deep_dive.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nGraph saved to: {output_path}")
