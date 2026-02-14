#!/usr/bin/env python3
"""
All-sensors 7-minute session analysis.
Compares thermistor, BMP280 pressure, and MAX30102 IR for breath detection.
Focus: Does pressure maintain signal strength where thermistor degrades?
"""

import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Helpers ──────────────────────────────────────────────────────────────────

def moving_average(signal, window):
    if len(signal) < window:
        return np.full_like(signal, np.mean(signal))
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

# ── Load data ────────────────────────────────────────────────────────────────

path = '/Users/mert/Downloads/allSensors7Min.gz'
with gzip.open(path, 'rt') as f:
    data = json.load(f)

samples = data['samples']
n = len(samples)

# v3 format: [timestamp, thermistor, ir, red, temp, humidity, pressureDelta]
ts = np.array([s[0] for s in samples], dtype=np.float64)
therm = np.array([s[1] for s in samples], dtype=np.float64)
ir = np.array([s[2] for s in samples], dtype=np.float64)
red = np.array([s[3] for s in samples], dtype=np.float64)
temp = np.array([s[4] for s in samples], dtype=np.float64)
pressure = np.array([s[6] for s in samples], dtype=np.float64)

t_sec = (ts - ts[0]) / 1000.0
sr = (n - 1) / t_sec[-1]
duration_min = t_sec[-1] / 60.0

print(f"Samples: {n}, Duration: {t_sec[-1]:.1f}s ({duration_min:.1f} min), Rate: {sr:.1f} Hz")

# ── Signal processing ────────────────────────────────────────────────────────

window_15s = int(15 * sr)
min_peak_dist = int(2.0 * sr)  # Min 2s between breaths

# Thermistor: baseline removal + light smoothing
therm_baseline = moving_average(therm, window_15s)
therm_ac = therm - therm_baseline
therm_smooth = moving_average(therm_ac, int(0.5 * sr))

# Pressure: baseline removal + light smoothing
press_baseline = moving_average(pressure, window_15s)
press_ac = pressure - press_baseline
press_smooth = moving_average(press_ac, int(0.3 * sr))

# IR: baseline removal (2.5s window for HRV is different from breath)
ir_baseline_breath = moving_average(ir, window_15s)
ir_ac_breath = ir - ir_baseline_breath
ir_smooth_breath = moving_average(ir_ac_breath, int(0.5 * sr))

# Peak detection — thermistor
therm_peaks = find_peaks(therm_smooth, min_distance=min_peak_dist,
                         prominence_threshold=therm_smooth.std() * 0.3)

# Peak detection — pressure (try both polarities)
press_peaks_pos = find_peaks(press_smooth, min_distance=min_peak_dist,
                             prominence_threshold=press_smooth.std() * 0.15)
press_peaks_neg = find_peaks(-press_smooth, min_distance=min_peak_dist,
                             prominence_threshold=press_smooth.std() * 0.15)
if len(press_peaks_neg) > len(press_peaks_pos):
    press_peaks = press_peaks_neg
    press_polarity = "inverted (exhale=negative)"
    press_display = -press_smooth  # flip for display so peaks are positive
else:
    press_peaks = press_peaks_pos
    press_polarity = "normal"
    press_display = press_smooth

# Peak detection — IR (breath frequency)
ir_peaks = find_peaks(ir_smooth_breath, min_distance=min_peak_dist,
                      prominence_threshold=ir_smooth_breath.std() * 0.3)

print(f"Thermistor peaks: {len(therm_peaks)}")
print(f"Pressure peaks: {len(press_peaks)} ({press_polarity})")
print(f"IR breath peaks: {len(ir_peaks)}")

# ── Minute-by-minute signal strength ─────────────────────────────────────────

minute_samples = int(60 * sr)
minutes = []
therm_stds = []
press_stds = []
ir_stds = []
therm_peak_counts = []
press_peak_counts = []

for m in range(int(t_sec[-1] // 60)):
    s_start = m * minute_samples
    s_end = min((m + 1) * minute_samples, n)
    if s_end - s_start < minute_samples * 0.5:
        break
    minutes.append(m)
    therm_stds.append(therm_ac[s_start:s_end].std())
    press_stds.append(press_ac[s_start:s_end].std())
    ir_stds.append(ir_ac_breath[s_start:s_end].std())
    therm_peak_counts.append(sum(1 for p in therm_peaks if s_start <= p < s_end))
    press_peak_counts.append(sum(1 for p in press_peaks if s_start <= p < s_end))

# Normalize to minute 0 for degradation comparison
therm_norm = [s / therm_stds[0] * 100 for s in therm_stds] if therm_stds[0] > 0 else therm_stds
press_norm = [s / press_stds[0] * 100 for s in press_stds] if press_stds[0] > 0 else press_stds
ir_norm = [s / ir_stds[0] * 100 for s in ir_stds] if ir_stds[0] > 0 else ir_stds

# ── Create figure ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 28))
gs = GridSpec(8, 3, figure=fig, hspace=0.4, wspace=0.3,
              height_ratios=[2.5, 2.5, 2.5, 2, 2, 2, 1.8, 1.8])

fig.suptitle('All Sensors — 7 min Recording (Thermistor vs Pressure vs IR)\n'
             f'{data["startTime"][:19]}  •  {n} samples @ {sr:.0f}Hz  •  {duration_min:.1f} min',
             fontsize=16, fontweight='bold', y=0.995)

colors = {
    'therm': '#1565C0',
    'press': '#7B1FA2',
    'ir': '#2E7D32',
    'temp': '#E65100',
    'red': '#C62828',
}

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1: Full session — Thermistor AC
# ═══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t_sec, therm_ac, color=colors['therm'], linewidth=0.3, alpha=0.3)
ax1.plot(t_sec, therm_smooth, color=colors['therm'], linewidth=1.2, label='Smoothed')
if therm_peaks:
    ax1.scatter(t_sec[therm_peaks], therm_smooth[therm_peaks],
                color='red', s=25, zorder=5, label=f'Peaks ({len(therm_peaks)})')
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
# Shade minute regions alternately
for m in range(int(duration_min) + 1):
    if m % 2 == 0:
        ax1.axvspan(m * 60, (m + 1) * 60, alpha=0.04, color='blue')
    ax1.axvline(x=m * 60, color='gray', linewidth=0.3, alpha=0.5)
    ax1.text(m * 60 + 30, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else therm_smooth.max() * 0.9,
             f'min {m}', ha='center', fontsize=8, color='gray')
ax1.set_ylabel('Thermistor AC (ADC counts)')
ax1.set_xlabel('Time (s)')
ax1.set_title('Thermistor — Detrended Breath Signal (15s baseline subtraction)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2: Full session — Pressure AC
# ═══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
ax2.plot(t_sec, press_ac, color=colors['press'], linewidth=0.3, alpha=0.3)
ax2.plot(t_sec, press_smooth, color=colors['press'], linewidth=1.2, label='Smoothed')
if press_peaks:
    # Show peaks on the smooth signal (not flipped)
    ax2.scatter(t_sec[press_peaks], press_smooth[press_peaks],
                color='red', s=25, zorder=5, label=f'Peaks ({len(press_peaks)})')
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
for m in range(int(duration_min) + 1):
    if m % 2 == 0:
        ax2.axvspan(m * 60, (m + 1) * 60, alpha=0.04, color='purple')
    ax2.axvline(x=m * 60, color='gray', linewidth=0.3, alpha=0.5)
ax2.set_ylabel('Pressure AC (Pa)')
ax2.set_xlabel('Time (s)')
ax2.set_title(f'BMP280 Pressure — Detrended Breath Signal ({press_polarity})', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3: Full session — IR AC (breath-rate component)
# ═══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
ax3.plot(t_sec, ir_ac_breath, color=colors['ir'], linewidth=0.3, alpha=0.3)
ax3.plot(t_sec, ir_smooth_breath, color=colors['ir'], linewidth=1.2, label='Smoothed')
if ir_peaks:
    ax3.scatter(t_sec[ir_peaks], ir_smooth_breath[ir_peaks],
                color='red', s=25, zorder=5, label=f'Peaks ({len(ir_peaks)})')
ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
for m in range(int(duration_min) + 1):
    if m % 2 == 0:
        ax3.axvspan(m * 60, (m + 1) * 60, alpha=0.04, color='green')
    ax3.axvline(x=m * 60, color='gray', linewidth=0.3, alpha=0.5)
ax3.set_ylabel('IR AC (counts)')
ax3.set_xlabel('Time (s)')
ax3.set_title('MAX30102 IR — Breath-Rate Component (15s baseline subtraction)', fontsize=13, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4: Zoomed — First 30 seconds (all 3 channels overlaid, normalized)
# ═══════════════════════════════════════════════════════════════════════════════
for col, (zoom_s, zoom_e, label) in enumerate([
    (15, 45, 'Start (15-45s)'),
    (180, 210, 'Middle (180-210s)'),
    (360, 390, 'End (360-390s)')
]):
    ax = fig.add_subplot(gs[3, col])
    mask = (t_sec >= zoom_s) & (t_sec <= zoom_e)

    # Normalize each to its own range for overlay
    t_z = t_sec[mask]
    th_z = therm_smooth[mask]
    pr_z = press_smooth[mask]

    if th_z.std() > 0:
        th_n = (th_z - th_z.mean()) / th_z.std()
    else:
        th_n = th_z - th_z.mean()
    if pr_z.std() > 0:
        pr_n = (pr_z - pr_z.mean()) / pr_z.std()
    else:
        pr_n = pr_z - pr_z.mean()

    ax.plot(t_z, th_n, color=colors['therm'], linewidth=1.8, label='Thermistor', alpha=0.8)
    ax.plot(t_z, pr_n, color=colors['press'], linewidth=1.8, label='Pressure', alpha=0.8)
    ax.axhline(y=0, color='gray', linewidth=0.3, linestyle='--')
    ax.set_title(f'{label}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized')
    ax.set_xlabel('Time (s)')
    if col == 0:
        ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 5: Side-by-side zoomed — raw signals at start vs end
# ═══════════════════════════════════════════════════════════════════════════════
# Thermistor start vs end
ax5a = fig.add_subplot(gs[4, 0])
m1 = (t_sec >= 15) & (t_sec <= 45)
m2 = (t_sec >= 360) & (t_sec <= 390)
ax5a.plot(t_sec[m1] - 15, therm_smooth[m1], color=colors['therm'], linewidth=2, label='Start (15-45s)')
ax5a.plot(t_sec[m2] - 360, therm_smooth[m2], color=colors['therm'], linewidth=2, linestyle='--', alpha=0.6, label='End (360-390s)')
ax5a.set_title('Thermistor: Start vs End', fontsize=11, fontweight='bold')
ax5a.set_ylabel('ADC counts (AC)')
ax5a.set_xlabel('Relative time (s)')
ax5a.legend(fontsize=8)
ax5a.grid(True, alpha=0.2)

# Pressure start vs end
ax5b = fig.add_subplot(gs[4, 1])
ax5b.plot(t_sec[m1] - 15, press_smooth[m1], color=colors['press'], linewidth=2, label='Start (15-45s)')
ax5b.plot(t_sec[m2] - 360, press_smooth[m2], color=colors['press'], linewidth=2, linestyle='--', alpha=0.6, label='End (360-390s)')
ax5b.set_title('Pressure: Start vs End', fontsize=11, fontweight='bold')
ax5b.set_ylabel('Pa (AC)')
ax5b.set_xlabel('Relative time (s)')
ax5b.legend(fontsize=8)
ax5b.grid(True, alpha=0.2)

# IR start vs end
ax5c = fig.add_subplot(gs[4, 2])
ax5c.plot(t_sec[m1] - 15, ir_smooth_breath[m1], color=colors['ir'], linewidth=2, label='Start (15-45s)')
ax5c.plot(t_sec[m2] - 360, ir_smooth_breath[m2], color=colors['ir'], linewidth=2, linestyle='--', alpha=0.6, label='End (360-390s)')
ax5c.set_title('IR: Start vs End', fontsize=11, fontweight='bold')
ax5c.set_ylabel('Counts (AC)')
ax5c.set_xlabel('Relative time (s)')
ax5c.legend(fontsize=8)
ax5c.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 6: Zoomed raw signals per minute (thermistor vs pressure, 10s windows)
# ═══════════════════════════════════════════════════════════════════════════════
# Show 10s windows from minute 0, 3, and 6 to see degradation
for col, minute in enumerate([0, 3, 6]):
    ax = fig.add_subplot(gs[5, col])
    center = minute * 60 + 35  # offset by 35s to avoid initial transient for min 0
    w_start = center - 5
    w_end = center + 5
    mask = (t_sec >= w_start) & (t_sec <= w_end)
    t_w = t_sec[mask]

    # Raw AC signals
    th_w = therm_smooth[mask]
    pr_w = press_smooth[mask]

    # Dual y-axis
    ax.plot(t_w, th_w, color=colors['therm'], linewidth=2, label='Thermistor')
    ax.set_ylabel('Thermistor AC (ADC)', color=colors['therm'], fontsize=9)
    ax.tick_params(axis='y', labelcolor=colors['therm'])

    ax2r = ax.twinx()
    ax2r.plot(t_w, pr_w, color=colors['press'], linewidth=2, label='Pressure')
    ax2r.set_ylabel('Pressure AC (Pa)', color=colors['press'], fontsize=9)
    ax2r.tick_params(axis='y', labelcolor=colors['press'])

    ax.set_title(f'Minute {minute} ({w_start:.0f}-{w_end:.0f}s)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.2)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 7: Signal strength degradation over time
# ═══════════════════════════════════════════════════════════════════════════════
ax7a = fig.add_subplot(gs[6, 0:2])
x = np.arange(len(minutes))
ax7a.plot(x, therm_norm, 'o-', color=colors['therm'], linewidth=2, markersize=8, label='Thermistor')
ax7a.plot(x, press_norm, 's-', color=colors['press'], linewidth=2, markersize=8, label='Pressure')
ax7a.plot(x, ir_norm, '^-', color=colors['ir'], linewidth=2, markersize=8, label='IR')
ax7a.axhline(y=100, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax7a.axhline(y=50, color='red', linewidth=1, linestyle=':', alpha=0.5, label='50% threshold')
ax7a.set_xlabel('Minute', fontsize=11)
ax7a.set_ylabel('Signal Strength (% of minute 0)', fontsize=11)
ax7a.set_title('CRITICAL: Signal Degradation Over Time (normalized to min 0)', fontsize=13, fontweight='bold')
ax7a.set_xticks(x)
ax7a.set_xticklabels([str(m) for m in minutes])
ax7a.legend(fontsize=10)
ax7a.grid(True, alpha=0.3)
ax7a.set_ylim(0, max(max(therm_norm), max(press_norm), max(ir_norm)) * 1.1)
# Annotate final values
for vals, name, color in [(therm_norm, 'Therm', colors['therm']),
                            (press_norm, 'Press', colors['press']),
                            (ir_norm, 'IR', colors['ir'])]:
    ax7a.annotate(f'{vals[-1]:.0f}%', xy=(len(minutes) - 1, vals[-1]),
                  xytext=(10, 0), textcoords='offset points',
                  fontsize=10, fontweight='bold', color=color)

# Temperature drift
ax7b = fig.add_subplot(gs[6, 2])
ax7b.plot(t_sec / 60, temp, color=colors['temp'], linewidth=1.5)
ax7b.set_xlabel('Time (min)')
ax7b.set_ylabel('Temperature (°C)')
ax7b.set_title(f'BMP280 Temp Drift ({temp[-1]-temp[0]:+.1f}°C)', fontsize=11, fontweight='bold')
ax7b.grid(True, alpha=0.3)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 8: Breath rate per minute + summary stats table
# ═══════════════════════════════════════════════════════════════════════════════
ax8a = fig.add_subplot(gs[7, 0:2])
width = 0.35
ax8a.bar(x - width / 2, therm_peak_counts, width, color=colors['therm'], alpha=0.8, label='Thermistor')
ax8a.bar(x + width / 2, press_peak_counts, width, color=colors['press'], alpha=0.8, label='Pressure')
ax8a.set_xlabel('Minute')
ax8a.set_ylabel('Breath Peaks Detected')
ax8a.set_title('Breaths Detected Per Minute', fontsize=13, fontweight='bold')
ax8a.set_xticks(x)
ax8a.set_xticklabels([str(m) for m in minutes])
ax8a.legend()
ax8a.grid(True, alpha=0.3, axis='y')

# Summary stats table
ax8b = fig.add_subplot(gs[7, 2])
ax8b.axis('off')

# Build comparison table
table_data = [
    ['Metric', 'Thermistor', 'Pressure'],
    ['Total peaks', str(len(therm_peaks)), str(len(press_peaks))],
    ['Signal @ min 0', f'{therm_stds[0]:.1f} ADC', f'{press_stds[0]:.2f} Pa'],
    ['Signal @ last', f'{therm_stds[-1]:.1f} ADC', f'{press_stds[-1]:.2f} Pa'],
    ['Retained', f'{therm_norm[-1]:.0f}%', f'{press_norm[-1]:.0f}%'],
    ['Temp drift', f'{temp[-1]-temp[0]:+.1f}°C', '—'],
]

if len(therm_peaks) >= 2:
    ti = np.diff(t_sec[therm_peaks])
    table_data.append(['Avg rate', f'{60/ti.mean():.1f} br/min', ''])
if len(press_peaks) >= 2:
    pi = np.diff(t_sec[press_peaks])
    table_data[-1][2] = f'{60/pi.mean():.1f} br/min'

table = ax8b.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.5)

# Color header row
for j in range(3):
    table[0, j].set_facecolor('#E0E0E0')
    table[0, j].set_text_props(fontweight='bold')

# Color the "Retained" row based on values
for j, color in [(1, colors['therm']), (2, colors['press'])]:
    table[4, j].set_text_props(fontweight='bold')

ax8b.set_title('Summary Comparison', fontsize=11, fontweight='bold')

# ── Save ─────────────────────────────────────────────────────────────────────
output_path = '/Users/mert/Developer/Hardware/BreathAnalysis/analysis/all_sensors_7min_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nGraph saved to: {output_path}")

# ── Print detailed stats ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"ALL SENSORS — 7 MINUTE SESSION ANALYSIS")
print(f"{'='*70}")
print(f"Duration: {t_sec[-1]:.1f}s ({duration_min:.1f} min), {n} samples @ {sr:.0f} Hz")
print(f"\n--- THERMISTOR ---")
print(f"  Raw range: {therm.min():.0f} to {therm.max():.0f} ADC")
print(f"  Peaks detected: {len(therm_peaks)}")
if len(therm_peaks) >= 2:
    ti = np.diff(t_sec[therm_peaks])
    print(f"  Avg breath interval: {ti.mean():.2f}s → {60/ti.mean():.1f} br/min")

print(f"\n--- BMP280 PRESSURE ---")
print(f"  Raw range: {pressure.min():.2f} to {pressure.max():.2f} Pa")
print(f"  Peaks detected: {len(press_peaks)} ({press_polarity})")
if len(press_peaks) >= 2:
    pi = np.diff(t_sec[press_peaks])
    print(f"  Avg breath interval: {pi.mean():.2f}s → {60/pi.mean():.1f} br/min")

print(f"\n--- MAX30102 IR ---")
print(f"  Raw range: {ir.min():.0f} to {ir.max():.0f}")
print(f"  Breath peaks detected: {len(ir_peaks)}")

print(f"\n--- TEMPERATURE ---")
print(f"  Start: {temp[:int(10*sr)].mean():.2f}°C → End: {temp[-int(10*sr):].mean():.2f}°C")
print(f"  Drift: {temp[-int(10*sr):].mean() - temp[:int(10*sr)].mean():+.2f}°C")

print(f"\n--- MINUTE-BY-MINUTE DEGRADATION ---")
print(f"{'Min':>4} {'Therm std':>10} {'Therm %':>8} {'Press std':>10} {'Press %':>8} {'IR std':>10} {'IR %':>8} {'T peaks':>8} {'P peaks':>8}")
print("-" * 82)
for i, m in enumerate(minutes):
    print(f"{m:4d} {therm_stds[i]:10.2f} {therm_norm[i]:7.0f}% {press_stds[i]:10.3f} {press_norm[i]:7.0f}% {ir_stds[i]:10.1f} {ir_norm[i]:7.0f}% {therm_peak_counts[i]:8d} {press_peak_counts[i]:8d}")

# Verdict
print(f"\n{'='*70}")
print(f"VERDICT")
print(f"{'='*70}")
therm_retained = therm_norm[-1]
press_retained = press_norm[-1]
print(f"Thermistor retains {therm_retained:.0f}% of initial signal at minute {minutes[-1]}")
print(f"Pressure retains {press_retained:.0f}% of initial signal at minute {minutes[-1]}")
if press_retained > therm_retained * 1.5:
    print(f"→ PRESSURE IS SIGNIFICANTLY MORE STABLE ({press_retained/therm_retained:.1f}x better retention)")
elif press_retained > therm_retained:
    print(f"→ Pressure is somewhat more stable ({press_retained/therm_retained:.1f}x better retention)")
elif abs(press_retained - therm_retained) < 10:
    print(f"→ Both sensors degrade similarly")
else:
    print(f"→ Thermistor is more stable (unexpected)")
