#!/usr/bin/env python3
"""
BMP280 pressure session analysis with graphs.
Compares thermistor vs pressure for breath detection.
"""

import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys

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

# ── Load data ────────────────────────────────────────────────────────────────

path = sys.argv[1] if len(sys.argv) > 1 else '/Users/mert/Downloads/BMP280_Just_Pressure.gz'
with gzip.open(path, 'rt') as f:
    data = json.load(f)

samples = data['samples']
n = len(samples)

ts = np.array([s[0] for s in samples], dtype=np.float64)
therm = np.array([s[1] for s in samples], dtype=np.float64)
ir = np.array([s[2] for s in samples], dtype=np.float64)
temp = np.array([s[4] for s in samples], dtype=np.float64)
pressure = np.array([s[6] for s in samples], dtype=np.float64)

t_sec = (ts - ts[0]) / 1000.0  # Time in seconds
sr = (n - 1) / t_sec[-1]

print(f"Samples: {n}, Duration: {t_sec[-1]:.1f}s ({t_sec[-1]/60:.1f} min), Rate: {sr:.1f} Hz")
print(f"Pressure range: {pressure.min():.2f} to {pressure.max():.2f} Pa")
print(f"Thermistor range: {therm.min():.0f} to {therm.max():.0f}")

# ── Signal processing ────────────────────────────────────────────────────────

window_15s = int(15 * sr)
window_3s = int(3 * sr)
min_peak_dist = int(2.0 * sr)

# Thermistor: baseline removal + smoothing
therm_baseline = moving_average(therm, window_15s)
therm_ac = therm - therm_baseline
therm_smooth = moving_average(therm_ac, int(0.5 * sr))

# Pressure: baseline removal + smoothing
press_baseline = moving_average(pressure, window_15s)
press_ac = pressure - press_baseline
press_smooth = moving_average(press_ac, int(0.3 * sr))

# Peak detection
therm_peaks = find_peaks(therm_smooth, min_distance=min_peak_dist,
                          prominence_threshold=therm_smooth.std() * 0.3)
press_peaks = find_peaks(press_smooth, min_distance=min_peak_dist,
                          prominence_threshold=press_smooth.std() * 0.15)
# Also try inverted pressure (exhale might be negative depending on sensor orientation)
press_peaks_inv = find_peaks(-press_smooth, min_distance=min_peak_dist,
                              prominence_threshold=press_smooth.std() * 0.15)

# Use whichever polarity found more regular peaks
if len(press_peaks_inv) > len(press_peaks):
    press_peaks = press_peaks_inv
    press_polarity = "inverted"
else:
    press_polarity = "normal"

print(f"Thermistor peaks: {len(therm_peaks)}")
print(f"Pressure peaks: {len(press_peaks)} ({press_polarity})")

# ── Create figure ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 22))
gs = GridSpec(6, 2, figure=fig, hspace=0.35, wspace=0.3,
              height_ratios=[2, 2, 2, 1.5, 1.5, 1.5])

fig.suptitle('BMP280 Pressure Session — 4 min Recording\n'
             'Subtle breathing + intentional blow at end',
             fontsize=16, fontweight='bold', y=0.98)

# ── Plot 1: Raw pressure (full session) ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t_sec, pressure, color='#7B1FA2', linewidth=0.5, alpha=0.6, label='Raw')
ax1.plot(t_sec, press_baseline, color='orange', linewidth=2, label='15s baseline')
ax1.set_ylabel('Pressure Δ (Pa)')
ax1.set_xlabel('Time (s)')
ax1.set_title('Raw Pressure Signal (BMP280)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
# Mark the intentional blow
blow_idx = np.argmax(pressure)
ax1.annotate(f'Intentional blow\n{pressure[blow_idx]:.1f} Pa',
             xy=(t_sec[blow_idx], pressure[blow_idx]),
             xytext=(t_sec[blow_idx]-30, pressure[blow_idx]-5),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red', fontweight='bold')

# ── Plot 2: Pressure AC (detrended) with peaks ──────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(t_sec, press_ac, color='#7B1FA2', linewidth=0.4, alpha=0.4, label='AC (raw)')
ax2.plot(t_sec, press_smooth, color='#7B1FA2', linewidth=1.5, label='AC (smoothed)')
if press_peaks:
    ax2.scatter(t_sec[press_peaks], press_smooth[press_peaks],
                color='red', s=40, zorder=5, label=f'Peaks ({len(press_peaks)})')
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax2.set_ylabel('Pressure AC (Pa)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Detrended Pressure + Breath Peaks', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# ── Plot 3: Thermistor AC with peaks (for comparison) ───────────────────────
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(t_sec, therm_ac, color='#1565C0', linewidth=0.4, alpha=0.4, label='AC (raw)')
ax3.plot(t_sec, therm_smooth, color='#1565C0', linewidth=1.5, label='AC (smoothed)')
if therm_peaks:
    ax3.scatter(t_sec[therm_peaks], therm_smooth[therm_peaks],
                color='red', s=40, zorder=5, label=f'Peaks ({len(therm_peaks)})')
ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax3.set_ylabel('Thermistor AC (ADC)')
ax3.set_xlabel('Time (s)')
ax3.set_title('Detrended Thermistor + Breath Peaks (Comparison)', fontsize=13, fontweight='bold')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# ── Plot 4: Zoomed 30s window (0-30s, start of session) ─────────────────────
zoom_start = 30  # seconds
zoom_end = 60
mask = (t_sec >= zoom_start) & (t_sec <= zoom_end)

ax4 = fig.add_subplot(gs[3, 0])
ax4.plot(t_sec[mask], press_smooth[mask], color='#7B1FA2', linewidth=2, label='Pressure')
zoom_press_peaks = [p for p in press_peaks if t_sec[p] >= zoom_start and t_sec[p] <= zoom_end]
if zoom_press_peaks:
    ax4.scatter(t_sec[zoom_press_peaks], press_smooth[zoom_press_peaks],
                color='red', s=60, zorder=5)
ax4.set_title(f'Pressure {zoom_start}-{zoom_end}s (normal breathing)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Pa')
ax4.set_xlabel('Time (s)')
ax4.grid(True, alpha=0.3)

ax4b = fig.add_subplot(gs[3, 1])
ax4b.plot(t_sec[mask], therm_smooth[mask], color='#1565C0', linewidth=2, label='Thermistor')
zoom_therm_peaks = [p for p in therm_peaks if t_sec[p] >= zoom_start and t_sec[p] <= zoom_end]
if zoom_therm_peaks:
    ax4b.scatter(t_sec[zoom_therm_peaks], therm_smooth[zoom_therm_peaks],
                 color='red', s=60, zorder=5)
ax4b.set_title(f'Thermistor {zoom_start}-{zoom_end}s (normal breathing)', fontsize=11, fontweight='bold')
ax4b.set_ylabel('ADC')
ax4b.set_xlabel('Time (s)')
ax4b.grid(True, alpha=0.3)

# ── Plot 5: Zoomed 30s window (subtle breathing ~180-210s) ──────────────────
sub_start = 180
sub_end = 210
mask2 = (t_sec >= sub_start) & (t_sec <= sub_end)

ax5 = fig.add_subplot(gs[4, 0])
ax5.plot(t_sec[mask2], press_smooth[mask2], color='#7B1FA2', linewidth=2)
sub_press_peaks = [p for p in press_peaks if t_sec[p] >= sub_start and t_sec[p] <= sub_end]
if sub_press_peaks:
    ax5.scatter(t_sec[sub_press_peaks], press_smooth[sub_press_peaks],
                color='red', s=60, zorder=5)
ax5.set_title(f'Pressure {sub_start}-{sub_end}s (subtle breathing)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Pa')
ax5.set_xlabel('Time (s)')
ax5.grid(True, alpha=0.3)

ax5b = fig.add_subplot(gs[4, 1])
ax5b.plot(t_sec[mask2], therm_smooth[mask2], color='#1565C0', linewidth=2)
sub_therm_peaks = [p for p in therm_peaks if t_sec[p] >= sub_start and t_sec[p] <= sub_end]
if sub_therm_peaks:
    ax5b.scatter(t_sec[sub_therm_peaks], therm_smooth[sub_therm_peaks],
                 color='red', s=60, zorder=5)
ax5b.set_title(f'Thermistor {sub_start}-{sub_end}s (subtle breathing)', fontsize=11, fontweight='bold')
ax5b.set_ylabel('ADC')
ax5b.set_xlabel('Time (s)')
ax5b.grid(True, alpha=0.3)

# ── Plot 6: Minute-by-minute signal strength comparison ──────────────────────
ax6 = fig.add_subplot(gs[5, 0])
minute_samples = int(60 * sr)
minutes = []
therm_stds = []
press_stds = []

for m in range(int(t_sec[-1] // 60)):
    s_start = m * minute_samples
    s_end = min((m + 1) * minute_samples, n)
    if s_end - s_start < minute_samples * 0.5:
        break
    minutes.append(m)
    therm_stds.append(therm_ac[s_start:s_end].std())
    press_stds.append(press_ac[s_start:s_end].std())

x = np.arange(len(minutes))
width = 0.35
ax6.bar(x - width/2, therm_stds, width, color='#1565C0', alpha=0.8, label='Thermistor')
ax6.bar(x + width/2, press_stds, width, color='#7B1FA2', alpha=0.8, label='Pressure')
ax6.set_xlabel('Minute')
ax6.set_ylabel('AC Std Dev')
ax6.set_title('Signal Strength by Minute', fontsize=11, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels([str(m) for m in minutes])
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# ── Plot 7: Temperature drift ───────────────────────────────────────────────
ax7 = fig.add_subplot(gs[5, 1])
ax7.plot(t_sec, temp, color='#E65100', linewidth=1.5)
ax7.set_xlabel('Time (s)')
ax7.set_ylabel('Temperature (°C)')
ax7.set_title(f'BMP280 Temperature (Δ{temp.max()-temp.min():.1f}°C drift)', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)

# ── Save ─────────────────────────────────────────────────────────────────────
output_path = '/Users/mert/Developer/Hardware/BreathAnalysis/analysis/bmp280_session_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nGraph saved to: {output_path}")

# ── Print stats ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"Duration: {t_sec[-1]:.1f}s ({t_sec[-1]/60:.1f} min)")
print(f"\nPressure signal:")
print(f"  Raw range: {pressure.min():.2f} to {pressure.max():.2f} Pa")
print(f"  AC std: {press_ac.std():.3f} Pa")
print(f"  Peaks detected: {len(press_peaks)}")
if len(press_peaks) >= 2:
    intervals = np.diff(t_sec[press_peaks])
    print(f"  Mean interval: {intervals.mean():.2f}s → {60/intervals.mean():.1f} br/min")

print(f"\nThermistor signal:")
print(f"  Raw range: {therm.min():.0f} to {therm.max():.0f}")
print(f"  AC std: {therm_ac.std():.2f}")
print(f"  Peaks detected: {len(therm_peaks)}")
if len(therm_peaks) >= 2:
    intervals = np.diff(t_sec[therm_peaks])
    print(f"  Mean interval: {intervals.mean():.2f}s → {60/intervals.mean():.1f} br/min")

print(f"\nTemperature:")
print(f"  Start: {temp[:int(10*sr)].mean():.2f}°C, End: {temp[-int(10*sr):].mean():.2f}°C")
print(f"  Drift: {temp[-int(10*sr):].mean() - temp[:int(10*sr)].mean():+.2f}°C")

# Per-minute breakdown
print(f"\n{'Min':>4} {'Therm std':>10} {'Press std':>10} {'Therm peaks':>12} {'Press peaks':>12}")
print("-" * 52)
for m in range(len(minutes)):
    s_start = m * minute_samples
    s_end = min((m + 1) * minute_samples, n)
    tp = sum(1 for p in therm_peaks if s_start <= p < s_end)
    pp = sum(1 for p in press_peaks if s_start <= p < s_end)
    print(f"{m:4d} {therm_stds[m]:10.2f} {press_stds[m]:10.3f} {tp:12d} {pp:12d}")
