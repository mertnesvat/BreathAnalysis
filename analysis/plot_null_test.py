#!/usr/bin/env python3
"""
Null Test Analysis — device recording with NO breathing nearby.
Runs the exact same analysis as the meditation session to check
if our breath detection is finding real signal or noise artifacts.
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# ── Load both sessions for comparison ──
with gzip.open('/Users/mert/Downloads/NullTest.gz', 'rt') as f:
    null_data = json.load(f)

with gzip.open('/Users/mert/Downloads/15MinsLatestChanges.gz', 'rt') as f:
    med_data = json.load(f)

# Parse null test
null_samples = np.array(null_data['samples'])
null_ts = null_samples[:, 0] / 1000.0
null_ir = null_samples[:, 1]
null_pressure = null_samples[:, 3]
null_duration = (null_ts[-1] - null_ts[0]) / 60
null_rate = (len(null_ts) - 1) / (null_ts[-1] - null_ts[0])

# Parse meditation
med_samples = np.array(med_data['samples'])
med_ts = med_samples[:, 0] / 1000.0
med_ir = med_samples[:, 1]
med_pressure = med_samples[:, 3]
med_duration = (med_ts[-1] - med_ts[0]) / 60
med_rate = (len(med_ts) - 1) / (med_ts[-1] - med_ts[0])

print(f"Null test: {null_duration:.1f} min, {len(null_ts)} samples")
print(f"Meditation: {med_duration:.1f} min, {len(med_ts)} samples")

# ═══════════════════════════════════════════════════
# Run IDENTICAL breath detection on both
# ═══════════════════════════════════════════════════

def detect_breaths(ts, pressure, rate):
    """Exact same algorithm as the meditation analysis."""
    window = int(15 * rate)
    baseline = np.convolve(pressure, np.ones(window)/window, mode='same')
    ac = pressure - baseline

    smooth_window = int(0.5 * rate)
    smooth = np.convolve(ac, np.ones(smooth_window)/smooth_window, mode='same')

    std = np.std(smooth)
    peaks, props = find_peaks(smooth,
                              height=std * 0.15,
                              distance=int(1.5 * rate),
                              prominence=std * 0.1)
    return ac, smooth, peaks

null_ac, null_smooth, null_peaks = detect_breaths(null_ts, null_pressure, null_rate)
med_ac, med_smooth, med_peaks = detect_breaths(med_ts, med_pressure, med_rate)

null_breaths = len(null_peaks)
med_breaths = len(med_peaks)
null_breath_rate = null_breaths / null_duration
med_breath_rate = med_breaths / med_duration

print(f"\n{'Metric':<30} {'Null Test':<20} {'Meditation':<20}")
print(f"{'─'*30} {'─'*20} {'─'*20}")
print(f"{'Detected breaths':<30} {null_breaths:<20} {med_breaths:<20}")
print(f"{'Breath rate (/min)':<30} {null_breath_rate:<20.1f} {med_breath_rate:<20.1f}")
print(f"{'Pressure raw std (Pa)':<30} {np.std(null_pressure):<20.2f} {np.std(med_pressure):<20.2f}")
print(f"{'Pressure AC std (Pa)':<30} {np.std(null_ac):<20.2f} {np.std(med_ac):<20.2f}")

if len(null_peaks) > 1:
    null_intervals = np.diff(null_ts[null_peaks])
    print(f"{'Mean interval (s)':<30} {np.mean(null_intervals):<20.2f}", end='')
else:
    print(f"{'Mean interval (s)':<30} {'N/A':<20}", end='')
if len(med_peaks) > 1:
    med_intervals = np.diff(med_ts[med_peaks])
    print(f" {np.mean(med_intervals):<20.2f}")
else:
    print(f" {'N/A':<20}")

# ═══════════════════════════════════════════════════
# Frequency analysis — the real discriminator
# ═══════════════════════════════════════════════════

def compute_psd(signal, rate):
    """Power spectral density via FFT."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0/rate)
    fft_vals = np.abs(np.fft.rfft(signal))**2 / n
    return freqs, fft_vals

null_freqs, null_psd = compute_psd(null_ac, null_rate)
med_freqs, med_psd = compute_psd(med_ac, med_rate)

# Breath band: 0.1 - 0.6 Hz (6-36 breaths/min)
breath_band = (0.1, 0.6)
null_breath_mask = (null_freqs >= breath_band[0]) & (null_freqs <= breath_band[1])
med_breath_mask = (med_freqs >= breath_band[0]) & (med_freqs <= breath_band[1])

null_breath_power = np.sum(null_psd[null_breath_mask])
med_breath_power = np.sum(med_psd[med_breath_mask])
null_total_power = np.sum(null_psd[null_freqs > 0.05])
med_total_power = np.sum(med_psd[med_freqs > 0.05])

null_breath_ratio = null_breath_power / null_total_power * 100 if null_total_power > 0 else 0
med_breath_ratio = med_breath_power / med_total_power * 100 if med_total_power > 0 else 0

print(f"\n{'Frequency Analysis':<30}")
print(f"{'─'*70}")
print(f"{'Breath band power (0.1-0.6Hz)':<30} {null_breath_power:<20.1f} {med_breath_power:<20.1f}")
print(f"{'Total power':<30} {null_total_power:<20.1f} {med_total_power:<20.1f}")
print(f"{'Breath band ratio':<30} {null_breath_ratio:<20.1f}% {med_breath_ratio:<20.1f}%")
print(f"{'SNR (med/null breath power)':<30} {med_breath_power/null_breath_power:.1f}x" if null_breath_power > 0 else "")

# ═══════════════════════════════════════════════════
# PLOTTING — side by side comparison
# ═══════════════════════════════════════════════════

fig, axes = plt.subplots(4, 2, figsize=(18, 16))
fig.suptitle('Null Test (No Breathing) vs Deep Meditation — Same Detection Algorithm',
             fontsize=14, fontweight='bold')

# ── Row 1: Raw pressure waveform ──
ax = axes[0, 0]
ax.plot(null_ts / 60, null_ac, color='#78909C', linewidth=0.5, alpha=0.7)
ax.plot(null_ts / 60, null_smooth, color='#37474F', linewidth=1.0)
if len(null_peaks) > 0:
    ax.plot(null_ts[null_peaks] / 60, null_smooth[null_peaks], 'v', color='#E91E63', markersize=5)
ax.set_title(f'NULL TEST — {null_breaths} "breaths" ({null_breath_rate:.1f}/min)', color='#F44336')
ax.set_ylabel('Pressure AC (Pa)')
ax.set_xlim(0, null_duration)
ax.grid(alpha=0.3)

ax = axes[0, 1]
# Show only first 9 minutes of meditation for equal comparison
med_mask = med_ts / 60 <= null_duration
ax.plot(med_ts[med_mask] / 60, med_ac[med_mask], color='#7E57C2', linewidth=0.5, alpha=0.7)
ax.plot(med_ts[med_mask] / 60, med_smooth[med_mask], color='#4527A0', linewidth=1.0)
med_peaks_mask = med_ts[med_peaks] / 60 <= null_duration
ax.plot(med_ts[med_peaks[med_peaks_mask]] / 60, med_smooth[med_peaks[med_peaks_mask]], 'v', color='#E91E63', markersize=5)
peaks_in_window = np.sum(med_peaks_mask)
ax.set_title(f'MEDITATION — {peaks_in_window} breaths (first {null_duration:.0f} min)', color='#4527A0')
ax.set_ylabel('Pressure AC (Pa)')
ax.set_xlim(0, null_duration)
ax.grid(alpha=0.3)

# ── Row 2: 30-second zoom ──
zoom_dur = 30  # seconds
for col, (ts_arr, ac_arr, smooth_arr, peaks_arr, label, color) in enumerate([
    (null_ts, null_ac, null_smooth, null_peaks, 'Null', '#37474F'),
    (med_ts, med_ac, med_smooth, med_peaks, 'Meditation', '#4527A0'),
]):
    ax = axes[1, col]
    mid = len(ts_arr) // 2
    rate_val = null_rate if col == 0 else med_rate
    half_win = int(zoom_dur / 2 * rate_val)
    s, e = max(0, mid - half_win), min(len(ts_arr), mid + half_win)
    t_offset = ts_arr[s]

    ax.plot(ts_arr[s:e] - t_offset, ac_arr[s:e], color=color, linewidth=0.5, alpha=0.5)
    ax.plot(ts_arr[s:e] - t_offset, smooth_arr[s:e], color=color, linewidth=1.5)

    zoom_peaks = peaks_arr[(peaks_arr >= s) & (peaks_arr < e)]
    if len(zoom_peaks) > 0:
        ax.plot(ts_arr[zoom_peaks] - t_offset, smooth_arr[zoom_peaks], 'v', color='#E91E63', markersize=8)

    ax.set_title(f'{label} — 30s Zoom (midpoint)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure AC (Pa)')
    ax.grid(alpha=0.3)

# ── Row 3: Power spectral density ──
ax = axes[2, 0]
# Smooth PSD for readability
def smooth_psd(freqs, psd, window=15):
    smoothed = np.convolve(psd, np.ones(window)/window, mode='same')
    return smoothed

ax.semilogy(null_freqs, smooth_psd(null_freqs, null_psd), color='#78909C', linewidth=1.5, label='Null test')
ax.semilogy(med_freqs, smooth_psd(med_freqs, med_psd), color='#4527A0', linewidth=1.5, label='Meditation')
ax.axvspan(breath_band[0], breath_band[1], alpha=0.15, color='#E91E63', label='Breath band (0.1-0.6 Hz)')
ax.set_xlim(0, 2)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (Pa²)')
ax.set_title('Power Spectral Density — Pressure Signal')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Row 3 right: Breath band power comparison ──
ax = axes[2, 1]
categories = ['Breath Band\n(0.1-0.6 Hz)', 'Total Power\n(>0.05 Hz)', 'Breath/Total\nRatio']
null_vals = [null_breath_power, null_total_power, null_breath_ratio]
med_vals = [med_breath_power, med_total_power, med_breath_ratio]

x = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x - width/2, null_vals, width, label='Null test', color='#78909C', alpha=0.8)
bars2 = ax.bar(x + width/2, med_vals, width, label='Meditation', color='#4527A0', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.set_title('Spectral Power Comparison')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        h = bar.get_height()
        if h > 100:
            ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.0f}',
                    ha='center', va='bottom', fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=8)

# ── Row 4: Amplitude histogram ──
ax = axes[3, 0]
bins = np.linspace(-15, 15, 60)
ax.hist(null_ac, bins=bins, color='#78909C', alpha=0.6, density=True, label=f'Null (std={np.std(null_ac):.2f} Pa)')
ax.hist(med_ac, bins=bins, color='#4527A0', alpha=0.6, density=True, label=f'Meditation (std={np.std(med_ac):.2f} Pa)')
ax.set_xlabel('Pressure AC (Pa)')
ax.set_ylabel('Density')
ax.set_title('Amplitude Distribution')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Row 4 right: Autocorrelation ──
ax = axes[3, 1]
max_lag = int(8 * null_rate)  # 8 seconds of lags

for arr, rate_val, label, color in [
    (null_smooth, null_rate, 'Null test', '#78909C'),
    (med_smooth[:len(null_smooth)], med_rate, 'Meditation', '#4527A0'),
]:
    centered = arr - np.mean(arr)
    autocorr = np.correlate(centered, centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # positive lags only
    autocorr = autocorr / autocorr[0]  # normalize
    lags = np.arange(min(max_lag, len(autocorr))) / rate_val

    ax.plot(lags, autocorr[:len(lags)], color=color, linewidth=1.5, label=label)

ax.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
ax.set_xlabel('Lag (seconds)')
ax.set_ylabel('Autocorrelation')
ax.set_title('Autocorrelation — Periodicity Test')
ax.legend(fontsize=9)
ax.set_xlim(0, 8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/null_test_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════
print("\n" + "="*60)
print("NULL TEST VERDICT")
print("="*60)

print(f"\n{'Metric':<35} {'Null':<15} {'Meditation':<15} {'Verdict'}")
print(f"{'─'*35} {'─'*15} {'─'*15} {'─'*20}")

# Peak detection
verdict_peaks = "REAL" if med_breath_rate > null_breath_rate * 1.5 else "SUSPICIOUS"
print(f"{'Detected breaths/min':<35} {null_breath_rate:<15.1f} {med_breath_rate:<15.1f} {verdict_peaks}")

# Breath band power
snr = med_breath_power / null_breath_power if null_breath_power > 0 else float('inf')
verdict_power = "REAL" if snr > 2 else "SUSPICIOUS"
print(f"{'Breath band power':<35} {null_breath_power:<15.1f} {med_breath_power:<15.1f} {verdict_power} ({snr:.1f}x)")

# Breath band ratio
verdict_ratio = "REAL" if med_breath_ratio > null_breath_ratio * 1.3 else "SUSPICIOUS"
print(f"{'Breath band % of total':<35} {null_breath_ratio:<15.1f}% {med_breath_ratio:<15.1f}% {verdict_ratio}")

# Overall
if snr > 2 and med_breath_rate > null_breath_rate * 1.3:
    print(f"\n✅ CONCLUSION: Breath signal is REAL — {snr:.1f}x more power in breath band than noise floor")
elif snr > 1.5:
    print(f"\n⚠️  CONCLUSION: Signal is MARGINAL — only {snr:.1f}x above noise. May need better isolation.")
else:
    print(f"\n❌ CONCLUSION: Signal may be NOISE — only {snr:.1f}x. Pressure sensor may be picking up ambient.")
