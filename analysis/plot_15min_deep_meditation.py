#!/usr/bin/env python3
"""
Analysis of 15-minute deep meditation session (firmware v2.0.0, packet v4).
Compares breath patterns and HR signal with previous recordings.
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# Load data
with gzip.open('/Users/mert/Downloads/15MinsLatestChanges.gz', 'rt') as f:
    data = json.load(f)

samples = np.array(data['samples'])
ts = samples[:, 0] / 1000.0       # seconds
ir = samples[:, 1]
red = samples[:, 2]
pressure = samples[:, 3]          # Pa relative to baseline (already ×100 decoded)

duration_min = (ts[-1] - ts[0]) / 60.0
n_samples = len(ts)
actual_rate = (n_samples - 1) / (ts[-1] - ts[0])

print(f"Session: {duration_min:.1f} min, {n_samples} samples, {actual_rate:.1f} Hz")

# ── BLE gap analysis ──
dt = np.diff(ts) * 1000  # ms
gaps = np.where(dt > 100)[0]
print(f"BLE gaps (>100ms): {len(gaps)}")
if len(gaps) > 0:
    print(f"  Worst gap: {dt[gaps].max():.0f} ms")
    print(f"  Mean gap: {dt[gaps].mean():.0f} ms")

# ═══════════════════════════════════════════════════
# 1. PRESSURE (BREATH) ANALYSIS
# ═══════════════════════════════════════════════════

# 15-second moving average baseline subtraction
window = int(15 * actual_rate)
pressure_baseline = np.convolve(pressure, np.ones(window)/window, mode='same')
pressure_ac = pressure - pressure_baseline

# Smooth for peak detection
smooth_window = int(0.5 * actual_rate)  # 0.5s smooth
pressure_smooth = np.convolve(pressure_ac, np.ones(smooth_window)/smooth_window, mode='same')

# Peak detection
pressure_std = np.std(pressure_smooth)
peaks, props = find_peaks(pressure_smooth,
                          height=pressure_std * 0.15,
                          distance=int(1.5 * actual_rate),  # min 1.5s between breaths
                          prominence=pressure_std * 0.1)

breath_count = len(peaks)
breath_rate = breath_count / duration_min
print(f"\nBreath Analysis:")
print(f"  Total breaths: {breath_count}")
print(f"  Avg breath rate: {breath_rate:.1f} breaths/min")

# Breath intervals
if len(peaks) > 1:
    intervals = np.diff(ts[peaks])
    print(f"  Mean interval: {np.mean(intervals):.2f}s (±{np.std(intervals):.2f}s)")
    print(f"  Min interval: {np.min(intervals):.2f}s")
    print(f"  Max interval: {np.max(intervals):.2f}s")

# Signal strength over time (rolling 3-minute windows)
window_3min = int(3 * 60 * actual_rate)
n_windows = max(1, n_samples // window_3min)
print(f"\nPressure signal strength (3-min windows):")
for i in range(n_windows):
    start_idx = i * window_3min
    end_idx = min((i + 1) * window_3min, n_samples)
    seg = pressure_ac[start_idx:end_idx]
    t_start = ts[start_idx] / 60
    t_end = ts[end_idx - 1] / 60
    std_val = np.std(seg)
    print(f"  {t_start:.1f}-{t_end:.1f} min: std={std_val:.2f} Pa")

# Remaining tail
if n_samples % window_3min > window_3min // 2:
    start_idx = n_windows * window_3min
    seg = pressure_ac[start_idx:]
    t_start = ts[start_idx] / 60
    std_val = np.std(seg)
    print(f"  {t_start:.1f}-{duration_min:.1f} min: std={std_val:.2f} Pa")

# ═══════════════════════════════════════════════════
# 2. IR SIGNAL (HEART RATE) ANALYSIS
# ═══════════════════════════════════════════════════

# Check finger contact
finger_on = ir > 50000
finger_pct = np.sum(finger_on) / len(ir) * 100
print(f"\nHR Analysis:")
print(f"  Finger contact: {finger_pct:.1f}%")

# DC removal: 2.5s moving average
dc_window = int(2.5 * actual_rate)
ir_dc = np.convolve(ir, np.ones(dc_window)/dc_window, mode='same')
ir_ac = ir - ir_dc

# Bandpass filter 0.7-3.5 Hz (42-210 BPM)
nyq = actual_rate / 2
try:
    b, a = butter(3, [0.7/nyq, 3.5/nyq], btype='band')
    ir_filtered = filtfilt(b, a, ir_ac)
except:
    ir_filtered = ir_ac

# Only use segments where finger is on
ir_filtered_masked = ir_filtered.copy()
ir_filtered_masked[~finger_on] = 0

# Peak detection on filtered IR
ir_std = np.std(ir_filtered_masked[finger_on]) if np.any(finger_on) else 1
hr_peaks, hr_props = find_peaks(ir_filtered_masked,
                                 height=ir_std * 0.3,
                                 distance=int(0.4 * actual_rate),  # min 0.4s (150 BPM max)
                                 prominence=ir_std * 0.2)

# Filter peaks to only finger-on segments
hr_peaks = hr_peaks[finger_on[hr_peaks]]

if len(hr_peaks) > 1:
    rr_intervals = np.diff(ts[hr_peaks])
    # Filter physiological range (0.33-1.5s = 40-180 BPM)
    valid_rr = rr_intervals[(rr_intervals > 0.33) & (rr_intervals < 1.5)]

    if len(valid_rr) > 10:
        avg_hr = 60.0 / np.mean(valid_rr)
        rmssd = np.sqrt(np.mean(np.diff(valid_rr * 1000)**2))
        sdnn = np.std(valid_rr * 1000)
        nn50 = np.sum(np.abs(np.diff(valid_rr * 1000)) > 50)
        pnn50 = nn50 / len(valid_rr) * 100

        print(f"  Detected beats: {len(hr_peaks)}")
        print(f"  Valid RR intervals: {len(valid_rr)}")
        print(f"  Avg HR: {avg_hr:.1f} BPM")
        print(f"  RMSSD: {rmssd:.1f} ms")
        print(f"  SDNN: {sdnn:.1f} ms")
        print(f"  pNN50: {pnn50:.1f}%")

        # Detection rate
        expected_beats = duration_min * avg_hr
        detection_rate = len(hr_peaks) / expected_beats * 100
        print(f"  Detection rate: {detection_rate:.0f}%")
    else:
        print(f"  Detected beats: {len(hr_peaks)} (insufficient valid RR)")
        valid_rr = np.array([])
        avg_hr = 0
else:
    print(f"  Detected beats: {len(hr_peaks)} (insufficient)")
    valid_rr = np.array([])
    avg_hr = 0

# ═══════════════════════════════════════════════════
# 3. BREATH RATE TREND (1-minute rolling)
# ═══════════════════════════════════════════════════

breath_rate_trend = []
trend_times = []
window_60s = 60  # seconds
step = 30  # 30s step for smoother trend

for t_start in np.arange(ts[0], ts[-1] - window_60s, step):
    t_end = t_start + window_60s
    mask = (ts[peaks] >= t_start) & (ts[peaks] < t_end)
    count = np.sum(mask)
    rate = count * (60.0 / window_60s)
    breath_rate_trend.append(rate)
    trend_times.append((t_start + t_end) / 2 / 60)  # midpoint in minutes

# ═══════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════

fig, axes = plt.subplots(5, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [3, 2, 3, 2, 2]})
fig.suptitle(f'15-Min Deep Meditation — Firmware v2.0.0 (Packet v4)\n'
             f'{breath_count} breaths ({breath_rate:.1f}/min), '
             f'{len(gaps)} BLE gaps, {finger_pct:.0f}% finger contact',
             fontsize=14, fontweight='bold')

# ── Panel 1: Full pressure waveform with peaks ──
ax1 = axes[0]
t_min = ts / 60
ax1.plot(t_min, pressure_ac, color='#7E57C2', linewidth=0.5, alpha=0.7, label='Pressure AC')
ax1.plot(t_min, pressure_smooth, color='#4527A0', linewidth=1.0, label='Smoothed')
ax1.plot(t_min[peaks], pressure_smooth[peaks], 'v', color='#E91E63', markersize=4, label=f'Breaths ({breath_count})')
ax1.set_ylabel('Pressure ΔP (Pa)')
ax1.set_title('Breath Pattern — BMP280 Pressure')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_xlim(0, duration_min)
ax1.grid(alpha=0.3)

# ── Panel 2: Breath rate trend ──
ax2 = axes[1]
ax2.plot(trend_times, breath_rate_trend, color='#E91E63', linewidth=2)
ax2.axhline(y=breath_rate, color='grey', linestyle='--', alpha=0.5, label=f'Avg: {breath_rate:.1f}/min')
ax2.fill_between(trend_times, breath_rate_trend, alpha=0.2, color='#E91E63')
ax2.set_ylabel('Breaths/min')
ax2.set_title('Breath Rate Trend (1-min rolling)')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xlim(0, duration_min)
ax2.set_ylim(0, max(breath_rate_trend) * 1.3 if breath_rate_trend else 30)
ax2.grid(alpha=0.3)

# ── Panel 3: IR signal (heartbeat) ──
ax3 = axes[2]
# Plot a 30-second zoomed section from the middle where finger is on
mid_idx = len(ts) // 2
zoom_start = mid_idx - int(15 * actual_rate)
zoom_end = mid_idx + int(15 * actual_rate)
zoom_start = max(0, zoom_start)
zoom_end = min(len(ts), zoom_end)

ax3.plot(ts[zoom_start:zoom_end] - ts[zoom_start], ir_filtered[zoom_start:zoom_end],
         color='#D32F2F', linewidth=0.8, label='IR Bandpass (0.7-3.5 Hz)')

# Mark detected peaks in zoom window
zoom_hr_peaks = hr_peaks[(hr_peaks >= zoom_start) & (hr_peaks < zoom_end)]
if len(zoom_hr_peaks) > 0:
    ax3.plot(ts[zoom_hr_peaks] - ts[zoom_start], ir_filtered[zoom_hr_peaks],
             'v', color='#1565C0', markersize=6, label=f'Heartbeats')

ax3.set_ylabel('IR AC (filtered)')
ax3.set_xlabel('Time (s) — 30s zoom from session midpoint')
ax3.set_title(f'Heartbeat Signal — MAX30102 IR (30s zoom)')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(alpha=0.3)

# ── Panel 4: HR trend (if enough data) ──
ax4 = axes[3]
if len(valid_rr) > 10:
    # Rolling HR from RR intervals
    hr_times = ts[hr_peaks[1:]]
    hr_instant = 60.0 / rr_intervals
    # Filter to valid range
    valid_mask = (hr_instant > 40) & (hr_instant < 180)
    hr_times_valid = hr_times[valid_mask] / 60
    hr_instant_valid = hr_instant[valid_mask]

    if len(hr_instant_valid) > 5:
        ax4.scatter(hr_times_valid, hr_instant_valid, s=3, color='#D32F2F', alpha=0.4)
        # Rolling average
        if len(hr_instant_valid) > 20:
            roll_size = min(20, len(hr_instant_valid) // 3)
            hr_rolling = np.convolve(hr_instant_valid, np.ones(roll_size)/roll_size, mode='valid')
            hr_t_rolling = hr_times_valid[roll_size//2:roll_size//2 + len(hr_rolling)]
            ax4.plot(hr_t_rolling, hr_rolling, color='#D32F2F', linewidth=2, label=f'Avg: {avg_hr:.0f} BPM')
        ax4.axhline(y=avg_hr, color='grey', linestyle='--', alpha=0.5)
        ax4.legend(loc='upper right', fontsize=9)
    ax4.set_ylabel('Heart Rate (BPM)')
    ax4.set_title('Heart Rate Trend')
else:
    ax4.text(0.5, 0.5, f'Insufficient HR data ({len(hr_peaks)} beats detected)\nHRV needs bandpass filter improvement',
             transform=ax4.transAxes, ha='center', va='center', fontsize=12, color='grey')
    ax4.set_title('Heart Rate Trend (N/A)')
ax4.set_xlim(0, duration_min)
ax4.grid(alpha=0.3)

# ── Panel 5: Breath interval histogram + comparison ──
ax5 = axes[4]
if len(peaks) > 1:
    breath_intervals = np.diff(ts[peaks])
    ax5.hist(breath_intervals, bins=30, color='#7E57C2', alpha=0.7, edgecolor='white',
             label=f'This session: {np.mean(breath_intervals):.2f}±{np.std(breath_intervals):.2f}s')
    ax5.axvline(x=np.mean(breath_intervals), color='#E91E63', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(breath_intervals):.2f}s ({60/np.mean(breath_intervals):.1f}/min)')
    # Reference lines for comparison
    ax5.axvline(x=3.34, color='#4CAF50', linestyle=':', linewidth=1.5, alpha=0.7,
                label='Prototype 1 avg (3.34s, 18.0/min)')
    ax5.axvline(x=2.84, color='#FF9800', linestyle=':', linewidth=1.5, alpha=0.7,
                label='Prototype 2 test (2.84s, 21.1/min)')
ax5.set_xlabel('Breath Interval (seconds)')
ax5.set_ylabel('Count')
ax5.set_title('Breath Interval Distribution (vs Previous Sessions)')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

for ax in axes:
    ax.set_xlabel('Time (min)' if ax != axes[4] else ax.get_xlabel())

plt.tight_layout()
plt.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/15min_deep_meditation.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════
print("\n" + "="*60)
print("VERDICT — 15-Min Deep Meditation")
print("="*60)

# Pressure signal quality
first_3min_std = np.std(pressure_ac[:int(3 * 60 * actual_rate)])
last_3min_std = np.std(pressure_ac[-int(3 * 60 * actual_rate):])
retention = last_3min_std / first_3min_std * 100 if first_3min_std > 0 else 0

print(f"\n📊 Pressure Signal:")
print(f"   First 3 min std:  {first_3min_std:.2f} Pa")
print(f"   Last 3 min std:   {last_3min_std:.2f} Pa")
print(f"   Signal retention:  {retention:.0f}% (prev sessions: 87%)")

print(f"\n🫁 Breathing:")
print(f"   Total breaths:    {breath_count}")
print(f"   Avg rate:         {breath_rate:.1f} breaths/min")
if len(peaks) > 1:
    print(f"   Interval:         {np.mean(breath_intervals):.2f}s ± {np.std(breath_intervals):.2f}s")
    print(f"   Regularity (CV):  {np.std(breath_intervals)/np.mean(breath_intervals):.2f}")

print(f"\n💓 Heart Rate:")
if avg_hr > 0:
    print(f"   Avg HR:           {avg_hr:.1f} BPM")
    print(f"   Detected beats:   {len(hr_peaks)}")
    if len(valid_rr) > 10:
        print(f"   RMSSD:            {rmssd:.1f} ms")
        print(f"   SDNN:             {sdnn:.1f} ms")
else:
    print(f"   Insufficient data ({len(hr_peaks)} peaks)")

print(f"\n📡 BLE Quality:")
print(f"   Gaps (>100ms):    {len(gaps)}")
print(f"   Sample rate:      {actual_rate:.1f} Hz")
print(f"   Duration:         {duration_min:.1f} min")

# Comparison table
print(f"\n📋 COMPARISON WITH PREVIOUS SESSIONS:")
print(f"   {'Metric':<25} {'Proto1 (17min)':<18} {'Proto2 (6min)':<18} {'This (15min)':<18}")
print(f"   {'─'*25} {'─'*18} {'─'*18} {'─'*18}")
print(f"   {'Breath rate':<25} {'18.0/min':<18} {'21.1/min':<18} {f'{breath_rate:.1f}/min':<18}")
print(f"   {'Signal retention':<25} {'87%':<18} {'N/A (short)':<18} {f'{retention:.0f}%':<18}")
print(f"   {'BLE gaps':<25} {'0':<18} {'22':<18} {f'{len(gaps)}':<18}")
print(f"   {'Pressure std':<25} {'1.5-2.5 Pa':<18} {'4.6 Pa':<18} {f'{np.std(pressure_ac):.2f} Pa':<18}")
