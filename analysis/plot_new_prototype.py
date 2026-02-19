#!/usr/bin/env python3
"""
New prototype analysis — 6-minute test session.
Focus: BMP280 pressure (primary breath) + MAX30102 IR (secondary/face detect).
No thermistor on this prototype.
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

def find_peaks_with_prom(signal, min_distance=10, prominence_threshold=0.0):
    peaks, proms = [], []
    n = len(signal)
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            left_min = min(signal[max(0, i - min_distance):i])
            right_min = min(signal[i + 1:min(n, i + min_distance + 1)])
            prominence = signal[i] - max(left_min, right_min)
            if prominence >= prominence_threshold:
                peaks.append(i)
                proms.append(prominence)
    if len(peaks) < 2:
        return peaks, proms
    filtered_p, filtered_pr = [peaks[0]], [proms[0]]
    for j in range(1, len(peaks)):
        if peaks[j] - filtered_p[-1] >= min_distance:
            filtered_p.append(peaks[j])
            filtered_pr.append(proms[j])
        elif proms[j] > filtered_pr[-1]:
            filtered_p[-1] = peaks[j]
            filtered_pr[-1] = proms[j]
    return filtered_p, filtered_pr

# ── Load data ────────────────────────────────────────────────────────────────

with gzip.open('/Users/mert/Downloads/raw_samples (6).gz', 'rt') as f:
    data = json.load(f)

samples = data['samples']
ts_raw = np.array([s[0] for s in samples], dtype=np.float64)
t_sec = (ts_raw - ts_raw[0]) / 1000.0
t_min = t_sec / 60.0
ir = np.array([s[2] for s in samples], dtype=np.float64)
red = np.array([s[3] for s in samples], dtype=np.float64)
temp = np.array([s[4] for s in samples], dtype=np.float64)
pressure = np.array([s[6] for s in samples], dtype=np.float64)
n_total = len(samples)
sr = 20.0

print(f"Loaded {n_total} samples, {t_sec[-1]:.1f}s = {t_min[-1]:.1f} min")

# ── BLE gap analysis ─────────────────────────────────────────────────────────

dt = np.diff(ts_raw)
gaps = np.where(dt > 100)[0]
gap_total_ms = dt[gaps].sum() if len(gaps) > 0 else 0
print(f"Gaps >100ms: {len(gaps)}, total lost: {gap_total_ms:.0f}ms ({gap_total_ms/1000:.1f}s)")

# ── Signal processing ────────────────────────────────────────────────────────

window_15s = int(15 * sr)
smooth_w = int(0.25 * sr)

press_bl = moving_average(pressure, window_15s)
press_ac = pressure - press_bl
press_sm = moving_average(press_ac, smooth_w)

ir_bl = moving_average(ir, window_15s)
ir_ac = ir - ir_bl
ir_sm = moving_average(ir_ac, smooth_w)

# Peak detection — pressure
min_peak_dist = int(1.8 * sr)
press_peaks, press_proms = find_peaks_with_prom(press_sm, min_peak_dist, press_sm.std() * 0.25)

# Peak detection — IR (for HRV, shorter distance)
ir_peaks, ir_proms = find_peaks_with_prom(ir_sm, int(0.4 * sr), ir_sm.std() * 0.3)

print(f"Pressure peaks (breaths): {len(press_peaks)}")
print(f"IR peaks: {len(ir_peaks)}")

# Breath intervals
if len(press_peaks) >= 2:
    peak_times = t_min[press_peaks]
    intervals_s = np.diff(peak_times) * 60
    interval_mid = (peak_times[:-1] + peak_times[1:]) / 2
    clean_ints = intervals_s[(intervals_s > 1.5) & (intervals_s < 8)]
    print(f"Breath rate: {60/np.median(clean_ints):.1f} br/min (median interval {np.median(clean_ints):.2f}s)")

# Rolling breathing energy
win_3s = int(3 * sr)
step = int(0.5 * sr)
energy_t, energy_p = [], []
for i in range(0, len(press_sm) - win_3s, step):
    energy_t.append(t_min[i + win_3s // 2])
    energy_p.append(np.std(press_sm[i:i + win_3s]))
energy_t = np.array(energy_t)
energy_p = np.array(energy_p)

# Per-minute stats
minute_labels = list(range(0, int(t_min[-1]) + 1))
m_press_std, m_ppeaks = [], []
for m in minute_labels:
    mask = (t_min >= m) & (t_min < m + 1)
    if mask.sum() < 100:
        m_press_std.append(0); m_ppeaks.append(0)
        continue
    m_press_std.append(press_ac[mask].std())
    m_ppeaks.append(sum(1 for p in press_peaks if m <= t_min[p] < m + 1))

ref_idx = 1  # normalize to minute 1 (skip settling)
p_norm = [s / m_press_std[ref_idx] * 100 if m_press_std[ref_idx] > 0 else 0 for s in m_press_std]

# ═══════════════════════════════════════════════════════════════════════════════
# CREATE FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 32))
gs = GridSpec(8, 3, figure=fig, hspace=0.42, wspace=0.30,
              height_ratios=[2.5, 2, 2.5, 2, 2, 2, 2, 1.0])

fig.suptitle('NEW PROTOTYPE — 6-Minute Test Session\n'
             'BMP280 pressure (primary) + MAX30102 IR (secondary)  •  No thermistor',
             fontsize=16, fontweight='bold', y=0.998)

CP = '#7B1FA2'
CI = '#2E7D32'

# ═══════════════════════ ROW 0: Pressure — full session ══════════════════════
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(t_min, press_ac, color=CP, linewidth=0.2, alpha=0.15)
ax0.plot(t_min, press_sm, color=CP, linewidth=1.2)
ax0.scatter(t_min[press_peaks], press_sm[press_peaks], color='red', s=18, zorder=5,
            label=f'{len(press_peaks)} breaths')
ax0.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
# Mark gap regions
for g in gaps:
    if dt[g] > 300:
        ax0.axvspan(t_min[g], t_min[min(g+1, n_total-1)], alpha=0.3, color='orange', zorder=0)
for m in range(0, 7):
    ax0.axvline(x=m, color='gray', linewidth=0.3, alpha=0.3)
ax0.set_ylabel('Pressure AC (Pa)')
ax0.set_title(f'BMP280 Pressure — {len(press_peaks)} breaths over {t_min[-1]:.1f} min',
              fontsize=14, fontweight='bold')
ax0.legend(loc='upper right')
ax0.grid(True, alpha=0.2)

# ═══════════════════════ ROW 1: IR — full session ════════════════════════════
ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
ax1.plot(t_min, ir_ac, color=CI, linewidth=0.3, alpha=0.3)
ax1.plot(t_min, ir_sm, color=CI, linewidth=1.0)
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
for g in gaps:
    if dt[g] > 300:
        ax1.axvspan(t_min[g], t_min[min(g+1, n_total-1)], alpha=0.3, color='orange', zorder=0)
ax1.set_ylabel('IR AC (counts)')
ax1.set_title(f'MAX30102 IR — Raw: {ir.mean():.0f} mean ({ir.max()-ir.min():.0f} range)',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.2)

# ═══════════════════════ ROW 2: Breathing energy ═════════════════════════════
ax2 = fig.add_subplot(gs[2, :], sharex=ax0)
ax2.plot(energy_t, energy_p, color=CP, linewidth=2)
ax2.fill_between(energy_t, 0, energy_p, color=CP, alpha=0.15)
ax2.set_ylabel('3s Rolling Std (Pa)')
ax2.set_title('Breathing Energy — Rolling Amplitude', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.2)
# Annotate mean
mean_energy = energy_p.mean()
ax2.axhline(y=mean_energy, color='red', linewidth=1, linestyle='--',
            label=f'Mean: {mean_energy:.2f} Pa')
ax2.legend(loc='upper right')

# ═══════════════════════ ROW 3: Breath intervals ═════════════════════════════
ax3 = fig.add_subplot(gs[3, :], sharex=ax0)
if len(press_peaks) >= 2:
    ax3.scatter(interval_mid, intervals_s, color=CP, s=20, alpha=0.7)
    # Running median
    if len(intervals_s) > 10:
        running_med = np.array([np.median(intervals_s[max(0,i-10):i+1]) for i in range(len(intervals_s))])
        ax3.plot(interval_mid, running_med, color='black', linewidth=2, alpha=0.7, label='Running median')
    ax3.axhline(y=np.median(clean_ints), color='green', linewidth=1.5, linestyle='--',
                label=f'Median: {np.median(clean_ints):.2f}s ({60/np.median(clean_ints):.1f} br/min)')
    ax3.axhline(y=3.3, color='blue', linewidth=1, linestyle=':', alpha=0.5, label='Target 3.3s')
    ax3.legend(loc='upper right', fontsize=9)
ax3.set_ylabel('Interval (s)')
ax3.set_title('Breath-to-Breath Intervals', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.2)
ax3.set_ylim(0, min(10, intervals_s.max() * 1.1) if len(press_peaks) >= 2 else 10)

# ═══════════════════════ ROW 4: BLE gap analysis ════════════════════════════
ax4a = fig.add_subplot(gs[4, 0:2])
# Histogram of all sample intervals
dt_clean = dt[dt < 500]  # cap for visibility
ax4a.hist(dt, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax4a.axvline(x=50, color='green', linewidth=2, label='Expected 50ms')
ax4a.axvline(x=100, color='red', linewidth=1.5, linestyle='--', label='Gap threshold')
ax4a.set_xlabel('Sample interval (ms)')
ax4a.set_ylabel('Count')
ax4a.set_title(f'BLE Sample Intervals — {len(gaps)} gaps >100ms', fontsize=11, fontweight='bold')
ax4a.legend(fontsize=8)
ax4a.set_xlim(0, min(2000, dt.max() * 1.1))
ax4a.grid(True, alpha=0.2)

# Gap timeline
ax4b = fig.add_subplot(gs[4, 2])
if len(gaps) > 0:
    ax4b.scatter(t_min[gaps], dt[gaps], color='red', s=30, alpha=0.7)
    ax4b.set_xlabel('Time (min)')
    ax4b.set_ylabel('Gap size (ms)')
    ax4b.set_title('Gap Location & Size', fontsize=11, fontweight='bold')
    ax4b.grid(True, alpha=0.2)
else:
    ax4b.text(0.5, 0.5, 'No gaps!', transform=ax4b.transAxes, ha='center', fontsize=14)

# ═══════════════════════ ROW 5: Signal stability ═════════════════════════════
ax5a = fig.add_subplot(gs[5, 0:2])
valid = [i for i, s in enumerate(m_press_std) if s > 0]
x = np.arange(len(valid))
labs = [minute_labels[i] for i in valid]
p_vals = [p_norm[i] for i in valid]
ax5a.plot(x, p_vals, 's-', color=CP, linewidth=2, markersize=8)
ax5a.axhline(y=100, color='gray', linewidth=1, linestyle='--', alpha=0.5)
for i, v in enumerate(p_vals):
    ax5a.annotate(f'{v:.0f}%', xy=(i, v), xytext=(0, 10), textcoords='offset points',
                  ha='center', fontsize=9, fontweight='bold', color=CP)
ax5a.set_xlabel('Minute')
ax5a.set_ylabel(f'% of minute {ref_idx}')
ax5a.set_title('Pressure Signal Stability', fontsize=13, fontweight='bold')
ax5a.set_xticks(x)
ax5a.set_xticklabels([str(l) for l in labs])
ax5a.grid(True, alpha=0.3)

# Temperature
ax5b = fig.add_subplot(gs[5, 2])
ax5b.plot(t_min, temp, color='#E65100', linewidth=2)
ax5b.set_xlabel('Time (min)')
ax5b.set_ylabel('°C')
ax5b.set_title(f'Temp: {temp[0]:.1f}→{temp[-1]:.1f}°C (+{temp[-1]-temp[0]:.1f}°C)',
               fontsize=11, fontweight='bold')
ax5b.grid(True, alpha=0.3)

# ═══════════════════════ ROW 6: Waveform + histogram ═════════════════════════
# Waveform overlay
ax6a = fig.add_subplot(gs[6, 0:2])
half_win = int(2.0 * sr)
waveforms = []
for p in press_peaks:
    if p - half_win >= 0 and p + half_win < n_total:
        wf = press_sm[p - half_win:p + half_win]
        waveforms.append(wf)
        ax6a.plot(np.linspace(-2, 2, len(wf)), wf, color=CP, alpha=0.1, linewidth=0.8)
if waveforms:
    mean_wf = np.mean(waveforms, axis=0)
    ax6a.plot(np.linspace(-2, 2, len(mean_wf)), mean_wf, color='red', linewidth=3,
              label=f'Mean (n={len(waveforms)})')
ax6a.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax6a.set_title('Breath Waveform Overlay', fontsize=13, fontweight='bold')
ax6a.set_xlabel('Time from peak (s)')
ax6a.set_ylabel('Pa')
ax6a.legend(fontsize=9)
ax6a.grid(True, alpha=0.2)

# Interval histogram
ax6b = fig.add_subplot(gs[6, 2])
if len(clean_ints) > 0:
    ax6b.hist(clean_ints, bins=20, color=CP, alpha=0.7, edgecolor='white')
    ax6b.axvline(x=np.median(clean_ints), color='green', linewidth=2,
                 label=f'Median: {np.median(clean_ints):.2f}s')
    ax6b.axvline(x=3.3, color='blue', linewidth=1, linestyle=':', label='Target 3.3s')
    ax6b.legend(fontsize=8)
ax6b.set_xlabel('Interval (s)')
ax6b.set_ylabel('Count')
ax6b.set_title('Interval Distribution', fontsize=11, fontweight='bold')
ax6b.grid(True, alpha=0.2)

# ═══════════════════════ ROW 7: Verdict ══════════════════════════════════════
ax7 = fig.add_subplot(gs[7, :])
ax7.axis('off')

# Compare to 30-min session from prototype 1
verdict = [
    f"NEW PROTOTYPE TEST — {n_total} samples, {t_min[-1]:.1f} min @ ~{n_total/t_sec[-1]:.0f} Hz",
    f"{'═' * 95}",
    f"  Pressure signal:    std={press_ac.std():.2f} Pa  |  range [{pressure.min():.1f}, {pressure.max():.1f}] Pa",
    f"  Breath detection:   {len(press_peaks)} peaks  |  "
    + (f"{60/np.median(clean_ints):.1f} br/min  |  median {np.median(clean_ints):.2f}s" if len(clean_ints) > 0 else "N/A"),
    f"  Signal stability:   min5 = {p_vals[-2] if len(p_vals) > 1 else 0:.0f}% of min1 baseline",
    f"  Temperature:        {temp[0]:.1f}→{temp[-1]:.1f}°C (+{temp[-1]-temp[0]:.1f}°C in {t_min[-1]:.0f} min)",
    f"",
    f"  BLE gaps:           {len(gaps)} gaps >100ms, largest {dt.max():.0f}ms, total lost {gap_total_ms/1000:.1f}s",
    f"  MAX30102 IR:        mean={ir.mean():.0f}, range={ir.max()-ir.min():.0f} — {'GOOD' if ir.mean() > 10000 else 'LOW'} signal",
]

# Overall assessment
gap_pct = gap_total_ms / (t_sec[-1] * 1000) * 100
if len(press_peaks) > 5 and gap_pct < 5:
    verdict.append(f"")
    verdict.append(f"  VERDICT:  ✓ Pressure breath detection WORKING  |  Ready for longer sessions")
elif len(press_peaks) > 5:
    verdict.append(f"")
    verdict.append(f"  VERDICT:  ✓ Breath detection OK  |  ⚠ BLE gaps need investigation ({gap_pct:.1f}% lost)")
else:
    verdict.append(f"")
    verdict.append(f"  VERDICT:  ✗ Insufficient breath peaks — check sensor placement")

ax7.text(0.02, 0.95, '\n'.join(verdict), transform=ax7.transAxes,
         fontsize=10.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.5))

# ── Save ─────────────────────────────────────────────────────────────────────
output = '/Users/mert/Developer/Hardware/BreathAnalysis/analysis/new_prototype_test.png'
plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output}")
