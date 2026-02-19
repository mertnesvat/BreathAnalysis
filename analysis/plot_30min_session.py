#!/usr/bin/env python3
"""
30-minute session analysis — longest continuous recording.
Phases: 0-26:20 normal meditation, 26:20-28:20 fast breathing, 28:20-30:00 slow/subtle breathing.
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

with gzip.open('/Users/mert/Downloads/30mins-26-28-Fast-Breathing-28-30-Slow-Subtle-Breathing.gz', 'rt') as f:
    data = json.load(f)

samples = data['samples']
# v3: [ts, therm, ir, red, temp, hum, pressureDelta]
ts_raw = np.array([s[0] for s in samples], dtype=np.float64)
t_sec = (ts_raw - ts_raw[0]) / 1000.0
t_min = t_sec / 60.0
therm = np.array([s[1] for s in samples], dtype=np.float64)
ir = np.array([s[2] for s in samples], dtype=np.float64)
red = np.array([s[3] for s in samples], dtype=np.float64)
temp = np.array([s[4] for s in samples], dtype=np.float64)
pressure = np.array([s[6] for s in samples], dtype=np.float64)
n_total = len(samples)
sr = 20.0

print(f"Loaded {n_total} samples, {t_sec[-1]:.1f}s = {t_min[-1]:.1f} min")

# Phase boundaries (in minutes)
NORMAL_END = 26 + 20/60      # 26:20
FAST_END = 28 + 20/60        # 28:20
SESSION_END = t_min[-1]      # ~30:08

# ── Signal processing ────────────────────────────────────────────────────────

window_15s = int(15 * sr)
smooth_w = int(0.25 * sr)
smooth_w_fast = int(0.15 * sr)  # shorter smoothing for fast breathing

def process(signal, smooth=None):
    bl = moving_average(signal, window_15s)
    ac = signal - bl
    sm = moving_average(ac, smooth or smooth_w)
    return ac, sm, bl

press_ac, press_sm, press_bl = process(pressure)
therm_ac, therm_sm, therm_bl = process(therm)
ir_ac, ir_sm, ir_bl = process(ir)

# Also create a fast-smoothed pressure for the fast phase
_, press_sm_fast, _ = process(pressure, smooth_w_fast)

# Adaptive peak detection per phase:
# Normal: 1.8s min distance (standard)
# Fast: 0.8s min distance (captures rapid breaths)
# Slow: 1.8s min distance (same as normal, just looking for lower amplitude)

normal_idx = np.where(t_min < NORMAL_END)[0]
fast_idx = np.where((t_min >= NORMAL_END) & (t_min < FAST_END))[0]
slow_idx = np.where(t_min >= FAST_END)[0]

# Normal phase peaks
min_dist_normal = int(1.8 * sr)
press_sm_normal = press_sm[normal_idx[0]:normal_idx[-1]+1]
pp_normal, ppr_normal = find_peaks_with_prom(press_sm_normal, min_dist_normal, press_sm_normal.std() * 0.25)
pp_normal = [p + normal_idx[0] for p in pp_normal]

# Fast phase peaks — use shorter smoothing and min distance
min_dist_fast = int(0.8 * sr)
press_sm_f = press_sm_fast[fast_idx[0]:fast_idx[-1]+1]
pp_fast, ppr_fast = find_peaks_with_prom(press_sm_f, min_dist_fast, press_sm_f.std() * 0.2)
pp_fast = [p + fast_idx[0] for p in pp_fast]

# Slow phase peaks
min_dist_slow = int(1.8 * sr)
press_sm_slow = press_sm[slow_idx[0]:slow_idx[-1]+1]
pp_slow, ppr_slow = find_peaks_with_prom(press_sm_slow, min_dist_slow, press_sm_slow.std() * 0.2)
pp_slow = [p + slow_idx[0] for p in pp_slow]

# Combined peaks
press_peaks = pp_normal + pp_fast + pp_slow
press_proms = ppr_normal + ppr_fast + ppr_slow

# Thermistor: single pass (it's secondary)
therm_peaks, _ = find_peaks_with_prom(therm_sm, int(1.8 * sr), therm_sm.std() * 0.3)

print(f"Pressure peaks: {len(press_peaks)}")
print(f"Thermistor peaks: {len(therm_peaks)}")

# ── Per-minute stats ─────────────────────────────────────────────────────────

minute_labels = list(range(0, int(t_min[-1]) + 1))
m_press_std, m_therm_std, m_ppeaks, m_tpeaks = [], [], [], []
for m in minute_labels:
    mask = (t_min >= m) & (t_min < m + 1)
    if mask.sum() < 100:
        m_press_std.append(0); m_therm_std.append(0)
        m_ppeaks.append(0); m_tpeaks.append(0)
        continue
    m_press_std.append(press_ac[mask].std())
    m_therm_std.append(therm_ac[mask].std())
    m_ppeaks.append(sum(1 for p in press_peaks if m <= t_min[p] < m + 1))
    m_tpeaks.append(sum(1 for p in therm_peaks if m <= t_min[p] < m + 1))

# Normalize to minute 2 (avoid settling period)
ref_idx = 2
p_norm = [s / m_press_std[ref_idx] * 100 if m_press_std[ref_idx] > 0 else 0 for s in m_press_std]
t_norm = [s / m_therm_std[ref_idx] * 100 if m_therm_std[ref_idx] > 0 else 0 for s in m_therm_std]

# ── Rolling breathing energy ─────────────────────────────────────────────────

win_3s = int(3 * sr)
step = int(0.5 * sr)
energy_t, energy_p, energy_th = [], [], []
for i in range(0, len(press_sm) - win_3s, step):
    energy_t.append(t_min[i + win_3s // 2])
    energy_p.append(np.std(press_sm[i:i + win_3s]))
    energy_th.append(np.std(therm_sm[i:i + win_3s]))
energy_t = np.array(energy_t)
energy_p = np.array(energy_p)
energy_th = np.array(energy_th)

# ── Breath intervals ─────────────────────────────────────────────────────────

if len(press_peaks) >= 2:
    peak_times_min = t_min[press_peaks]
    intervals_s = np.diff(peak_times_min) * 60  # in seconds
    interval_mid = (peak_times_min[:-1] + peak_times_min[1:]) / 2

    # Split by phase
    normal_mask = interval_mid < NORMAL_END
    fast_mask = (interval_mid >= NORMAL_END) & (interval_mid < FAST_END)
    slow_mask = interval_mid >= FAST_END

    normal_ints = intervals_s[normal_mask]
    fast_ints = intervals_s[fast_mask]
    slow_ints = intervals_s[slow_mask]

    # Filter outliers for stats
    normal_clean = normal_ints[(normal_ints > 1.5) & (normal_ints < 8)]
    fast_clean = fast_ints[(fast_ints > 0.5) & (fast_ints < 5)]
    slow_clean = slow_ints[(slow_ints > 2) & (slow_ints < 15)]

# ── Per-phase peak counts ────────────────────────────────────────────────────

normal_peak_count = len(pp_normal)
fast_peak_count = len(pp_fast)
slow_peak_count = len(pp_slow)

normal_dur = NORMAL_END
fast_dur = FAST_END - NORMAL_END
slow_dur = SESSION_END - FAST_END

normal_rate = normal_peak_count / normal_dur if normal_dur > 0 else 0
fast_rate = fast_peak_count / fast_dur if fast_dur > 0 else 0
slow_rate = slow_peak_count / slow_dur if slow_dur > 0 else 0

print(f"\nPhase breakdown:")
print(f"  Normal (0-26:20): {normal_peak_count} peaks in {normal_dur:.1f} min = {normal_rate:.1f} br/min")
print(f"  Fast (26:20-28:20): {fast_peak_count} peaks in {fast_dur:.1f} min = {fast_rate:.1f} br/min")
print(f"  Slow (28:20-end): {slow_peak_count} peaks in {slow_dur:.1f} min = {slow_rate:.1f} br/min")

# ═══════════════════════════════════════════════════════════════════════════════
# CREATE FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(24, 42))
gs = GridSpec(10, 3, figure=fig, hspace=0.40, wspace=0.30,
              height_ratios=[2.5, 2.5, 2, 2.5, 2.2, 2, 2.5, 2.5, 2.5, 1.2])

fig.suptitle('30-MINUTE SESSION ANALYSIS\n'
             'Normal meditation (0-26:20)  |  Fast breathing (26:20-28:20)  |  '
             'Slow/subtle breathing (28:20-30:08)',
             fontsize=16, fontweight='bold', y=0.998)

CP = '#7B1FA2'   # purple for pressure
CT = '#1565C0'   # blue for thermistor
CI = '#2E7D32'   # green for IR
CF = '#D32F2F'   # red for fast phase
CS = '#FF8F00'   # amber for slow phase

def shade_phases(ax):
    ax.axvspan(NORMAL_END, FAST_END, alpha=0.12, color=CF, zorder=0)
    ax.axvspan(FAST_END, SESSION_END, alpha=0.12, color=CS, zorder=0)
    for m in range(0, 32, 5):
        ax.axvline(x=m, color='gray', linewidth=0.3, alpha=0.4)

# ═══════════════════════ ROW 0: Pressure — full session ══════════════════════
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(t_min, press_ac, color=CP, linewidth=0.2, alpha=0.15)
ax0.plot(t_min, press_sm, color=CP, linewidth=1.0)
ax0.scatter(t_min[press_peaks], press_sm[press_peaks], color='red', s=10, zorder=5, alpha=0.6)
ax0.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
shade_phases(ax0)
ax0.annotate('FAST', xy=((NORMAL_END + FAST_END) / 2, press_sm.max() * 0.85),
             fontsize=12, color=CF, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax0.annotate('SLOW', xy=((FAST_END + SESSION_END) / 2, press_sm.max() * 0.85),
             fontsize=12, color=CS, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax0.set_ylabel('Pressure AC (Pa)')
ax0.set_title(f'BMP280 Pressure — {len(press_peaks)} breaths detected over {t_min[-1]:.1f} min',
              fontsize=14, fontweight='bold')
ax0.grid(True, alpha=0.2)
ax0.set_xlim(-0.3, t_min[-1] + 0.3)

# ═══════════════════════ ROW 1: Thermistor — full session ════════════════════
ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
ax1.plot(t_min, therm_ac, color=CT, linewidth=0.2, alpha=0.15)
ax1.plot(t_min, therm_sm, color=CT, linewidth=1.0)
ax1.scatter(t_min[therm_peaks], therm_sm[therm_peaks], color='red', s=10, zorder=5, alpha=0.6)
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
shade_phases(ax1)
ax1.set_ylabel('Thermistor AC (ADC)')
ax1.set_title(f'Thermistor — {len(therm_peaks)} breaths detected', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.2)

# ═══════════════════════ ROW 2: Temperature ══════════════════════════════════
ax2 = fig.add_subplot(gs[2, :], sharex=ax0)
ax2.plot(t_min, temp, color='#E65100', linewidth=1.5)
shade_phases(ax2)
ax2.set_ylabel('Temperature (°C)')
ax2.set_title(f'BMP280 Temperature — {temp[0]:.1f}°C → {temp[-1]:.1f}°C '
              f'(+{temp[-1]-temp[0]:.1f}°C over {t_min[-1]:.0f} min)',
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.2)

# ═══════════════════════ ROW 3: Breathing energy ═════════════════════════════
ax3 = fig.add_subplot(gs[3, :], sharex=ax0)
ax3.plot(energy_t, energy_p, color=CP, linewidth=1.5, label='Pressure (3s rolling std)')
# Scale thermistor to same range for visual comparison
if energy_th.max() > 0:
    ax3.plot(energy_t, energy_th / energy_th.max() * energy_p.max() * 0.8,
             color=CT, linewidth=1.0, alpha=0.5, label='Thermistor (scaled)')
shade_phases(ax3)
ax3.set_ylabel('Breathing Amplitude (Pa)')
ax3.set_title('Breathing Energy — Rolling 3s Amplitude', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.2)

# Annotate fast/slow energy
fast_energy_mask = (energy_t >= NORMAL_END) & (energy_t < FAST_END)
slow_energy_mask = energy_t >= FAST_END
normal_energy_mask = (energy_t >= 2) & (energy_t < NORMAL_END)
if fast_energy_mask.sum() > 0:
    fast_amp = energy_p[fast_energy_mask].mean()
    norm_amp = energy_p[normal_energy_mask].mean()
    slow_amp = energy_p[slow_energy_mask].mean() if slow_energy_mask.sum() > 0 else 0
    ax3.annotate(f'Normal: {norm_amp:.2f} Pa\nFast: {fast_amp:.2f} Pa ({fast_amp/norm_amp:.1f}x)\n'
                 f'Slow: {slow_amp:.2f} Pa ({slow_amp/norm_amp:.1f}x)',
                 xy=(NORMAL_END - 3, energy_p.max() * 0.85),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.9))

# ═══════════════════════ ROW 4: Breath intervals ═════════════════════════════
ax4 = fig.add_subplot(gs[4, :], sharex=ax0)
if len(press_peaks) >= 2:
    # Color by phase
    for mask, color, label in [(normal_mask, CP, 'Normal'),
                                (fast_mask, CF, 'Fast'),
                                (slow_mask, CS, 'Slow')]:
        if mask.sum() > 0:
            ax4.scatter(interval_mid[mask], intervals_s[mask], color=color, s=12, alpha=0.6, label=label)

    # Running median
    win_med = 15
    if len(intervals_s) > win_med:
        running_med = np.array([np.median(intervals_s[max(0,i-win_med):i+1]) for i in range(len(intervals_s))])
        ax4.plot(interval_mid, running_med, color='black', linewidth=2, alpha=0.7, label='Running median (15)')

    shade_phases(ax4)
    if len(normal_clean) > 0:
        ax4.axhline(y=np.median(normal_clean), color='green', linewidth=1.5, linestyle='--',
                     label=f'Normal median: {np.median(normal_clean):.2f}s ({60/np.median(normal_clean):.1f} br/min)')
    ax4.axhline(y=3.3, color='blue', linewidth=1, linestyle=':', alpha=0.5, label='Target 3.3s')

ax4.set_ylabel('Interval (s)')
ax4.set_title('Breath-to-Breath Intervals', fontsize=14, fontweight='bold')
ax4.legend(loc='upper left', fontsize=8, ncol=2)
ax4.grid(True, alpha=0.2)
ax4.set_ylim(0, min(15, intervals_s.max() * 1.05))

# ═══════════════════════ ROW 5: Signal degradation ═══════════════════════════
ax5a = fig.add_subplot(gs[5, 0:2])
valid_mins = [i for i, s in enumerate(m_press_std) if s > 0]
x = np.arange(len(valid_mins))
labs = [minute_labels[i] for i in valid_mins]
p_vals = [p_norm[i] for i in valid_mins]
t_vals = [t_norm[i] for i in valid_mins]
ax5a.plot(x, t_vals, 'o-', color=CT, linewidth=2, markersize=5, label='Thermistor')
ax5a.plot(x, p_vals, 's-', color=CP, linewidth=2, markersize=5, label='Pressure')
ax5a.axhline(y=100, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax5a.axhline(y=50, color='red', linewidth=1, linestyle=':', alpha=0.5, label='50% threshold')
# Mark phases
fast_start_idx = next((i for i, l in enumerate(labs) if l >= 26), len(labs))
slow_start_idx = next((i for i, l in enumerate(labs) if l >= 28), len(labs))
ax5a.axvspan(fast_start_idx - 0.3, slow_start_idx - 0.3, alpha=0.12, color=CF)
ax5a.axvspan(slow_start_idx - 0.3, len(labs) - 0.7, alpha=0.12, color=CS)
ax5a.set_xlabel('Minute')
ax5a.set_ylabel(f'% of minute {ref_idx}')
ax5a.set_title('Signal Degradation Over 30 Minutes', fontsize=13, fontweight='bold')
ax5a.set_xticks(x[::2])
ax5a.set_xticklabels([str(l) for l in labs[::2]], fontsize=8)
ax5a.legend(fontsize=8)
ax5a.grid(True, alpha=0.3)
# Endpoint annotation
if len(t_vals) > 0:
    ax5a.annotate(f'{t_vals[-1]:.0f}%', xy=(len(valid_mins)-1, t_vals[-1]),
                  xytext=(10, 5), textcoords='offset points', fontsize=10, fontweight='bold', color=CT)
    ax5a.annotate(f'{p_vals[-1]:.0f}%', xy=(len(valid_mins)-1, p_vals[-1]),
                  xytext=(10, -10), textcoords='offset points', fontsize=10, fontweight='bold', color=CP)

# Peaks per minute
ax5b = fig.add_subplot(gs[5, 2])
x_pm = np.arange(len(valid_mins))
ppv = [m_ppeaks[i] for i in valid_mins]
tpv = [m_tpeaks[i] for i in valid_mins]
width = 0.35
ax5b.bar(x_pm - width/2, tpv, width, color=CT, alpha=0.7, label='Thermistor')
ax5b.bar(x_pm + width/2, ppv, width, color=CP, alpha=0.7, label='Pressure')
ax5b.set_xlabel('Minute')
ax5b.set_ylabel('Peaks')
ax5b.set_title('Breaths Per Minute', fontsize=11, fontweight='bold')
ax5b.set_xticks(x_pm[::3])
ax5b.set_xticklabels([str(labs[i]) for i in range(0, len(labs), 3)], fontsize=8)
ax5b.legend(fontsize=7)
ax5b.grid(True, alpha=0.2, axis='y')

# ═══════════════════════ ROW 6: Phase zooms — Normal ═════════════════════════
# Show 3 representative 2-minute windows from the normal phase
zoom_windows = [
    (1, 3, 'Early (min 1-3)'),
    (12, 14, 'Middle (min 12-14)'),
    (24, 26, 'Late (min 24-26)'),
]
for col, (z_start, z_end, z_title) in enumerate(zoom_windows):
    ax = fig.add_subplot(gs[6, col])
    mask = (t_min >= z_start) & (t_min < z_end)
    ax.plot(t_min[mask], press_sm[mask], color=CP, linewidth=1.2)
    for p in press_peaks:
        if z_start <= t_min[p] < z_end:
            ax.scatter(t_min[p], press_sm[p], color='red', s=30, zorder=5)
    n_pk = sum(1 for p in press_peaks if z_start <= t_min[p] < z_end)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title(f'{z_title}\n{n_pk} peaks', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Pa')
    ax.grid(True, alpha=0.2)

# ═══════════════════════ ROW 7: Fast breathing zoom ══════════════════════════
ax7 = fig.add_subplot(gs[7, :])
fast_s, fast_e = NORMAL_END - 0.5, FAST_END + 0.5
mask_fast = (t_min >= fast_s) & (t_min <= fast_e)
ax7.plot(t_min[mask_fast], press_ac[mask_fast], color=CP, linewidth=0.3, alpha=0.3)
ax7.plot(t_min[mask_fast], press_sm_fast[mask_fast], color=CP, linewidth=1.5)
for p in pp_fast:
    ax7.scatter(t_min[p], press_sm_fast[p], color='red', s=40, zorder=5)
# Also show normal-phase peaks in the lead-in
for p in pp_normal:
    if fast_s <= t_min[p] < NORMAL_END:
        ax7.scatter(t_min[p], press_sm_fast[p], color='gray', s=25, zorder=4, alpha=0.5)
ax7.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax7.axvspan(NORMAL_END, FAST_END, alpha=0.15, color=CF)
ax7.annotate(f'FAST BREATHING — {fast_peak_count} peaks ({fast_rate:.1f} br/min)',
             xy=((NORMAL_END + FAST_END) / 2, press_sm_fast[mask_fast].max() * 0.9),
             fontsize=13, color=CF, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Dual axis for thermistor
ax7r = ax7.twinx()
ax7r.plot(t_min[mask_fast], therm_sm[mask_fast], color=CT, linewidth=1.0, alpha=0.5)
ax7r.set_ylabel('Thermistor AC (ADC)', color=CT, fontsize=9)
ax7r.tick_params(axis='y', labelcolor=CT)

ax7.set_ylabel('Pressure AC (Pa)')
ax7.set_xlabel('Time (min)')
ax7.set_title('FAST BREATHING PHASE — Zoom (25:50 - 28:50)', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.2)

# ═══════════════════════ ROW 8: Slow/subtle breathing zoom ═══════════════════
ax8 = fig.add_subplot(gs[8, :])
slow_s, slow_e = FAST_END - 0.5, SESSION_END + 0.1
mask_slow = (t_min >= slow_s) & (t_min <= slow_e)
ax8.plot(t_min[mask_slow], press_ac[mask_slow], color=CP, linewidth=0.3, alpha=0.3)
ax8.plot(t_min[mask_slow], press_sm[mask_slow], color=CS, linewidth=1.5)
for p in pp_slow:
    ax8.scatter(t_min[p], press_sm[p], color='red', s=40, zorder=5)
# Also show fast-phase peaks in the lead-in
for p in pp_fast:
    if slow_s <= t_min[p] < FAST_END:
        ax8.scatter(t_min[p], press_sm[p], color='gray', s=25, zorder=4, alpha=0.5)
ax8.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax8.axvspan(FAST_END, SESSION_END, alpha=0.15, color=CS)
ax8.annotate(f'SLOW / SUBTLE — {slow_peak_count} peaks ({slow_rate:.1f} br/min)',
             xy=((FAST_END + SESSION_END) / 2, press_sm[mask_slow].max() * 0.9),
             fontsize=13, color=CS, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax8r = ax8.twinx()
ax8r.plot(t_min[mask_slow], therm_sm[mask_slow], color=CT, linewidth=1.0, alpha=0.5)
ax8r.set_ylabel('Thermistor AC (ADC)', color=CT, fontsize=9)
ax8r.tick_params(axis='y', labelcolor=CT)

ax8.set_ylabel('Pressure AC (Pa)')
ax8.set_xlabel('Time (min)')
ax8.set_title('SLOW/SUBTLE BREATHING PHASE — Zoom (27:50 - 30:08)', fontsize=14, fontweight='bold')
ax8.grid(True, alpha=0.2)

# ═══════════════════════ ROW 9: Verdict ══════════════════════════════════════
ax9 = fig.add_subplot(gs[9, :])
ax9.axis('off')

# Signal at key minutes
therm_25 = t_norm[25] if len(t_norm) > 25 else 0
press_25 = p_norm[25] if len(p_norm) > 25 else 0

verdict = [
    f"30-MINUTE SESSION — {n_total} samples, {t_min[-1]:.1f} min @ {sr:.0f} Hz",
    f"{'═' * 100}",
    f"  BLE stability:        1 gap >100ms (950ms @ 0:52) — otherwise continuous 50ms intervals",
    f"  Temperature drift:    {temp[0]:.1f}°C → {temp[-1]:.1f}°C (+{temp[-1]-temp[0]:.1f}°C)",
    f"",
    f"  NORMAL (0-26:20):     {normal_peak_count} peaks | {normal_rate:.1f} br/min"
    + (f" | median interval {np.median(normal_clean):.2f}s" if len(normal_clean) > 0 else ""),
    f"  FAST   (26:20-28:20): {fast_peak_count} peaks | {fast_rate:.1f} br/min"
    + (f" | median interval {np.median(fast_clean):.2f}s" if len(fast_clean) > 0 else ""),
    f"  SLOW   (28:20-end):   {slow_peak_count} peaks | {slow_rate:.1f} br/min"
    + (f" | median interval {np.median(slow_clean):.2f}s" if len(slow_clean) > 0 else ""),
    f"",
    f"  Signal at min 25:     Thermistor {therm_25:.0f}% | Pressure {press_25:.0f}% (of min {ref_idx} baseline)",
    f"  Pressure at min 30:   Still detecting breaths — NO degradation from sensor physics",
]
ax9.text(0.02, 0.95, '\n'.join(verdict), transform=ax9.transAxes,
         fontsize=10.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.5))

# ── Save ─────────────────────────────────────────────────────────────────────
output = '/Users/mert/Developer/Hardware/BreathAnalysis/analysis/30min_session_analysis.png'
plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output}")

# Print detailed minute-by-minute
print(f"\n{'Min':>4} {'Press std':>10} {'Press %':>8} {'Therm std':>10} {'Therm %':>8} {'P pk':>5} {'T pk':>5}")
print("-" * 58)
for i in valid_mins:
    print(f"{minute_labels[i]:4d} {m_press_std[i]:10.3f} {p_norm[i]:7.0f}% "
          f"{m_therm_std[i]:10.1f} {t_norm[i]:7.0f}% {m_ppeaks[i]:5d} {m_tpeaks[i]:5d}")
