#!/usr/bin/env python3
"""
Extended session analysis: minutes 5-15 (two recordings stitched together).
Session A: 5-10 min window (normal breathing throughout)
Session B: 10-15 min window (breath hold at ~12:00-12:25 global, compensatory mouth breathing after)
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

files = [
    ('/Users/mert/Downloads/5-10MinSnapshot.gz', 5.0),
    ('/Users/mert/Downloads/10-15MinSnapshot-(2-2.25-stopped-breathing)', 10.0),
]

sessions = []
for path, global_start_min in files:
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    samples = data['samples']
    ts = np.array([s[0] for s in samples], dtype=np.float64)
    t_sec = (ts - ts[0]) / 1000.0
    sessions.append({
        'therm': np.array([s[1] for s in samples], dtype=np.float64),
        'ir': np.array([s[2] for s in samples], dtype=np.float64),
        'pressure': np.array([s[6] for s in samples], dtype=np.float64),
        'temp': np.array([s[4] for s in samples], dtype=np.float64),
        't_sec': t_sec,
        't_global_min': global_start_min + t_sec / 60.0,
        'global_start': global_start_min,
        'n': len(samples),
    })

# Stitch into global arrays
t_min = np.concatenate([s['t_global_min'] for s in sessions])
therm = np.concatenate([s['therm'] for s in sessions])
pressure = np.concatenate([s['pressure'] for s in sessions])
ir = np.concatenate([s['ir'] for s in sessions])
temp = np.concatenate([s['temp'] for s in sessions])
n_total = len(t_min)
n_a = sessions[0]['n']
sr = 20.0

# Breath hold: session B 2:00-2:25 = global 12:00-12:25
breath_hold_start = 12.0
breath_hold_end = 12.0 + 25 / 60  # 12.417

# ── Process each session with its own baseline ───────────────────────────────

window_15s = int(15 * sr)
smooth_w = int(0.25 * sr)

def process(signal):
    bl = moving_average(signal, window_15s)
    ac = signal - bl
    sm = moving_average(ac, smooth_w)
    return ac, sm

therm_ac_a, therm_sm_a = process(sessions[0]['therm'])
press_ac_a, press_sm_a = process(sessions[0]['pressure'])
therm_ac_b, therm_sm_b = process(sessions[1]['therm'])
press_ac_b, press_sm_b = process(sessions[1]['pressure'])
ir_ac_a, ir_sm_a = process(sessions[0]['ir'])
ir_ac_b, ir_sm_b = process(sessions[1]['ir'])

therm_sm = np.concatenate([therm_sm_a, therm_sm_b])
press_sm = np.concatenate([press_sm_a, press_sm_b])
therm_ac = np.concatenate([therm_ac_a, therm_ac_b])
press_ac = np.concatenate([press_ac_a, press_ac_b])
ir_ac = np.concatenate([ir_ac_a, ir_ac_b])
ir_sm = np.concatenate([ir_sm_a, ir_sm_b])

# Peak detection per session
min_peak_dist = int(1.8 * sr)
pp_a, ppr_a = find_peaks_with_prom(press_sm_a, min_peak_dist, press_sm_a.std() * 0.3)
pp_b, ppr_b = find_peaks_with_prom(press_sm_b, min_peak_dist, press_sm_b.std() * 0.3)
tp_a, _ = find_peaks_with_prom(therm_sm_a, min_peak_dist, therm_sm_a.std() * 0.3)
tp_b, _ = find_peaks_with_prom(therm_sm_b, min_peak_dist, therm_sm_b.std() * 0.3)

# Map to global indices
pp_b_g = [p + n_a for p in pp_b]
tp_b_g = [p + n_a for p in tp_b]
all_pp = pp_a + pp_b_g
all_tp = tp_a + tp_b_g
all_ppr = ppr_a + ppr_b

print(f"Session A: {len(pp_a)} pressure, {len(tp_a)} therm peaks")
print(f"Session B: {len(pp_b)} pressure, {len(tp_b)} therm peaks")

# ── Compute rolling breathing energy (key for breath hold detection) ──────

# 3-second sliding window amplitude
win_3s = int(3 * sr)
step = int(0.5 * sr)
energy_t_b = []
energy_press_b = []
energy_therm_b = []
t_b = sessions[1]['t_sec']
for i in range(0, len(press_sm_b) - win_3s, step):
    energy_t_b.append(10.0 + t_b[i + win_3s // 2] / 60)
    energy_press_b.append(np.std(press_sm_b[i:i + win_3s]))
    energy_therm_b.append(np.std(therm_sm_b[i:i + win_3s]))

energy_t_b = np.array(energy_t_b)
energy_press_b = np.array(energy_press_b)
energy_therm_b = np.array(energy_therm_b)

# ── Per-minute stats ─────────────────────────────────────────────────────────

minute_labels = list(range(5, 16))
m_therm_std, m_press_std, m_tpeaks, m_ppeaks = [], [], [], []
for m in minute_labels:
    mask = (t_min >= m) & (t_min < m + 1)
    if mask.sum() < 100:
        m_therm_std.append(0); m_press_std.append(0)
        m_tpeaks.append(0); m_ppeaks.append(0)
        continue
    m_therm_std.append(therm_ac[mask].std())
    m_press_std.append(press_ac[mask].std())
    m_ppeaks.append(sum(1 for p in all_pp if m <= t_min[p] < m + 1))
    m_tpeaks.append(sum(1 for p in all_tp if m <= t_min[p] < m + 1))

valid = [i for i, s in enumerate(m_press_std) if s > 0]
t_norm = [m_therm_std[i] / m_therm_std[valid[0]] * 100 if m_therm_std[valid[0]] > 0 else 0 for i in valid]
p_norm = [m_press_std[i] / m_press_std[valid[0]] * 100 if m_press_std[valid[0]] > 0 else 0 for i in valid]

# ── Create figure ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(22, 34))
gs = GridSpec(10, 3, figure=fig, hspace=0.42, wspace=0.3,
              height_ratios=[2.2, 2.2, 2.2, 2.2, 2.5, 2, 1.8, 1.8, 1.8, 1.0])

fig.suptitle('Extended Session — Minutes 5-15\n'
             'Session A (min 5-10) + Session B (min 10-15)  •  '
             'Breath hold @ ~12:00-12:25',
             fontsize=16, fontweight='bold', y=0.997)

CP = '#7B1FA2'
CT = '#1565C0'
CI = '#2E7D32'

def annotate_ax(ax):
    ax.axvline(x=10.0, color='black', linewidth=2, alpha=0.5)
    ax.axvspan(breath_hold_start, breath_hold_end, alpha=0.2, color='red')
    for m in range(5, 16):
        ax.axvline(x=m, color='gray', linewidth=0.3, alpha=0.3)

# ═══════════════════════ ROW 0: Pressure full session ═════════════════════════
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(t_min, press_ac, color=CP, linewidth=0.3, alpha=0.2)
ax0.plot(t_min, press_sm, color=CP, linewidth=1.2)
ax0.scatter(t_min[all_pp], press_sm[all_pp], color='red', s=18, zorder=5,
            label=f'{len(all_pp)} peaks')
ax0.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
annotate_ax(ax0)
ax0.annotate('HOLD', xy=((breath_hold_start + breath_hold_end) / 2, 0),
             fontsize=10, color='red', fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax0.set_ylabel('Pressure AC (Pa)')
ax0.set_xlabel('Time (min)')
ax0.set_title('BMP280 Pressure — Full Session', fontsize=13, fontweight='bold')
ax0.legend(loc='upper right')
ax0.grid(True, alpha=0.2)

# ═══════════════════════ ROW 1: Thermistor full session ═══════════════════════
ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
ax1.plot(t_min, therm_ac, color=CT, linewidth=0.3, alpha=0.2)
ax1.plot(t_min, therm_sm, color=CT, linewidth=1.2)
ax1.scatter(t_min[all_tp], therm_sm[all_tp], color='red', s=18, zorder=5,
            label=f'{len(all_tp)} peaks')
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
annotate_ax(ax1)
ax1.annotate('HOLD', xy=((breath_hold_start + breath_hold_end) / 2, 0),
             fontsize=10, color='red', fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.set_ylabel('Thermistor AC (ADC)')
ax1.set_xlabel('Time (min)')
ax1.set_title('Thermistor — Full Session', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)

# ═══════════════════════ ROW 2: IR full session ═══════════════════════════════
ax2 = fig.add_subplot(gs[2, :], sharex=ax0)
ax2.plot(t_min, ir_ac, color=CI, linewidth=0.3, alpha=0.2)
ax2.plot(t_min, ir_sm, color=CI, linewidth=1.2)
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
annotate_ax(ax2)
ax2.annotate('HOLD', xy=((breath_hold_start + breath_hold_end) / 2, 0),
             fontsize=10, color='red', fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.set_ylabel('IR AC (counts)')
ax2.set_xlabel('Time (min)')
ax2.set_title('MAX30102 IR — Full Session', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.2)

# ═══════════════════════ ROW 3: Breathing energy (rolling amplitude) ═════════
ax3 = fig.add_subplot(gs[3, :])

# Session A energy
win_3s_a = int(3 * sr)
step_a = int(0.5 * sr)
en_t_a, en_p_a, en_th_a = [], [], []
t_a_sec = sessions[0]['t_sec']
for i in range(0, len(press_sm_a) - win_3s_a, step_a):
    en_t_a.append(5.0 + t_a_sec[i + win_3s_a // 2] / 60)
    en_p_a.append(np.std(press_sm_a[i:i + win_3s_a]))
    en_th_a.append(np.std(therm_sm_a[i:i + win_3s_a]))

# Combine
all_en_t = np.array(en_t_a + energy_t_b.tolist())
all_en_p = np.array(en_p_a + energy_press_b.tolist())
all_en_th = np.array(en_th_a + energy_therm_b.tolist())

ax3.plot(all_en_t, all_en_p, color=CP, linewidth=1.5, label='Pressure amplitude')
ax3.plot(all_en_t, all_en_th / all_en_th.max() * all_en_p.max(), color=CT, linewidth=1.2,
         alpha=0.6, label='Thermistor (scaled)')
ax3.axvspan(breath_hold_start, breath_hold_end, alpha=0.3, color='red', label='Breath hold')
ax3.axvline(x=10.0, color='black', linewidth=2, alpha=0.5)

# Mark the quiet zone
hold_mask = (all_en_t >= breath_hold_start) & (all_en_t <= breath_hold_end)
if hold_mask.sum() > 0:
    hold_amp = all_en_p[hold_mask].mean()
    normal_amp = all_en_p[~hold_mask].mean()
    ax3.annotate(f'Hold avg: {hold_amp:.2f} Pa\nNormal avg: {normal_amp:.2f} Pa\n'
                 f'Ratio: {hold_amp/normal_amp*100:.0f}%',
                 xy=(breath_hold_end + 0.1, hold_amp),
                 fontsize=9, color='red',
                 bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))

for m in range(5, 16):
    ax3.axvline(x=m, color='gray', linewidth=0.3, alpha=0.3)

ax3.set_ylabel('3s Rolling Std (Pa)')
ax3.set_xlabel('Time (min)')
ax3.set_title('BREATHING ENERGY — Rolling Amplitude (key for breath hold detection)',
              fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.2)

# ═══════════════════════ ROW 4: Breath hold zoom ═════════════════════════════
# Wide zoom: 11:00 to 13:30 (2.5 min window around the hold)
ax4 = fig.add_subplot(gs[4, :])
zoom_s, zoom_e = 11.0, 13.5
mask_z = (t_min >= zoom_s) & (t_min <= zoom_e)
t_z = t_min[mask_z]

# Dual y-axis: pressure (left) and thermistor (right)
ax4.plot(t_z, press_sm[mask_z], color=CP, linewidth=2, label='Pressure')
ax4r = ax4.twinx()
ax4r.plot(t_z, therm_sm[mask_z], color=CT, linewidth=1.5, alpha=0.7, label='Thermistor')
ax4r.set_ylabel('Thermistor AC (ADC)', color=CT)
ax4r.tick_params(axis='y', labelcolor=CT)

ax4.axvspan(breath_hold_start, breath_hold_end, alpha=0.3, color='red')
# Mark compensatory fast mouth breathing (approx 30s after hold)
comp_end = breath_hold_end + 0.5
ax4.axvspan(breath_hold_end, comp_end, alpha=0.2, color='orange')
ax4.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

# Annotate phases
ax4.annotate('Normal\nnose breathing', xy=(11.3, press_sm[mask_z].max() * 0.7),
             fontsize=10, ha='center', fontweight='bold', color=CP)
ax4.annotate('BREATH\nHOLD', xy=((breath_hold_start + breath_hold_end) / 2, 0),
             fontsize=12, ha='center', fontweight='bold', color='red',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
ax4.annotate('Fast mouth\nbreathing', xy=((breath_hold_end + comp_end) / 2,
             press_sm[mask_z].min() * 0.7),
             fontsize=10, ha='center', fontweight='bold', color='darkorange')
ax4.annotate('Recovery', xy=(13.2, press_sm[mask_z].max() * 0.5),
             fontsize=10, ha='center', fontweight='bold', color='green')

# Mark peaks in zoom
for p in all_pp:
    if zoom_s <= t_min[p] <= zoom_e:
        in_hold = breath_hold_start <= t_min[p] <= breath_hold_end
        ax4.scatter(t_min[p], press_sm[p], color='orange' if in_hold else 'red',
                    s=40, zorder=5, marker='x' if in_hold else 'o')

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4r.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
ax4.set_ylabel('Pressure AC (Pa)', color=CP)
ax4.set_xlabel('Time (min)')
ax4.set_title('BREATH HOLD ZOOM (11:00 - 13:30) — Pressure + Thermistor',
              fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.2)

# ═══════════════════════ ROW 5: Breath intervals ═════════════════════════════
ax5 = fig.add_subplot(gs[5, :])
# Session A intervals
if len(pp_a) >= 2:
    t_a_g = sessions[0]['t_global_min']
    int_t_a = [(t_a_g[pp_a[i]] + t_a_g[pp_a[i+1]]) / 2 for i in range(len(pp_a)-1)]
    int_v_a = np.diff(t_a_g[pp_a]) * 60  # seconds
    ax5.plot(int_t_a, int_v_a, 'o-', color=CP, markersize=3, linewidth=1, alpha=0.7)

# Session B intervals
if len(pp_b) >= 2:
    t_b_g = sessions[1]['t_global_min']
    int_t_b = [(t_b_g[pp_b[i]] + t_b_g[pp_b[i+1]]) / 2 for i in range(len(pp_b)-1)]
    int_v_b = np.diff(t_b_g[pp_b]) * 60
    ax5.plot(int_t_b, int_v_b, 'o-', color=CP, markersize=3, linewidth=1, alpha=0.7)

    # Color intervals during hold differently
    for i in range(len(int_t_b)):
        if breath_hold_start <= int_t_b[i] <= comp_end:
            ax5.scatter(int_t_b[i], int_v_b[i], color='red', s=50, zorder=6)

# Normal intervals stats (exclude hold period)
all_ints = []
if len(pp_a) >= 2:
    all_ints.extend(int_v_a.tolist())
if len(pp_b) >= 2:
    normal_b = [int_v_b[i] for i in range(len(int_v_b))
                if not (breath_hold_start - 0.2 <= int_t_b[i] <= comp_end + 0.2)]
    all_ints.extend(normal_b)
normal_ints = np.array([x for x in all_ints if x < 8])

ax5.axhline(y=np.median(normal_ints), color='green', linewidth=2, linestyle='--',
            label=f'Median: {np.median(normal_ints):.2f}s ({60/np.median(normal_ints):.1f} br/min)')
ax5.axhline(y=3.3, color='blue', linewidth=1, linestyle=':', label='Target: 3.3s', alpha=0.7)

ax5.axvspan(breath_hold_start, breath_hold_end, alpha=0.2, color='red')
ax5.axvspan(breath_hold_end, comp_end, alpha=0.15, color='orange')
ax5.axvline(x=10.0, color='black', linewidth=2, alpha=0.4)

ax5.set_ylabel('Interval (s)')
ax5.set_xlabel('Time (min)')
ax5.set_title('Breath-to-Breath Intervals', fontsize=13, fontweight='bold')
ax5.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.2)
ax5.set_ylim(0, min(12, max(all_ints) * 1.05 if all_ints else 10))

# ═══════════════════════ ROW 6: Signal degradation ═══════════════════════════
ax6a = fig.add_subplot(gs[6, 0:2])
x = np.arange(len(valid))
labs = [minute_labels[i] for i in valid]
ax6a.plot(x, t_norm, 'o-', color=CT, linewidth=2, markersize=8, label='Thermistor')
ax6a.plot(x, p_norm, 's-', color=CP, linewidth=2, markersize=8, label='Pressure')
ax6a.axhline(y=100, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax6a.axhline(y=50, color='red', linewidth=1, linestyle=':', alpha=0.5, label='50% threshold')
# Mark breath hold minute
hold_x = labs.index(12) if 12 in labs else None
if hold_x is not None:
    ax6a.axvspan(hold_x - 0.3, hold_x + 0.3, alpha=0.15, color='red')
ax6a.set_xlabel('Minute')
ax6a.set_ylabel('% of minute 5')
ax6a.set_title('Signal Degradation — Min 5 to 15', fontsize=13, fontweight='bold')
ax6a.set_xticks(x)
ax6a.set_xticklabels([str(l) for l in labs])
ax6a.legend(fontsize=9)
ax6a.grid(True, alpha=0.3)
for vals, color in [(t_norm, CT), (p_norm, CP)]:
    ax6a.annotate(f'{vals[-1]:.0f}%', xy=(len(valid) - 1, vals[-1]),
                  xytext=(10, 0), textcoords='offset points',
                  fontsize=10, fontweight='bold', color=color)

# Temperature
ax6b = fig.add_subplot(gs[6, 2])
ax6b.plot(t_min, temp, color='#E65100', linewidth=1.5)
ax6b.axvline(x=10.0, color='black', linewidth=1.5, alpha=0.4)
ax6b.set_xlabel('Time (min)')
ax6b.set_ylabel('°C')
ax6b.set_title(f'Temp ({temp[0]:.1f}→{temp[-1]:.1f}°C)', fontsize=11, fontweight='bold')
ax6b.grid(True, alpha=0.3)

# ═══════════════════════ ROW 7: Peaks per minute + waveforms ═════════════════
ax7a = fig.add_subplot(gs[7, 0])
width = 0.35
tpv = [m_tpeaks[i] for i in valid]
ppv = [m_ppeaks[i] for i in valid]
ax7a.bar(x - width/2, tpv, width, color=CT, alpha=0.8, label='Thermistor')
ax7a.bar(x + width/2, ppv, width, color=CP, alpha=0.8, label='Pressure')
ax7a.set_xlabel('Minute')
ax7a.set_ylabel('Peaks')
ax7a.set_title('Breaths Per Minute', fontsize=11, fontweight='bold')
ax7a.set_xticks(x)
ax7a.set_xticklabels([str(l) for l in labs])
ax7a.legend(fontsize=8)
ax7a.grid(True, alpha=0.2, axis='y')

# Waveform overlay: Session A
ax7b = fig.add_subplot(gs[7, 1])
half_win = int(2.0 * sr)
wf_a = []
for p in pp_a:
    if p - half_win >= 0 and p + half_win < n_a:
        wf_a.append(press_sm_a[p - half_win:p + half_win])
        ax7b.plot(np.linspace(-2, 2, len(wf_a[-1])), wf_a[-1], color=CP, alpha=0.12, linewidth=0.8)
if wf_a:
    mean_wf = np.mean(wf_a, axis=0)
    ax7b.plot(np.linspace(-2, 2, len(mean_wf)), mean_wf, color='red', linewidth=2.5,
              label=f'Mean (n={len(wf_a)})')
ax7b.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax7b.set_title('Waveform: Min 5-10', fontsize=11, fontweight='bold')
ax7b.set_xlabel('Time from peak (s)')
ax7b.set_ylabel('Pa')
ax7b.legend(fontsize=8)
ax7b.grid(True, alpha=0.2)

# Waveform overlay: Session B
ax7c = fig.add_subplot(gs[7, 2])
wf_b = []
for p in pp_b:
    if p - half_win >= 0 and p + half_win < sessions[1]['n']:
        wf_b.append(press_sm_b[p - half_win:p + half_win])
        ax7c.plot(np.linspace(-2, 2, len(wf_b[-1])), wf_b[-1], color=CP, alpha=0.12, linewidth=0.8)
if wf_b:
    mean_wf = np.mean(wf_b, axis=0)
    ax7c.plot(np.linspace(-2, 2, len(mean_wf)), mean_wf, color='red', linewidth=2.5,
              label=f'Mean (n={len(wf_b)})')
ax7c.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax7c.set_title('Waveform: Min 10-15', fontsize=11, fontweight='bold')
ax7c.set_xlabel('Time from peak (s)')
ax7c.set_ylabel('Pa')
ax7c.legend(fontsize=8)
ax7c.grid(True, alpha=0.2)

# ═══════════════════════ ROW 8: Interval histogram + close-up of hold ════════
ax8a = fig.add_subplot(gs[8, 0:2])
ax8a.hist(normal_ints, bins=30, color=CP, alpha=0.7, edgecolor='white')
ax8a.axvline(x=np.mean(normal_ints), color='red', linewidth=2,
             label=f'Mean: {np.mean(normal_ints):.2f}s ({60/np.mean(normal_ints):.1f} br/min)')
ax8a.axvline(x=np.median(normal_ints), color='green', linewidth=2, linestyle='--',
             label=f'Median: {np.median(normal_ints):.2f}s')
ax8a.axvline(x=3.3, color='blue', linewidth=1, linestyle=':', label='Target 3.3s')
ax8a.set_xlabel('Breath Interval (s)')
ax8a.set_ylabel('Count')
ax8a.set_title(f'Interval Distribution (n={len(normal_ints)})', fontsize=11, fontweight='bold')
ax8a.legend(fontsize=8)
ax8a.grid(True, alpha=0.2)

# Close-up: raw pressure during hold (not baseline-subtracted)
ax8b = fig.add_subplot(gs[8, 2])
mask_hold = (sessions[1]['t_sec'] >= 110) & (sessions[1]['t_sec'] <= 160)
t_raw = sessions[1]['t_sec'][mask_hold]
p_raw = sessions[1]['pressure'][mask_hold]
ax8b.plot(t_raw, p_raw, color=CP, linewidth=1)
ax8b.axvspan(120, 145, alpha=0.2, color='red')
ax8b.set_xlabel('Session B time (s)')
ax8b.set_ylabel('Raw pressure delta (Pa)')
ax8b.set_title('Raw Pressure @ Hold', fontsize=11, fontweight='bold')
ax8b.grid(True, alpha=0.2)

# ═══════════════════════ ROW 9: Verdict ══════════════════════════════════════
ax9 = fig.add_subplot(gs[9, :])
ax9.axis('off')

# Compute hold stats
hold_peaks = sum(1 for p in all_pp if breath_hold_start <= t_min[p] <= breath_hold_end)
hold_mask_en = (all_en_t >= breath_hold_start) & (all_en_t <= breath_hold_end)
hold_energy = all_en_p[hold_mask_en].mean() if hold_mask_en.sum() > 0 else 0
normal_energy = all_en_p[~hold_mask_en].mean()

verdict = [
    f"SESSION ANALYSIS — Minutes 5-15 (combined {n_total} samples)",
    f"{'═' * 95}",
    f"  Session A (min 5-10):  Pressure {len(pp_a)} peaks ({len(pp_a)/5:.0f}/min) | Thermistor {len(tp_a)} peaks",
    f"  Session B (min 10-15): Pressure {len(pp_b)} peaks ({len(pp_b)/5:.0f}/min) | Thermistor {len(tp_b)} peaks",
    f"  Breath rate (normal):  {60/np.mean(normal_ints):.1f} br/min (median interval {np.median(normal_ints):.2f}s)",
    f"  Signal at min 15:      Thermistor {t_norm[-1]:.0f}% | Pressure {p_norm[-1]:.0f}% (of min 5 baseline)",
    f"",
    f"  Breath hold @ 12:00:   {hold_peaks} peaks detected in hold zone | Energy {hold_energy:.2f} vs normal {normal_energy:.2f} Pa",
    f"  Temperature (min 15):  {temp[-1]:.1f}°C (approaching body equilibrium)",
]
ax9.text(0.02, 0.95, '\n'.join(verdict), transform=ax9.transAxes,
         fontsize=10.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.5))

# ── Save ─────────────────────────────────────────────────────────────────────
output = '/Users/mert/Developer/Hardware/BreathAnalysis/analysis/extended_sessions_5_15min.png'
plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output}")

# ── Print stats ──────────────────────────────────────────────────────────────
print(f"\n{'Min':>4} {'Therm std':>10} {'Therm %':>8} {'Press std':>10} {'Press %':>8} {'T pk':>5} {'P pk':>5}")
print("-" * 55)
for i in range(len(valid)):
    m = minute_labels[valid[i]]
    print(f"{m:4d} {m_therm_std[valid[i]]:10.2f} {t_norm[i]:7.0f}% {m_press_std[valid[i]]:10.3f} {p_norm[i]:7.0f}% {m_tpeaks[valid[i]]:5d} {m_ppeaks[valid[i]]:5d}")
