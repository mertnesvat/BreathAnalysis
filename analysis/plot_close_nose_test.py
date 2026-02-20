#!/usr/bin/env python3
"""
Close Nose Test — BMP280 ~1mm from nostrils
=============================================
Phase 1: 0:00 - 1:00  — Normal breath (+ mouth blow at ~40-50s)
Phase 2: 1:00 - 1:45  — Breath hold (no air out)
Phase 3: 1:45 - 2:45  — Deep breath, full lungs
Phase 4: 2:45 - 3:45  — Fast panting
Phase 5: 3:45 - 5:00  — Return to normal rhythm
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, welch
import matplotlib.gridspec as gridspec


def detect_breaths_v4b(ts, pressure, sample_rate,
                       bandpass=(0.08, 0.7),
                       amplitude_threshold_pa=2.0,
                       envelope_window_s=8,
                       min_cluster_breaths=5,
                       cluster_window_s=20,
                       min_breath_period_s=1.5):
    n = len(ts)
    nyq = sample_rate / 2
    b, a = butter(3, [bandpass[0] / nyq, bandpass[1] / nyq], btype='band')
    bp_signal = filtfilt(b, a, pressure)

    env_win = int(envelope_window_s * sample_rate)
    padded = np.pad(bp_signal ** 2, (env_win // 2, env_win // 2), mode='edge')
    rolling_mean_sq = np.convolve(padded, np.ones(env_win) / env_win, mode='valid')[:n]
    envelope = np.sqrt(np.maximum(rolling_mean_sq, 0))

    gate_mask = envelope > amplitude_threshold_pa
    min_prom = amplitude_threshold_pa * 0.3
    all_peaks, _ = find_peaks(bp_signal, prominence=min_prom,
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


# ═══════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════

print("Loading data...")
with gzip.open('/Users/mert/Downloads/5minCloseNoseTest.gz', 'rt') as f:
    data = json.load(f)

# Also load the 10-min null test for comparison
with gzip.open('/Users/mert/Downloads/10minsNullTest2.gz', 'rt') as f:
    null_data = json.load(f)

samples = np.array(data['samples'])
ts = samples[:, 0] / 1000.0
ir = samples[:, 1]
red = samples[:, 2]
pressure = samples[:, 3]
sample_rate = (len(ts) - 1) / (ts[-1] - ts[0])
duration = (ts[-1] - ts[0]) / 60

null_samples = np.array(null_data['samples'])
null_ts = null_samples[:, 0] / 1000.0
null_pressure = null_samples[:, 3]
null_rate = (len(null_ts) - 1) / (null_ts[-1] - null_ts[0])
null_duration = (null_ts[-1] - null_ts[0]) / 60

print(f"Close nose test: {duration:.1f} min, {len(ts)} samples, {sample_rate:.1f} Hz")
print(f"Null test:       {null_duration:.1f} min, {len(null_ts)} samples, {null_rate:.1f} Hz")

# Quick look at raw signal range
print(f"\nRaw pressure range: {np.min(pressure):.1f} to {np.max(pressure):.1f} Pa")
print(f"Raw pressure std:   {np.std(pressure):.2f} Pa")
print(f"Null pressure std:  {np.std(null_pressure):.2f} Pa")
print(f"Signal amplification vs null: {np.std(pressure)/np.std(null_pressure):.1f}x")


# ═══════════════════════════════════════════════════
# RUN v4b
# ═══════════════════════════════════════════════════

bp, env, final, gated, all_peaks, gate = \
    detect_breaths_v4b(ts, pressure, sample_rate)
null_bp, null_env, null_final, null_gated, null_all, null_gate = \
    detect_breaths_v4b(null_ts, null_pressure, null_rate)

print(f"\nv4b results:")
print(f"  Close nose: {len(final)} breaths ({len(final)/duration:.1f}/min)")
print(f"  Null test:  {len(null_final)} breaths ({len(null_final)/null_duration:.1f}/min)")
selectivity = (len(final)/duration) / (len(null_final)/null_duration) \
    if len(null_final) > 0 else float('inf')
print(f"  Selectivity: {'∞' if selectivity == float('inf') else f'{selectivity:.1f}x'}")


# ═══════════════════════════════════════════════════
# PHASE DEFINITIONS
# ═══════════════════════════════════════════════════

phases = [
    (0,    60,   'Normal + Mouth Blow',  '#4CAF50'),
    (60,   105,  'Breath Hold',          '#F44336'),
    (105,  165,  'Deep Full Lungs',      '#2196F3'),
    (165,  225,  'Fast Panting',         '#FF9800'),
    (225,  300,  'Return to Normal',     '#9C27B0'),
]


# ═══════════════════════════════════════════════════
# PHASE ANALYSIS
# ═══════════════════════════════════════════════════

print(f"\n{'Phase-by-Phase Analysis':─<80}")
print(f"  {'Phase':<24} {'Time':<12} {'Breaths':<9} {'Rate':<9} "
      f"{'Depth':<11} {'Interval':<14} {'Env Mean':<10} {'Env Max'}")
print(f"  {'─'*24} {'─'*12} {'─'*9} {'─'*9} {'─'*11} {'─'*14} {'─'*10} {'─'*8}")

phase_stats = []
for t_start, t_end, name, color in phases:
    if len(final) > 0:
        mask = (ts[final] >= t_start) & (ts[final] < t_end)
        pp = final[mask]
    else:
        pp = np.array([], dtype=int)

    dur = (t_end - t_start) / 60
    n = len(pp)
    bpm = n / dur if dur > 0 else 0

    depth = np.mean(np.abs(bp[pp])) if n > 0 else 0
    max_depth = np.max(np.abs(bp[pp])) if n > 0 else 0

    if n > 1:
        intervals = np.diff(ts[pp])
        avg_int = np.mean(intervals)
        std_int = np.std(intervals)
    else:
        avg_int = std_int = 0

    env_mask = (ts >= t_start) & (ts < t_end)
    env_mean = np.mean(env[env_mask])
    env_max = np.max(env[env_mask])

    phase_stats.append({
        'name': name, 'color': color, 't_start': t_start, 't_end': t_end,
        'n': n, 'bpm': bpm, 'depth': depth, 'max_depth': max_depth,
        'avg_int': avg_int, 'std_int': std_int, 'env_mean': env_mean,
        'env_max': env_max, 'peaks': pp,
    })

    time_str = f"{t_start//60:.0f}:{t_start%60:02.0f}-{t_end//60:.0f}:{t_end%60:02.0f}"
    int_str = f"{avg_int:.2f}s ±{std_int:.2f}" if avg_int > 0 else "N/A"
    print(f"  {name:<24} {time_str:<12} {n:<9} {bpm:<9.1f} "
          f"{depth:<11.2f} {int_str:<14} {env_mean:<10.2f} {env_max:.2f}")


# ═══════════════════════════════════════════════════
# FIGURE 1: Overview — raw signal + v4b detection
# ═══════════════════════════════════════════════════

fig1 = plt.figure(figsize=(20, 22))
gs = gridspec.GridSpec(5, 1, figure=fig1, hspace=0.35,
                       height_ratios=[1.0, 1.3, 1.0, 1.0, 0.8])
fig1.suptitle('Close Nose Test (~1mm) — BMP280 Breath Detection\n'
              f'{len(final)} breaths in {duration:.1f} min ({len(final)/duration:.1f}/min) '
              f'│ Null: {len(null_final)} FP in {null_duration:.1f} min',
              fontsize=14, fontweight='bold', y=0.99)

t_min = ts / 60

# ── Panel 1: Raw pressure with phase bands ──
ax = fig1.add_subplot(gs[0])
for t_s, t_e, name, c in phases:
    ax.axvspan(t_s/60, t_e/60, alpha=0.12, color=c)
    ax.text((t_s+t_e)/2/60, np.max(pressure)*0.95, name, ha='center', va='top',
            fontsize=9, color=c, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=c))

ax.plot(t_min, pressure, color='#4527A0', linewidth=0.4, alpha=0.7)
ax.set_ylabel('Raw Pressure δ (Pa)')
ax.set_title('Raw Pressure Signal (before filtering)', fontweight='bold')
ax.set_xlim(0, duration)
ax.grid(alpha=0.2)

# ── Panel 2: Bandpass + envelope + detected breaths ──
ax = fig1.add_subplot(gs[1])
for t_s, t_e, _, c in phases:
    ax.axvspan(t_s/60, t_e/60, alpha=0.06, color=c)

ax.fill_between(t_min, bp.min()*1.3, bp.max()*1.3,
                where=gate, alpha=0.10, color='#4CAF50')
ax.plot(t_min, bp, color='#4527A0', linewidth=0.5, alpha=0.6)
ax.plot(t_min, env, color='#FF9800', linewidth=2, zorder=3, label='RMS envelope')
ax.plot(t_min, -env, color='#FF9800', linewidth=2, alpha=0.4, zorder=3)
ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1.2, zorder=4, label='Threshold (2.0 Pa)')
ax.axhline(y=-2.0, color='#F44336', linestyle='--', linewidth=1.2, alpha=0.4, zorder=4)

cluster_rej = np.setdiff1d(gated, final)
if len(cluster_rej) > 0:
    ax.plot(t_min[cluster_rej], bp[cluster_rej], 'x', color='#FFAB91',
            markersize=4, label=f'Isolated ({len(cluster_rej)})')
if len(final) > 0:
    for ps in phase_stats:
        if ps['n'] > 0:
            ax.plot(t_min[ps['peaks']], bp[ps['peaks']], 'v', color=ps['color'],
                    markersize=7, zorder=5, markeredgecolor='black', markeredgewidth=0.3)

ax.set_ylabel('Bandpass Signal (Pa)')
ax.set_title(f'v4b Detection — {len(final)} breaths, {len(cluster_rej)} rejected (isolated)',
             fontweight='bold')
ax.set_xlim(0, duration)
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.2)

# ── Panel 3: Envelope with phase labels ──
ax = fig1.add_subplot(gs[2])
ax.plot(t_min, env, color='#FF9800', linewidth=2)
ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1.5)
ax.fill_between(t_min, 0, env, where=env>2.0, alpha=0.3, color='#4CAF50', label='Above threshold')
ax.fill_between(t_min, 0, env, where=env<=2.0, alpha=0.3, color='#EF9A9A', label='Below threshold')

# Also plot null envelope mean as reference
ax.axhline(y=np.mean(null_env), color='#78909C', linestyle=':', linewidth=1.5,
           label=f'Null env mean ({np.mean(null_env):.2f} Pa)')

for t_s, t_e, name, c in phases:
    ax.axvspan(t_s/60, t_e/60, alpha=0.06, color=c)
    ax.text((t_s+t_e)/2/60, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 1 else 8,
            name, ha='center', va='top', fontsize=8, color=c, fontweight='bold')

ax.set_ylabel('RMS Envelope (Pa)')
ax.set_title('Amplitude Envelope — Breath vs Silence', fontweight='bold')
ax.set_xlim(0, duration)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

# ── Panel 4: Rolling breath rate ──
ax = fig1.add_subplot(gs[3])
ax2 = ax.twinx()

for t_s, t_e, _, c in phases:
    ax.axvspan(t_s/60, t_e/60, alpha=0.06, color=c)

if len(final) >= 3:
    peak_ts_sec = ts[final]
    rate_t, rate_v = [], []
    for t_c in np.arange(15, ts[-1]-15, 3):
        in_win = np.sum((peak_ts_sec >= t_c-15) & (peak_ts_sec < t_c+15))
        rate_t.append(t_c / 60)
        rate_v.append(in_win * 2)  # 30s window → per minute

    ax.plot(rate_t, rate_v, color='#E91E63', linewidth=2.5, label='Breath Rate')
    ax.fill_between(rate_t, 0, rate_v, alpha=0.15, color='#E91E63')
    ax.set_ylabel('Breath Rate (/min)', color='#E91E63')
    ax.tick_params(axis='y', labelcolor='#E91E63')

    # Depth on right axis
    peak_depths = np.abs(bp[final])
    peak_t_min = ts[final] / 60
    ax2.bar(peak_t_min, peak_depths, width=0.015, color='#4527A0', alpha=0.4, label='Breath Depth')
    ax2.set_ylabel('Breath Depth (Pa)', color='#4527A0')
    ax2.tick_params(axis='y', labelcolor='#4527A0')

ax.set_xlabel('Time (min)')
ax.set_title('Instantaneous Breath Rate & Depth', fontweight='bold')
ax.set_xlim(0, duration)
ax.grid(alpha=0.2)

# ── Panel 5: Comparison — envelope distributions ──
ax = fig1.add_subplot(gs[4])
bins = np.linspace(0, max(np.max(env), np.max(null_env))*1.1, 80)
ax.hist(null_env, bins=bins, color='#78909C', alpha=0.5, density=True,
        label=f'Null test (mean={np.mean(null_env):.2f} Pa)', edgecolor='white', linewidth=0.3)
ax.hist(env, bins=bins, color='#4527A0', alpha=0.5, density=True,
        label=f'Close nose (mean={np.mean(env):.2f} Pa)', edgecolor='white', linewidth=0.3)
ax.axvline(x=2.0, color='#F44336', linestyle='--', linewidth=2.5, label='Threshold (2.0 Pa)')
ax.set_xlabel('RMS Envelope (Pa)')
ax.set_ylabel('Density')
ax.set_title('Signal Separation: Close Nose vs Null Test', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

plt.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/close_nose_overview.png',
            dpi=150, bbox_inches='tight')
plt.close(fig1)
print("\nSaved: close_nose_overview.png")


# ═══════════════════════════════════════════════════
# FIGURE 2: Per-phase zooms
# ═══════════════════════════════════════════════════

fig2, axes = plt.subplots(5, 1, figsize=(20, 22), gridspec_kw={'height_ratios': [1]*5})
fig2.suptitle('Close Nose Test — Individual Phase Zooms\n'
              'Each panel shows the full phase duration with bandpass signal + envelope + detections',
              fontsize=14, fontweight='bold', y=0.99)

for i, ps in enumerate(phase_stats):
    ax = axes[i]
    t_s, t_e = ps['t_start'], ps['t_end']
    mask = (ts >= t_s) & (ts < t_e)
    t_sec = ts[mask] - t_s

    ax.plot(t_sec, bp[mask], color=ps['color'], linewidth=0.8, alpha=0.7)
    ax.plot(t_sec, env[mask], color='#FF9800', linewidth=2.5, zorder=3)
    ax.plot(t_sec, -env[mask], color='#FF9800', linewidth=2.5, alpha=0.4, zorder=3)
    ax.axhline(y=2.0, color='#F44336', linestyle='--', linewidth=1.2, zorder=4)
    ax.axhline(y=-2.0, color='#F44336', linestyle='--', linewidth=1.2, alpha=0.4, zorder=4)

    # Null envelope reference
    ax.axhline(y=np.mean(null_env), color='#78909C', linestyle=':', linewidth=1,
               label=f'Null mean ({np.mean(null_env):.2f} Pa)')

    if ps['n'] > 0:
        ax.plot(ts[ps['peaks']] - t_s, bp[ps['peaks']], 'v', color='#E91E63',
                markersize=10, zorder=5, markeredgecolor='black', markeredgewidth=0.5)

    # Stats annotation
    stats = (f"Breaths: {ps['n']}  │  Rate: {ps['bpm']:.1f}/min  │  "
             f"Avg Depth: {ps['depth']:.2f} Pa  │  Max: {ps['max_depth']:.2f} Pa  │  "
             f"Env Mean: {ps['env_mean']:.2f} Pa  │  Env Max: {ps['env_max']:.2f} Pa")
    if ps['avg_int'] > 0:
        stats += f"  │  Interval: {ps['avg_int']:.2f}s ±{ps['std_int']:.2f}"

    ax.text(0.5, 0.97, stats, transform=ax.transAxes, ha='center', va='top',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                      edgecolor=ps['color'], linewidth=2))

    time_label = f"{t_s//60:.0f}:{t_s%60:02.0f} - {t_e//60:.0f}:{t_e%60:02.0f}"
    ax.set_title(f'Phase {i+1}: {ps["name"]} ({time_label})',
                 fontweight='bold', color=ps['color'], fontsize=12)
    ax.set_xlabel('Time within phase (s)')
    ax.set_ylabel('Pressure (Pa)')
    ax.set_xlim(0, t_e - t_s)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.2)

plt.tight_layout()
fig2.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/close_nose_phases.png',
             dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved: close_nose_phases.png")


# ═══════════════════════════════════════════════════
# FIGURE 3: Breath-by-breath intervals + comparison with null
# ═══════════════════════════════════════════════════

if len(final) > 1:
    fig3, axes3 = plt.subplots(2, 2, figsize=(18, 12))
    fig3.suptitle('Close Nose Test — Signal Quality Assessment',
                  fontsize=14, fontweight='bold')

    # Top left: intervals
    ax = axes3[0, 0]
    intervals = np.diff(ts[final])
    int_times = ts[final][1:] / 60
    for ps in phase_stats:
        ax.axvspan(ps['t_start']/60, ps['t_end']/60, alpha=0.10, color=ps['color'])
    ax.plot(int_times, intervals, 'o-', color='#5C6BC0', markersize=4, linewidth=0.8, alpha=0.7)
    if len(intervals) >= 6:
        roll = np.convolve(intervals, np.ones(6)/6, mode='valid')
        ax.plot(int_times[2:2+len(roll)], roll, color='#E91E63', linewidth=3, zorder=4,
                label='6-breath rolling avg')
    ax.axhspan(2.5, 5.0, alpha=0.06, color='#4CAF50', label='Normal range')
    ax.set_ylabel('Breath Interval (s)')
    ax.set_title('Breath Intervals', fontweight='bold')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, min(np.max(intervals)*1.2, 20))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Top right: depth over time
    ax = axes3[0, 1]
    depths = np.abs(bp[final])
    dtimes = ts[final] / 60
    for ps in phase_stats:
        ax.axvspan(ps['t_start']/60, ps['t_end']/60, alpha=0.10, color=ps['color'])
    ax.bar(dtimes, depths, width=0.02, color='#4527A0', alpha=0.6)
    if len(depths) >= 6:
        roll_d = np.convolve(depths, np.ones(6)/6, mode='valid')
        ax.plot(dtimes[2:2+len(roll_d)], roll_d, color='#FF9800', linewidth=3, zorder=4,
                label='6-breath rolling avg')
    ax.set_ylabel('Breath Depth (Pa)')
    ax.set_title('Breath Depth', fontweight='bold')
    ax.set_xlim(0, duration)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Bottom left: PSD comparison
    ax = axes3[1, 0]
    f_br, pxx_br = welch(bp, fs=sample_rate, nperseg=min(1024, len(bp)))
    f_null, pxx_null = welch(null_bp, fs=null_rate, nperseg=min(1024, len(null_bp)))
    ax.semilogy(f_null, pxx_null, color='#78909C', linewidth=1.5, label='Null test')
    ax.semilogy(f_br, pxx_br, color='#4527A0', linewidth=1.5, label='Close nose')
    ax.axvspan(0.08, 0.7, alpha=0.12, color='#E91E63', label='Breath band')
    ax.set_xlim(0, 2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (Pa²/Hz)')
    ax.set_title('Power Spectral Density', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Compute SNR in breath band
    breath_mask_br = (f_br >= 0.08) & (f_br <= 0.7)
    breath_mask_null = (f_null >= 0.08) & (f_null <= 0.7)
    breath_power = np.sum(pxx_br[breath_mask_br])
    null_power = np.sum(pxx_null[breath_mask_null])
    snr = breath_power / null_power if null_power > 0 else float('inf')

    ax.text(0.95, 0.95, f'Breath band SNR: {snr:.1f}x',
            transform=ax.transAxes, ha='right', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Bottom right: phase comparison bar chart
    ax = axes3[1, 1]
    x = np.arange(len(phase_stats))
    names = [ps['name'] for ps in phase_stats]
    colors = [ps['color'] for ps in phase_stats]
    rates = [ps['bpm'] for ps in phase_stats]
    env_means = [ps['env_mean'] for ps in phase_stats]

    w = 0.35
    bars1 = ax.bar(x - w/2, rates, w, color=colors, alpha=0.8, label='Rate (/min)')
    bars2 = ax.bar(x + w/2, env_means, w, color=colors, alpha=0.4, hatch='//', label='Env Mean (Pa)')

    for bar, val in zip(bars1, rates):
        if val > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=np.mean(null_env), color='#78909C', linestyle=':', linewidth=1.5,
               label=f'Null env mean')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_title('Phase Comparison', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis='y')

    plt.tight_layout()
    fig3.savefig('/Users/mert/Developer/Hardware/BreathAnalysis/analysis/close_nose_quality.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("Saved: close_nose_quality.png")


# ═══════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════

print("\n" + "=" * 80)
print("CLOSE NOSE TEST — VERDICT")
print("=" * 80)

print(f"\n  Signal amplification: {np.std(pressure)/np.std(null_pressure):.1f}x (raw), "
      f"{np.std(bp)/np.std(null_bp):.1f}x (bandpass)")
print(f"  Envelope mean: {np.mean(env):.2f} Pa (vs null {np.mean(null_env):.2f} Pa)")
print(f"  Envelope max:  {np.max(env):.2f} Pa (vs null {np.max(null_env):.2f} Pa)")

if len(final) > 0 and len(null_final) > 0:
    sel = (len(final)/duration) / (len(null_final)/null_duration)
    print(f"  Selectivity: {sel:.1f}x")
elif len(null_final) == 0:
    print(f"  Selectivity: ∞ (zero null FP)")
else:
    print(f"  Selectivity: N/A")

print(f"\n  Detection: {len(final)} breaths in {duration:.1f} min ({len(final)/duration:.1f}/min)")
print(f"  Null FP:   {len(null_final)} in {null_duration:.1f} min ({len(null_final)/null_duration:.1f}/min)")

print(f"\n  Phase Verdicts:")
for ps in phase_stats:
    if "Normal" in ps['name'] and "Return" not in ps['name']:
        expect = "expect 14-20/min"
    elif "Hold" in ps['name']:
        expect = "expect ~0/min (silence)"
    elif "Deep" in ps['name']:
        expect = "expect <12/min, high depth"
    elif "Fast" in ps['name'] or "Pant" in ps['name']:
        expect = "expect >25/min"
    elif "Return" in ps['name']:
        expect = "expect 14-20/min"
    else:
        expect = ""

    verdict = ""
    if "Hold" in ps['name']:
        verdict = "PASS" if ps['bpm'] < 5 else "PARTIAL" if ps['bpm'] < 10 else "FAIL"
    elif "Fast" in ps['name']:
        verdict = "PASS" if ps['bpm'] > 25 else "PARTIAL" if ps['bpm'] > 18 else "FAIL"
    elif "Deep" in ps['name']:
        verdict = "PASS" if ps['bpm'] < 14 and ps['depth'] > 3 else "PARTIAL"
    elif "Normal" in ps['name']:
        verdict = "PASS" if 10 <= ps['bpm'] <= 24 else "PARTIAL"

    print(f"    {ps['name']:<24} {ps['bpm']:>5.1f}/min  depth={ps['depth']:.2f} Pa  "
          f"env={ps['env_mean']:.2f} Pa  ({expect}) [{verdict}]")

print("\n" + "=" * 80)
