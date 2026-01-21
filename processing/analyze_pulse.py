#!/usr/bin/env python3
"""
Improved pulse and HRV analysis for MAX30102 data.
"""

import numpy as np
import csv
from pathlib import Path
from scipy import signal
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def load_ir_data(filepath):
    """Load IR data from session CSV"""
    timestamps = []
    ir = []

    with open(filepath) as f:
        reader = csv.DictReader((row for row in f if not row.startswith('#')))
        for row in reader:
            timestamps.append(int(row['timestamp_ms']))
            ir.append(int(row['ir']))

    return np.array(timestamps) / 1000.0, np.array(ir)


def process_ppg_signal(ir_data, sample_rate=50):
    """
    Process PPG signal to extract pulse waveform.

    The MAX30102 outputs a PPG signal where:
    - Large DC component (~200,000)
    - Small AC component from pulse (~1-2% variation)
    """
    # 1. Remove DC offset using high-pass filter
    # Cutoff at 0.5 Hz removes baseline drift but keeps pulse
    nyquist = sample_rate / 2
    high_cutoff = 0.5 / nyquist
    b_high, a_high = signal.butter(2, high_cutoff, btype='high')
    ac_signal = signal.filtfilt(b_high, a_high, ir_data)

    # 2. Low-pass filter to remove high frequency noise
    # Cutoff at 5 Hz (pulse rate max ~3 Hz, harmonics up to ~5 Hz)
    low_cutoff = 5.0 / nyquist
    b_low, a_low = signal.butter(3, low_cutoff, btype='low')
    filtered = signal.filtfilt(b_low, a_low, ac_signal)

    # 3. Invert signal (MAX30102 signal typically inverts - peaks are valleys)
    # Check if we need to invert by looking at skewness
    if np.mean(filtered) < 0:
        filtered = -filtered

    return filtered


def find_peaks_adaptive(signal_data, sample_rate=50):
    """
    Adaptive peak detection for pulse signal.
    """
    # Minimum peak distance: 0.4 seconds (150 BPM max)
    min_distance = int(0.4 * sample_rate)

    # Use adaptive prominence based on signal statistics
    signal_std = np.std(signal_data)
    min_prominence = signal_std * 0.5

    # Find peaks
    peaks, properties = signal.find_peaks(
        signal_data,
        distance=min_distance,
        prominence=min_prominence,
        height=0  # Only positive peaks (after proper processing)
    )

    return peaks, properties


def calculate_hrv(peak_indices, timestamps):
    """Calculate HRV metrics from detected peaks"""
    if len(peak_indices) < 3:
        return None

    # Get peak times
    peak_times = timestamps[peak_indices]

    # Calculate RR intervals in milliseconds
    rr_intervals = np.diff(peak_times) * 1000

    # Filter physiologically plausible intervals (40-180 BPM -> 333-1500ms)
    valid_rr = rr_intervals[(rr_intervals > 333) & (rr_intervals < 1500)]

    if len(valid_rr) < 2:
        return None

    # Time-domain HRV metrics
    mean_rr = np.mean(valid_rr)
    sdnn = np.std(valid_rr, ddof=1)

    # RMSSD
    successive_diff = np.diff(valid_rr)
    rmssd = np.sqrt(np.mean(successive_diff ** 2))

    # pNN50
    pnn50 = np.sum(np.abs(successive_diff) > 50) / len(successive_diff) * 100

    # Heart rate
    mean_hr = 60000 / mean_rr

    # Instantaneous HR
    instant_hr = 60000 / valid_rr

    return {
        'mean_hr': mean_hr,
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'pnn50': pnn50,
        'num_beats': len(valid_rr) + 1,
        'instant_hr': instant_hr,
        'rr_intervals': valid_rr,
        'valid_peak_times': peak_times[:-1][
            (rr_intervals > 333) & (rr_intervals < 1500)
        ] if len(peak_times) > 1 else np.array([])
    }


def analyze_and_plot(filepath):
    """Main analysis function"""
    print(f"\n{'='*60}")
    print("  Pulse & HRV Analysis")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading: {filepath.name}")
    timestamps, ir_data = load_ir_data(filepath)

    duration = timestamps[-1] - timestamps[0]
    sample_rate = len(ir_data) / duration

    print(f"Duration: {duration:.1f} seconds")
    print(f"Samples: {len(ir_data)}")
    print(f"Sample rate: {sample_rate:.1f} Hz")

    # Check finger contact
    finger_on = ir_data > 50000
    contact_pct = np.sum(finger_on) / len(finger_on) * 100
    print(f"Finger contact: {contact_pct:.1f}%")

    # Process signal
    print("\nProcessing PPG signal...")
    pulse_signal = process_ppg_signal(ir_data, sample_rate)

    # Find peaks
    peaks, props = find_peaks_adaptive(pulse_signal, sample_rate)
    print(f"Detected peaks: {len(peaks)}")

    # Calculate HRV
    hrv = calculate_hrv(peaks, timestamps)

    if hrv:
        print(f"\n{'─'*60}")
        print("  Heart Rate Variability Results")
        print(f"{'─'*60}")
        print(f"  Heart Rate:    {hrv['mean_hr']:.1f} BPM")
        print(f"  RR Interval:   {hrv['mean_rr']:.1f} ms")
        print(f"  SDNN:          {hrv['sdnn']:.1f} ms")
        print(f"  RMSSD:         {hrv['rmssd']:.1f} ms")
        print(f"  pNN50:         {hrv['pnn50']:.1f}%")
        print(f"  Valid Beats:   {hrv['num_beats']}")
        print(f"{'─'*60}")

        # Interpretation for meditation
        print("\n  Meditation Relevance:")
        print(f"  ├─ Higher RMSSD = more parasympathetic (rest/digest) activity")
        print(f"  ├─ During deep meditation, RMSSD typically increases")
        print(f"  └─ Your RMSSD: {hrv['rmssd']:.1f} ms", end="")
        if hrv['rmssd'] > 40:
            print(" (Good range)")
        elif hrv['rmssd'] > 25:
            print(" (Moderate)")
        else:
            print(" (Lower - could indicate stress)")

    # Create visualization
    fig, axes = plt.subplots(5, 1, figsize=(14, 14))
    fig.suptitle(f"Pulse Analysis: {filepath.stem}", fontsize=14, fontweight='bold')

    # 1. Raw IR signal
    ax = axes[0]
    ax.plot(timestamps, ir_data, 'r-', linewidth=0.5, alpha=0.7)
    ax.set_ylabel('IR Raw')
    ax.set_title('Raw MAX30102 Signal', loc='left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Processed pulse signal with peaks
    ax = axes[1]
    ax.plot(timestamps, pulse_signal, 'b-', linewidth=0.5)
    if len(peaks) > 0:
        ax.plot(timestamps[peaks], pulse_signal[peaks], 'ro', markersize=3,
                label=f'{len(peaks)} peaks detected')
    ax.set_ylabel('Filtered')
    ax.set_title('Processed Pulse Waveform', loc='left', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Zoomed view (5 seconds)
    ax = axes[2]
    zoom_start, zoom_end = 30, 35  # 5 second window
    mask = (timestamps >= zoom_start) & (timestamps <= zoom_end)
    peak_mask = (timestamps[peaks] >= zoom_start) & (timestamps[peaks] <= zoom_end)

    ax.plot(timestamps[mask], pulse_signal[mask], 'b-', linewidth=1.5)
    if np.any(peak_mask):
        ax.plot(timestamps[peaks][peak_mask], pulse_signal[peaks][peak_mask],
                'ro', markersize=8, label='Heartbeats')
    ax.set_ylabel('Filtered')
    ax.set_title(f'Zoomed View ({zoom_start}-{zoom_end}s) - Individual Heartbeats',
                 loc='left', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(zoom_start, zoom_end)

    # 4. Instantaneous heart rate
    ax = axes[3]
    if hrv and len(hrv['instant_hr']) > 0:
        ax.plot(hrv['valid_peak_times'], hrv['instant_hr'], 'g-o',
                linewidth=1, markersize=3)
        ax.axhline(hrv['mean_hr'], color='orange', linestyle='--',
                   label=f"Mean: {hrv['mean_hr']:.1f} BPM")
        ax.set_ylim(max(40, hrv['mean_hr'] - 30), min(140, hrv['mean_hr'] + 30))
        ax.legend(loc='upper right', fontsize=8)
    ax.set_ylabel('BPM')
    ax.set_title('Instantaneous Heart Rate Over Time', loc='left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 5. RR interval tachogram
    ax = axes[4]
    if hrv and len(hrv['rr_intervals']) > 0:
        ax.plot(hrv['rr_intervals'], 'b-o', linewidth=1, markersize=3)
        ax.axhline(hrv['mean_rr'], color='orange', linestyle='--',
                   label=f"Mean: {hrv['mean_rr']:.0f} ms")
        ax.set_ylim(max(400, hrv['mean_rr'] - 200), min(1200, hrv['mean_rr'] + 200))
        ax.legend(loc='upper right', fontsize=8)
    ax.set_ylabel('RR (ms)')
    ax.set_xlabel('Beat Number')
    ax.set_title('RR Interval Tachogram (Beat-to-Beat Intervals)', loc='left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = filepath.with_name(filepath.stem + '_pulse_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    return hrv


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        # Find latest session
        data_dir = Path('data/sessions')
        sessions = list(data_dir.glob('*.csv'))
        filepath = max(sessions, key=lambda p: p.stat().st_mtime)
        print(f"Using latest: {filepath.name}")

    analyze_and_plot(filepath)
