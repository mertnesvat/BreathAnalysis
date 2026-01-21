#!/usr/bin/env python3
"""
Analyze breath analysis session data.
Extracts pulse waveform, heart rate, and HRV metrics from MAX30102 data.
"""

import sys
import csv
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.ndimage import uniform_filter1d

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install matplotlib: pip install matplotlib")
    sys.exit(1)


def load_session(filepath: Path) -> dict:
    """Load session CSV, skipping comment lines"""
    data = {
        "timestamp": [],
        "thermistor": [],
        "piezo": [],
        "ir": [],
        "red": []
    }
    metadata = {}

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("# "):
                if ":" in line:
                    key, value = line[2:].split(":", 1)
                    metadata[key.strip()] = value.strip()
            elif line.startswith("#"):
                continue
            elif line.strip():
                break

        f.seek(0)
        reader = csv.DictReader(
            (row for row in f if not row.startswith("#")),
        )
        for row in reader:
            data["timestamp"].append(int(row["timestamp_ms"]))
            data["thermistor"].append(int(row["thermistor"]))
            data["piezo"].append(int(row["piezo"]))
            data["ir"].append(int(row["ir"]))
            data["red"].append(int(row["red"]))

    for key in data:
        data[key] = np.array(data[key])

    data["time_sec"] = data["timestamp"] / 1000.0

    return data, metadata


def extract_pulse_waveform(ir_data: np.ndarray, sample_rate: float = 50.0) -> np.ndarray:
    """
    Extract pulse waveform from raw IR data using bandpass filter.
    Heart rate typically 0.5-3 Hz (30-180 BPM)
    """
    # Remove DC component and very low frequencies
    # Bandpass filter: 0.5 Hz to 4 Hz
    nyquist = sample_rate / 2
    low = 0.5 / nyquist
    high = 4.0 / nyquist

    # Ensure high doesn't exceed 1.0
    high = min(high, 0.99)

    b, a = signal.butter(3, [low, high], btype='band')

    # Apply filter
    filtered = signal.filtfilt(b, a, ir_data)

    return filtered


def detect_peaks(waveform: np.ndarray, sample_rate: float = 50.0) -> np.ndarray:
    """Detect peaks in the pulse waveform"""
    # Minimum distance between peaks: 0.3 seconds (200 BPM max)
    min_distance = int(0.3 * sample_rate)

    # Find peaks
    peaks, properties = signal.find_peaks(
        waveform,
        distance=min_distance,
        prominence=np.std(waveform) * 0.3  # Adaptive threshold
    )

    return peaks


def calculate_hrv_metrics(peak_times: np.ndarray) -> dict:
    """Calculate HRV metrics from peak times (in seconds)"""
    if len(peak_times) < 3:
        return {}

    # RR intervals (time between beats) in milliseconds
    rr_intervals = np.diff(peak_times) * 1000

    # Filter out unrealistic intervals (< 300ms or > 2000ms)
    rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]

    if len(rr_intervals) < 2:
        return {}

    # Time-domain metrics
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)  # Standard deviation of NN intervals

    # RMSSD: Root mean square of successive differences
    successive_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))

    # pNN50: Percentage of successive RR intervals that differ by more than 50ms
    pnn50 = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100

    # Heart rate
    mean_hr = 60000 / mean_rr  # BPM

    return {
        "mean_hr_bpm": mean_hr,
        "mean_rr_ms": mean_rr,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50_percent": pnn50,
        "total_beats": len(rr_intervals) + 1
    }


def analyze_session(filepath: Path):
    """Main analysis function"""
    print(f"\nLoading: {filepath}")
    data, metadata = load_session(filepath)

    # Check for valid data
    ir = data["ir"]
    time_sec = data["time_sec"]

    # Calculate actual sample rate
    duration = time_sec[-1] - time_sec[0]
    actual_sample_rate = len(ir) / duration
    print(f"Duration: {duration:.1f} seconds")
    print(f"Samples: {len(ir)}")
    print(f"Actual sample rate: {actual_sample_rate:.1f} Hz")

    # Check if finger was on sensor (IR > 50000)
    finger_on = ir > 50000
    finger_on_percent = np.sum(finger_on) / len(finger_on) * 100
    print(f"Finger contact: {finger_on_percent:.1f}% of session")

    if finger_on_percent < 50:
        print("WARNING: Low finger contact - results may be unreliable")

    # Extract pulse waveform
    print("\nExtracting pulse waveform...")
    pulse_waveform = extract_pulse_waveform(ir, actual_sample_rate)

    # Detect peaks
    peaks = detect_peaks(pulse_waveform, actual_sample_rate)
    peak_times = time_sec[peaks]
    print(f"Detected {len(peaks)} heartbeats")

    # Calculate HRV metrics
    print("\nCalculating HRV metrics...")
    hrv = calculate_hrv_metrics(peak_times)

    if hrv:
        print(f"\n{'='*50}")
        print("  HRV Analysis Results")
        print(f"{'='*50}")
        print(f"  Mean Heart Rate:  {hrv['mean_hr_bpm']:.1f} BPM")
        print(f"  Mean RR Interval: {hrv['mean_rr_ms']:.1f} ms")
        print(f"  SDNN:             {hrv['sdnn_ms']:.1f} ms")
        print(f"  RMSSD:            {hrv['rmssd_ms']:.1f} ms")
        print(f"  pNN50:            {hrv['pnn50_percent']:.1f}%")
        print(f"  Total Beats:      {hrv['total_beats']}")
        print(f"{'='*50}")

        # Interpretation
        print("\n  Interpretation:")
        if hrv['rmssd_ms'] > 40:
            print("  - RMSSD > 40ms: Good parasympathetic activity")
        elif hrv['rmssd_ms'] > 20:
            print("  - RMSSD 20-40ms: Moderate HRV")
        else:
            print("  - RMSSD < 20ms: Low HRV (may indicate stress)")

    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f"Session Analysis: {filepath.stem}", fontsize=14, fontweight="bold")

    # 1. Raw IR signal
    ax = axes[0]
    ax.plot(time_sec, ir, 'r-', linewidth=0.5, alpha=0.8)
    ax.set_ylabel("IR Raw")
    ax.set_title("Raw MAX30102 IR Signal", loc="left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_sec[0], time_sec[-1])

    # 2. Filtered pulse waveform with peaks
    ax = axes[1]
    ax.plot(time_sec, pulse_waveform, 'b-', linewidth=0.8)
    ax.plot(peak_times, pulse_waveform[peaks], 'ro', markersize=4, label=f'{len(peaks)} beats')
    ax.set_ylabel("Filtered")
    ax.set_title("Pulse Waveform (Bandpass Filtered) with Detected Peaks", loc="left")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_sec[0], time_sec[-1])

    # 3. Zoomed view (10 seconds)
    ax = axes[2]
    zoom_start = 10  # Start at 10 seconds
    zoom_end = 20    # End at 20 seconds
    zoom_mask = (time_sec >= zoom_start) & (time_sec <= zoom_end)
    zoom_peaks_mask = (peak_times >= zoom_start) & (peak_times <= zoom_end)

    ax.plot(time_sec[zoom_mask], pulse_waveform[zoom_mask], 'b-', linewidth=1.5)
    ax.plot(peak_times[zoom_peaks_mask],
            pulse_waveform[peaks][zoom_peaks_mask[:len(peaks)] if len(zoom_peaks_mask) >= len(peaks) else zoom_peaks_mask],
            'ro', markersize=8)
    ax.set_ylabel("Filtered")
    ax.set_title(f"Zoomed View ({zoom_start}-{zoom_end}s) - Individual Heartbeats", loc="left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(zoom_start, zoom_end)

    # 4. Heart rate over time
    ax = axes[3]
    if len(peak_times) > 1:
        rr_intervals = np.diff(peak_times)
        instant_hr = 60 / rr_intervals  # BPM
        hr_times = peak_times[1:]  # Time of each HR measurement

        # Filter outliers
        valid = (instant_hr > 40) & (instant_hr < 180)

        ax.plot(hr_times[valid], instant_hr[valid], 'g-', linewidth=1, marker='o', markersize=3)
        ax.axhline(y=hrv.get('mean_hr_bpm', 0), color='orange', linestyle='--',
                   label=f"Mean: {hrv.get('mean_hr_bpm', 0):.1f} BPM")
        ax.set_ylabel("BPM")
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Instantaneous Heart Rate", loc="left")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_sec[0], time_sec[-1])
        ax.set_ylim(40, 120)

    plt.tight_layout()

    # Save figure
    output_path = filepath.with_name(filepath.stem + "_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")

    plt.show()

    return hrv


def find_latest_session() -> Path:
    """Find the most recent session file"""
    data_dir = Path(__file__).parent.parent / "data" / "sessions"
    sessions = list(data_dir.glob("*.csv"))
    if not sessions:
        print("No session files found in data/sessions/")
        sys.exit(1)
    return max(sessions, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        filepath = find_latest_session()
        print(f"Using latest session: {filepath.name}")

    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    analyze_session(filepath)
