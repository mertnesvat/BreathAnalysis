#!/usr/bin/env python3
"""
Quick visualization of breath analysis session data.

Usage:
    python visualize.py path/to/session.csv
    python visualize.py  # Uses most recent session
"""

import sys
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Please install dependencies: pip install matplotlib numpy")
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
                # Parse metadata
                if ":" in line:
                    key, value = line[2:].split(":", 1)
                    metadata[key.strip()] = value.strip()
            elif line.startswith("#"):
                continue
            elif line.strip():
                break

        # Reset and read CSV
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

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    # Convert timestamp to seconds
    data["time_sec"] = data["timestamp"] / 1000.0

    return data, metadata


def plot_session(data: dict, metadata: dict, filepath: Path):
    """Create multi-panel visualization of session data"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Meditation Session: {filepath.stem}", fontsize=14, fontweight="bold")

    time = data["time_sec"]
    duration_min = time[-1] / 60

    # 1. Thermistor (breath airflow)
    ax = axes[0]
    ax.plot(time, data["thermistor"], "b-", linewidth=0.5, alpha=0.8)
    ax.set_ylabel("Thermistor\n(ADC)")
    ax.set_title("Nasal Airflow Temperature (breath cycles)", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)

    # 2. Piezoelectric (chest/belly expansion)
    ax = axes[1]
    ax.plot(time, data["piezo"], "g-", linewidth=0.5, alpha=0.8)
    ax.set_ylabel("Piezo\n(ADC)")
    ax.set_title("Chest/Belly Expansion", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)

    # 3. MAX30102 IR (pulse/HRV)
    ax = axes[2]
    ax.plot(time, data["ir"], "r-", linewidth=0.5, alpha=0.8)
    ax.set_ylabel("IR\n(raw)")
    ax.set_title("Pulse Oximeter IR (for HRV extraction)", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)

    # 4. MAX30102 Red (SpO2 calculation)
    ax = axes[3]
    ax.plot(time, data["red"], "darkred", linewidth=0.5, alpha=0.8)
    ax.set_ylabel("Red\n(raw)")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Pulse Oximeter Red", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)

    # Add session info
    info_text = f"Duration: {duration_min:.1f} min | Samples: {len(time)}"
    if "Sample Rate" in metadata:
        info_text += f" | Rate: {metadata['Sample Rate']}"
    fig.text(0.99, 0.01, info_text, ha="right", va="bottom", fontsize=9, color="gray")

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save figure
    output_path = filepath.with_suffix(".png")
    plt.savefig(output_path, dpi=150)
    print(f"[Saved] {output_path}")

    plt.show()


def find_latest_session() -> Path:
    """Find the most recent session file"""
    data_dir = Path(__file__).parent.parent / "data" / "sessions"
    sessions = list(data_dir.glob("session_*.csv"))
    if not sessions:
        print("[Error] No session files found in data/sessions/")
        sys.exit(1)
    return max(sessions, key=lambda p: p.stat().st_mtime)


def main():
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        filepath = find_latest_session()
        print(f"[Using latest] {filepath}")

    if not filepath.exists():
        print(f"[Error] File not found: {filepath}")
        sys.exit(1)

    print(f"[Loading] {filepath}")
    data, metadata = load_session(filepath)

    print(f"[Data] {len(data['timestamp'])} samples over {data['time_sec'][-1]:.1f} seconds")

    plot_session(data, metadata, filepath)


if __name__ == "__main__":
    main()
