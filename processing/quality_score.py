#!/usr/bin/env python3
"""
Meditation Quality Score Calculator

Calculates a technique-aware quality score from breath and HRV data.
Different meditation techniques have different physiological signatures,
so scoring is adjusted based on the practice type.

Techniques supported:
- anapana: Focused attention on breath (expects slow, regular breathing, high HRV)
- vipassana: Body scanning (expects natural breathing, stable HRV)
- open_awareness: Choiceless awareness (expects variable breathing, high HRV)
- general: Generic meditation (balanced scoring)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path
import csv
from scipy import signal as sig


@dataclass
class SessionMetrics:
    """Container for all session metrics"""
    duration_min: float

    # Breath metrics
    breath_rate: float
    breath_count: int
    breath_interval_mean: float
    breath_interval_std: float
    breath_regularity: float  # 0-1, higher = more regular

    # HRV metrics
    heart_rate: float
    heartbeat_count: int
    rmssd: float
    sdnn: float
    pnn50: float

    # Temporal trends
    rmssd_start: float
    rmssd_end: float
    rmssd_trend: float  # positive = increasing
    hr_start: float
    hr_end: float
    hr_trend: float  # negative = decreasing (good)

    # Signal quality
    finger_contact_pct: float
    breath_signal_range: float


@dataclass
class QualityScore:
    """Meditation quality score with components"""
    total: float  # 0-100
    technique: str

    # Component scores (0-100 each)
    relaxation: float      # Based on RMSSD level
    deepening: float       # Based on RMSSD trend
    stability: float       # Based on HR stability
    breath_quality: float  # Based on breath pattern (technique-dependent)

    # Interpretation
    level: str  # "excellent", "good", "moderate", "developing"
    insights: List[str]


# Technique-specific scoring weights
TECHNIQUE_WEIGHTS = {
    "anapana": {
        "relaxation": 0.30,
        "deepening": 0.25,
        "stability": 0.15,
        "breath_quality": 0.30,  # Breath is central to anapana
    },
    "vipassana": {
        "relaxation": 0.35,
        "deepening": 0.30,
        "stability": 0.25,
        "breath_quality": 0.10,  # Breath not the focus
    },
    "open_awareness": {
        "relaxation": 0.40,
        "deepening": 0.30,
        "stability": 0.20,
        "breath_quality": 0.10,
    },
    "general": {
        "relaxation": 0.30,
        "deepening": 0.25,
        "stability": 0.25,
        "breath_quality": 0.20,
    }
}

# Reference values for scoring (based on meditation research)
REFERENCE = {
    "rmssd_excellent": 100,  # ms - very high parasympathetic
    "rmssd_good": 50,        # ms - good HRV
    "rmssd_moderate": 25,    # ms - average

    "breath_rate_anapana_optimal": 6,   # breaths/min for RSA
    "breath_rate_natural": 12,          # normal resting

    "hr_resting": 70,        # BPM baseline
}


def load_session(filepath: Path) -> tuple:
    """Load session data from CSV"""
    timestamps, therm, ir = [], [], []

    with open(filepath) as f:
        reader = csv.DictReader((row for row in f if not row.startswith('#')))
        for row in reader:
            timestamps.append(int(row['timestamp_ms']))
            therm.append(int(row['thermistor']))
            ir.append(int(row['ir']))

    return (
        np.array(timestamps) / 1000.0,
        np.array(therm, dtype=float),
        np.array(ir, dtype=float)
    )


def extract_metrics(filepath: Path) -> SessionMetrics:
    """Extract all metrics from a session file"""
    t, therm, ir = load_session(filepath)

    # Skip first 10 seconds
    mask = t >= 10
    t = t[mask] - t[mask][0]
    therm = therm[mask]
    ir = ir[mask]

    fs = len(t) / t[-1]
    duration_min = t[-1] / 60

    # === Breath Analysis ===
    b_bp, a_bp = sig.butter(2, [0.05/(fs/2), 0.5/(fs/2)], btype='band')
    breath = sig.filtfilt(b_bp, a_bp, therm - np.mean(therm))

    breath_peaks, _ = sig.find_peaks(breath, distance=int(1.5*fs),
                                      prominence=np.std(breath)*0.3)
    breath_count = len(breath_peaks)
    breath_rate = breath_count / duration_min

    if len(breath_peaks) > 1:
        breath_intervals = np.diff(t[breath_peaks])
        breath_interval_mean = np.mean(breath_intervals)
        breath_interval_std = np.std(breath_intervals)
        # Regularity: inverse of coefficient of variation
        cv = breath_interval_std / breath_interval_mean if breath_interval_mean > 0 else 1
        breath_regularity = max(0, 1 - cv)
    else:
        breath_interval_mean = 0
        breath_interval_std = 0
        breath_regularity = 0

    # === HRV Analysis ===
    b_hp, a_hp = sig.butter(2, 0.5/(fs/2), btype='high')
    ir_hp = sig.filtfilt(b_hp, a_hp, ir)
    b_bp2, a_bp2 = sig.butter(2, [0.7/(fs/2), 3.5/(fs/2)], btype='band')
    pulse = -sig.filtfilt(b_bp2, a_bp2, ir_hp)
    pulse = (pulse - np.mean(pulse)) / np.std(pulse)

    hr_peaks, _ = sig.find_peaks(pulse, distance=int(0.35*fs),
                                  prominence=0.4, height=0)
    heartbeat_count = len(hr_peaks)

    peak_times = t[hr_peaks]
    rr = np.diff(peak_times) * 1000
    valid_mask = (rr > 333) & (rr < 1500)
    valid_rr = rr[valid_mask]

    if len(valid_rr) > 5:
        heart_rate = 60000 / np.mean(valid_rr)
        sdnn = np.std(valid_rr, ddof=1)
        rmssd = np.sqrt(np.mean(np.diff(valid_rr)**2))
        pnn50 = 100 * np.sum(np.abs(np.diff(valid_rr)) > 50) / len(np.diff(valid_rr))

        # Temporal analysis
        third = len(valid_rr) // 3
        rr_start = valid_rr[:third]
        rr_end = valid_rr[2*third:]

        hr_start = 60000 / np.mean(rr_start)
        hr_end = 60000 / np.mean(rr_end)
        rmssd_start = np.sqrt(np.mean(np.diff(rr_start)**2)) if len(rr_start) > 1 else rmssd
        rmssd_end = np.sqrt(np.mean(np.diff(rr_end)**2)) if len(rr_end) > 1 else rmssd

        rmssd_trend = (rmssd_end - rmssd_start) / rmssd_start if rmssd_start > 0 else 0
        hr_trend = (hr_end - hr_start) / hr_start if hr_start > 0 else 0
    else:
        heart_rate = sdnn = rmssd = pnn50 = 0
        hr_start = hr_end = rmssd_start = rmssd_end = 0
        rmssd_trend = hr_trend = 0

    # Signal quality
    finger_contact_pct = 100 * np.sum(ir > 50000) / len(ir)
    breath_signal_range = np.max(therm) - np.min(therm)

    return SessionMetrics(
        duration_min=duration_min,
        breath_rate=breath_rate,
        breath_count=breath_count,
        breath_interval_mean=breath_interval_mean,
        breath_interval_std=breath_interval_std,
        breath_regularity=breath_regularity,
        heart_rate=heart_rate,
        heartbeat_count=heartbeat_count,
        rmssd=rmssd,
        sdnn=sdnn,
        pnn50=pnn50,
        rmssd_start=rmssd_start,
        rmssd_end=rmssd_end,
        rmssd_trend=rmssd_trend,
        hr_start=hr_start,
        hr_end=hr_end,
        hr_trend=hr_trend,
        finger_contact_pct=finger_contact_pct,
        breath_signal_range=breath_signal_range,
    )


def calculate_quality_score(metrics: SessionMetrics, technique: str = "general") -> QualityScore:
    """
    Calculate meditation quality score based on metrics and technique.

    Returns a score from 0-100 with component breakdowns.
    """
    weights = TECHNIQUE_WEIGHTS.get(technique, TECHNIQUE_WEIGHTS["general"])
    insights = []

    # === 1. Relaxation Score (RMSSD level) ===
    # Scale: 0-25ms = 0-25, 25-50ms = 25-60, 50-100ms = 60-85, 100+ = 85-100
    if metrics.rmssd >= 100:
        relaxation = 85 + min(15, (metrics.rmssd - 100) / 10)
    elif metrics.rmssd >= 50:
        relaxation = 60 + (metrics.rmssd - 50) / 2
    elif metrics.rmssd >= 25:
        relaxation = 25 + (metrics.rmssd - 25) * 1.4
    else:
        relaxation = metrics.rmssd
    relaxation = min(100, max(0, relaxation))

    if metrics.rmssd > 80:
        insights.append(f"Excellent parasympathetic activation (RMSSD: {metrics.rmssd:.0f}ms)")
    elif metrics.rmssd > 50:
        insights.append(f"Good relaxation response (RMSSD: {metrics.rmssd:.0f}ms)")

    # === 2. Deepening Score (RMSSD trend) ===
    # Positive trend = deepening, negative = not settling
    if metrics.rmssd_trend > 0.2:
        deepening = 80 + min(20, metrics.rmssd_trend * 50)
        insights.append("RMSSD increased during session - deepening meditation")
    elif metrics.rmssd_trend > 0:
        deepening = 50 + metrics.rmssd_trend * 150
    elif metrics.rmssd_trend > -0.2:
        deepening = 50 + metrics.rmssd_trend * 100  # Slight penalty
    else:
        deepening = max(20, 50 + metrics.rmssd_trend * 100)
    deepening = min(100, max(0, deepening))

    # === 3. Stability Score (HR variability over session) ===
    # Less HR jumping = more stable state
    hr_change = abs(metrics.hr_trend)
    if hr_change < 0.02:
        stability = 90 + min(10, (0.02 - hr_change) * 500)
        insights.append("Very stable heart rate throughout session")
    elif hr_change < 0.05:
        stability = 70 + (0.05 - hr_change) * 666
    elif hr_change < 0.1:
        stability = 50 + (0.1 - hr_change) * 400
    else:
        stability = max(20, 50 - hr_change * 200)
    stability = min(100, max(0, stability))

    # === 4. Breath Quality Score (technique-dependent) ===
    if technique == "anapana":
        # For anapana: slow, regular breathing is ideal
        # Optimal around 6 breaths/min, good up to 10
        if metrics.breath_rate <= 6:
            rate_score = 100
        elif metrics.breath_rate <= 10:
            rate_score = 100 - (metrics.breath_rate - 6) * 10
        elif metrics.breath_rate <= 15:
            rate_score = 60 - (metrics.breath_rate - 10) * 4
        else:
            rate_score = max(20, 40 - (metrics.breath_rate - 15) * 2)

        # Regularity bonus for anapana
        regularity_score = metrics.breath_regularity * 100
        breath_quality = rate_score * 0.6 + regularity_score * 0.4

        if metrics.breath_rate < 8:
            insights.append(f"Excellent breath control ({metrics.breath_rate:.1f}/min)")

    elif technique == "vipassana":
        # For vipassana: natural breathing is fine, focus elsewhere
        # Penalize only extreme rates
        if 8 <= metrics.breath_rate <= 16:
            breath_quality = 80
        elif 6 <= metrics.breath_rate <= 20:
            breath_quality = 60
        else:
            breath_quality = 40
        insights.append("Natural breathing pattern (appropriate for body scan)")

    else:
        # General: moderate preference for slower breathing
        if metrics.breath_rate <= 10:
            breath_quality = 80 + min(20, (10 - metrics.breath_rate) * 4)
        elif metrics.breath_rate <= 15:
            breath_quality = 60 + (15 - metrics.breath_rate) * 4
        else:
            breath_quality = max(30, 60 - (metrics.breath_rate - 15) * 3)

    breath_quality = min(100, max(0, breath_quality))

    # === Calculate Total Score ===
    total = (
        weights["relaxation"] * relaxation +
        weights["deepening"] * deepening +
        weights["stability"] * stability +
        weights["breath_quality"] * breath_quality
    )

    # Determine level
    if total >= 80:
        level = "excellent"
    elif total >= 65:
        level = "good"
    elif total >= 50:
        level = "moderate"
    else:
        level = "developing"

    # Add overall insight
    if total >= 80:
        insights.insert(0, f"Outstanding {technique} session!")
    elif total >= 65:
        insights.insert(0, f"Solid {technique} practice")

    return QualityScore(
        total=round(total, 1),
        technique=technique,
        relaxation=round(relaxation, 1),
        deepening=round(deepening, 1),
        stability=round(stability, 1),
        breath_quality=round(breath_quality, 1),
        level=level,
        insights=insights,
    )


def print_score_report(score: QualityScore, metrics: SessionMetrics):
    """Print a formatted score report"""
    print()
    print("=" * 65)
    print(f"  MEDITATION QUALITY REPORT")
    print(f"  Technique: {score.technique.upper()}")
    print("=" * 65)
    print()
    print(f"  ╔═══════════════════════════════════════════════════════════╗")
    print(f"  ║                                                           ║")
    print(f"  ║     OVERALL SCORE:  {score.total:5.1f} / 100  ({score.level.upper()})          ║")
    print(f"  ║                                                           ║")
    print(f"  ╚═══════════════════════════════════════════════════════════╝")
    print()
    print("  Component Scores:")
    print("  ─────────────────────────────────────────────────────────────")

    # Visual bars
    def bar(value, width=30):
        filled = int(value / 100 * width)
        return "█" * filled + "░" * (width - filled)

    print(f"  Relaxation:     [{bar(score.relaxation)}] {score.relaxation:5.1f}")
    print(f"  Deepening:      [{bar(score.deepening)}] {score.deepening:5.1f}")
    print(f"  Stability:      [{bar(score.stability)}] {score.stability:5.1f}")
    print(f"  Breath Quality: [{bar(score.breath_quality)}] {score.breath_quality:5.1f}")
    print()
    print("  Key Metrics:")
    print("  ─────────────────────────────────────────────────────────────")
    print(f"  Duration:     {metrics.duration_min:.1f} min")
    print(f"  Breath Rate:  {metrics.breath_rate:.1f} /min ({metrics.breath_count} breaths)")
    print(f"  Heart Rate:   {metrics.heart_rate:.1f} BPM")
    print(f"  RMSSD:        {metrics.rmssd:.1f} ms (trend: {metrics.rmssd_trend:+.1%})")
    print(f"  pNN50:        {metrics.pnn50:.1f}%")
    print()
    print("  Insights:")
    print("  ─────────────────────────────────────────────────────────────")
    for insight in score.insights:
        print(f"  • {insight}")
    print()
    print("=" * 65)


def compare_sessions(sessions: List[tuple]) -> None:
    """
    Compare multiple sessions.
    sessions: List of (filepath, technique) tuples
    """
    print()
    print("=" * 70)
    print("  SESSION COMPARISON")
    print("=" * 70)
    print()

    results = []
    for filepath, technique in sessions:
        metrics = extract_metrics(Path(filepath))
        score = calculate_quality_score(metrics, technique)
        results.append((filepath, technique, metrics, score))

    # Header
    print(f"  {'Session':<30} {'Technique':<12} {'Score':>8} {'RMSSD':>8} {'BR':>6}")
    print("  " + "─" * 66)

    for filepath, technique, metrics, score in results:
        name = Path(filepath).stem[:28]
        print(f"  {name:<30} {technique:<12} {score.total:>7.1f} {metrics.rmssd:>7.1f} {metrics.breath_rate:>5.1f}")

    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Default: analyze latest session
        data_dir = Path("data/sessions")
        sessions = list(data_dir.glob("meditation_*.csv"))
        if not sessions:
            print("No meditation sessions found")
            sys.exit(1)
        filepath = max(sessions, key=lambda p: p.stat().st_mtime)
        technique = "general"
    else:
        filepath = Path(sys.argv[1])
        technique = sys.argv[2] if len(sys.argv) > 2 else "general"

    print(f"Analyzing: {filepath.name}")
    metrics = extract_metrics(filepath)
    score = calculate_quality_score(metrics, technique)
    print_score_report(score, metrics)
