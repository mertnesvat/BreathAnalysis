# Coherent Breathing HRV Experiment — March 2026

## What This Is

A personal n=1 experiment to test whether a 2–4 week daily coherent breathing practice
(6-second inhale / 6-second exhale) produces a measurable increase in overnight HRV,
using 84 days of Garmin baseline data as the reference.

---

## The Science

### Why 5 breaths/min?

At 5 br/min (one breath every 12 seconds), your breathing frequency aligns with the
natural resonance frequency of the cardiovascular system's baroreflex loop — roughly
0.1 Hz. The baroreflex is the feedback system that constantly adjusts heart rate to
maintain blood pressure. When you breathe at its natural oscillation period, you drive
it into resonance, producing the largest possible heart-rate swing per breath.

This is called **resonance breathing** or **coherent breathing**. The 6-6 pattern
(6s inhale, 6s exhale) achieves exactly 5 br/min.

### What actually changes in your body during a session

Your heart rate rises during the inhale (vagus nerve briefly inhibited by lung stretch)
and falls during the exhale (vagus nerve reactivated). At normal breathing this swing is
small (~5–8 BPM). At 5 br/min, the swing expands to ~15–25 BPM because the heart has
12 full seconds to complete each phase. This phenomenon is **Respiratory Sinus Arrhythmia
(RSA)** — it is mediated entirely by the vagus nerve, the main channel of parasympathetic
(rest-and-digest) activity.

### Why session RMSSD ≠ your real HRV

The session recording from the BreathMonitor device showed RMSSD = 129.7 ms during the
6-6 practice. Your Garmin baseline is 49.2 ms (measured overnight at natural breathing).
These numbers are not comparable. At 5 br/min, each 12-second breath cycle drives a
large HR swing that mechanically inflates RMSSD — even if your autonomic state hadn't
changed at all. Think of it as measuring how hard a spring bounces when you push it at
its resonant frequency vs. at random. The spring's stiffness hasn't changed; you've just
found the right push.

The **actual long-term benefits** accumulate between sessions, not during them:

1. **Baroreflex sensitivity improves** — your cardiovascular system becomes more
   responsive to the vagal signal even at rest.
2. **Resting parasympathetic tone increases** — this shows up as a higher overnight
   RMSSD on your Garmin after weeks of practice.
3. **RSA at normal breathing grows** — the HR swing during ordinary breathing
   gradually increases, visible in morning HRV trends.

---

## Your Baseline

**Source:** Garmin Forerunner overnight HRV (RMSSD), measured during sleep.
**Period:** Dec 8, 2025 → Mar 1, 2026 (84 days).

| Metric | Value |
|--------|-------|
| Mean | **49.2 ms** |
| Standard deviation | **4.1 ms** |
| Normal range (±1 SD) | **45–53 ms** |
| Day-to-day average swing | **4.4 ms** |
| Coefficient of variation | **8.3%** |

Monthly breakdown shows a very stable baseline:

| Month | Mean | SD |
|-------|------|----|
| Dec 2025 | 49.2 ms | 3.7 ms |
| Jan 2026 | 48.4 ms | 3.1 ms |
| Feb 2026 | 50.0 ms | 5.3 ms |

February has two outliers (32 ms and 62 ms), almost certainly confounded days
(illness, alcohol, very late night). The underlying baseline is stable.

See `hrv_baseline_analysis.png` for the full timeline and distribution.

---

## What Change to Expect

The noise floor of your measurement is ±4.1 ms (daily SD). To declare a change
real rather than random noise, the 7-day rolling mean at the end of the experiment
needs to move by:

| Practice length | Min detectable shift (80% power, p < 0.05) |
|-----------------|---------------------------------------------|
| 7 days | > 3.8 ms |
| 10 days | > 3.2 ms |
| 14 days | > 2.7 ms |

What the research on resonance breathing actually shows:

| Duration | Typical RMSSD change |
|----------|----------------------|
| 1 week | +1–3 ms (likely within noise) |
| 2 weeks | +3–6 ms (borderline detectable) |
| 4 weeks | +5–12 ms (reliably detectable) |

**Realistic expectation for 2 weeks:** If the 7-day rolling mean on Day 14 is at or
above ~52–54 ms, that is a meaningful signal. A single day reading of 55 ms is not —
that's within your normal range.

---

## How to Evaluate Results

**Primary metric:** 7-day rolling mean HRV at the end of Week 2 (and Week 4 if you
continue), compared to your 49.2 ms baseline.

**Threshold for a positive result:**
- Suggestive: 7-day mean ≥ 52 ms (+3 ms, just above noise floor)
- Clear: 7-day mean ≥ 54 ms (+5 ms, outside your normal ±1 SD band)

**Track confounders daily** (a one-line note is enough):
- Sleep duration (roughly)
- Alcohol the evening before
- Unusual stress or illness
- Whether you did the practice

Any confounded day should be noted and treated with caution in the comparison.

---

## What to Do After the Experiment

**If HRV increased:** Continue the practice and extend to 4–8 weeks for a stronger
signal. Consider building toward two sessions/day or longer sessions (15–20 min) if
the 10-minute format feels easy.

**If no change after 2 weeks:** That is not failure. Most studies find reliable effects
at 4+ weeks. Extend to 4 weeks before drawing conclusions. Also check adherence —
fewer than 10 out of 14 practice days weakens the experiment significantly.

**Either way:** The session RMSSD data from the BreathMonitor provides a secondary
dataset. If session quality (measured as coherence, peak amplitude, or breath regularity)
improves over the experiment, that suggests the technique is being executed better
even if the overnight baseline hasn't shifted yet.
