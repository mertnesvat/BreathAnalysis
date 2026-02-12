# Thermistor Replacement Research Findings

## Problem Statement

Glass bead NTC thermistors offer the fastest thermal response for breath detection but are extremely fragile. During the new casing build, the glass bead thermistor may have been damaged by wire bending (measured 9K ohm — thermistor element is OK, but ADC reads 0-18, suggesting a wiring/connection issue in the voltage divider circuit).

## Thermal Time Constant Comparison

The **thermal time constant (τ)** is the time for a thermistor to reach 63.2% of a step temperature change. For breath detection with ~4-5 second cycles, τ must be well under 2 seconds.

| Thermistor Type | τ (still air) | Durability | Signal Attenuation at 5s breath | Suitability |
|----------------|--------------|------------|--------------------------------|-------------|
| **Glass bead** (bare) | 0.5–1.5s | Very fragile | 38% loss (τ=1s) | Best for breath |
| **SMD 0402 chip** | ~3s | Robust | 75% loss | Usable with software compensation |
| **SMD 0603 chip** | ~5s | Robust | 85% loss | Marginal |
| **SMD 0805 chip** | ~10s | Robust | 97% loss — nearly flat | Not suitable |
| **Epoxy-coated bead** | 3–10s | Moderate | 75–97% loss | Marginal to poor |
| **Metal case (DO-35)** | 5–15s | Very sturdy | 85–99% loss | Not suitable |

### Signal Attenuation Math

For a sinusoidal breath signal with period T and a thermistor with time constant τ:

```
Attenuation = 1 / √(1 + (2πτ/T)²)
```

| τ | T=4s | T=5s | T=7s (slow breathing) |
|---|------|------|------|
| 0.5s | 78% | 85% | 91% |
| 1.0s | 54% | 62% | 74% |
| 3.0s | 20% | 25% | 35% |
| 5.0s | 12% | 16% | 22% |

**Interpretation:** A glass bead (τ=1s) passes 62% of a 5-second breath signal. An SMD 0402 (τ=3s) passes only 25% — a 2.5x reduction in signal-to-noise ratio.

## Recommendation: Keep Glass Bead, Fix Mounting

Glass bead remains the best sensor for breath detection. The fragility issue should be solved mechanically:

### Strain Relief Techniques

1. **Silicone wire leads** — Solder glass bead to 30AWG silicone-insulated wire. Silicone wire is flexible and absorbs bending forces before they reach the glass.

2. **Hot glue anchor** — Apply a small drop of hot glue 3-5mm from the bead to anchor the wire to the casing. Bending forces transfer to the anchor point, not the glass-wire junction.

3. **PTFE sleeve** — Slide thin PTFE tubing over the leads up to the bead. This stiffens the lead-bead junction where fractures typically occur.

4. **Kapton tape wrap** — Wrap kapton tape around the wire near the bead junction for additional strain relief.

### SMD 0402 as Backup Plan

If glass bead breakage becomes chronic:

- **Part numbers:**
  - TDK NTCG103JF103FT1 (10K, 0402)
  - Murata NCP15XH103F03RC (10K, 0402)
  - EPCOS B57230V2103F260 (10K, 0402, τ ≈ 3s)

- **Software compensation needed:**
  - Tighter bandpass filtering
  - Lower prominence threshold for peak detection
  - Adaptive thresholds based on signal amplitude
  - Sessions may still degrade after 8-10 minutes

- **Trade-off:** More robust sensor, but ~2.5x weaker breath signal requiring more aggressive signal processing and more susceptible to noise.

### Buy Glass Beads in Bulk

Glass bead 10K NTC thermistors cost ~$0.30-0.50 each. Keep 10+ spares on hand.

## Session Data Evidence

### 15-Minute Session (old casing, glass bead working)

The thermistor signal degraded from std=25.8 (minute 0) to std=2.5 (minute 14) — a 10x reduction over 15 minutes. Analysis showed this was partially **physical sensor drift** (the sensor moved away from the nostril), not purely thermal equilibrium:

- Baseline drop paused at minutes 4-7 (inconsistent with pure thermal drift)
- HR spiked during the pause period (user noticed sensor moving, got anxious)
- Breath amplitude dropped faster than thermal models predict

### 1-Minute Session (new casing, thermistor wiring issue)

- Thermistor ADC readings: 0-18 (should be 1500-1800)
- Thermistor measures 9K ohm — element is fine
- **Diagnosis:** Voltage divider pull-up resistor disconnected, or signal wire to GPIO1 broken during bending
- **Fix:** Check continuity from 3.3V through fixed resistor to ADC pin

## Sources

- [Vishay NTHS Series Datasheet](https://www.vishay.com/docs/33008/nths.pdf)
- [Vishay NTCS0402E3 Datasheet](https://www.vishay.com/docs/29003/ntcs0402e3t.pdf)
- [TDK EPCOS 0402 NTC](https://www.tdk-electronics.tdk.com/inf/50/db/ntc/NTC_SMD_Standard_series_0402.pdf)
- [Ametherm: Thermal Time Constant Study](https://www.ametherm.com/blog/thermistors/thermal-time-constant-ntc-thermistors/)
