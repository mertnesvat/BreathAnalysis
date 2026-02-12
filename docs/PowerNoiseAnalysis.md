# LED Optical Crosstalk Analysis: Why HRV Failed on Battery

## The Problem

HRV worked perfectly on USB power but failed on battery power. Initial hypothesis was electrical power supply noise. **Actual root cause: optical interference from the red COB LED strip.**

## The Evidence

Two sessions recorded on **battery power only**, identical conditions except LED:

| Metric | LED ON | LED OFF |
|--------|--------|---------|
| IR range | 4,037 - 71,426 (67K swing!) | 68,240 - 71,270 (3K swing) |
| IR std | 4,835 | 848 |
| pulseStd | 3,566 | **148** |
| HR peaks found | 3 | **63** |
| Valid RR intervals | 2 | **62** |
| Avg Heart Rate | 133 BPM (wrong) | **72.1 BPM** (correct) |
| RMSSD | 0 (failed) | **74.4 ms** (healthy) |

The IR range with LED ON is **67,000 counts** — the external LED was flooding the photodiode with modulated light, creating a signal 20x larger than the cardiac waveform.

## Why It Appeared to Be a USB vs Battery Issue

On USB, the XIAO was connected to the computer, and the user held the device differently — the LED happened to be further from the finger/sensor. On battery, the portable form factor brought the LED strip closer to the MAX30102, allowing more light leakage.

The correlation with power source was coincidental. The real variable was **physical proximity** of the LED to the sensor.

## How Optical Crosstalk Works

### The MAX30102 Measurement Principle

The MAX30102 measures blood flow by:
1. Driving internal LEDs (Red at 660nm, IR at 880nm) into your fingertip
2. A photodiode measures the reflected/transmitted light
3. An 18-bit ADC digitizes the photodiode current
4. The tiny pulsatile component (~1-2% of DC) reveals each heartbeat

```
Normal operation (no external light):

    MAX30102 IR LED → )))  finger  ((( → Photodiode
                         blood flow
                         modulates
                         absorption

    DC component:  ~60,000 counts (steady)
    AC component:  ~500-1300 counts (heartbeat, ~1% of DC)
```

### What the External LED Does

The COB LED strip (2200K warm white) emits strongly in red and near-infrared — the **exact wavelengths** the MAX30102 is designed to detect (660nm red, 880nm IR).

```
With external red LED:

    MAX30102 IR LED → )))  finger  ((( → Photodiode  ← ))) External LED
                         blood flow                       (PWM modulated
                         modulates                         0-100mA at 200Hz)
                         absorption

    DC component:  ~60,000 counts (steady)
    AC cardiac:    ~500-1300 counts (heartbeat, ~1% of DC)
    AC from LED:   ~67,000 counts swing (!!! drowns everything)
```

The photodiode cannot distinguish between:
- Light from MAX30102's own LEDs (the signal we want)
- Light from the external COB LED (interference)

### Why PWM Makes It Catastrophic

The LED brightness changes via PWM — rapidly switching between full ON and full OFF. This creates a massive alternating signal:

```
External LED output (PWM, breath-reactive brightness):

Bright ┤ ██  ██  ██  ██  ██    ██  ██  ██  ██  ██  ██
       │ █ █ █ █ █ █ █ █ █ █   █ █ █ █ █ █ █ █ █ █ █ █
  Off  ┤─┘ └─┘ └─┘ └─┘ └─┘ └──┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─
       └─────────────────────────────────────────────────→ time
       ← exhale (dimmer) →    ← inhale (brighter) →

What photodiode sees: huge on/off pulses + slow breath envelope
What we need to see: tiny 1% cardiac oscillation
```

The breath-reactive brightness adds another layer: the LED intensity follows the thermistor signal, creating a slow modulation (at breathing frequency) on top of the fast PWM switching. The peak detector sees these as enormous "peaks" that dwarf the real heartbeats.

## Why the Initial Power Noise Theory Was Wrong

The original analysis focused on:
- PWM current spikes on the 3.3V rail
- LDO dropout voltage on battery
- ESR/ESL of the electrolytic capacitor

While these are real concerns in general, the experimental data disproves this as the primary cause:
- With the LED **physically disconnected** (removing both optical AND electrical effects), pulseStd dropped from 3,566 to 148
- If it were purely electrical, lowering PWM frequency from 5kHz to 200Hz should have helped significantly — but the problem persisted until the LED was removed entirely
- The IR range of 4,037-71,426 (67K swing) is far too large to be explained by power supply ripple alone — this is direct optical contamination

## Solutions

### For Prototype (Immediate)

**Option A: Turn LED off during recording (firmware)**
```cpp
void updateLED() {
    if (isRecording && !isPaused) {
        ledcWrite(PIN_LED, 0);
        return;
    }
    // ... rest of LED logic
}
```

**Option B: Physical light barrier**
Wrap opaque black heat-shrink tubing or electrical tape around the MAX30102 module to block external light.

### For PCB Design (Production)

1. **Physical separation**: Place LED on opposite side of enclosure from MAX30102. Minimum 3cm distance with an opaque barrier between them.

2. **Finger clip design**: The finger clip/sensor housing should be:
   - Made of opaque material (black plastic/silicone)
   - Completely enclosed around the finger contact area
   - No light leaks from any direction
   - Similar to commercial pulse oximeter clips

3. **LED wavelength**: Avoid warm white (2200K) or red LEDs near the sensor. If an indicator LED is needed near the sensor area, use **green or blue** — the MAX30102's photodiode has minimal sensitivity at 500nm (green) or 470nm (blue).

4. **Layout**: Even with separation, ensure no PCB traces or copper pours can act as light guides between the LED and sensor areas.

### Power Design Remains Important

Even though optical crosstalk was the primary issue, the electrical power recommendations from the original analysis are still good practice for the PCB:

- **Separate power domains** (analog LDO for MAX30102, digital LDO for ESP32+LED) — prevents any future electrical noise coupling
- **Drive LED from VBAT via MOSFET** — keeps LED current off the regulated 3.3V rail
- **Bypass capacitors** close to MAX30102 — always good practice for precision analog

## Key Lesson

When debugging sensor issues, consider **all coupling mechanisms**:
1. **Conducted** (through shared power rails) — the initial hypothesis
2. **Radiated** (EMI through the air) — less likely at these frequencies
3. **Optical** (light contamination) — the actual cause!

The MAX30102 is an optical sensor. Any light source near it that overlaps its measurement wavelengths (660nm, 880nm) will interfere. This is especially insidious because:
- The interference only appears with certain physical arrangements
- It can look like electrical noise in the data
- It correlates with power source if the form factor changes between USB and battery testing
