# The NoseHub: A Multi-Modal Nasal Breath & HRV Sensor

*A creative custom solution that moves everything to the nose*

## The Problem (Why Single-Thermistor Fails)

Three compounding issues make nasal thermistor breath detection unreliable for meditation sessions >5 minutes:

| Problem | What Happens | Data Evidence |
|---------|-------------|---------------|
| **Thermal drift** | Thermistor bead equilibrates to local temperature, shrinking inhale/exhale delta | Signal std drops 25 → 2.5 over 15 minutes |
| **Alignment sensitivity** | 3-5mm position shift moves bead out of direct airflow | HR spike at minute 4 when user noticed sensor moving |
| **Nasal cycle** | Every 2-6 hours, one nostril becomes dominant while the other mostly closes | Up to 80% of airflow through one side — a single sensor on the wrong side is blind |

No single fix addresses all three. But combining multiple sensing principles does.

## The Concept: NoseHub

A lightweight nose-bridge clip (~8g) that consolidates ALL sensing onto the face. Nothing on the finger. Nothing on the chest. One device, one location.

```
                          ┌─────────┐
                          │  PCB    │  ← XIAO ESP32S3 + SDP32 + battery
                          │ (behind │     (sits behind the head or on
                          │  head)  │      a headband — wired to clip)
                          └────┬────┘
                               │ thin 4-wire cable
                               │
                    ┌──────────┴──────────┐
                    │    NOSE BRIDGE CLIP  │  ← soft silicone, ~30mm wide
                    │  ┌────────────────┐  │
                    │  │   MAX30102     │  │  ← PPG on nose bridge skin
                    │  │  (nose pad)    │  │     angular artery → HRV + breath
                    │  └────────────────┘  │
                    │  [REF thermistor]     │  ← reference bead (skin temp, not airflow)
                    └──┬──────────────┬────┘
                       │              │
                  ┌────┴────┐   ┌────┴────┐
                  │  Left   │   │  Right  │   ← soft silicone prongs
                  │  Prong  │   │  Prong  │      3-5mm into nostril opening
                  │         │   │         │
                  │ •therm  │   │ •therm  │   ← glass bead thermistor at tip
                  │ ○tube   │   │ ○tube   │   ← pressure tube lumen (to SDP32)
                  └─────────┘   └─────────┘
                       │              │
                       └──────┬───────┘
                              │
                         Y-connector → thin silicone tube → SDP32 (Port A)
                                                            Port B = ambient
```

### What Each Sensor Measures

| Sensor | Location | Signal | Frequency | Addresses |
|--------|----------|--------|-----------|-----------|
| Left thermistor | Inside left nostril prong tip | Temperature oscillation from airflow | 0.1-0.5 Hz | Nasal cycle (covers left) |
| Right thermistor | Inside right nostril prong tip | Temperature oscillation from airflow | 0.1-0.5 Hz | Nasal cycle (covers right) |
| Reference thermistor | Nose bridge skin (no airflow) | Skin baseline temperature | DC / very slow drift | **Thermal drift** (cancellation) |
| SDP32 diff. pressure | Y-tube from both prongs | Pressure from airflow (both nostrils combined) | 0.1-0.5 Hz | **Alignment** (cannula-style, always in flow) |
| MAX30102 PPG | Nose bridge pad (angular artery) | Blood volume changes | 0.1-0.5 Hz (breath) + 0.8-2 Hz (cardiac) | **HRV** + breath confirmation |

**Five independent breath-correlated signals + HRV from one device.**

## Why This Is Better Than Anything Out There

### vs. Sleep Lab Nasal Cannula
Sleep labs use cannula + pressure sensor — they get airflow but no HRV. You have to add a separate finger pulse oximeter. The NoseHub gets both from one device on the nose.

### vs. MORFEA (Nasal Septum Device)
MORFEA sticks to the septum with adhesive. The NoseHub clips on — no adhesive, no skin irritation, easy to remove. MORFEA has PPG + accelerometer. NoseHub adds thermistors + differential pressure for much richer breath waveform data.

### vs. Consumer Devices (Spire, Prana)
These measure chest expansion — they get breath rate but lose the detailed nasal airflow waveform (depth, asymmetry, pauses, transitions). You also can't detect which nostril is dominant, which is relevant to yoga/pranayama practitioners.

### vs. Current BreathMonitor v1
- **Thermal drift**: Eliminated by reference thermistor subtraction
- **Alignment**: Eliminated by prong-in-nostril design
- **Nasal cycle**: Covered by dual thermistors
- **Finger inconvenience**: Eliminated — MAX30102 moves to nose
- **Optical crosstalk**: Eliminated — LED is nowhere near the nose

## The Physics: Why Each Part Works

### 1. Reference Thermistor Drift Cancellation

This is the same principle behind a **Wheatstone bridge** or **differential thermocouple**. Two identical thermistors at the same ambient temperature will drift together. If one is in the airflow and the other isn't, the difference signal isolates only the breath component.

```
Signal_breath = Thermistor_nostril - Thermistor_reference

                         Thermal drift
                         (affects both equally)
                              │
    Nostril thermistor:  ─────┼───╲╱╲╱╲╱──  (drift + breath oscillation)
    Reference thermistor: ────┼────────────  (drift only, no airflow)
                              │
    Difference:          ─────┼───╲╱╲╱╲╱──  (breath only, drift cancelled!)
```

**Key requirement**: The reference thermistor must be at approximately the same ambient temperature but NOT in the airflow path. The nose bridge skin is ideal — same body region, thermally coupled to the same tissue, but above the nostrils where airflow doesn't reach.

**How much drift does this cancel?** Consider our 15-minute session where baseline dropped steadily. Both thermistors would see the same environmental warming (body heat radiating upward, ambient temperature changes). The differential would cancel this common-mode drift, leaving only the true breath-induced temperature oscillation.

**Quantitative estimate**: If thermal drift causes both thermistors to warm by 0.5C over 15 minutes, both ADC values drift by ~200 counts. The difference stays near zero. The breath signal (0.1-0.3C swing at the nostril, absent at the bridge) remains fully visible. This could extend usable signal from 5 minutes to 30+ minutes.

### 2. Why Prongs Fix Alignment

Current setup: thermistor bead dangles near the nostril opening on a wire. Movement of the head, jaw, or device shifts it out of the airflow cone.

```
CURRENT — position-sensitive:

    Nostril        Thermistor bead
    ┌──┐           •  ← anywhere in this zone
    │░░│          /     (sensitive to 3-5mm shift)
    │░░│  airflow/
    │░░│────────►
    │░░│
    └──┘

NOSEHUB — position-locked:

    Nostril    Silicone prong
    ┌──┐     ┌──────┐
    │░░├─────┤ •    │  ← thermistor INSIDE the prong tip
    │░░│ air │ bead │     always in direct airflow path
    │░░│ ══► │      │     can't shift without moving the whole clip
    │░░├─────┤      │
    └──┘     └──────┘
```

The prong mechanically locks the thermistor in the center of the airflow. The only way to lose alignment is to physically remove the device. This converts a continuous alignment problem into a binary one (on/off).

**Thermal insulation bonus**: The silicone prong body acts as thermal insulator around the bead. Only the exposed tip contacts the airflow directly. Body heat from surrounding nasal tissue conducts through silicone slowly (thermal conductivity of silicone: 0.2 W/mK vs air: 0.025 W/mK). This partially shields the bead from the conductive warming that causes equilibration. Not a complete fix, but it helps — and the reference subtraction handles the rest.

### 3. Dual Nostrils and the Nasal Cycle

The nasal cycle alternates dominant airflow between nostrils every 2-6 hours, controlled by the autonomic nervous system (specifically the stellate ganglion). During meditation, this cycle can shift — some practitioners report it evening out.

```
Typical nasal cycle during a 30-min session:

Left nostril:   ████████████████░░░░░░░░░░░░░░░░  (dominant first half)
Right nostril:  ░░░░░░░░░░░░░░░░████████████████  (dominant second half)

Single sensor on left:
Signal:         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░  (signal dies mid-session!)

Dual sensor (pick dominant):
Signal:         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (continuous coverage)
```

**Algorithm**: At each sample, compare the AC amplitude (rolling 5-second std) of left vs right thermistor. Use the one with higher amplitude as the primary breath signal. Or sum them — the dominant side contributes more naturally. Either way, you never lose the signal to nasal cycle switching.

**Bonus data**: The ratio of left/right airflow is itself meaningful. In yoga/pranayama traditions, left-dominant breathing (ida nadi) is associated with parasympathetic activation, right-dominant (pingala nadi) with sympathetic. Whether or not you believe the tradition, the airflow ratio correlates with autonomic balance — potentially a useful meditation quality metric.

### 4. Pressure Sensing as the Alignment-Proof Backbone

Even with prongs, the thermistors are sensitive to how hard the clip presses and exact positioning. The SDP32 differential pressure through cannula-style tubing is fundamentally alignment-insensitive because it measures the bulk pressure change inside a sealed tube.

```
                  Left prong          Right prong
                  ┌────────┐          ┌────────┐
                  │ ○ tube │          │ ○ tube │
                  └───┬────┘          └───┬────┘
                      │                   │
                      └─────────┬─────────┘
                           Y-connector
                                │
                          thin silicone tube
                                │
                          ┌─────┴─────┐
                          │  SDP32    │
                          │  Port A   │ ← sees combined nasal pressure
                          │  Port B   │ ← open to ambient
                          └───────────┘
```

The pressure signal is:
- **Zero drift** (returns to exactly zero when no airflow)
- **Bidirectional** (positive = exhale, negative = inhale)
- **Quantitative** (proportional to flow² via Bernoulli)
- **Fast** (speed of sound in air, effectively instantaneous)

This acts as the reliable backbone signal. The thermistors add sensitivity for very shallow breaths where the pressure delta is tiny (thermistors respond to ANY temperature change, even from the faintest whisper of air). The two modalities complement each other:

| Breath Depth | Pressure Signal | Thermistor Signal |
|-------------|----------------|-------------------|
| Deep breath | Strong | Strong |
| Normal breath | Moderate | Moderate |
| Shallow meditation breath | Weak | Moderate (still detectable) |
| Breath pause | Zero | Slowly decays to reference |

### 5. MAX30102 on the Nose: Why It Works Here

The angular artery runs along the side of the nose, and the lateral nasal artery supplies the nose bridge area. These are branches of the facial artery — robust, superficial vessels ideal for PPG.

**MORFEA proved this**: Their study placed a MAX30102 on the nasal septum and got clean PPG signals with 90% sensitivity for apnea detection. The nose bridge is even more accessible.

**Two signals from one PPG**:

The raw IR signal from the MAX30102 on the nose contains two frequency components:

```
Raw MAX30102 IR signal from nose:

    ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮   ← cardiac pulse (1 Hz)
───╯   ╰─╯   ╰─╯   ╰─╯   ╰─╯   ╰─╯   ╰─╯   ╰─╯   ╰───
            ╲                                    ╱
             ╲──────────────────────────────────╱            ← respiratory modulation
                    (slow breath envelope)                      (0.2 Hz)

After bandpass filtering:

    Low-pass (< 0.5 Hz):  Breath signal  → rate, depth, regularity
    Bandpass (0.5-3 Hz):   Cardiac signal → HR, HRV (RR intervals)
```

Respiratory sinus arrhythmia (RSA) also modulates the cardiac rhythm — heart rate naturally increases during inhale and decreases during exhale. This is ANOTHER breath signal embedded in the HRV data. You already calculate this in your Cloud Function; now it would come from a cleaner source (nose PPG vs finger PPG with optical crosstalk issues).

**Why nose PPG is actually better than finger for this project**:
- No finger clip to forget or bump
- Closer to the brain (less pulse transit time delay)
- Less motion artifact during meditation (head is still, hands might shift)
- Optically isolated from the LED (device is behind the head, not near the finger)
- One location for everything = simpler form factor

## Hardware Design

### Bill of Materials

| Component | Part Number | Interface | Cost | Notes |
|-----------|-------------|-----------|------|-------|
| MCU | XIAO ESP32S3 (existing) | - | $7 | Same board, just rewired |
| PPG sensor | MAX30102 (existing) | I2C (0x57) | $3 | Move from finger to nose pad |
| Diff pressure | Sensirion SDP32-125PA | I2C (0x25) | $28 | +-125 Pa, 0.002 Pa resolution |
| Left thermistor | NTC 10K glass bead | ADC (GPIO1) | $0.40 | Embedded in left prong |
| Right thermistor | NTC 10K glass bead | ADC (GPIO2) | $0.40 | Embedded in right prong |
| Reference thermistor | NTC 10K glass bead | ADC (GPIO7) | $0.40 | On nose bridge, no airflow |
| Nasal prongs | Custom silicone molded | Mechanical | ~$5 | Or modified cannula prongs |
| Bridge clip | 3D printed + silicone pad | Mechanical | ~$2 | Spring clip or friction fit |
| Tubing | Silicone 2mm ID, ~20cm | Mechanical | $2 | From prongs to SDP32 |
| Y-connector | Medical grade, 2mm barb | Mechanical | $1 | Merges left + right tubes |
| Battery | Existing 18650 | Existing | - | Already have this |
| **Total new parts** | | | **~$37** | MAX30102 and thermistors reused |

### Pin Allocation (XIAO ESP32S3)

| Pin | Function | Signal |
|-----|----------|--------|
| GPIO1 (A0) | ADC | Left nostril thermistor (existing pin, repurposed) |
| GPIO2 (A1) | ADC | Right nostril thermistor (new) |
| GPIO7 (A3) | ADC | Reference thermistor (new) |
| GPIO5 (D4) | I2C SDA | MAX30102 (0x57) + SDP32 (0x25) — shared bus |
| GPIO6 (D5) | I2C SCL | MAX30102 (0x57) + SDP32 (0x25) — shared bus |
| GPIO3 (D2) | PWM | LED (existing) |
| GPIO4 (D3) | Digital | Power button (existing) |

**I2C bus**: MAX30102 (0x57) and SDP32 (0x25) have different addresses — they coexist on the same bus with no conflicts. Both support 400 kHz fast mode.

### Nose Clip Construction

```
Cross-section of the nose bridge clip:

        ┌──────── Spring steel or flexible plastic core ────────┐
        │                                                        │
    ┌───┴───┐                                              ┌────┴────┐
    │Silicone│          Nose bridge skin                   │Silicone │
    │ pad L  │══════════════════════════════════════════════│ pad R   │
    │MAX30102│          ↑ REF thermistor                   │(cushion)│
    │(under) │          (bonded to bridge skin side)       │         │
    └───┬───┘                                              └────┬────┘
        │                                                        │
        │              ┌────┐     ┌────┐                        │
        └──────────────┤Left├─────┤Right├───────────────────────┘
                       │prng│     │prng │
                       │  • │     │ •   │  ← thermistor beads at tips
                       │  ○ │     │ ○   │  ← tube lumens
                       └────┘     └────┘

                       ↑ These sit ~3-5mm inside nostril openings
```

**Prong construction detail**:
```
Single prong cross-section:

    ┌─────────────────────┐
    │     Silicone body    │  ← medical-grade, Shore 20A (very soft)
    │                     │
    │   ┌───┐   ┌─────┐  │
    │   │ • │   │  ○  │  │
    │   │bead│   │tube │  │  ← 1mm ID silicone tube (to SDP32)
    │   │wire│   │lumen│  │
    │   └───┘   └─────┘  │
    │                     │
    └─────────────────────┘

    Outer diameter: ~5-6mm (similar to standard nasal cannula prong)
    Length: ~8mm (only 3-5mm inserts into nostril)

    Thermistor bead: exposed at the tip (direct airflow contact)
    Wire leads: 30AWG silicone-insulated, run through prong body
    Tube: exits the back of the prong, joins Y-connector
```

**Silicone molding**: Create a simple 2-part mold from 3D-printed shells. Pour medical silicone (Smooth-On Dragon Skin 10, Shore 10A) around the pre-positioned thermistor and tube. Cure 4 hours at room temperature. This is a well-documented DIY process.

## Firmware Architecture

### Sensor Packet (Extended)

Current 14-byte packet:
```c
struct SensorPacket {          // 14 bytes — CURRENT
    uint32_t timestamp_ms;     // 4
    uint16_t thermistor;       // 2
    uint32_t ir_value;         // 4
    uint32_t red_value;        // 4
};
```

New 24-byte packet:
```c
struct SensorPacketV2 {        // 24 bytes — NOSEHUB
    uint32_t timestamp_ms;     // 4  — session time
    uint16_t therm_left;       // 2  — left nostril thermistor ADC
    uint16_t therm_right;      // 2  — right nostril thermistor ADC
    uint16_t therm_reference;  // 2  — bridge reference thermistor ADC
    int16_t  pressure_pa;      // 2  — SDP32 differential pressure (Pa * 10)
    uint32_t ir_value;         // 4  — MAX30102 IR (nose PPG)
    uint32_t red_value;        // 4  — MAX30102 Red (nose PPG)
    uint16_t signal_flags;     // 2  — bitfield (see below)
    uint16_t reserved;         // 2  — future use / alignment
};

// signal_flags bitfield:
// bit 0:    left thermistor valid (ADC 101-3999)
// bit 1:    right thermistor valid
// bit 2:    reference thermistor valid
// bit 3:    SDP32 data ready
// bit 4:    MAX30102 data ready
// bit 5:    left nostril dominant (higher AC amplitude)
// bits 6-15: reserved
```

24 bytes fits well within the BLE MTU (default 23 bytes payload with ATT header, but NimBLE supports MTU negotiation up to 512). Request MTU 64 during connection.

### Sampling Loop

```
At 20 Hz (every 50ms):
  1. Read left thermistor      → analogRead(GPIO1)          ~100µs
  2. Read right thermistor     → analogRead(GPIO2)          ~100µs
  3. Read reference thermistor → analogRead(GPIO7)          ~100µs
  4. Read SDP32 pressure       → I2C triggered measurement  ~500µs
  5. Read MAX30102 FIFO        → I2C FIFO read              ~200µs
  6. Pack SensorPacketV2       → 24 bytes
  7. BLE notify                → ~1ms

  Total: ~2ms per sample. Plenty of headroom for 50ms interval.
```

### Pseudocode: Signal Processing on ESP32

Keep the firmware simple — just read and transmit raw values. All the heavy signal processing happens in the Cloud Function. But for the real-time LED feedback, we need basic breath detection on the ESP32:

```cpp
// Drift-compensated breath signal (runs on ESP32 for LED feedback)
float getBreathSignal() {
    float left  = analogRead(PIN_THERM_LEFT);
    float right = analogRead(PIN_THERM_RIGHT);
    float ref   = analogRead(PIN_THERM_REF);

    // Subtract reference to cancel thermal drift
    float leftCorrected  = left - ref;
    float rightCorrected = right - ref;

    // Pick dominant nostril (higher rolling variance)
    // Or simply sum them (dominant side naturally contributes more)
    float breath = leftCorrected + rightCorrected;

    return breath;
}
```

## Signal Fusion Algorithm (Cloud Function)

The cloud analysis function receives five time series. Here's how to fuse them for robust breath detection:

### Layer 1: Individual Signal Extraction

```
For each sample at time t:
  T_left(t)  = therm_left - therm_reference     // drift-cancelled left
  T_right(t) = therm_right - therm_reference     // drift-cancelled right
  P(t)       = pressure_pa                        // already zero-centered
  PPG_breath(t) = lowpass(ir_value, cutoff=0.5Hz) // respiratory component
  PPG_cardiac(t) = bandpass(ir_value, 0.5-3Hz)    // cardiac component
```

### Layer 2: Nostril Selection

```
For each 5-second window:
  var_left  = variance(T_left in window)
  var_right = variance(T_right in window)

  if var_left > 2 * var_right:
      dominant = 'left'
      T_breath = T_left
  else if var_right > 2 * var_left:
      dominant = 'right'
      T_breath = T_right
  else:
      dominant = 'both'
      T_breath = T_left + T_right  // both contributing

  nostril_ratio(window) = var_left / (var_left + var_right)  // 0=right, 1=left
```

### Layer 3: Multi-Signal Breath Detection

Each signal can independently detect breaths. By requiring agreement, we reject false positives:

```
For each candidate breath peak in T_breath:
    confirmed = false

    // Check 1: Does pressure signal show corresponding zero-crossing?
    if P(t) crosses zero within ±500ms of thermistor peak:
        confidence += 0.4

    // Check 2: Does PPG respiratory envelope show corresponding valley?
    if PPG_breath(t) has a local minimum within ±1000ms:
        confidence += 0.3

    // Check 3: Does HRV show RSA pattern? (HR increase = inhale)
    if PPG_cardiac rate increases around this time:
        confidence += 0.3

    if confidence >= 0.5:
        confirmed = true
        // Use pressure signal for precise timing (fastest response)
        // Use thermistor signal for depth estimation
        // Use PPG for validation
```

### Layer 4: Graceful Degradation

Not all sensors will always be available. The algorithm works with any subset:

```
Priority cascade:
  1. SDP32 pressure (most reliable, no drift, always aligned)
     → alone: detects breath rate and relative depth accurately

  2. Thermistor differential (good waveform detail, especially at low flow)
     → alone: works for ~10 min before drift overwhelms reference cancellation
     → with pressure: beautiful, adds shallow breath sensitivity

  3. PPG respiratory (indirect but always-on if sensor contacts skin)
     → alone: breath rate detection, limited depth info
     → with others: provides independent confirmation + HRV overlay

  4. Raw thermistors without reference (current behavior)
     → fallback if reference thermistor fails
     → works for ~5 min (current limitation)

Degradation is transparent: signal_flags in each packet tells the cloud
function exactly which sensors are active, so it adjusts processing.
```

## New Meditation Quality Metrics

With five signals instead of one, we can compute metrics that no existing consumer device offers:

### 1. Nostril Dominance Timeline
```
Left ██████████████░░░░░░░░░░░░░░░░░░  Session start: left dominant
Right░░░░░░░░░░░░░░████████████████████  Session end: switched to right

Interpretation: The nasal cycle shifted during meditation.
In yogic tradition, balanced nostril breathing (sushumna)
is associated with deep meditative states.
```

### 2. Breath Depth Confidence Score
Instead of estimating depth from one noisy thermistor, we have:
- Pressure integral (quantitative airflow volume)
- Thermistor amplitude (thermal proxy for flow)
- PPG respiratory modulation depth (vascular proxy)

Three independent estimates cross-validate each other. If all three agree the breath was shallow, we're confident. If they disagree, we flag uncertainty.

### 3. Breath Symmetry (Inhale vs Exhale)
The SDP32 is bidirectional: positive = exhale, negative = inhale. We can measure:
- Inhale duration vs exhale duration (I:E ratio)
- Inhale flow rate vs exhale flow rate
- Pause duration between inhale and exhale

In meditation, longer exhales (I:E ratio > 1:2) indicate parasympathetic activation. This is directly measurable with pressure sensing but impossible with a thermistor alone (which only sees temperature oscillation, losing the direction information in the drift).

### 4. Apnea Detection
The SDP32 reads exactly zero during breath holds. Combined with the PPG (which keeps working — blood still flows), we can detect intentional breath retentions used in certain meditation techniques and measure their duration precisely.

### 5. Respiratory Sinus Arrhythmia (RSA) Coupling
RSA is the natural variation in heart rate with breathing. Strong RSA coupling indicates healthy vagal tone. With simultaneous breath waveform (from pressure/thermistors) and cardiac timing (from nose PPG), we can compute the cross-correlation — a direct measure of parasympathetic engagement during meditation.

## Build Phases

### Phase 0: Validate Nose PPG (1-2 days)
Before building the full NoseHub, verify that the MAX30102 works on the nose:
1. Take the existing MAX30102 breakout
2. Hold it against the nose bridge with a finger (or tape it)
3. Run the existing firmware — look for cardiac pulse in IR signal
4. If you see a clean ~1Hz oscillation with ~1-2% AC/DC ratio, nose PPG works

**If it works**: Proceed to Phase 1.
**If signal is weak**: Try the septum (between nostrils) or the lateral nose (side of nostril). MORFEA used the septum successfully.

### Phase 1: Dual Thermistor + Reference (1 week)
Add two more thermistor voltage dividers to the breadboard:
- Left nostril thermistor on GPIO2
- Reference thermistor on GPIO7
- Mount all three on a crude nose clip (bent wire + tape)
- Modify firmware to read 3 ADC channels
- Record a 15-minute session
- Compare drift of (raw thermistor) vs (thermistor - reference)
- **Expected result**: The drift-cancelled signal stays usable for 15+ minutes

### Phase 2: Add SDP32 Pressure (1 week)
- Wire SDP32 eval board to I2C bus alongside MAX30102
- Connect nasal cannula tubing to SDP32 Port A
- Wear cannula + nose clip simultaneously
- Record same session on all 5 channels
- Compare breath detection accuracy across modalities
- **Expected result**: Pressure gives the cleanest, most reliable breath waveform

### Phase 3: Integrated NoseHub Clip (2 weeks)
- Design and 3D-print the nose clip frame
- Mold silicone prongs with embedded thermistors and tubing
- Mount MAX30102 in the bridge pad
- Build the full 24-byte packet firmware
- Update the Flutter debug screen to show all channels
- Record 30-minute meditation sessions
- **Expected result**: Reliable, drift-free, alignment-insensitive breath + HRV from one device

### Phase 4: Software Integration (1 week)
- Update BLE packet parsing in `breath_data.dart`
- Update Cloud Function to process 5-channel data
- Implement signal fusion algorithm
- Add nostril dominance and breath symmetry metrics
- Update debug screen with new signal waveforms

## Risks and Unknowns

| Risk | Mitigation |
|------|-----------|
| MAX30102 signal too weak on nose bridge | Try septum, ala (nostril wing), or earlobe as alternatives |
| Silicone prongs uncomfortable | Test various Shore hardness (10A-30A); standard cannula prongs are well-tolerated |
| SDP32 pressure too weak for shallow meditation breaths | SDP32 resolution is 0.002 Pa — orders of magnitude below nasal breath pressure (~1-5 Pa) |
| Three ADC reads slow down sample loop | Each analogRead ~100µs; three = 300µs; well within 50ms budget |
| I2C bus contention (MAX30102 + SDP32) | Both devices support 400kHz; bus time ~700µs total; no contention |
| 24-byte packet exceeds BLE MTU | NimBLE supports MTU negotiation; request 64 bytes at connection time |
| Reference thermistor picks up some airflow | Mount it on the TOP of the nose bridge, above the nostrils; airflow doesn't reach there |
| Silicone prong irritates nasal mucosa | Use medical-grade silicone (FDA Class VI); same material as cannula prongs worn 24/7 in hospitals |

## Why This Could Actually Be Special

Most breath-sensing research focuses on one modality and tries to optimize it. The NoseHub's approach is fundamentally different: **use five OK signals instead of one great signal**. Each sensor has weaknesses, but they have *different* weaknesses:

- Thermistors drift → Pressure doesn't
- Pressure loses shallow breaths → Thermistors catch them
- Both are external → PPG measures internal vasodilation
- PPG is indirect → Thermistors and pressure are direct

When you fuse signals that fail in different ways, the combined system is more reliable than any individual sensor could be — even a theoretically "perfect" one. This is the same principle behind GPS (multiple weak satellite signals → precise position) and cochlear implants (multiple electrodes → rich sound).

No consumer meditation device currently does multi-modal nasal breath sensing. If this works, it's genuinely novel.
