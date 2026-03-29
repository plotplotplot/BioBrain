# BioBrain — Biologically-Realistic Neural Simulation Engine

## Context

BioBrain is a research simulation tool that models a biologically-faithful spiking neural network in C++. It integrates with a webcam to perceive the real world via a retinal encoding model, processes visual input through a hierarchy of brain regions (LGN → V1 → V2/V4 → IT), and uses a dopamine-modulated reinforcement learning loop (VTA → Striatum → Motor) to learn responses. The simulation runs in real biological time (1ms = 1ms wall-clock) on Apple M5 hardware using a hybrid CPU event-driven + Metal GPU compute architecture. A Qt6 frontend provides real-time visualization and per-region backend configuration.

**Target hardware:** Apple M5, 10 cores, 32GB unified memory, Metal 4 GPU.

---

## Neuron Model

**Primary model: Izhikevich (2003)**

Two coupled ODEs reproducing 20+ biological firing patterns:

```
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)
if v >= 30mV: v = c, u = u + d
```

Parameters (a, b, c, d) determine neuron type:

| Type | a | b | c | d | Region | Role |
|------|---|---|---|---|--------|------|
| Regular Spiking (RS) | 0.02 | 0.2 | -65 | 8 | V1/V2/IT excitatory | Cortical pyramidal cells |
| Fast Spiking (FS) | 0.1 | 0.2 | -65 | 2 | All cortex inhibitory | Interneurons |
| Intrinsically Bursting (IB) | 0.02 | 0.2 | -55 | 4 | V1 layer 5 | Feedback to LGN |
| Low-Threshold Spiking (LTS) | 0.02 | 0.25 | -65 | 2 | Cortex inhibitory | Rhythmic inhibition |
| Tonic Spiking | 0.02 | 0.2 | -65 | 6 | LGN relay | Thalamic relay |
| Medium Spiny (D1) | 0.02 | 0.2 | -80 | 8 | Striatum | Go pathway |
| Medium Spiny (D2) | 0.02 | 0.2 | -80 | 8 | Striatum | NoGo pathway |
| Dopaminergic | 0.02 | 0.2 | -50 | 2 | VTA | Reward signal |

**Alternative models (selectable per-region at runtime):**
- Hodgkin-Huxley: Full ion-channel dynamics (Na+, K+, Ca2+, leak). 4 ODEs per neuron.
- AdEx (Adaptive Exponential): 2 ODEs with exponential spike initiation.
- LIF (Leaky Integrate-and-Fire): Simplest, 1 ODE. For baseline comparison.

All models implement a common `Neuron` interface: `step(dt, I_syn) → bool spiked`.

---

## Synapse Model

### Conductance-Based Transmission

Each synapse type modeled as a conductance with distinct kinetics:

| Receptor | Type | Rise τ | Decay τ | Reversal E | Role |
|----------|------|--------|---------|------------|------|
| AMPA | Excitatory | 0.5ms | 2ms | 0mV | Fast excitation |
| NMDA | Excitatory | 2ms | 80ms | 0mV | Slow, voltage-gated (Mg²⁺ block) |
| GABA-A | Inhibitory | 0.5ms | 6ms | -70mV | Fast inhibition |
| GABA-B | Inhibitory | 30ms | 150ms | -90mV | Slow inhibition |

Synaptic current: `I_syn = g(t) * (V - E_rev)`

Each receptor type toggleable per-region via the Qt config panel.

### Short-Term Plasticity (Tsodyks-Markram)

Models synaptic facilitation and depression:
- **x**: fraction of available neurotransmitter (depression variable)
- **u**: release probability (facilitation variable)
- Each spike: `u → u + U*(1-u)`, `x → x - u*x`, transmitter released = `u*x`

### Axonal Conduction Delays

- **Myelinated axons**: 1-5ms delay (long-range inter-region connections)
- **Unmyelinated axons**: 5-20ms delay (local inhibitory interneurons)
- Conduction velocity scales with axon diameter and myelination ratio
- Per-region myelination ratio configurable (slider, 0.0-1.0)
- Delays implemented via the spike router's priority queue

---

## Plasticity: STDP + Dopamine (Three-Factor Rule)

### Spike-Timing-Dependent Plasticity

Standard asymmetric STDP window:
- Pre before post (Δt > 0): LTP, `Δw = A+ * exp(-Δt / τ+)`, τ+ = 20ms
- Post before pre (Δt < 0): LTD, `Δw = -A- * exp(Δt / τ-)`, τ- = 20ms

### Dopamine Modulation (Three-Factor)

STDP alone sets an **eligibility trace** `e(t)` that decays exponentially (τ_e = 1000ms). Actual weight change only occurs when dopamine arrives:

```
de/dt = -e/τ_e + STDP(Δt) * δ(t_spike)
dw/dt = DA(t) * e(t)
```

Where `DA(t)` is the dopamine concentration from VTA neurons, encoding reward prediction error.

### Alternative plasticity rules (selectable):
- **STDP only**: No dopamine modulation, pure Hebbian
- **Full neuromodulatory**: DA + serotonin + acetylcholine + norepinephrine
- **None**: Fixed weights for controlled experiments

---

## Brain Regions

### Architecture: Visual Cortex Pipeline + Basal Ganglia RL Loop

```
Webcam → [Retinal Encoder] → LGN → V1 → V2/V4 → IT
                                                    ↓
                              Motor ← Striatum ← VTA
                                        ↑           ↑
                                      (D1/D2)    (reward)
```

### Region Specifications

| Region | Neurons | Neuron Types | Connectivity | Role |
|--------|---------|-------------|--------------|------|
| Retina | 8,192 | ON-center, OFF-center RGCs | Topographic → LGN | Luminance → spike encoding |
| LGN | 5,000 | Tonic relay, interneurons | 1:1 from retina + V1 feedback | Temporal filtering, attention gating |
| V1 | 80,000 | RS, FS, IB, LTS (layers 2/3, 4, 5, 6) | Cortical columns, orientation maps | Edge detection, orientation selectivity |
| V2/V4 | 60,000 | RS, FS | Convergent from V1 | Shape, texture, color processing |
| IT | 30,000 | RS, FS | Convergent from V2/V4 | Object representation |
| VTA | 2,000 | Dopaminergic, GABAergic | Diffuse → Striatum, cortex | Reward prediction error |
| Striatum | 20,000 | D1 MSN, D2 MSN, interneurons | From IT + VTA modulation | Go/NoGo action selection |
| Motor | 5,000 | RS, FS | From Striatum | Action output, population decoding |

**Total: ~210,000 neurons**

### Retinal Encoder

Converts webcam frames (30fps, 640×480) to spike trains:
1. **Luminance extraction**: RGB → grayscale
2. **Center-surround filtering**: Difference-of-Gaussians (DoG) at multiple scales
3. **ON/OFF pathways**: ON-center cells fire for bright spots, OFF-center for dark
4. **Rate coding**: Filter response magnitude → Poisson spike rate (0-200 Hz)
5. **Spatial downsampling**: 640×480 → 64×64 grid → 8,192 RGCs (ON + OFF)

### V1 Cortical Columns

V1 organized into cortical minicolumns:
- Each column: ~80 neurons (excitatory + inhibitory)
- Orientation selectivity: columns tuned to 0°, 22.5°, 45°, ... 157.5° (8 orientations)
- Lateral inhibition between columns with different preferred orientations
- Layer 4 receives LGN input, layers 2/3 do lateral processing, layer 5 sends feedback

---

## Simulation Engine: Hybrid CPU + Metal GPU

### Event-Driven Spike Router (CPU)

The spike router is the central nervous system of the simulation:

1. **Priority queue** ordered by spike arrival time (accounting for axonal delays)
2. When a spike arrives at a target neuron, the neuron's synaptic conductances are updated
3. If multiple spikes arrive at neurons in the same region within a batch window (0.1ms), they're collected and dispatched to Metal for parallel update
4. Neurons that cross threshold generate new spike events, inserted back into the queue

```
while (sim_time < wall_clock_time):
    event = queue.pop()          // next spike event
    region = event.target_region
    region.deliver_spike(event)  // update synaptic conductance

    if region.pending_spikes >= batch_threshold:
        metal_backend.update_neurons(region)  // GPU batch update
    else:
        cpu_backend.update_neuron(event.target) // single neuron on CPU
```

### Metal Compute Backend (GPU)

For batch neuron updates when a region receives a burst of input:

- **Izhikevich kernel**: Each GPU thread updates one neuron's (v, u) state for one timestep
- **Synapse kernel**: Parallel conductance decay + spike delivery
- **Unified memory**: M5's shared CPU/GPU memory eliminates copy overhead — neuron state arrays are directly accessible by both

Metal shaders for each neuron model:
- `izhikevich.metal`: 2-equation update, ~4 FLOPs per neuron per step
- `hodgkin_huxley.metal`: 4-equation update with ion channel gating variables
- `synapse_update.metal`: Conductance decay for all 4 receptor types

### Real-Time Clock Synchronization

- Simulation timestep: 0.1ms (100μs) for numerical stability
- Wall-clock target: 1ms simulation = 1ms real time
- Each 1ms wall-clock tick: run 10 simulation substeps
- If simulation runs faster than real-time: sleep to maintain sync
- If simulation falls behind: log warning, do not drop spikes

Thread allocation (10 cores):
- 1 thread: Spike router (priority queue management)
- 1 thread: Qt GUI event loop
- 1 thread: Webcam capture + retinal encoding
- 1 thread: Recording / HDF5 output
- 6 threads: CPU neuron updates (when not batched to Metal)
- Metal GPU: Handles burst updates independently

---

## Qt6 Frontend

### Main Window Layout

Four dockable panels:

1. **Brain Region Tree** (left dock)
   - Hierarchical QTreeWidget: Region → Layer → Neuron types
   - Shows neuron count, current firing rate, compute backend indicator
   - Click to select region for visualization and configuration

2. **Visualization Area** (center)
   - **Spike Raster** (QCustomPlot): Scrolling 100ms window, neuron ID vs time
   - **Webcam Feed** (QLabel + QCamera): Live 640×480 RGB
   - **Activity Map**: Per-region circles with glow intensity proportional to firing rate

3. **Backend Config Panel** (right dock)
   - Per-region dropdowns for:
     - Neuron model (Izhikevich / HH / AdEx / LIF)
     - Compute backend (CPU Event-Driven / Metal GPU / Hybrid auto)
     - Plasticity rule (STDP+DA / STDP / Full Neuromodulatory / None)
   - Synapse receptor checkboxes (AMPA / NMDA / GABA-A / GABA-B)
   - Myelination ratio slider
   - Raw neuron parameter editors (a, b, c, d for Izhikevich; gNa, gK for HH; etc.)
   - Changes apply at next simulation pause or hot-swap with queue drain

4. **Toolbar + Status Bar**
   - Run / Pause / Stop controls
   - Simulation time display
   - Real-time scaling factor
   - Active neuron count / total
   - GPU utilization, CPU thread usage, memory, spikes/sec

### Signal/Slot Architecture

- `Simulation` emits `spikesBatch(RegionID, vector<SpikeEvent>)` at 60Hz (batched for GUI)
- `SpikeRasterWidget` consumes spike batches, renders via QCustomPlot
- `BackendConfigPanel` emits `configChanged(RegionID, RegionConfig)` → `Simulation` hot-swaps
- `WebcamCapture` feeds frames to both `WebcamWidget` (display) and `RetinalEncoder` (processing)

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| Qt6 | 6.7+ | GUI framework (Widgets, Multimedia for webcam) |
| Metal/MetalKit | macOS 15+ | GPU compute shaders |
| QCustomPlot | 2.1+ | Spike raster plotting |
| HDF5 | 1.14+ | Spike data recording |
| CMake | 3.28+ | Build system |
| C++20 | clang 16+ | Language standard (concepts, ranges, jthread) |

---

## Project Structure

```
BioBrain/
├── CMakeLists.txt
├── README.md
├── CLAUDE.md
├── src/
│   ├── core/
│   │   ├── Neuron.h              # Base neuron interface (virtual step/reset)
│   │   ├── IzhikevichNeuron.h/cpp
│   │   ├── HodgkinHuxleyNeuron.h/cpp
│   │   ├── AdExNeuron.h/cpp
│   │   ├── LIFNeuron.h/cpp
│   │   ├── Synapse.h/cpp         # Conductance-based, 4 receptor types
│   │   ├── SpikeEvent.h          # {source, target, time, delay}
│   │   ├── SpikeRouter.h/cpp     # Priority queue + axonal delay routing
│   │   ├── BrainRegion.h/cpp     # Region container, topology, neuron factory
│   │   └── Simulation.h/cpp      # Master clock, thread orchestration, real-time sync
│   ├── plasticity/
│   │   ├── PlasticityRule.h      # Interface
│   │   ├── STDP.h/cpp
│   │   ├── DopamineSTDP.h/cpp    # Three-factor eligibility trace
│   │   └── NeuromodulatorySTDP.h/cpp
│   ├── input/
│   │   ├── WebcamCapture.h/cpp   # AVFoundation (macOS) / V4L2 (Linux)
│   │   └── RetinalEncoder.h/cpp  # DoG filtering, ON/OFF, rate coding
│   ├── regions/
│   │   ├── Retina.h/cpp
│   │   ├── LGN.h/cpp
│   │   ├── V1.h/cpp              # Cortical columns, orientation maps
│   │   ├── V2V4.h/cpp
│   │   ├── ITCortex.h/cpp
│   │   ├── VTA.h/cpp             # Dopamine system
│   │   ├── Striatum.h/cpp        # D1/D2 MSN pathways
│   │   └── MotorCortex.h/cpp
│   ├── compute/
│   │   ├── ComputeBackend.h      # Interface: updateNeurons(region, dt)
│   │   ├── CPUBackend.h/cpp      # Event-driven, thread pool
│   │   └── MetalBackend.h/mm     # Metal command queues + compute pipelines
│   ├── metal/
│   │   ├── izhikevich.metal
│   │   ├── hodgkin_huxley.metal
│   │   ├── adex.metal
│   │   └── synapse_update.metal
│   ├── gui/
│   │   ├── MainWindow.h/cpp
│   │   ├── RegionTreeWidget.h/cpp
│   │   ├── SpikeRasterWidget.h/cpp
│   │   ├── ActivityMapWidget.h/cpp
│   │   ├── WebcamWidget.h/cpp
│   │   ├── BackendConfigPanel.h/cpp
│   │   └── NeuronParamEditor.h/cpp
│   ├── recording/
│   │   └── SpikeRecorder.h/cpp   # HDF5 spike raster output
│   └── main.cpp
└── docs/
```

---

## Verification Plan

1. **Single neuron validation**: Inject known current waveforms into each Izhikevich neuron type, verify firing patterns match published figures (Izhikevich 2003, Fig 1)
2. **Synapse validation**: Verify AMPA/NMDA/GABA conductance kinetics against Dayan & Abbott textbook curves
3. **STDP validation**: Measure weight change vs spike timing offset, compare to Bi & Poo (1998) experimental curve
4. **Retinal encoder**: Feed static images, verify ON/OFF center-surround responses produce expected edge-enhancement patterns
5. **Real-time check**: Measure wall-clock drift over 60s of simulation; should stay within ±5ms
6. **Metal GPU parity**: Run same 1000-neuron network on CPU-only vs Metal, compare spike rasters (must be identical within floating-point tolerance)
7. **Qt frontend**: Verify spike raster updates at 60fps, backend config changes apply without crash, webcam feed displays correctly
8. **End-to-end**: Run full pipeline with webcam input, verify spikes propagate Retina→LGN→V1→V2→IT→Striatum→Motor, dopamine modulates weights over time
