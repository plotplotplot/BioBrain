# BioBrain

A biologically-realistic spiking neural simulation engine that models the visual cortex pipeline and basal ganglia reinforcement learning loop. Written in C++ with Metal GPU compute acceleration and a Qt6 frontend for real-time visualization and configuration.

## Architecture

```
Webcam → [Retinal Encoder] → LGN → V1 → V2/V4 → IT
                                                    ↓
                              Motor ← Striatum ← VTA
                                        ↑           ↑
                                      (D1/D2)    (reward)
```

### Brain Regions

| Region | Neurons | Role |
|--------|---------|------|
| Retina | 8,192 | ON/OFF center-surround → spike encoding |
| LGN | 5,000 | Thalamic relay, temporal filtering |
| V1 | 80,000 | Orientation-selective cortical columns |
| V2/V4 | 60,000 | Shape, texture, color processing |
| IT | 30,000 | Object representation |
| VTA | 2,000 | Dopamine reward prediction error |
| Striatum | 20,000 | D1/D2 Go/NoGo action selection |
| Motor | 5,000 | Action output, population decoding |
| **Total** | **~210,000** | |

### Simulation Engine

**Hybrid CPU + Metal GPU** — event-driven spike routing on CPU with batch neuron updates dispatched to Metal compute shaders when spike bursts arrive.

- **Real-time**: 1ms simulation = 1ms wall-clock, 0.1ms substeps
- **Event-driven**: neurons only compute when receiving spikes (no wasted cycles on silent neurons)
- **Metal acceleration**: batch updates on GPU via unified memory (zero-copy on Apple Silicon)

### Neuron Models (selectable per-region)

- **Izhikevich** (default) — 20+ firing patterns from 2 equations
- **Hodgkin-Huxley** — full ion-channel dynamics (Na⁺, K⁺, Ca²⁺)
- **AdEx** — adaptive exponential integrate-and-fire
- **LIF** — leaky integrate-and-fire (baseline comparison)

### Synapse Model

- Conductance-based: AMPA, NMDA (voltage-gated Mg²⁺ block), GABA-A, GABA-B
- Short-term plasticity: Tsodyks-Markram facilitation/depression
- Axonal delays: myelinated (1-5ms) and unmyelinated (5-20ms)

### Plasticity (selectable per-region)

- **STDP + Dopamine** (default) — three-factor eligibility trace rule
- **STDP** — pure Hebbian spike-timing
- **Full neuromodulatory** — DA + serotonin + acetylcholine + norepinephrine
- **None** — fixed weights for controlled experiments

## Qt6 Frontend

Four-panel dashboard:
1. **Brain Region Tree** — hierarchical region browser with neuron counts and firing rates
2. **Spike Raster** — real-time scrolling spike plot for selected region
3. **Webcam + Activity Map** — live camera feed and per-region activity visualization
4. **Backend Config** — per-region dropdowns for neuron model, compute backend, synapse types, plasticity rule, myelination, and raw parameters

## Building

### macOS (Metal GPU)

```bash
# Requirements: Xcode, Qt6, HDF5
brew install qt@6 hdf5

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
codesign --force --deep --sign - ./build/BioBrain.app
open ./build/BioBrain.app
```

### Ubuntu/Linux (NVIDIA CUDA GPU)

```bash
# Requirements: CUDA toolkit, Qt6, PulseAudio, V4L2, HDF5
sudo apt install -y qt6-base-dev libqt6multimedia6 \
    libpulse-dev libhdf5-dev cmake g++ \
    nvidia-cuda-toolkit libv4l-dev

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/BioBrain
```

### Debug API

The simulation exposes a REST debug API on port 9090:
```bash
curl http://localhost:9090/api/sim/status     # simulation state
curl http://localhost:9090/api/regions         # brain region stats
curl http://localhost:9090/api/webcam/cameras  # list cameras
curl -X POST http://localhost:9090/api/webcam/switch?id=DEVICE_ID
```

Open http://localhost:9090 in a browser for the interactive dashboard.

## Project Structure

```
src/
├── core/           # Neuron models, synapses, spike routing, simulation clock
├── plasticity/     # STDP, dopamine-modulated STDP, neuromodulatory rules
├── input/          # Webcam capture, retinal encoder
├── regions/        # Brain region implementations (LGN, V1, V2/V4, IT, VTA, Striatum, Motor)
├── compute/        # CPU, Metal (macOS), and CUDA (Linux) compute backends
├── metal/          # Metal compute shaders (.metal files)
├── cuda/           # CUDA compute kernels (.cu files)
├── audio/          # Vocal synthesizer (CoreAudio macOS / PulseAudio Linux)
├── gui/            # Qt6 frontend widgets
├── recording/      # HDF5 spike data recorder
└── main.cpp        # Application entry point
```

## References

- Izhikevich, E.M. (2003). Simple model of spiking neurons. IEEE Trans Neural Networks.
- Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. J Neuroscience.
- Tsodyks, M. & Markram, H. (1997). The neural code between neocortical pyramidal neurons. PNAS.
- Dayan, P. & Abbott, L.F. (2001). Theoretical Neuroscience. MIT Press.

## License

MIT
