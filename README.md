# BioBrain

A biologically-realistic spiking neural simulation engine that models the visual cortex pipeline, basal ganglia reinforcement learning loop, and language production circuit. Written in C++ with GPU compute acceleration (Metal on macOS, CUDA on Linux) and a Qt6 frontend for real-time visualization.

**Cross-platform**: macOS (Apple Silicon / Metal) and Linux (NVIDIA CUDA / RTX).

## Architecture

```
Webcam → [Retinal Encoder] → LGN → V1 → V2/V4 → IT ─┬→ Wernicke's → Broca's → Speaker
                                                       └→ Striatum → Motor
                                                VTA ───→ (dopamine modulation)
```

### Brain Regions (228,192 neurons)

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
| Wernicke's | 10,000 | Language comprehension (semantic attractors) |
| Broca's | 8,000 | Speech production (6 vocal output pools) |
| **Total** | **228,192** | |

### Simulation Engine

**Hybrid CPU + GPU** — event-driven spike routing on CPU with batch neuron updates on GPU.

- **Real-time target**: 1ms simulation = 1ms wall-clock, 0.1ms substeps
- **Event-driven**: neurons only compute when receiving spikes
- **GPU acceleration**: Metal (macOS) or CUDA (Linux) for batch neuron updates
- **Vocal synthesis**: Broca's area output pools drive a formant-based synthesizer via CoreAudio (macOS) or PulseAudio (Linux)

### Platform Support

| Component | macOS | Linux |
|-----------|-------|-------|
| GPU compute | Metal (Apple Silicon) | CUDA (NVIDIA RTX) |
| Webcam | AVFoundation | V4L2 |
| Audio output | CoreAudio | PulseAudio |
| GUI | Qt6 | Qt6 |
| Build | CMake + Clang | CMake + GCC/nvcc |

### Neuron Models (selectable per-region at runtime)

- **Izhikevich** (default) — 20+ firing patterns from 2 equations
- **Hodgkin-Huxley** — full ion-channel dynamics (Na+, K+, Ca2+)
- **AdEx** — adaptive exponential integrate-and-fire
- **LIF** — leaky integrate-and-fire (baseline comparison)

### Synapse Model

- Conductance-based: AMPA, NMDA (voltage-gated Mg2+ block), GABA-A, GABA-B
- Short-term plasticity: Tsodyks-Markram facilitation/depression
- Axonal delays: myelinated (1-5ms) and unmyelinated (5-20ms)

### Plasticity (selectable per-region)

- **STDP + Dopamine** (default) — three-factor eligibility trace rule
- **STDP** — pure Hebbian spike-timing
- **Full neuromodulatory** — DA + serotonin + acetylcholine + norepinephrine
- **None** — fixed weights for controlled experiments

## Qt6 Frontend

Four-panel dark-themed dashboard:
1. **Brain Region Tree** — hierarchical region browser with neuron counts and firing rates
2. **Spike Raster** — real-time scrolling spike plot for selected region
3. **Webcam + Activity Map** — live camera feed with camera selector, per-region activity glow
4. **Backend Config** — per-region dropdowns for neuron model, compute backend, synapse types, plasticity rule, myelination, and raw parameters

## Building

### macOS (Metal GPU)

```bash
brew install qt@6 hdf5

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
codesign --force --deep --sign - ./build/BioBrain.app
open ./build/BioBrain.app
```

### Ubuntu/Linux (NVIDIA CUDA GPU)

```bash
sudo apt install -y qt6-base-dev libqt6multimedia6 \
    libpulse-dev libhdf5-dev cmake g++ \
    nvidia-cuda-toolkit libv4l-dev

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/BioBrain
```

### Docker (Linux + NVIDIA GPU)

```bash
docker build -t biobrain .
docker run --gpus all -p 9090:9090 biobrain
```

The Docker container runs the headless REST harness on port 9090. For the full GUI, use the native build or run with X11 forwarding:
```bash
docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device /dev/video0 -p 9090:9090 biobrain ./BioBrain
```

### Debug REST API

The simulation exposes a REST API on port 9090:
```bash
curl http://localhost:9090/                          # web dashboard
curl http://localhost:9090/api/sim/status             # simulation state
curl http://localhost:9090/api/regions                # brain region stats
curl http://localhost:9090/api/region/2               # V1 detail + voltages
curl http://localhost:9090/api/neuron/1/42            # single LGN neuron state
curl http://localhost:9090/api/profile                # step timing + real-time ratio
curl http://localhost:9090/api/screenshot             # PNG screenshot (base64)
curl http://localhost:9090/api/webcam/cameras         # list cameras
curl -X POST http://localhost:9090/api/webcam/switch?id=DEVICE_ID
curl -X POST http://localhost:9090/api/sim/inject/region?id=9&count=500  # stimulate Broca's
curl http://localhost:9090/api/debug/trace            # projection wiring map
curl http://localhost:9090/api/memory                 # memory usage estimate
```

## Project Structure

```
src/
├── core/           # Neuron models, synapses, spike routing, simulation clock
├── plasticity/     # STDP, dopamine-modulated STDP, neuromodulatory rules
├── input/          # Webcam capture (AVFoundation macOS / V4L2 Linux), retinal encoder
├── regions/        # Brain regions: Retina, LGN, V1, V2/V4, IT, VTA, Striatum, Motor, Wernicke's, Broca's
├── compute/        # CPU, Metal (macOS), and CUDA (Linux) compute backends
├── metal/          # Metal compute shaders (.metal)
├── cuda/           # CUDA compute kernels (.cu)
├── audio/          # Vocal synthesizer (CoreAudio macOS / PulseAudio Linux)
├── gui/            # Qt6 frontend widgets
├── harness/        # REST debug API and test harness
├── recording/      # HDF5 spike data recorder
└── main.cpp        # Application entry point
```

## References

- Izhikevich, E.M. (2003). Simple model of spiking neurons. IEEE Trans Neural Networks.
- Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. J Neuroscience.
- Tsodyks, M. & Markram, H. (1997). The neural code between neocortical pyramidal neurons. PNAS.
- Dayan, P. & Abbott, L.F. (2001). Theoretical Neuroscience. MIT Press.
- Izhikevich, E.M. (2007). Solving the distal reward problem through STDP and dopamine. Cerebral Cortex.

## License

MIT
