# BioBrain — Development Guide

## Project Overview

Biologically-realistic spiking neural simulation engine in C++20 with Metal GPU compute and Qt6 frontend. Research-focused: biological accuracy over performance shortcuts.

## Build Commands

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(sysctl -n hw.ncpu)

# Run
./build/BioBrain

# Clean
rm -rf build
```

## Architecture

### Simulation Engine: Hybrid CPU + Metal GPU

- **Event-driven spike router** (CPU): priority queue ordered by spike arrival time, handles axonal delays
- **Metal compute** (GPU): batch neuron updates when spike bursts exceed threshold
- **Real-time clock**: 0.1ms substeps, synced to wall-clock (1ms sim = 1ms real)
- **Thread layout**: 1 router, 1 GUI, 1 webcam, 1 recorder, 6 CPU compute

### Key Abstractions

- `Neuron` (interface) → `IzhikevichNeuron`, `HodgkinHuxleyNeuron`, `AdExNeuron`, `LIFNeuron`
- `PlasticityRule` (interface) → `STDP`, `DopamineSTDP`, `NeuromodulatorySTDP`
- `ComputeBackend` (interface) → `CPUBackend`, `MetalBackend`
- `BrainRegion` holds a neuron factory + compute backend + plasticity rule (all swappable at runtime)

### Data Flow

```
WebcamCapture → RetinalEncoder → SpikeRouter → BrainRegions → SpikeRouter (loop)
                                                     ↕
                                              ComputeBackend (CPU or Metal)
```

## Conventions

- C++20: use concepts, ranges, jthread, span where appropriate
- Header + implementation split (.h/.cpp), no header-only except trivial structs
- Metal Objective-C++ files use .mm extension
- Metal shader files in src/metal/ with .metal extension
- Neuron parameters: use biologically-grounded values with references in comments
- Time units: milliseconds (double) throughout the codebase
- Voltage units: millivolts (double)
- Conductance units: nanosiemens (double)

## Dependencies

- Qt6 6.7+ (Widgets, Multimedia)
- Metal/MetalKit (macOS system framework)
- QCustomPlot 2.1+ (spike raster plotting)
- HDF5 1.14+ (data recording)
- CMake 3.28+
- C++20 / Apple Clang 16+

## Brain Regions (9 total, ~210K neurons)

Retina (8192) → LGN (5000) → V1 (80000) → V2/V4 (60000) → IT (30000)
VTA (2000) → Striatum (20000) → Motor (5000)
