#pragma once

#include <cstdint>
#include <string>

namespace biobrain {

// Detected hardware capabilities and derived simulation parameters.
// Call HardwareProfile::detect() at startup, then use the scaled
// neuron counts when constructing brain regions.
struct HardwareProfile {
    // ── Raw hardware specs ──
    int      cpu_cores         = 1;
    uint64_t ram_bytes         = 0;
    double   ram_gb            = 0;
    bool     has_gpu           = false;
    std::string gpu_name;
    uint64_t gpu_vram_bytes    = 0;
    double   gpu_vram_gb       = 0;
    int      gpu_compute_units = 0;   // CUDA cores or Metal GPU cores

    // ── Scaling tier ──
    // Determines neuron count multipliers based on hardware.
    //   Tier 0: Minimal    (~23K neurons)  — 1 core, <4GB RAM, no GPU
    //   Tier 1: Light      (~57K neurons)  — 2-4 cores, 4-8GB RAM
    //   Tier 2: Standard   (~115K neurons) — 4-8 cores, 8-16GB RAM
    //   Tier 3: Full       (~228K neurons) — 8+ cores, 16-32GB RAM, GPU
    //   Tier 4: Extended   (~456K neurons) — 16+ cores, 64GB+ RAM, strong GPU
    int tier = 2;

    // ── Scaled region neuron counts ──
    // These replace the hard-coded constants in each region's create() call.
    uint32_t retina_neurons     = 8192;
    uint32_t lgn_neurons        = 5000;
    uint32_t v1_neurons         = 80000;
    uint32_t v2v4_neurons       = 60000;
    uint32_t it_neurons         = 30000;
    uint32_t vta_neurons        = 2000;
    uint32_t striatum_neurons   = 20000;
    uint32_t motor_neurons      = 5000;
    uint32_t wernicke_neurons   = 10000;
    uint32_t broca_neurons      = 8000;
    uint32_t total_neurons      = 228192;

    // ── Scaled simulation parameters ──
    int      cpu_sim_threads    = 1;       // threads for CPU backend
    double   synapse_density    = 1.0;     // multiplier on connection probability
    int      webcam_resolution  = 480;     // 240, 480, 720, 1080
    int      retinal_grid       = 64;      // retinal encoder grid size

    // ── Detect hardware and compute scaling ──
    static HardwareProfile detect();

    // Print summary to stderr
    void print() const;

private:
    void computeScaling();
};

} // namespace biobrain
