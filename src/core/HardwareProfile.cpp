#include "core/HardwareProfile.h"
#include <cstdio>
#include <algorithm>
#include <thread>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

#ifdef __linux__
#include <fstream>
#include <string>
#include <unistd.h>
#endif

// Forward declare CUDA query (defined in CUDABackend.cu if CUDA is available)
#ifndef NO_CUDA
extern "C" {
    int biobrain_cuda_device_count();
    int biobrain_cuda_get_cores(int device);
    size_t biobrain_cuda_get_vram(int device);
    const char* biobrain_cuda_get_name(int device);
}
#endif

namespace biobrain {

HardwareProfile HardwareProfile::detect() {
    HardwareProfile hw;

    // ── CPU cores ──
    hw.cpu_cores = static_cast<int>(std::thread::hardware_concurrency());
    if (hw.cpu_cores < 1) hw.cpu_cores = 1;

    // ── RAM ──
#ifdef __APPLE__
    {
        int64_t mem = 0;
        size_t len = sizeof(mem);
        sysctlbyname("hw.memsize", &mem, &len, nullptr, 0);
        hw.ram_bytes = static_cast<uint64_t>(mem);
    }
#elif defined(__linux__)
    {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            hw.ram_bytes = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
        }
    }
#endif
    hw.ram_gb = hw.ram_bytes / (1024.0 * 1024.0 * 1024.0);

    // ── GPU (Metal on macOS) ──
#ifdef __APPLE__
    // Metal GPU info via sysctl
    {
        char brand[256] = {};
        size_t len = sizeof(brand);
        if (sysctlbyname("machdep.cpu.brand_string", brand, &len, nullptr, 0) == 0) {
            // Apple Silicon has unified GPU — estimate cores from chip name
            hw.gpu_name = brand;
        }
        // On Apple Silicon, GPU shares RAM — estimate VRAM as fraction of total
        hw.gpu_vram_bytes = hw.ram_bytes / 2;  // conservative: half of unified memory
        hw.gpu_vram_gb = hw.gpu_vram_bytes / (1024.0 * 1024.0 * 1024.0);
        hw.has_gpu = true;  // all modern Macs have Metal

        // Estimate GPU compute units from chip
        int gpu_cores = 0;
        len = sizeof(gpu_cores);
        // Try IOKit-style detection via sysctl
        if (sysctlbyname("hw.perflevel0.logicalcpu", &gpu_cores, &len, nullptr, 0) != 0) {
            gpu_cores = hw.cpu_cores;  // fallback
        }
        hw.gpu_compute_units = gpu_cores;
    }
#endif

    // ── GPU (CUDA on Linux) ──
#ifndef NO_CUDA
    {
        int count = 0;
        // Try to query CUDA — this is safe even if no GPU is present
        // (biobrain_cuda_device_count returns 0 if CUDA init fails)
        // These functions are weak-linked or conditionally compiled
        #ifdef __linux__
        count = biobrain_cuda_device_count();
        #endif
        if (count > 0) {
            hw.has_gpu = true;
            hw.gpu_compute_units = biobrain_cuda_get_cores(0);
            hw.gpu_vram_bytes = biobrain_cuda_get_vram(0);
            hw.gpu_vram_gb = hw.gpu_vram_bytes / (1024.0 * 1024.0 * 1024.0);
            const char* name = biobrain_cuda_get_name(0);
            if (name) hw.gpu_name = name;
        }
    }
#endif

    hw.computeScaling();
    return hw;
}

void HardwareProfile::computeScaling() {
    // ── Determine tier ──
    // Based on RAM (primary constraint for neuron/synapse storage)
    // and CPU cores (affects simulation throughput)
    if (ram_gb >= 64 && cpu_cores >= 16) {
        tier = 4;  // Extended
    } else if (ram_gb >= 16 && cpu_cores >= 8) {
        tier = 3;  // Full
    } else if (ram_gb >= 8 && cpu_cores >= 4) {
        tier = 2;  // Standard
    } else if (ram_gb >= 4 && cpu_cores >= 2) {
        tier = 1;  // Light
    } else {
        tier = 0;  // Minimal
    }

    // Bump tier if strong GPU available
    if (has_gpu && gpu_vram_gb >= 8.0 && tier < 4) {
        tier = std::min(tier + 1, 4);
    }

    // ── Scale neuron counts by tier ──
    // Tier 3 = 1.0x (the original 228K design)
    // Each tier halves/doubles from there
    static const double tier_scale[] = {0.1, 0.25, 0.5, 1.0, 2.0};
    double s = tier_scale[tier];

    // Round to multiples of 100 for clean column sizes
    auto scale = [s](uint32_t base) -> uint32_t {
        uint32_t scaled = static_cast<uint32_t>(base * s);
        return std::max(scaled - (scaled % 100), uint32_t(100));
    };

    retina_neurons   = scale(8192);
    lgn_neurons      = scale(5000);
    v1_neurons       = scale(80000);
    v2v4_neurons     = scale(60000);
    it_neurons       = scale(30000);
    vta_neurons      = scale(2000);
    striatum_neurons = scale(20000);
    motor_neurons    = scale(5000);
    wernicke_neurons = scale(10000);
    broca_neurons    = scale(8000);

    total_neurons = retina_neurons + lgn_neurons + v1_neurons + v2v4_neurons
                  + it_neurons + vta_neurons + striatum_neurons + motor_neurons
                  + wernicke_neurons + broca_neurons;

    // ── Scale simulation parameters ──
    cpu_sim_threads = std::max(1, cpu_cores - 2);  // reserve 2 for GUI + webcam
    synapse_density = s;  // fewer synapses on lower tiers

    // Webcam resolution
    if (tier >= 4)      webcam_resolution = 1080;
    else if (tier >= 3) webcam_resolution = 720;
    else if (tier >= 2) webcam_resolution = 480;
    else                webcam_resolution = 240;

    // Retinal grid (sqrt of retina neuron count / 2 for ON+OFF)
    retinal_grid = static_cast<int>(std::sqrt(retina_neurons / 2.0));
    if (retinal_grid < 8) retinal_grid = 8;
}

void HardwareProfile::print() const {
    static const char* tier_names[] = {"Minimal", "Light", "Standard", "Full", "Extended"};

    fprintf(stderr, "\n");
    fprintf(stderr, "Hardware Profile:\n");
    fprintf(stderr, "  CPU:  %d cores\n", cpu_cores);
    fprintf(stderr, "  RAM:  %.1f GB\n", ram_gb);
    if (has_gpu) {
        fprintf(stderr, "  GPU:  %s (%d compute units, %.1f GB VRAM)\n",
                gpu_name.c_str(), gpu_compute_units, gpu_vram_gb);
    } else {
        fprintf(stderr, "  GPU:  none (CPU-only mode)\n");
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "Scaling Tier: %d (%s)\n", tier, tier_names[tier]);
    fprintf(stderr, "  Total neurons:     %u\n", total_neurons);
    fprintf(stderr, "  Synapse density:   %.0f%%\n", synapse_density * 100);
    fprintf(stderr, "  Sim threads:       %d\n", cpu_sim_threads);
    fprintf(stderr, "  Webcam resolution: %dp\n", webcam_resolution);
    fprintf(stderr, "  Retinal grid:      %dx%d\n", retinal_grid, retinal_grid);
    fprintf(stderr, "\n");
    fprintf(stderr, "  Region breakdown:\n");
    fprintf(stderr, "    Retina:    %6u    V1:       %6u    IT:       %6u\n",
            retina_neurons, v1_neurons, it_neurons);
    fprintf(stderr, "    LGN:       %6u    V2/V4:    %6u    VTA:      %6u\n",
            lgn_neurons, v2v4_neurons, vta_neurons);
    fprintf(stderr, "    Striatum:  %6u    Motor:    %6u\n",
            striatum_neurons, motor_neurons);
    fprintf(stderr, "    Wernicke:  %6u    Broca:    %6u\n",
            wernicke_neurons, broca_neurons);
    fprintf(stderr, "\n");
}

} // namespace biobrain
