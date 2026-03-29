#pragma once

#include <cstdint>
#include <vector>
#include <span>

// Forward declarations
namespace biobrain { class BrainRegion; }
using biobrain::BrainRegion;

// Results from a batch neuron update
struct UpdateResult {
    std::vector<uint32_t> spiked_neuron_ids;
    std::vector<double> spike_times;
};

// Abstract interface for neuron computation backends.
// Each BrainRegion holds a ComputeBackend pointer that can be swapped at runtime.
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // Update all neurons in a region for one timestep.
    // I_syn: synaptic currents indexed by local neuron index.
    // Returns IDs of neurons that spiked.
    virtual UpdateResult updateNeurons(BrainRegion& region, double dt,
                                       std::span<const double> I_syn) = 0;

    // Name for UI display
    virtual const char* name() const = 0;
};

enum class ComputeBackendType { CPUEventDriven, MetalGPU, HybridAuto };
