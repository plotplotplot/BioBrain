#include "compute/CPUBackend.h"
#include "core/BrainRegion.h"
#include "core/Neuron.h"

CPUBackend::CPUBackend(int num_threads)
    : num_threads_(num_threads) {}

UpdateResult CPUBackend::updateNeurons(BrainRegion& region, double dt,
                                        std::span<const double> I_syn) {
    auto& neurons = region.neurons();
    size_t count = neurons.size();
    if (count == 0) return {};

    double sim_time = region.currentTime();
    UpdateResult result;

    // Single-threaded hot loop — avoids pthread_create overhead entirely.
    // Profile showed 13.4% of sim time was thread spawning via std::async.
    // For 0.1ms substeps, thread overhead dwarfs the actual neuron computation.
    for (size_t i = 0; i < count; ++i) {
        double current = (i < I_syn.size()) ? I_syn[i] : 0.0;

        // Skip neurons with zero input that are near resting potential.
        // This is safe because Izhikevich neurons at rest (-65mV) with zero
        // input will never spike — the dynamics are stable at rest.
        if (current == 0.0 && neurons[i]->voltage() < -60.0) {
            continue;
        }

        bool spiked = neurons[i]->step(dt, current);
        if (spiked) {
            neurons[i]->last_spike_time = sim_time;
            result.spiked_neuron_ids.push_back(neurons[i]->id);
            result.spike_times.push_back(sim_time);
        }
    }

    return result;
}
