#include "compute/CPUBackend.h"
#include "core/BrainRegion.h"
#include "core/Neuron.h"
#include <future>
#include <numeric>

CPUBackend::CPUBackend(int num_threads)
    : num_threads_(num_threads) {}

UpdateResult CPUBackend::updateNeurons(BrainRegion& region, double dt,
                                        std::span<const double> I_syn) {
    auto& neurons = region.neurons();
    size_t count = neurons.size();

    if (count == 0) return {};

    // Partition neurons across threads
    size_t chunk = (count + num_threads_ - 1) / num_threads_;

    std::vector<std::future<std::vector<std::pair<uint32_t, double>>>> futures;
    futures.reserve(num_threads_);

    double sim_time = region.currentTime();

    for (int t = 0; t < num_threads_; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, count);
        if (start >= end) break;

        futures.push_back(std::async(std::launch::async,
            [&neurons, &I_syn, start, end, dt, sim_time]() {
                std::vector<std::pair<uint32_t, double>> spikes;
                for (size_t i = start; i < end; ++i) {
                    double current = (i < I_syn.size()) ? I_syn[i] : 0.0;
                    bool spiked = neurons[i]->step(dt, current);
                    if (spiked) {
                        neurons[i]->last_spike_time = sim_time;
                        spikes.emplace_back(neurons[i]->id, sim_time);
                    }
                }
                return spikes;
            }
        ));
    }

    // Collect results
    UpdateResult result;
    for (auto& f : futures) {
        auto spikes = f.get();
        for (auto& [id, time] : spikes) {
            result.spiked_neuron_ids.push_back(id);
            result.spike_times.push_back(time);
        }
    }

    return result;
}
