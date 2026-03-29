#include "core/Simulation.h"
#include "compute/ComputeBackend.h"
#include "recording/SpikeRecorder.h"
#include <algorithm>
#include <cmath>

namespace biobrain {

Simulation::Simulation()
    : last_gui_update_(std::chrono::steady_clock::now()) {}

Simulation::~Simulation() {
    stop();
}

void Simulation::addRegion(std::shared_ptr<BrainRegion> region) {
    regions_.push_back(std::move(region));
}

BrainRegion* Simulation::getRegion(uint32_t id) const {
    for (const auto& r : regions_) {
        if (r->id() == id) {
            return r.get();
        }
    }
    return nullptr;
}

void Simulation::start() {
    if (running_.load()) return;

    running_.store(true);
    paused_.store(false);
    last_gui_update_ = std::chrono::steady_clock::now();

    sim_thread_ = std::jthread([this](std::stop_token token) {
        simulationLoop(token);
    });
}

void Simulation::pause() {
    paused_.store(true);
}

void Simulation::resume() {
    paused_.store(false);
}

void Simulation::stop() {
    running_.store(false);
    if (sim_thread_.joinable()) {
        sim_thread_.request_stop();
        sim_thread_.join();
    }
}

void Simulation::injectSpikes(const std::vector<SpikeEvent>& events) {
    router_.submitSpikes(events);
}

uint32_t Simulation::totalActiveNeurons() const {
    uint32_t total = 0;
    for (const auto& r : regions_) {
        total += r->activeNeuronCount();
    }
    return total;
}

double Simulation::spikesPerSecond() const {
    return static_cast<double>(spike_count_last_second_.load());
}

// ---------- Main simulation loop ----------

void Simulation::simulationLoop(std::stop_token stop_token) {
    constexpr int substeps_per_ms = 10; // 10 * 0.1ms = 1ms
    constexpr auto one_ms = std::chrono::microseconds(1000);
    // ~60 Hz GUI update interval: ~16.67ms
    constexpr auto gui_interval = std::chrono::microseconds(16667);

    while (!stop_token.stop_requested() && running_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto step_start = std::chrono::steady_clock::now();

        // Run 10 substeps = 1ms of simulation time
        for (int i = 0; i < substeps_per_ms; ++i) {
            stepSimulation();
        }

        // Real-time synchronization: 1ms sim should take ~1ms wall-clock
        auto step_end = std::chrono::steady_clock::now();
        auto elapsed = step_end - step_start;
        if (elapsed < one_ms) {
            std::this_thread::sleep_for(one_ms - elapsed);
        }
        // Note: if elapsed > one_ms, we're falling behind real-time
        // but we do NOT drop events -- just continue

        // GUI update at ~60Hz
        auto now = std::chrono::steady_clock::now();
        if (now - last_gui_update_ >= gui_interval) {
            flushGuiBatch();
            last_gui_update_ = now;
        }
    }

    // Flush any remaining GUI batches on shutdown
    flushGuiBatch();
}

void Simulation::stepSimulation() {
    double t = sim_time_.load();

    // 1. Get all events that should be delivered at or before current time
    auto events = router_.getEventsUntil(t);

    // 2. Deliver spikes to target neurons (updates synaptic conductances)
    deliverSpikes(events);

    // 3. For each region: compute synaptic currents and update neurons
    for (auto& region : regions_) {
        auto* backend = region->computeBackend();
        if (!backend) continue;

        size_t n = region->neurons().size();
        std::vector<double> I_syn(n, 0.0);

        // Accumulate synaptic currents from internal synapses
        for (auto& syn : region->internalSynapses()) {
            // Compute local neuron index from post_id
            uint32_t local_idx = syn.post_id - region->baseNeuronId();
            if (local_idx < n) {
                double V_post = region->neurons()[local_idx]->voltage();
                I_syn[local_idx] += syn.computeCurrent(V_post, DT);
            }
        }

        // Update all neurons via the compute backend
        UpdateResult result = backend->updateNeurons(*region, DT, I_syn);

        // 4. For spiked neurons: generate new spike events for outgoing synapses
        for (size_t i = 0; i < result.spiked_neuron_ids.size(); ++i) {
            uint32_t neuron_id = result.spiked_neuron_ids[i];
            double spike_time = result.spike_times[i];

            // Record spike time for firing rate stats
            region->recordSpikeTime(spike_time);

            // Internal synapses: create events for post-synaptic targets
            for (const auto& syn : region->internalSynapses()) {
                if (syn.pre_id == neuron_id) {
                    SpikeEvent ev;
                    ev.source_id = neuron_id;
                    ev.target_id = syn.post_id;
                    ev.time = spike_time;
                    ev.delay = syn.delay;
                    ev.source_region = region->id();
                    ev.target_region = region->id();
                    router_.submitSpike(ev);
                }
            }

            // Inter-region projections
            for (const auto& [target_region_id, synapses] : region->projections()) {
                for (const auto& syn : synapses) {
                    if (syn.pre_id == neuron_id) {
                        SpikeEvent ev;
                        ev.source_id = neuron_id;
                        ev.target_id = syn.post_id;
                        ev.time = spike_time;
                        ev.delay = syn.delay;
                        ev.source_region = region->id();
                        ev.target_region = target_region_id;
                        router_.submitSpike(ev);
                    }
                }
            }

            // Batch for GUI callback
            bool found = false;
            for (auto& batch : gui_batch_) {
                if (batch.region_id == region->id()) {
                    batch.neuron_ids.push_back(neuron_id);
                    batch.times.push_back(spike_time);
                    found = true;
                    break;
                }
            }
            if (!found) {
                gui_batch_.push_back({region->id(), {neuron_id}, {spike_time}});
            }

            // Record to SpikeRecorder
            if (recorder_) {
                recorder_->recordSpike(neuron_id, region->id(), spike_time);
            }

            spike_count_current_.fetch_add(1, std::memory_order_relaxed);
        }

        region->setCurrentTime(t + DT);
    }

    // 5. Advance simulation time
    double new_time = t + DT;
    sim_time_.store(new_time);

    // Update spikes-per-second counter every 1000ms
    if (new_time - last_rate_update_time_ >= 1000.0) {
        spike_count_last_second_.store(spike_count_current_.load());
        spike_count_current_.store(0);
        last_rate_update_time_ = new_time;
    }
}

void Simulation::deliverSpikes(const std::vector<SpikeEvent>& events) {
    for (const auto& ev : events) {
        // Find the target region
        BrainRegion* target = getRegion(ev.target_region);
        if (!target) continue;

        // Deliver the spike to matching synapses within the target region
        double arrival_time = ev.time + ev.delay;

        // Check internal synapses of the target region
        for (auto& syn : target->internalSynapses()) {
            if (syn.pre_id == ev.source_id && syn.post_id == ev.target_id) {
                syn.deliverSpike(arrival_time);
            }
        }

        // Check incoming projections -- the synapse lives in the source region's
        // projections map, but the conductance state we need is there too.
        BrainRegion* source = getRegion(ev.source_region);
        if (source && source->id() != target->id()) {
            auto& proj = source->projections();
            auto it = proj.find(ev.target_region);
            if (it != proj.end()) {
                for (auto& syn : const_cast<std::vector<Synapse>&>(it->second)) {
                    if (syn.pre_id == ev.source_id && syn.post_id == ev.target_id) {
                        syn.deliverSpike(arrival_time);
                    }
                }
            }
        }
    }
}

void Simulation::flushGuiBatch() {
    if (!spike_callback_ || gui_batch_.empty()) return;

    for (const auto& batch : gui_batch_) {
        spike_callback_(batch.region_id, batch.neuron_ids, batch.times);
    }
    gui_batch_.clear();
}

} // namespace biobrain
