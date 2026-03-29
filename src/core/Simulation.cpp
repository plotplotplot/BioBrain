#include "core/Simulation.h"
#include "compute/ComputeBackend.h"
#include "recording/SpikeRecorder.h"
#include <algorithm>
#include <cmath>
#include <random>

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

    // Build synapse indices for O(1) lookup instead of O(N) scan
    for (auto& r : regions_) {
        r->buildSynapseIndex();
    }

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
    // ~60 Hz GUI update interval: ~16.67ms
    constexpr auto gui_interval = std::chrono::microseconds(16667);
    // Budget: spend at most 8ms per iteration to leave CPU for GUI
    constexpr auto max_step_budget = std::chrono::milliseconds(8);

    while (!stop_token.stop_requested() && running_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        auto iter_start = std::chrono::steady_clock::now();

        // Run as many substeps as we can within the time budget
        int steps_done = 0;
        while (std::chrono::steady_clock::now() - iter_start < max_step_budget) {
            stepSimulation();
            steps_done++;
            // Yield periodically to prevent starvation
            if (steps_done % 5 == 0) {
                std::this_thread::yield();
            }
        }

        // Always sleep at least 2ms to give GUI thread breathing room
        std::this_thread::sleep_for(std::chrono::milliseconds(2));

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

    // 2. Deliver spikes and track which neurons received input
    deliverSpikes(events);

    // 3. For each region: only update neurons that have pending synaptic input
    // This is the key optimization — skip regions/neurons with no activity
    for (auto& region : regions_) {
        auto* backend = region->computeBackend();
        if (!backend) continue;

        size_t n = region->neurons().size();

        // Only compute synaptic currents for neurons targeted by recent events
        std::vector<double> I_syn(n, 0.0);
        bool has_activity = false;

        // Add externally injected currents (persist across substeps,
        // decaying 10% per step to simulate a brief pulse)
        auto& injected = region->injectedCurrents();
        if (!injected.empty()) {
            for (size_t i = 0; i < std::min(n, injected.size()); ++i) {
                if (injected[i] != 0.0) {
                    I_syn[i] += injected[i];
                    has_activity = true;
                    injected[i] *= 0.9;  // decay (reaches ~35% after 10 substeps = 1ms)
                    if (std::abs(injected[i]) < 0.01) injected[i] = 0.0;
                }
            }
        }

        // Check active synapses (conductance > 0) — use the active set
        auto& synapses = region->internalSynapses();

        // Quick check: do we have any events targeting this region?
        bool region_has_events = false;
        for (const auto& ev : events) {
            if (ev.target_region == region->id()) {
                region_has_events = true;
                break;
            }
        }

        if (region_has_events || has_activity) {
            // Internal synapses: only scan if region has recent activity.
            // Skip entirely for quiet regions (saves scanning millions of synapses).
            uint32_t base = region->baseNeuronId();
            for (size_t si = 0; si < synapses.size(); ++si) {
                auto& syn = synapses[si];
                if (syn.conductance() <= 0.0) continue;
                uint32_t local_idx = syn.post_id - base;
                if (local_idx < n) {
                    double V_post = region->neurons()[local_idx]->voltage();
                    I_syn[local_idx] += syn.computeCurrent(V_post, DT);
                    has_activity = true;
                }
            }
        }

        // Inter-region synaptic current is handled via direct current injection
        // in deliverSpikes() (weight * 10 nA per spike). This avoids the O(800K)
        // scan of all projection synapses that was consuming 56.8% of sim time.

        // Minimal ion channel noise (~1 nA) — only for regions with injected
        // current or active synapses, to help neurons near threshold cross it.
        if (has_activity) {
            static thread_local std::mt19937 noise_rng(std::random_device{}());
            std::normal_distribution<double> noise(0.5, 1.0);
            for (size_t i = 0; i < n; ++i) {
                if (I_syn[i] > 0.0) {
                    I_syn[i] += noise(noise_rng);
                }
            }
        }

        // Update neurons via the compute backend
        UpdateResult result = backend->updateNeurons(*region, DT, I_syn);

        // 4. For spiked neurons: generate new spike events for outgoing synapses
        for (size_t i = 0; i < result.spiked_neuron_ids.size(); ++i) {
            uint32_t neuron_id = result.spiked_neuron_ids[i];
            double spike_time = result.spike_times[i];

            // Record spike time for firing rate stats
            region->recordSpikeTime(spike_time);

            // Internal synapses: use pre-synaptic index for O(1) lookup
            auto& region_synapses = region->internalSynapses();
            for (uint32_t si : region->getSynapsesForPreNeuron(neuron_id)) {
                auto& syn = region_synapses[si];
                SpikeEvent ev;
                ev.source_id = neuron_id;
                ev.target_id = syn.post_id;
                ev.time = spike_time;
                ev.delay = syn.delay;
                ev.source_region = region->id();
                ev.target_region = region->id();
                router_.submitSpike(ev);
            }

            // Inter-region projections (these are smaller, linear scan is OK)
            for (const auto& [target_region_id, proj_synapses] : region->projections()) {
                for (const auto& syn : proj_synapses) {
                    if (syn.pre_id == neuron_id) {
                        SpikeEvent ev;
                        ev.source_id = neuron_id;
                        ev.target_id = syn.post_id;
                        ev.time = spike_time;
                        ev.delay = syn.delay;
                        ev.source_region = region->id();
                        ev.target_region = target_region_id;
                        router_.submitSpike(ev);
                        debug_inter_region_events_.fetch_add(1, std::memory_order_relaxed);
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
        double arrival_time = ev.time + ev.delay;

        if (ev.source_region == ev.target_region) {
            // Internal spike: use pre-synaptic index
            BrainRegion* region = getRegion(ev.target_region);
            if (!region) continue;
            auto& synapses = region->internalSynapses();
            for (uint32_t si : region->getSynapsesForPreNeuron(ev.source_id)) {
                auto& syn = synapses[si];
                if (syn.post_id == ev.target_id) {
                    syn.deliverSpike(arrival_time);
                }
            }
        } else {
            // Inter-region spike: check source's projection to target
            BrainRegion* source = getRegion(ev.source_region);
            if (!source) continue;
            auto& proj = source->projections();
            auto it = proj.find(ev.target_region);
            debug_delivered_events_.fetch_add(1, std::memory_order_relaxed);
            if (it != proj.end()) {
                for (auto& syn : const_cast<std::vector<Synapse>&>(it->second)) {
                    if (syn.pre_id == ev.source_id && syn.post_id == ev.target_id) {
                        syn.deliverSpike(arrival_time);
                        debug_projection_matches_.fetch_add(1, std::memory_order_relaxed);

                        // Also inject direct current into target neuron
                        // (models strong thalamocortical / feedforward synapses)
                        BrainRegion* target = getRegion(ev.target_region);
                        if (target && ev.target_id < target->neurons().size()) {
                            target->injectCurrent(ev.target_id, syn.weight * 10.0);
                        }
                        break;
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
