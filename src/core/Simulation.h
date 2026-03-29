#pragma once

#include "core/SpikeRouter.h"
#include "core/BrainRegion.h"
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>

class ComputeBackend;
class SpikeRecorder;

namespace biobrain {

class Simulation {
public:
    Simulation();
    ~Simulation();

    // --- Region management ---
    void addRegion(std::shared_ptr<BrainRegion> region);
    BrainRegion* getRegion(uint32_t id) const;
    const std::vector<std::shared_ptr<BrainRegion>>& regions() const { return regions_; }

    // --- Simulation control ---
    void start();
    void pause();
    void resume();
    void stop();

    bool isRunning() const { return running_.load(); }
    bool isPaused() const { return paused_.load(); }

    /// Current simulation time (ms).
    double currentTime() const { return sim_time_.load(); }

    /// Real-time scaling factor (1.0 = real-time).
    double timeScale() const { return 1.0; }

    /// Inject external spikes (e.g., from retinal encoder).
    void injectSpikes(const std::vector<SpikeEvent>& events);

    /// Spike callback for GUI updates (called at ~60Hz with batched spikes).
    using SpikeCallback = std::function<void(uint32_t region_id,
                                              const std::vector<uint32_t>& neuron_ids,
                                              const std::vector<double>& times)>;
    void setSpikeCallback(SpikeCallback cb) { spike_callback_ = std::move(cb); }

    /// Set a spike recorder for HDF5 output.
    void setRecorder(std::shared_ptr<SpikeRecorder> recorder) { recorder_ = recorder; }

    // --- Statistics ---
    uint32_t totalActiveNeurons() const;
    double spikesPerSecond() const;

    /// Simulation timestep (ms): 0.1ms substep.
    static constexpr double DT = 0.1;

private:
    std::vector<std::shared_ptr<BrainRegion>> regions_;
    SpikeRouter router_;

    std::atomic<double> sim_time_{0.0};
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};

    std::jthread sim_thread_;

    SpikeCallback spike_callback_;
    std::shared_ptr<SpikeRecorder> recorder_;

    // Spike batching for GUI callback
    struct SpikeBatch {
        uint32_t region_id;
        std::vector<uint32_t> neuron_ids;
        std::vector<double> times;
    };
    std::vector<SpikeBatch> gui_batch_;
    std::chrono::steady_clock::time_point last_gui_update_;

    // Total spikes in current second (for spikesPerSecond)
    std::atomic<uint64_t> spike_count_current_{0};
    std::atomic<uint64_t> spike_count_last_second_{0};
    double last_rate_update_time_ = 0.0;

public:
    // Debug counters
    std::atomic<uint64_t> debug_inter_region_events_{0};
    std::atomic<uint64_t> debug_delivered_events_{0};
    std::atomic<uint64_t> debug_projection_matches_{0};
private:

    // --- Internal methods ---
    void simulationLoop(std::stop_token stop_token);
    void stepSimulation();
    void deliverSpikes(const std::vector<SpikeEvent>& events);
    void flushGuiBatch();
};

} // namespace biobrain
