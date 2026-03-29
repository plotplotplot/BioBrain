#pragma once

#include "harness/RestHarness.h"
#include <memory>
#include <chrono>
#include <mutex>
#include <deque>
#include <atomic>

namespace biobrain { class Simulation; class BrainRegion; }
class WebcamCapture;
class MainWindow;

// Debug REST API that attaches to a running BioBrain simulation.
// Provides profiling, screenshots, real-time inspection, and debugging.
// Start alongside the GUI app on port 9090.
class DebugAPI {
public:
    DebugAPI(std::shared_ptr<biobrain::Simulation> sim,
             WebcamCapture* webcam,
             MainWindow* window,
             int port = 9090);
    ~DebugAPI();

    void start();
    void stop();

    // Call once per simulation timestep for profiling
    void recordStepTiming(double step_ms);

    // Record spike batch for activity logging
    void recordSpikeBatch(uint32_t region_id, uint32_t count, double sim_time);

private:
    void registerRoutes();

    // --- Profiling ---
    struct TimingSample {
        double step_duration_ms;
        double sim_time;
        std::chrono::steady_clock::time_point wall_time;
    };
    std::mutex timing_mutex_;
    std::deque<TimingSample> timing_samples_;
    static constexpr size_t MAX_TIMING_SAMPLES = 5000;

    // --- Spike activity log ---
    struct SpikeBatch {
        uint32_t region_id;
        uint32_t spike_count;
        double sim_time;
    };
    std::mutex spike_mutex_;
    std::deque<SpikeBatch> spike_log_;
    static constexpr size_t MAX_SPIKE_LOG = 20000;

    // --- Screenshot ---
    std::string captureScreenshotBase64();

    // --- State ---
    std::shared_ptr<biobrain::Simulation> sim_;
    WebcamCapture* webcam_;
    MainWindow* window_;
    RestHarness server_;
    std::chrono::steady_clock::time_point start_time_;
};
