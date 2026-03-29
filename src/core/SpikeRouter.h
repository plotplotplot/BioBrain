#pragma once

#include "core/SpikeEvent.h"
#include <queue>
#include <vector>
#include <mutex>
#include <functional>
#include <limits>

namespace biobrain {

class SpikeRouter {
public:
    SpikeRouter() = default;

    /// Submit a spike event (thread-safe).
    void submitSpike(const SpikeEvent& event);

    /// Submit multiple spikes (thread-safe).
    void submitSpikes(const std::vector<SpikeEvent>& events);

    /// Get all events whose arrival time (time + delay) <= the given time.
    /// Returns them sorted by arrival time (earliest first).
    std::vector<SpikeEvent> getEventsUntil(double time);

    /// Peek at next event arrival time. Returns +infinity if empty.
    double nextEventTime() const;

    /// Check if the queue is empty.
    bool empty() const;

    /// Number of pending events.
    size_t pendingCount() const;

    /// Clear all pending events.
    void clear();

private:
    struct EventCompare {
        bool operator()(const SpikeEvent& a, const SpikeEvent& b) const {
            return (a.time + a.delay) > (b.time + b.delay); // min-heap
        }
    };

    std::priority_queue<SpikeEvent, std::vector<SpikeEvent>, EventCompare> queue_;
    mutable std::mutex mutex_;
};

} // namespace biobrain
