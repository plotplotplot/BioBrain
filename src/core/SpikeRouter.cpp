#include "core/SpikeRouter.h"
#include <limits>

namespace biobrain {

void SpikeRouter::submitSpike(const SpikeEvent& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(event);
}

void SpikeRouter::submitSpikes(const std::vector<SpikeEvent>& events) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& e : events) {
        queue_.push(e);
    }
}

std::vector<SpikeEvent> SpikeRouter::getEventsUntil(double time) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<SpikeEvent> result;
    while (!queue_.empty()) {
        const auto& top = queue_.top();
        if ((top.time + top.delay) <= time) {
            result.push_back(top);
            queue_.pop();
        } else {
            break;
        }
    }
    return result;
}

double SpikeRouter::nextEventTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const auto& top = queue_.top();
    return top.time + top.delay;
}

bool SpikeRouter::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

size_t SpikeRouter::pendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void SpikeRouter::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    // priority_queue has no clear(); swap with empty queue
    std::priority_queue<SpikeEvent, std::vector<SpikeEvent>, EventCompare> empty;
    std::swap(queue_, empty);
}

} // namespace biobrain
