#pragma once

#include <cstdint>

namespace biobrain {

struct SpikeEvent {
    uint32_t source_id      = 0;
    uint32_t target_id      = 0;
    double   time           = 0.0;   // ms
    double   delay          = 0.0;   // ms
    uint32_t source_region  = 0;
    uint32_t target_region  = 0;

    // Priority queue ordering: earliest arrival first
    bool operator>(const SpikeEvent& other) const {
        return (time + delay) > (other.time + other.delay);
    }
};

} // namespace biobrain
