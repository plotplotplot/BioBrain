#pragma once

#include <cstdint>

namespace biobrain {

class Neuron {
public:
    virtual ~Neuron() = default;

    /// Advance neuron state by dt milliseconds with synaptic input current I_syn.
    /// Returns true if the neuron fired a spike during this step.
    virtual bool step(double dt, double I_syn) = 0;

    /// Reset neuron to resting state.
    virtual void reset() = 0;

    /// Current membrane voltage (mV).
    virtual double voltage() const = 0;

    /// Recovery / adaptation variable (model-dependent units).
    virtual double recoveryVariable() const = 0;

    uint32_t id              = 0;
    double   last_spike_time = -1e9;  // ms
};

} // namespace biobrain
