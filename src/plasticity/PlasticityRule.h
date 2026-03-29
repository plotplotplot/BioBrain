#pragma once

#include <cstdint>

namespace biobrain {

/// Interface for synaptic plasticity rules.
/// Attached per-synapse (or shared per-region). The simulation engine calls
/// onPreSpike/onPostSpike when neurons fire and computeWeightChange each timestep.
class PlasticityRule {
public:
    virtual ~PlasticityRule() = default;

    /// Called when the presynaptic neuron fires.
    virtual void onPreSpike(double t, uint32_t synapse_id) = 0;

    /// Called when the postsynaptic neuron fires.
    virtual void onPostSpike(double t, uint32_t synapse_id) = 0;

    /// Called each timestep to compute weight change based on neuromodulatory signals.
    /// @param dt          timestep duration (ms)
    /// @param modulator_concentration  primary modulator (dopamine) concentration
    /// @return weight change to apply this timestep
    virtual double computeWeightChange(double dt, double modulator_concentration) = 0;

    /// Get the current eligibility trace value.
    virtual double eligibilityTrace() const = 0;

    /// Reset all internal state.
    virtual void reset() = 0;
};

enum class PlasticityType {
    None,
    STDP,
    DopamineSTDP,
    FullNeuromodulatory
};

} // namespace biobrain
