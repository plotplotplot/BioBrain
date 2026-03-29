#pragma once

#include "PlasticityRule.h"
#include <cmath>
#include <cstdint>

namespace biobrain {

/// Three-factor learning rule: STDP + Dopamine eligibility trace.
///
/// Correlated pre/post spike pairs create an eligibility trace instead of
/// directly modifying weights. Dopamine must arrive within the eligibility
/// window (~1s) to convert the trace into lasting weight changes.
///
/// de/dt = -e/tau_e + STDP(dt) * delta(t_spike)
/// dw/dt = (DA(t) - DA_tonic) * e(t)
///
/// This implements reward-modulated Hebbian learning: synapses that fired
/// together are "tagged" by the eligibility trace, and dopamine arriving
/// within ~1s converts those tags into lasting weight changes.
///
/// References:
///   Izhikevich (2007) "Solving the distal reward problem through linkage
///   of STDP and dopamine signaling", Cerebral Cortex.
struct DopamineSTDPParams {
    double A_plus    = 0.01;
    double A_minus   = 0.012;
    double tau_plus   = 20.0;   // ms
    double tau_minus  = 20.0;   // ms
    double tau_e     = 1000.0;  // eligibility trace decay time constant (ms)
    double DA_tonic  = 0.01;    // tonic dopamine baseline
    double w_min     = 0.0;
    double w_max     = 1.0;
};

class DopamineSTDP : public PlasticityRule {
public:
    using Params = DopamineSTDPParams;

    explicit DopamineSTDP(Params params = {});

    void   onPreSpike(double t, uint32_t synapse_id) override;
    void   onPostSpike(double t, uint32_t synapse_id) override;
    double computeWeightChange(double dt, double modulator_concentration) override;
    double eligibilityTrace() const override;
    void   reset() override;

    const Params& params() const { return params_; }

private:
    /// Compute raw STDP weight change for a given spike-time difference.
    double stdpWindow(double delta_t) const;

    Params params_;

    double last_pre_spike_  = -1e9;
    double last_post_spike_ = -1e9;
    double eligibility_     = 0.0;   // current eligibility trace value
    double last_update_t_   = 0.0;   // time of last eligibility decay update
};

} // namespace biobrain
