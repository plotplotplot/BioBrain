#pragma once

#include "PlasticityRule.h"
#include <cmath>
#include <cstdint>

namespace biobrain {

/// Standard Spike-Timing-Dependent Plasticity (Bi & Poo, 1998).
///
/// Pre-before-post (causal, dt > 0):  dw = +A_plus  * exp(-dt / tau_plus)   [LTP]
/// Post-before-pre (anticausal, dt < 0): dw = -A_minus * exp( dt / tau_minus)  [LTD]
///
/// Slightly asymmetric: A_minus > A_plus so LTD dominates on average,
/// consistent with experimental findings (Bi & Poo 1998, J. Neuroscience).
class STDP : public PlasticityRule {
public:
    struct Params {
        double A_plus   = 0.01;    // LTP amplitude
        double A_minus  = 0.012;   // LTD amplitude (slightly larger → LTD-dominant)
        double tau_plus  = 20.0;   // LTP time constant (ms)
        double tau_minus = 20.0;   // LTD time constant (ms)
        double w_min     = 0.0;    // minimum synaptic weight
        double w_max     = 1.0;    // maximum synaptic weight
    };

    explicit STDP(Params params = {});

    void   onPreSpike(double t, uint32_t synapse_id) override;
    void   onPostSpike(double t, uint32_t synapse_id) override;
    double computeWeightChange(double dt, double modulator_concentration) override;
    double eligibilityTrace() const override;
    void   reset() override;

    const Params& params() const { return params_; }

private:
    Params params_;

    double last_pre_spike_  = -1e9;  // ms
    double last_post_spike_ = -1e9;  // ms
    double accumulated_dw_  = 0.0;   // accumulated weight change since last read
};

} // namespace biobrain
