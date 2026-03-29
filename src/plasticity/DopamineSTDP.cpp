#include "DopamineSTDP.h"
#include <algorithm>
#include <cmath>

namespace biobrain {

DopamineSTDP::DopamineSTDP(Params params)
    : params_(params)
{}

double DopamineSTDP::stdpWindow(double delta_t) const {
    if (delta_t > 0.0) {
        // Pre before post → LTP
        return params_.A_plus * std::exp(-delta_t / params_.tau_plus);
    } else if (delta_t < 0.0) {
        // Post before pre → LTD
        return -params_.A_minus * std::exp(delta_t / params_.tau_minus);
    }
    return 0.0;
}

void DopamineSTDP::onPreSpike(double t, uint32_t /*synapse_id*/) {
    // Decay eligibility trace to current time before adding new contribution.
    double elapsed = t - last_update_t_;
    if (elapsed > 0.0) {
        eligibility_ *= std::exp(-elapsed / params_.tau_e);
        last_update_t_ = t;
    }

    // Post-before-pre → LTD contribution to eligibility trace.
    if (last_post_spike_ > -1e8) {
        double delta_t = last_post_spike_ - t;  // negative
        eligibility_ += stdpWindow(delta_t);
    }

    last_pre_spike_ = t;
}

void DopamineSTDP::onPostSpike(double t, uint32_t /*synapse_id*/) {
    // Decay eligibility trace to current time.
    double elapsed = t - last_update_t_;
    if (elapsed > 0.0) {
        eligibility_ *= std::exp(-elapsed / params_.tau_e);
        last_update_t_ = t;
    }

    // Pre-before-post → LTP contribution to eligibility trace.
    if (last_pre_spike_ > -1e8) {
        double delta_t = t - last_pre_spike_;  // positive
        eligibility_ += stdpWindow(delta_t);
    }

    last_post_spike_ = t;
}

double DopamineSTDP::computeWeightChange(double dt, double modulator_concentration) {
    // Decay eligibility trace over this timestep.
    eligibility_ *= std::exp(-dt / params_.tau_e);
    last_update_t_ += dt;

    // Weight change: dopamine modulates the eligibility trace.
    // Subtracting tonic baseline means no net change at baseline DA levels.
    double da_signal = modulator_concentration - params_.DA_tonic;
    return da_signal * eligibility_ * dt;
}

double DopamineSTDP::eligibilityTrace() const {
    return eligibility_;
}

void DopamineSTDP::reset() {
    last_pre_spike_  = -1e9;
    last_post_spike_ = -1e9;
    eligibility_     = 0.0;
    last_update_t_   = 0.0;
}

} // namespace biobrain
