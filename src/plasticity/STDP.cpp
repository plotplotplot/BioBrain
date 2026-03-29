#include "STDP.h"
#include <algorithm>
#include <cmath>

namespace biobrain {

STDP::STDP(Params params)
    : params_(params)
{}

void STDP::onPreSpike(double t, uint32_t /*synapse_id*/) {
    // Pre spike arrived — check timing relative to last post spike (LTD window).
    // Post-before-pre: dt = t_pre - t_post > 0, but in STDP convention this is
    // the anticausal case where dt_effective = t_post - t_pre < 0 → LTD.
    double dt = last_post_spike_ - t;  // negative when post was before pre
    if (last_post_spike_ > -1e8) {
        // dt < 0 → post before pre → LTD
        accumulated_dw_ += -params_.A_minus * std::exp(dt / params_.tau_minus);
    }
    last_pre_spike_ = t;
}

void STDP::onPostSpike(double t, uint32_t /*synapse_id*/) {
    // Post spike arrived — check timing relative to last pre spike (LTP window).
    double dt = t - last_pre_spike_;  // positive when pre was before post
    if (last_pre_spike_ > -1e8) {
        // dt > 0 → pre before post → LTP
        accumulated_dw_ += params_.A_plus * std::exp(-dt / params_.tau_plus);
    }
    last_post_spike_ = t;
}

double STDP::computeWeightChange(double /*dt*/, double /*modulator_concentration*/) {
    // Standard STDP ignores modulators — return accumulated dw directly.
    double dw = accumulated_dw_;
    accumulated_dw_ = 0.0;
    return dw;
}

double STDP::eligibilityTrace() const {
    // Standard STDP has no eligibility trace; return accumulated change magnitude.
    return accumulated_dw_;
}

void STDP::reset() {
    last_pre_spike_  = -1e9;
    last_post_spike_ = -1e9;
    accumulated_dw_  = 0.0;
}

} // namespace biobrain
