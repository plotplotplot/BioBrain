#include "NeuromodulatorySTDP.h"
#include <algorithm>
#include <cmath>

namespace biobrain {

NeuromodulatorySTDP::NeuromodulatorySTDP(Params params)
    : params_(params)
{}

void NeuromodulatorySTDP::setModulator(ModulatorType type, double concentration) {
    modulators_[static_cast<int>(type)] = concentration;
}

double NeuromodulatorySTDP::getModulator(ModulatorType type) const {
    return modulators_[static_cast<int>(type)];
}

double NeuromodulatorySTDP::effectiveTauE() const {
    double ne = modulators_[static_cast<int>(ModulatorType::Norepinephrine)];
    return params_.tau_e * (1.0 + ne);
}

double NeuromodulatorySTDP::stdpWindow(double delta_t) const {
    double ach = modulators_[static_cast<int>(ModulatorType::Acetylcholine)];
    double serotonin = modulators_[static_cast<int>(ModulatorType::Serotonin)];

    // ACh modulates overall learning rate (both LTP and LTD amplitudes).
    double ach_factor = 1.0 + ach;
    double a_plus_eff  = params_.A_plus  * ach_factor;
    double a_minus_eff = params_.A_minus * ach_factor;

    if (delta_t > 0.0) {
        // Pre before post → LTP
        return a_plus_eff * std::exp(-delta_t / params_.tau_plus);
    } else if (delta_t < 0.0) {
        // Post before pre → LTD, scaled by serotonin.
        double ltd = -a_minus_eff * std::exp(delta_t / params_.tau_minus);
        ltd *= (1.0 + serotonin);  // 5-HT increases depression
        return ltd;
    }
    return 0.0;
}

void NeuromodulatorySTDP::onPreSpike(double t, uint32_t /*synapse_id*/) {
    // Decay eligibility trace to current time using NE-modulated tau.
    double tau_eff = effectiveTauE();
    double elapsed = t - last_update_t_;
    if (elapsed > 0.0) {
        eligibility_ *= std::exp(-elapsed / tau_eff);
        last_update_t_ = t;
    }

    // Post-before-pre → LTD contribution to eligibility trace.
    if (last_post_spike_ > -1e8) {
        double delta_t = last_post_spike_ - t;  // negative
        eligibility_ += stdpWindow(delta_t);
    }

    last_pre_spike_ = t;
}

void NeuromodulatorySTDP::onPostSpike(double t, uint32_t /*synapse_id*/) {
    // Decay eligibility trace to current time using NE-modulated tau.
    double tau_eff = effectiveTauE();
    double elapsed = t - last_update_t_;
    if (elapsed > 0.0) {
        eligibility_ *= std::exp(-elapsed / tau_eff);
        last_update_t_ = t;
    }

    // Pre-before-post → LTP contribution to eligibility trace.
    if (last_pre_spike_ > -1e8) {
        double delta_t = t - last_pre_spike_;  // positive
        eligibility_ += stdpWindow(delta_t);
    }

    last_post_spike_ = t;
}

double NeuromodulatorySTDP::computeWeightChange(double dt, double modulator_concentration) {
    // Update dopamine from the primary modulator parameter.
    modulators_[static_cast<int>(ModulatorType::Dopamine)] = modulator_concentration;

    // Decay eligibility trace over this timestep using NE-modulated tau.
    double tau_eff = effectiveTauE();
    eligibility_ *= std::exp(-dt / tau_eff);
    last_update_t_ += dt;

    // Weight change: dopamine modulates the eligibility trace.
    double da_signal = modulator_concentration - params_.DA_tonic;
    return da_signal * eligibility_ * dt;
}

double NeuromodulatorySTDP::eligibilityTrace() const {
    return eligibility_;
}

void NeuromodulatorySTDP::reset() {
    last_pre_spike_  = -1e9;
    last_post_spike_ = -1e9;
    eligibility_     = 0.0;
    last_update_t_   = 0.0;
    modulators_.fill(0.0);
}

} // namespace biobrain
