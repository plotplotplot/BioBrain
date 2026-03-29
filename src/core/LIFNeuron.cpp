#include "LIFNeuron.h"

namespace biobrain {

LIFNeuron::LIFNeuron() {
    reset();
}

bool LIFNeuron::step(double dt, double I_syn) {
    // Handle refractory period
    if (refractory_timer_ > 0.0) {
        refractory_timer_ -= dt;
        V_ = V_reset;
        return false;
    }

    // tau_m * dV/dt = -(V - V_rest) + R * I
    double dV = (-(V_ - V_rest) + R * I_syn) / tau_m;
    V_ += dt * dV;

    // Spike detection
    if (V_ >= V_thresh) {
        V_ = V_reset;
        refractory_timer_ = tau_ref;
        return true;
    }
    return false;
}

void LIFNeuron::reset() {
    V_ = V_rest;
    refractory_timer_ = 0.0;
    last_spike_time = -1e9;
}

double LIFNeuron::voltage() const {
    return V_;
}

double LIFNeuron::recoveryVariable() const {
    // LIF has no recovery variable; return refractory timer as proxy
    return refractory_timer_;
}

} // namespace biobrain
