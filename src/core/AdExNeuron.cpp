#include "AdExNeuron.h"
#include <cmath>

namespace biobrain {

AdExNeuron::AdExNeuron() {
    reset();
}

bool AdExNeuron::step(double dt, double I_syn) {
    // All quantities in consistent sub-units:
    //   C in pF, g_L in nS, voltages in mV, w in pA, I_syn in pA
    //   nS * mV = pA, pA / pF = mV/ms -- everything is consistent.

    // Exponential term with overflow guard
    double exp_arg = (V_ - V_T) / Delta_T;
    double exp_term = (exp_arg < 50.0) ? Delta_T * std::exp(exp_arg) : Delta_T * std::exp(50.0);

    // dV/dt = (1/C) * [-g_L*(V-E_L) + g_L*Delta_T*exp((V-V_T)/Delta_T) - w + I]
    double dV = (-g_L * (V_ - E_L) + g_L * exp_term - w_ + I_syn) / C;

    // dw/dt = (1/tau_w) * [a*(V-E_L) - w]
    double dw = (a * (V_ - E_L) - w_) / tau_w;

    V_ += dt * dV;
    w_ += dt * dw;

    // Spike detection and reset
    if (V_ > V_cutoff) {
        V_ = V_reset;
        w_ += b;
        return true;
    }
    return false;
}

void AdExNeuron::reset() {
    V_ = E_L;
    w_ = 0.0;
    last_spike_time = -1e9;
}

double AdExNeuron::voltage() const {
    return V_;
}

double AdExNeuron::recoveryVariable() const {
    return w_;
}

} // namespace biobrain
