#include "HodgkinHuxleyNeuron.h"
#include <cmath>
#include <algorithm>

namespace biobrain {

HodgkinHuxleyNeuron::HodgkinHuxleyNeuron() {
    reset();
}

// --- Rate functions (Hodgkin & Huxley 1952) ---

double HodgkinHuxleyNeuron::alpha_m(double V) {
    double dV = V + 40.0;
    if (std::abs(dV) < 1e-7) return 1.0;  // L'Hopital limit
    return 0.1 * dV / (1.0 - std::exp(-dV / 10.0));
}

double HodgkinHuxleyNeuron::beta_m(double V) {
    return 4.0 * std::exp(-(V + 65.0) / 18.0);
}

double HodgkinHuxleyNeuron::alpha_h(double V) {
    return 0.07 * std::exp(-(V + 65.0) / 20.0);
}

double HodgkinHuxleyNeuron::beta_h(double V) {
    return 1.0 / (1.0 + std::exp(-(V + 35.0) / 10.0));
}

double HodgkinHuxleyNeuron::alpha_n(double V) {
    double dV = V + 55.0;
    if (std::abs(dV) < 1e-7) return 0.1;  // L'Hopital limit
    return 0.01 * dV / (1.0 - std::exp(-dV / 10.0));
}

double HodgkinHuxleyNeuron::beta_n(double V) {
    return 0.125 * std::exp(-(V + 65.0) / 80.0);
}

double HodgkinHuxleyNeuron::m_inf(double V) {
    double am = alpha_m(V);
    return am / (am + beta_m(V));
}

double HodgkinHuxleyNeuron::h_inf(double V) {
    double ah = alpha_h(V);
    return ah / (ah + beta_h(V));
}

double HodgkinHuxleyNeuron::n_inf(double V) {
    double an = alpha_n(V);
    return an / (an + beta_n(V));
}

bool HodgkinHuxleyNeuron::step(double dt, double I_syn) {
    // Gating variable derivatives
    double dm = alpha_m(V_) * (1.0 - m_) - beta_m(V_) * m_;
    double dh = alpha_h(V_) * (1.0 - h_) - beta_h(V_) * h_;
    double dn = alpha_n(V_) * (1.0 - n_) - beta_n(V_) * n_;

    // Ionic currents
    double m3h  = m_ * m_ * m_ * h_;
    double n4   = n_ * n_ * n_ * n_;
    double I_Na = g_Na * m3h * (V_ - E_Na);
    double I_K  = g_K  * n4  * (V_ - E_K);
    double I_L  = g_L         * (V_ - E_L);

    // Membrane potential derivative: C_m * dV/dt = -I_Na - I_K - I_L + I_ext
    double dV = (-I_Na - I_K - I_L + I_syn) / C_m;

    // Euler integration
    V_ += dt * dV;
    m_ += dt * dm;
    h_ += dt * dh;
    n_ += dt * dn;

    // Clamp gating variables to [0, 1]
    m_ = std::clamp(m_, 0.0, 1.0);
    h_ = std::clamp(h_, 0.0, 1.0);
    n_ = std::clamp(n_, 0.0, 1.0);

    // Spike detection: upward crossing of 0 mV threshold
    bool currently_above = (V_ >= 0.0);
    bool spiked = currently_above && prev_below_threshold_;
    prev_below_threshold_ = !currently_above;

    return spiked;
}

void HodgkinHuxleyNeuron::reset() {
    V_ = -65.0;
    m_ = m_inf(V_);
    h_ = h_inf(V_);
    n_ = n_inf(V_);
    prev_below_threshold_ = true;
    last_spike_time = -1e9;
}

double HodgkinHuxleyNeuron::voltage() const {
    return V_;
}

double HodgkinHuxleyNeuron::recoveryVariable() const {
    // Return the sodium inactivation variable h as the "recovery" analogue
    return h_;
}

} // namespace biobrain
