#pragma once

#include "Neuron.h"

namespace biobrain {

/// Full Hodgkin-Huxley (1952) conductance-based neuron model.
/// C_m * dV/dt = -g_Na*m^3*h*(V-E_Na) - g_K*n^4*(V-E_K) - g_L*(V-E_L) + I
class HodgkinHuxleyNeuron : public Neuron {
public:
    HodgkinHuxleyNeuron();

    bool   step(double dt, double I_syn) override;
    void   reset() override;
    double voltage() const override;
    double recoveryVariable() const override;

    // Maximal conductances (mS/cm^2, but we treat voltage in mV and current in uA/cm^2)
    double g_Na = 120.0;    // sodium
    double g_K  = 36.0;     // potassium
    double g_L  = 0.3;      // leak

    // Reversal potentials (mV)
    double E_Na = 50.0;
    double E_K  = -77.0;
    double E_L  = -54.387;

    // Membrane capacitance (uF/cm^2)
    double C_m  = 1.0;

private:
    double V_ = -65.0;  // membrane potential (mV)
    double m_ = 0.0;    // sodium activation gating variable
    double h_ = 0.0;    // sodium inactivation gating variable
    double n_ = 0.0;    // potassium activation gating variable

    bool prev_below_threshold_ = true;  // for upward zero-crossing detection

    // Rate functions (Hodgkin & Huxley 1952)
    static double alpha_m(double V);
    static double beta_m(double V);
    static double alpha_h(double V);
    static double beta_h(double V);
    static double alpha_n(double V);
    static double beta_n(double V);

    // Steady-state gating variable
    static double m_inf(double V);
    static double h_inf(double V);
    static double n_inf(double V);
};

} // namespace biobrain
