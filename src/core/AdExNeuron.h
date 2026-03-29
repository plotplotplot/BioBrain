#pragma once

#include "Neuron.h"

namespace biobrain {

/// Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.
/// Brette & Gerstner (2005).
/// C*dV/dt = -g_L*(V-E_L) + g_L*Delta_T*exp((V-V_T)/Delta_T) - w + I
/// tau_w*dw/dt = a*(V-E_L) - w
/// if V > V_cutoff: V = V_reset, w += b
class AdExNeuron : public Neuron {
public:
    AdExNeuron();

    bool   step(double dt, double I_syn) override;
    void   reset() override;
    double voltage() const override;
    double recoveryVariable() const override;

    // Parameters (Brette & Gerstner 2005, default values)
    // Units: pF, nS, mV, pA, ms -- all mutually consistent
    double C        = 281.0;    // membrane capacitance (pF)
    double g_L      = 30.0;     // leak conductance (nS)
    double E_L      = -70.6;    // leak reversal potential (mV)
    double V_T      = -50.4;    // spike threshold parameter (mV)
    double Delta_T  = 2.0;      // slope factor (mV)
    double a        = 4.0;      // subthreshold adaptation (nS)
    double b        = 80.5;     // spike-triggered adaptation (pA) [= 0.0805 nA]
    double tau_w    = 144.0;    // adaptation time constant (ms)
    double V_reset  = -70.6;    // reset potential (mV)
    double V_cutoff = -30.0;    // spike detection cutoff (mV)

private:
    double V_ = -70.6;  // membrane potential (mV)
    double w_ = 0.0;    // adaptation current (pA)
};

} // namespace biobrain
