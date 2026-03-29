#pragma once

#include "Neuron.h"

namespace biobrain {

/// Leaky Integrate-and-Fire (LIF) neuron model.
/// tau_m * dV/dt = -(V - V_rest) + R * I
/// if V >= V_thresh: V = V_reset, refractory for tau_ref
class LIFNeuron : public Neuron {
public:
    LIFNeuron();

    bool   step(double dt, double I_syn) override;
    void   reset() override;
    double voltage() const override;
    double recoveryVariable() const override;

    // Parameters
    double tau_m   = 20.0;   // membrane time constant (ms)
    double V_rest  = -65.0;  // resting potential (mV)
    double V_thresh = -50.0; // spike threshold (mV)
    double V_reset = -65.0;  // reset potential (mV)
    double R       = 1.0;    // membrane resistance (MOhm) => R*I: MOhm * nA = mV
    double tau_ref = 2.0;    // refractory period (ms)

private:
    double V_ = -65.0;             // membrane potential (mV)
    double refractory_timer_ = 0.0; // time remaining in refractory period (ms)
};

} // namespace biobrain
