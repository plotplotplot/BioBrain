#pragma once

#include "Neuron.h"
#include <memory>

namespace biobrain {

/// Izhikevich (2003) simple model of spiking neurons.
/// v' = 0.04v^2 + 5v + 140 - u + I
/// u' = a(bv - u)
/// if v >= 30 mV: v = c, u += d
class IzhikevichNeuron : public Neuron {
public:
    enum class Type {
        RegularSpiking,
        FastSpiking,
        IntrinsicallyBursting,
        LowThresholdSpiking,
        TonicSpiking,
        MediumSpinyD1,
        MediumSpinyD2,
        Dopaminergic
    };

    /// Factory: create a neuron with biologically-grounded parameter presets.
    static std::unique_ptr<IzhikevichNeuron> create(Type type);

    IzhikevichNeuron(double a, double b, double c, double d);

    bool   step(double dt, double I_syn) override;
    void   reset() override;
    double voltage() const override;
    double recoveryVariable() const override;

    // Parameters (Izhikevich 2003, Table 1)
    double a = 0.02;   // time scale of recovery variable u
    double b = 0.2;    // sensitivity of u to subthreshold fluctuations of v
    double c = -65.0;  // after-spike reset value of v (mV)
    double d = 8.0;    // after-spike reset increment of u

private:
    double v_ = -65.0; // membrane potential (mV)
    double u_ = 0.0;   // recovery variable

    static constexpr double kSpikeThreshold = 30.0; // mV
};

} // namespace biobrain
