#include "IzhikevichNeuron.h"

namespace biobrain {

std::unique_ptr<IzhikevichNeuron> IzhikevichNeuron::create(Type type) {
    // Parameter presets from Izhikevich (2003)
    switch (type) {
        case Type::RegularSpiking:
            return std::make_unique<IzhikevichNeuron>(0.02, 0.2, -65.0, 8.0);
        case Type::FastSpiking:
            return std::make_unique<IzhikevichNeuron>(0.1, 0.2, -65.0, 2.0);
        case Type::IntrinsicallyBursting:
            return std::make_unique<IzhikevichNeuron>(0.02, 0.2, -55.0, 4.0);
        case Type::LowThresholdSpiking:
            return std::make_unique<IzhikevichNeuron>(0.02, 0.25, -65.0, 2.0);
        case Type::TonicSpiking:
            return std::make_unique<IzhikevichNeuron>(0.02, 0.2, -65.0, 6.0);
        case Type::MediumSpinyD1:
            return std::make_unique<IzhikevichNeuron>(0.02, 0.2, -80.0, 8.0);
        case Type::MediumSpinyD2:
            return std::make_unique<IzhikevichNeuron>(0.02, 0.2, -80.0, 8.0);
        case Type::Dopaminergic:
            return std::make_unique<IzhikevichNeuron>(0.02, 0.2, -50.0, 2.0);
    }
    // Fallback: regular spiking
    return std::make_unique<IzhikevichNeuron>(0.02, 0.2, -65.0, 8.0);
}

IzhikevichNeuron::IzhikevichNeuron(double a_, double b_, double c_, double d_)
    : a(a_), b(b_), c(c_), d(d_)
{
    v_ = c;
    u_ = b * v_;
}

bool IzhikevichNeuron::step(double dt, double I_syn) {
    // Euler integration (Izhikevich recommends splitting into two 0.5-steps for v)
    // v' = 0.04v^2 + 5v + 140 - u + I
    // Two half-steps for numerical stability at dt=0.5ms equivalent
    double half_dt = dt * 0.5;
    v_ += half_dt * (0.04 * v_ * v_ + 5.0 * v_ + 140.0 - u_ + I_syn);
    v_ += half_dt * (0.04 * v_ * v_ + 5.0 * v_ + 140.0 - u_ + I_syn);

    // u' = a(bv - u)
    u_ += dt * a * (b * v_ - u_);

    // Spike detection and reset
    if (v_ >= kSpikeThreshold) {
        v_ = c;
        u_ += d;
        return true;
    }
    return false;
}

void IzhikevichNeuron::reset() {
    v_ = c;
    u_ = b * v_;
    last_spike_time = -1e9;
}

double IzhikevichNeuron::voltage() const {
    return v_;
}

double IzhikevichNeuron::recoveryVariable() const {
    return u_;
}

} // namespace biobrain
