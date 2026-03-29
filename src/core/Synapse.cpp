#include "Synapse.h"
#include <cmath>

namespace biobrain {

// --- Receptor kinetic parameters ---

ReceptorKinetics Synapse::kinetics(ReceptorType type) {
    switch (type) {
        case ReceptorType::AMPA:
            return { .tau_rise = 0.5,  .tau_decay = 2.0,   .E_rev = 0.0 };
        case ReceptorType::NMDA:
            return { .tau_rise = 2.0,  .tau_decay = 80.0,  .E_rev = 0.0 };
        case ReceptorType::GABA_A:
            return { .tau_rise = 0.5,  .tau_decay = 6.0,   .E_rev = -70.0 };
        case ReceptorType::GABA_B:
            return { .tau_rise = 30.0, .tau_decay = 150.0, .E_rev = -90.0 };
    }
    return { .tau_rise = 0.5, .tau_decay = 2.0, .E_rev = 0.0 }; // fallback
}

double Synapse::nmda_mg_block(double V, double Mg_concentration) {
    return 1.0 / (1.0 + std::exp(-0.062 * V) * Mg_concentration / 3.57);
}

// --- Synapse implementation ---

Synapse::Synapse(uint32_t pre, uint32_t post, const SynapseParams& params)
    : pre_id(pre)
    , post_id(post)
    , weight(params.weight)
    , delay(params.delay)
    , receptor(params.receptor)
{
    // Myelination reduces conduction delay
    if (params.myelinated) {
        delay *= 0.2;  // ~5x faster conduction velocity
    }
}

void Synapse::deliverSpike(double t) {
    double dt_since = t - last_spike_;

    // Tsodyks-Markram short-term plasticity update
    // Recovery of x toward 1.0
    if (last_spike_ > -1e8) {
        x = 1.0 - (1.0 - x) * std::exp(-dt_since / tau_rec);

        // Facilitation of u (if tau_fac > 0)
        if (tau_fac > 0.0) {
            u = U + (u - U) * std::exp(-dt_since / tau_fac);
        }
    }

    // Update utilization
    if (tau_fac > 0.0) {
        u = u + U * (1.0 - u);  // facilitation
    } else {
        u = U;  // no facilitation, constant U
    }

    // Resources consumed by this spike
    double delta_g = weight * u * x;
    x -= u * x;

    // Add to conductance (rising component for dual-exponential waveform)
    g_rise_ += delta_g;
    g_      += delta_g;

    last_spike_ = t;
}

double Synapse::computeCurrent(double V_post, double dt) {
    auto kin = kinetics(receptor);

    // Dual-exponential conductance decay
    // g_rise decays with tau_rise, g decays with tau_decay
    double decay_rise  = std::exp(-dt / kin.tau_rise);
    double decay_total = std::exp(-dt / kin.tau_decay);

    g_rise_ *= decay_rise;
    g_      *= decay_total;

    // Effective conductance = g - g_rise (dual exponential shape)
    double g_eff = g_ - g_rise_;
    if (g_eff < 0.0) g_eff = 0.0;

    // Voltage-dependent modulation for NMDA
    double modulation = 1.0;
    if (receptor == ReceptorType::NMDA) {
        modulation = nmda_mg_block(V_post);
    }

    // I_syn = g_eff * modulation * (E_rev - V_post)
    // Positive current for excitatory (E_rev > V_post), negative for inhibitory
    return g_eff * modulation * (kin.E_rev - V_post);
}

double Synapse::conductance() const {
    double g_eff = g_ - g_rise_;
    return (g_eff > 0.0) ? g_eff : 0.0;
}

} // namespace biobrain
