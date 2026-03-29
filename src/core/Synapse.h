#pragma once

#include <cstdint>

namespace biobrain {

enum class ReceptorType {
    AMPA,
    NMDA,
    GABA_A,
    GABA_B
};

struct SynapseParams {
    double       weight     = 1.0;   // synaptic weight (conductance scaling, nS)
    double       delay      = 1.0;   // axonal conduction delay (ms)
    ReceptorType receptor   = ReceptorType::AMPA;
    bool         myelinated = false;  // affects delay calculation
};

/// Receptor kinetic parameters.
struct ReceptorKinetics {
    double tau_rise;   // rise time constant (ms)
    double tau_decay;  // decay time constant (ms)
    double E_rev;      // reversal potential (mV)
};

/// Conductance-based synapse with Tsodyks-Markram short-term plasticity.
class Synapse {
public:
    Synapse() = default;
    Synapse(uint32_t pre_id, uint32_t post_id, const SynapseParams& params);

    /// Called when a presynaptic spike arrives (after delay).
    /// Updates Tsodyks-Markram facilitation/depression state and adds to conductance.
    void deliverSpike(double t);

    /// Compute the postsynaptic current given the postsynaptic membrane voltage.
    /// Applies conductance decay and voltage-dependent modulation (NMDA Mg block).
    /// Returns I_syn (sign-correct: negative for inhibitory, positive for excitatory
    /// when driving toward reversal potential with standard conventions).
    double computeCurrent(double V_post, double dt);

    /// Current synaptic conductance (nS).
    double conductance() const;

    // --- Public state for inspection ---
    uint32_t     pre_id   = 0;
    uint32_t     post_id  = 0;
    double       weight   = 1.0;      // nS
    double       delay    = 1.0;      // ms
    ReceptorType receptor = ReceptorType::AMPA;

    // Tsodyks-Markram short-term plasticity state
    double x = 1.0;     // fraction of available resources (recovered)
    double u = 0.0;     // fraction of resources used per spike (utilization)
    double U = 0.5;     // baseline utilization
    double tau_rec = 800.0;  // recovery time constant (ms)
    double tau_fac = 0.0;    // facilitation time constant (ms), 0 = depression-dominant

    /// Get kinetic parameters for a receptor type.
    static ReceptorKinetics kinetics(ReceptorType type);

    /// NMDA Mg2+ voltage-dependent block factor: B(V) = 1/(1 + exp(-0.062*V)*[Mg]/3.57)
    static double nmda_mg_block(double V, double Mg_concentration = 1.0);

private:
    double g_         = 0.0;   // current conductance (nS)
    double g_rise_    = 0.0;   // rising component for dual-exponential kinetics
    double last_spike_ = -1e9; // time of last presynaptic spike delivery (ms)
};

} // namespace biobrain
