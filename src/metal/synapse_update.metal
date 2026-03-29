#include <metal_stdlib>
using namespace metal;

// Synapse state for conductance-based model with 4 receptor types
struct SynapseState {
    float g;           // current conductance (nS)
    float g_rise;      // rising phase conductance
    float weight;      // synaptic weight (scaling factor)

    // Kinetics (set per receptor type)
    float tau_rise;    // rise time constant (ms)
    float tau_decay;   // decay time constant (ms)
    float E_rev;       // reversal potential (mV)

    // Tsodyks-Markram short-term plasticity
    float x;           // available neurotransmitter fraction (depression)
    float u_tm;        // release probability (facilitation)
    float U;           // baseline release probability
    float tau_rec;     // recovery time constant (ms)
    float tau_fac;     // facilitation time constant (ms)

    // Spike delivery flag
    uint  spike_arrived;  // 1 if a presynaptic spike arrived this step

    // Receptor type (0=AMPA, 1=NMDA, 2=GABA_A, 3=GABA_B)
    uint  receptor_type;
};

// Decay conductances and handle spike delivery for all synapses in parallel.
// Each thread processes one synapse.
kernel void synapse_step(
    device SynapseState* synapses [[buffer(0)]],
    constant float& dt [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    device SynapseState& s = synapses[id];

    // Tsodyks-Markram recovery (runs every step)
    // dx/dt = (1 - x) / tau_rec
    // du/dt = (U - u) / tau_fac
    s.x += dt * (1.0f - s.x) / s.tau_rec;
    if (s.tau_fac > 0.0f) {
        s.u_tm += dt * (s.U - s.u_tm) / s.tau_fac;
    }

    // Handle spike arrival
    if (s.spike_arrived) {
        // Tsodyks-Markram: update facilitation then depression
        if (s.tau_fac > 0.0f) {
            s.u_tm += s.U * (1.0f - s.u_tm);
        } else {
            s.u_tm = s.U;
        }
        float release = s.u_tm * s.x;
        s.x -= release;

        // Add to rising conductance
        s.g_rise += s.weight * release;
        s.spike_arrived = 0;
    }

    // Dual-exponential conductance kinetics
    // g_rise decays with tau_rise, g decays with tau_decay
    // Net conductance = g - g_rise (difference of exponentials)
    s.g     -= dt * s.g     / s.tau_decay;
    s.g_rise -= dt * s.g_rise / s.tau_rise;

    // Transfer from rising to total
    // The rising component feeds into the main conductance
    float dg = s.g_rise * dt / s.tau_rise;
    s.g += dg;
}

// Compute synaptic current for a postsynaptic neuron.
// For NMDA, applies Mg²⁺ voltage-dependent block.
// V_post provided per-neuron, synapses grouped by postsynaptic neuron.
struct CurrentOutput {
    float I_syn;  // total synaptic current (nA)
};

kernel void compute_synaptic_current(
    device const SynapseState* synapses [[buffer(0)]],
    device const float* V_post [[buffer(1)]],          // postsynaptic voltage per synapse
    device float* I_out [[buffer(2)]],                  // output current per synapse
    uint id [[thread_position_in_grid]]
) {
    device const SynapseState& s = synapses[id];
    float V = V_post[id];
    float g_eff = s.g;

    // NMDA Mg²⁺ block: B(V) = 1 / (1 + exp(-0.062*V) * [Mg²⁺]/3.57)
    // [Mg²⁺] = 1.0 mM (physiological)
    if (s.receptor_type == 1) {  // NMDA
        float B = 1.0f / (1.0f + exp(-0.062f * V) * (1.0f / 3.57f));
        g_eff *= B;
    }

    // I = g * (V - E_rev)
    I_out[id] = g_eff * (V - s.E_rev);
}
