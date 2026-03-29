// CUDA kernel for synapse conductance update.
// Handles: dual-exponential conductance decay, spike delivery with
// Tsodyks-Markram short-term plasticity, and NMDA Mg2+ voltage-dependent block.
//
// Matches the CPU Synapse::computeCurrent() and Synapse::deliverSpike() logic.

#ifdef __CUDACC__

#include <cstdint>
#include <cmath>

// Receptor type encoding (matches biobrain::ReceptorType)
enum CUDAReceptorType : uint32_t {
    RECEPTOR_AMPA   = 0,
    RECEPTOR_NMDA   = 1,
    RECEPTOR_GABA_A = 2,
    RECEPTOR_GABA_B = 3
};

// Per-synapse state uploaded to GPU
struct CUDASynapseState {
    // Conductance state
    float g;           // total conductance (nS)
    float g_rise;      // rising component (nS)

    // Tsodyks-Markram STP state
    float x;           // available resources fraction
    float u_stp;       // utilization variable
    float U;           // baseline utilization
    float tau_rec;     // recovery time constant (ms)
    float tau_fac;     // facilitation time constant (ms)

    // Synapse identity
    float weight;      // synaptic weight (nS)
    uint32_t receptor; // CUDAReceptorType

    // Spike delivery flag and timing
    uint32_t spike_pending; // 1 if a presynaptic spike should be delivered this step
    float dt_since_last;    // time since last presynaptic spike (ms)

    // Post-synaptic voltage (for NMDA Mg block and current computation)
    float V_post;

    // Output: computed synaptic current (sign-correct)
    float I_syn_out;
};

// Receptor kinetic parameters (matches Synapse::kinetics)
__device__ void get_kinetics(uint32_t receptor, float& tau_rise, float& tau_decay, float& E_rev) {
    switch (receptor) {
        case RECEPTOR_AMPA:
            tau_rise = 0.5f; tau_decay = 2.0f; E_rev = 0.0f; break;
        case RECEPTOR_NMDA:
            tau_rise = 2.0f; tau_decay = 80.0f; E_rev = 0.0f; break;
        case RECEPTOR_GABA_A:
            tau_rise = 0.5f; tau_decay = 6.0f; E_rev = -70.0f; break;
        case RECEPTOR_GABA_B:
            tau_rise = 30.0f; tau_decay = 150.0f; E_rev = -90.0f; break;
        default:
            tau_rise = 0.5f; tau_decay = 2.0f; E_rev = 0.0f; break;
    }
}

// NMDA Mg2+ voltage-dependent block: B(V) = 1/(1 + exp(-0.062*V)*[Mg]/3.57)
// Default [Mg2+] = 1.0 mM
__device__ float nmda_mg_block(float V, float Mg = 1.0f) {
    return 1.0f / (1.0f + expf(-0.062f * V) * Mg / 3.57f);
}

__global__ void synapse_update_kernel(CUDASynapseState* __restrict__ synapses,
                                      const float* __restrict__ dt_ptr,
                                      uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    CUDASynapseState s = synapses[idx];
    float dt = *dt_ptr;

    // --- Phase 1: Spike delivery (Tsodyks-Markram STP) ---
    if (s.spike_pending) {
        float dt_since = s.dt_since_last;

        // Recovery of x toward 1.0
        if (dt_since < 1e8f) {
            s.x = 1.0f - (1.0f - s.x) * expf(-dt_since / s.tau_rec);
            if (s.tau_fac > 0.0f) {
                s.u_stp = s.U + (s.u_stp - s.U) * expf(-dt_since / s.tau_fac);
            }
        }

        // Update utilization
        if (s.tau_fac > 0.0f) {
            s.u_stp = s.u_stp + s.U * (1.0f - s.u_stp);  // facilitation
        } else {
            s.u_stp = s.U;  // no facilitation
        }

        // Resources consumed
        float delta_g = s.weight * s.u_stp * s.x;
        s.x -= s.u_stp * s.x;

        // Add to conductance
        s.g_rise += delta_g;
        s.g      += delta_g;
    }

    // --- Phase 2: Conductance decay (dual-exponential) ---
    float tau_rise, tau_decay, E_rev;
    get_kinetics(s.receptor, tau_rise, tau_decay, E_rev);

    float decay_rise  = expf(-dt / tau_rise);
    float decay_total = expf(-dt / tau_decay);

    s.g_rise *= decay_rise;
    s.g      *= decay_total;

    // Effective conductance
    float g_eff = s.g - s.g_rise;
    if (g_eff < 0.0f) g_eff = 0.0f;

    // --- Phase 3: Current computation with NMDA Mg block ---
    float modulation = 1.0f;
    if (s.receptor == RECEPTOR_NMDA) {
        modulation = nmda_mg_block(s.V_post);
    }

    // I_syn = g_eff * modulation * (E_rev - V_post)
    s.I_syn_out = g_eff * modulation * (E_rev - s.V_post);

    // Write back
    synapses[idx].g        = s.g;
    synapses[idx].g_rise   = s.g_rise;
    synapses[idx].x        = s.x;
    synapses[idx].u_stp    = s.u_stp;
    synapses[idx].I_syn_out = s.I_syn_out;
}

#endif // __CUDACC__
