// CUDA kernel for the Hodgkin-Huxley (1952) conductance-based neuron model.
// Each thread updates one neuron using forward Euler integration.
// Rate functions match the CPU implementation in HodgkinHuxleyNeuron.cpp.

#ifdef __CUDACC__

#include <cstdint>
#include <cmath>

struct HHState {
    // State variables
    float V, m, h, n;
    // Parameters
    float g_Na, g_K, g_L;
    float E_Na, E_K, E_L;
    float C_m;
    // Input
    float I_syn;
    // Output
    uint32_t spiked;
    uint32_t prev_below_threshold;
};

// --- HH rate functions (Hodgkin & Huxley 1952) ---

__device__ float hh_alpha_m(float V) {
    float dV = V + 40.0f;
    if (fabsf(dV) < 1e-7f) return 1.0f;  // L'Hopital limit
    return 0.1f * dV / (1.0f - expf(-dV / 10.0f));
}

__device__ float hh_beta_m(float V) {
    return 4.0f * expf(-(V + 65.0f) / 18.0f);
}

__device__ float hh_alpha_h(float V) {
    return 0.07f * expf(-(V + 65.0f) / 20.0f);
}

__device__ float hh_beta_h(float V) {
    return 1.0f / (1.0f + expf(-(V + 35.0f) / 10.0f));
}

__device__ float hh_alpha_n(float V) {
    float dV = V + 55.0f;
    if (fabsf(dV) < 1e-7f) return 0.1f;  // L'Hopital limit
    return 0.01f * dV / (1.0f - expf(-dV / 10.0f));
}

__device__ float hh_beta_n(float V) {
    return 0.125f * expf(-(V + 65.0f) / 80.0f);
}

__global__ void hodgkin_huxley_step(HHState* __restrict__ states,
                                    const float* __restrict__ dt_ptr,
                                    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    HHState s = states[idx];
    float dt = *dt_ptr;

    float V = s.V;
    float m = s.m;
    float h = s.h;
    float n = s.n;

    // Gating variable derivatives
    float dm = hh_alpha_m(V) * (1.0f - m) - hh_beta_m(V) * m;
    float dh = hh_alpha_h(V) * (1.0f - h) - hh_beta_h(V) * h;
    float dn = hh_alpha_n(V) * (1.0f - n) - hh_beta_n(V) * n;

    // Ionic currents
    float m3h  = m * m * m * h;
    float n4   = n * n * n * n;
    float I_Na = s.g_Na * m3h * (V - s.E_Na);
    float I_K  = s.g_K  * n4  * (V - s.E_K);
    float I_L  = s.g_L         * (V - s.E_L);

    // Membrane potential: C_m * dV/dt = -I_Na - I_K - I_L + I_syn
    float dV = (-I_Na - I_K - I_L + s.I_syn) / s.C_m;

    // Euler integration
    V += dt * dV;
    m += dt * dm;
    h += dt * dh;
    n += dt * dn;

    // Clamp gating variables to [0, 1]
    m = fminf(fmaxf(m, 0.0f), 1.0f);
    h = fminf(fmaxf(h, 0.0f), 1.0f);
    n = fminf(fmaxf(n, 0.0f), 1.0f);

    // Spike detection: upward crossing of 0 mV threshold
    uint32_t currently_above = (V >= 0.0f) ? 1 : 0;
    uint32_t spiked = currently_above && s.prev_below_threshold;

    states[idx].V = V;
    states[idx].m = m;
    states[idx].h = h;
    states[idx].n = n;
    states[idx].spiked = spiked;
    states[idx].prev_below_threshold = currently_above ? 0 : 1;
}

#endif // __CUDACC__
