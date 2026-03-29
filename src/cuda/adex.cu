// CUDA kernel for the Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.
// Brette & Gerstner (2005). Each thread updates one neuron.
//
// C*dV/dt = -g_L*(V-E_L) + g_L*Delta_T*exp((V-V_T)/Delta_T) - w + I
// tau_w*dw/dt = a*(V-E_L) - w
// if V > V_cutoff: V = V_reset, w += b

#ifdef __CUDACC__

#include <cstdint>
#include <cmath>

struct AdExState {
    // State variables
    float V, w;
    // Parameters (pF, nS, mV, pA, ms)
    float C, g_L, E_L, V_T, Delta_T;
    float a, b, tau_w, V_reset, V_cutoff;
    // Input
    float I_syn;
    // Output
    uint32_t spiked;
};

__global__ void adex_step(AdExState* __restrict__ states,
                          const float* __restrict__ dt_ptr,
                          uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    AdExState s = states[idx];
    float dt = *dt_ptr;

    float V = s.V;
    float w = s.w;

    // Exponential term with overflow guard
    float exp_arg = (V - s.V_T) / s.Delta_T;
    float exp_term = (exp_arg < 50.0f)
        ? s.Delta_T * expf(exp_arg)
        : s.Delta_T * expf(50.0f);

    // dV/dt = (1/C) * [-g_L*(V-E_L) + g_L*Delta_T*exp((V-V_T)/Delta_T) - w + I]
    float dV = (-s.g_L * (V - s.E_L) + s.g_L * exp_term - w + s.I_syn) / s.C;

    // dw/dt = (1/tau_w) * [a*(V-E_L) - w]
    float dw = (s.a * (V - s.E_L) - w) / s.tau_w;

    V += dt * dV;
    w += dt * dw;

    // Spike detection and reset
    uint32_t spiked = 0;
    if (V > s.V_cutoff) {
        V = s.V_reset;
        w += s.b;
        spiked = 1;
    }

    states[idx].V = V;
    states[idx].w = w;
    states[idx].spiked = spiked;
}

#endif // __CUDACC__
