// CUDA kernel for Izhikevich (2003) spiking neuron model.
// Matches the Metal compute shader: split half-step Euler integration.
// Each thread updates one neuron.

#ifdef __CUDACC__

#include <cstdint>

struct IzhikevichState {
    float v, u, a, b, c, d, I_syn;
    uint32_t spiked;
};

// Izhikevich model:
//   v' = 0.04v^2 + 5v + 140 - u + I
//   u' = a(bv - u)
//   if v >= 30: v = c, u += d
//
// Integration: two half-steps for v (numerical stability), full step for u.

__global__ void izhikevich_step(IzhikevichState* __restrict__ states,
                                const float* __restrict__ dt_ptr,
                                uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    IzhikevichState s = states[idx];

    float dt = *dt_ptr;
    float half_dt = dt * 0.5f;

    // Two half-steps for v (Izhikevich recommended split)
    float v = s.v;
    float u = s.u;
    float I = s.I_syn;

    v += half_dt * (0.04f * v * v + 5.0f * v + 140.0f - u + I);
    v += half_dt * (0.04f * v * v + 5.0f * v + 140.0f - u + I);

    // Full step for u
    u += dt * s.a * (s.b * v - u);

    // Spike detection and reset
    uint32_t spiked = 0;
    if (v >= 30.0f) {
        v = s.c;
        u += s.d;
        spiked = 1;
    }

    states[idx].v = v;
    states[idx].u = u;
    states[idx].spiked = spiked;
}

#endif // __CUDACC__
