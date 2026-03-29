#include <metal_stdlib>
using namespace metal;

// Izhikevich neuron state buffer layout
struct IzhikevichState {
    float v;           // membrane potential (mV)
    float u;           // recovery variable
    float a;           // time scale of recovery
    float b;           // sensitivity of recovery to subthreshold fluctuations
    float c;           // after-spike reset value of v
    float d;           // after-spike reset value of u
    float I_syn;       // total synaptic current for this step
    uint  spiked;      // output: 1 if neuron spiked, 0 otherwise
};

// Batch update kernel: each thread updates one neuron for one timestep.
// Implements: dv/dt = 0.04v² + 5v + 140 - u + I
//             du/dt = a(bv - u)
//             if v >= 30: v = c, u += d
kernel void izhikevich_step(
    device IzhikevichState* neurons [[buffer(0)]],
    constant float& dt [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    device IzhikevichState& n = neurons[id];

    float v = n.v;
    float u = n.u;
    float I = n.I_syn;

    // Euler integration with 2 half-steps for numerical stability
    // (Izhikevich recommends this for 0.5ms steps; we use 0.1ms but keep it)
    float half_dt = dt * 0.5f;

    // First half-step
    v += half_dt * (0.04f * v * v + 5.0f * v + 140.0f - u + I);
    u += half_dt * (n.a * (n.b * v - u));

    // Second half-step
    v += half_dt * (0.04f * v * v + 5.0f * v + 140.0f - u + I);
    u += half_dt * (n.a * (n.b * v - u));

    // Spike check
    if (v >= 30.0f) {
        n.spiked = 1;
        v = n.c;
        u += n.d;
    } else {
        n.spiked = 0;
    }

    n.v = v;
    n.u = u;
    n.I_syn = 0.0f;  // clear synaptic current for next step
}
