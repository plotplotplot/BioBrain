#include <metal_stdlib>
using namespace metal;

// Adaptive Exponential Integrate-and-Fire state
struct AdExState {
    float V;           // membrane potential (mV)
    float w;           // adaptation current (nA)
    float I_syn;       // total synaptic current (nA)
    uint  spiked;      // output flag

    // Parameters
    float C;           // membrane capacitance (pF), default 281
    float g_L;         // leak conductance (nS), default 30
    float E_L;         // leak reversal (mV), default -70.6
    float V_T;         // threshold potential (mV), default -50.4
    float Delta_T;     // slope factor (mV), default 2.0
    float a;           // subthreshold adaptation (nS), default 4.0
    float b;           // spike-triggered adaptation (nA), default 0.0805
    float tau_w;       // adaptation time constant (ms), default 144
    float V_reset;     // reset potential (mV), default -70.6
    float V_cutoff;    // spike cutoff (mV), default -30.0
};

kernel void adex_step(
    device AdExState* neurons [[buffer(0)]],
    constant float& dt [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    device AdExState& s = neurons[id];

    float V = s.V;
    float w = s.w;
    float I = s.I_syn;

    // dV/dt = (-g_L*(V-E_L) + g_L*Delta_T*exp((V-V_T)/Delta_T) - w + I) / C
    float exp_term = s.g_L * s.Delta_T * exp((V - s.V_T) / s.Delta_T);
    float dVdt = (-s.g_L * (V - s.E_L) + exp_term - w + I) / s.C;

    // dw/dt = (a*(V-E_L) - w) / tau_w
    float dwdt = (s.a * (V - s.E_L) - w) / s.tau_w;

    V += dt * dVdt;
    w += dt * dwdt;

    // Spike condition
    if (V >= s.V_cutoff) {
        s.spiked = 1;
        V = s.V_reset;
        w += s.b;
    } else {
        s.spiked = 0;
    }

    s.V = V;
    s.w = w;
    s.I_syn = 0.0f;
}
