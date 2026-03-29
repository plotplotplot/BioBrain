#include <metal_stdlib>
using namespace metal;

// Hodgkin-Huxley neuron state buffer layout
struct HHState {
    float V;           // membrane potential (mV)
    float m;           // Na+ activation gate
    float h;           // Na+ inactivation gate
    float n;           // K+ activation gate
    float I_syn;       // total synaptic current
    float V_prev;      // previous voltage (for spike detection)
    uint  spiked;      // output: 1 if spiked

    // Parameters (per-neuron to allow heterogeneity)
    float g_Na;        // Na+ max conductance (mS/cm²)
    float g_K;         // K+ max conductance
    float g_L;         // leak conductance
    float E_Na;        // Na+ reversal potential (mV)
    float E_K;         // K+ reversal potential
    float E_L;         // leak reversal potential
    float C_m;         // membrane capacitance (μF/cm²)
};

// Rate functions (Hodgkin & Huxley 1952)
inline float alpha_m(float V) {
    float dV = V + 40.0f;
    if (abs(dV) < 1e-6f) return 1.0f;  // L'Hôpital limit
    return 0.1f * dV / (1.0f - exp(-dV / 10.0f));
}

inline float beta_m(float V) {
    return 4.0f * exp(-(V + 65.0f) / 18.0f);
}

inline float alpha_h(float V) {
    return 0.07f * exp(-(V + 65.0f) / 20.0f);
}

inline float beta_h(float V) {
    return 1.0f / (1.0f + exp(-(V + 35.0f) / 10.0f));
}

inline float alpha_n(float V) {
    float dV = V + 55.0f;
    if (abs(dV) < 1e-6f) return 0.1f;  // L'Hôpital limit
    return 0.01f * dV / (1.0f - exp(-dV / 10.0f));
}

inline float beta_n(float V) {
    return 0.125f * exp(-(V + 65.0f) / 80.0f);
}

kernel void hodgkin_huxley_step(
    device HHState* neurons [[buffer(0)]],
    constant float& dt [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    device HHState& s = neurons[id];

    float V = s.V;
    float m = s.m;
    float h = s.h;
    float n = s.n;
    float I = s.I_syn;

    // Ionic currents
    float I_Na = s.g_Na * m * m * m * h * (V - s.E_Na);
    float I_K  = s.g_K * n * n * n * n * (V - s.E_K);
    float I_L  = s.g_L * (V - s.E_L);

    // Membrane potential (Euler)
    float dVdt = (-I_Na - I_K - I_L + I) / s.C_m;
    V += dt * dVdt;

    // Gating variables (Euler)
    float am = alpha_m(V); float bm = beta_m(V);
    float ah = alpha_h(V); float bh = beta_h(V);
    float an = alpha_n(V); float bn = beta_n(V);

    m += dt * (am * (1.0f - m) - bm * m);
    h += dt * (ah * (1.0f - h) - bh * h);
    n += dt * (an * (1.0f - n) - bn * n);

    // Clamp gating variables to [0, 1]
    m = clamp(m, 0.0f, 1.0f);
    h = clamp(h, 0.0f, 1.0f);
    n = clamp(n, 0.0f, 1.0f);

    // Spike detection: upward zero-crossing
    s.spiked = (s.V_prev < 0.0f && V >= 0.0f) ? 1 : 0;

    s.V_prev = V;
    s.V = V;
    s.m = m;
    s.h = h;
    s.n = n;
    s.I_syn = 0.0f;
}
