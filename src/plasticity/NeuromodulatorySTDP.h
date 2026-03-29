#pragma once

#include "PlasticityRule.h"
#include <array>
#include <cmath>
#include <cstdint>

namespace biobrain {

/// Neuromodulator types for the full neuromodulatory plasticity model.
enum class ModulatorType {
    Dopamine,       // DA  — reward signal, scales LTP
    Serotonin,      // 5-HT — aversive/punishment signal, scales LTD
    Acetylcholine,  // ACh — attention signal, scales learning rate
    Norepinephrine  // NE  — arousal/salience signal, modulates eligibility trace duration
};

/// Full neuromodulatory STDP with 4 modulator channels.
///
/// Extends DopamineSTDP with additional neuromodulatory influences:
///
/// - **Dopamine (DA)**: Scales LTP via eligibility trace (same as DopamineSTDP).
/// - **Serotonin (5-HT)**: Scales LTD magnitude.
///     Effect: dw_LTD *= (1 + [5-HT])
/// - **Acetylcholine (ACh)**: Modulates overall learning rate (attention).
///     Effect: A_plus *= (1 + [ACh]), A_minus *= (1 + [ACh])
/// - **Norepinephrine (NE)**: Modulates eligibility trace duration (arousal).
///     Effect: tau_e_effective = tau_e * (1 + [NE])
///
/// References:
///   Izhikevich (2007); Frémaux & Gerstner (2016) "Neuromodulated STDP";
///   Doya (2002) "Metalearning and neuromodulation".
struct NeuromodulatorySTDPParams {
    double A_plus    = 0.01;
    double A_minus   = 0.012;
    double tau_plus   = 20.0;   // ms
    double tau_minus  = 20.0;   // ms
    double tau_e     = 1000.0;  // ms
    double DA_tonic  = 0.01;
    double w_min     = 0.0;
    double w_max     = 1.0;
};

class NeuromodulatorySTDP : public PlasticityRule {
public:
    using Params = NeuromodulatorySTDPParams;

    explicit NeuromodulatorySTDP(Params params = {});

    void   onPreSpike(double t, uint32_t synapse_id) override;
    void   onPostSpike(double t, uint32_t synapse_id) override;
    double computeWeightChange(double dt, double modulator_concentration) override;
    double eligibilityTrace() const override;
    void   reset() override;

    /// Set concentration for a specific neuromodulator.
    void setModulator(ModulatorType type, double concentration);

    /// Get current concentration for a specific neuromodulator.
    double getModulator(ModulatorType type) const;

    const Params& params() const { return params_; }

private:
    static constexpr int kNumModulators = 4;

    /// Compute STDP window with ACh and 5-HT modulation.
    double stdpWindow(double delta_t) const;

    /// Effective eligibility trace time constant (modulated by NE).
    double effectiveTauE() const;

    Params params_;

    double last_pre_spike_  = -1e9;
    double last_post_spike_ = -1e9;
    double eligibility_     = 0.0;
    double last_update_t_   = 0.0;

    /// Modulator concentrations indexed by ModulatorType.
    std::array<double, kNumModulators> modulators_ = {0.0, 0.0, 0.0, 0.0};
};

} // namespace biobrain
