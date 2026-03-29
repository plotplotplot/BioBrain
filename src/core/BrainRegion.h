#pragma once

#include "core/Neuron.h"
#include "core/Synapse.h"
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <unordered_map>

class ComputeBackend;

namespace biobrain {

class PlasticityRule;

enum class NeuronModelType { Izhikevich, HodgkinHuxley, AdEx, LIF };

class BrainRegion {
public:
    BrainRegion(uint32_t id, const std::string& name, uint32_t neuron_count);

    /// Initialize all neurons with a specific model type.
    void initializeNeurons(NeuronModelType model);

    // --- Accessors ---
    uint32_t id() const { return id_; }
    const std::string& name() const { return name_; }
    std::vector<std::unique_ptr<Neuron>>& neurons() { return neurons_; }
    const std::vector<std::unique_ptr<Neuron>>& neurons() const { return neurons_; }

    /// Internal synapses (within this region).
    std::vector<Synapse>& internalSynapses() { return internal_synapses_; }

    /// Add an outgoing projection to another region.
    void addProjection(uint32_t target_region_id, std::vector<Synapse> synapses);
    const std::unordered_map<uint32_t, std::vector<Synapse>>& projections() const { return projections_; }

    /// Runtime backend swap.
    void setComputeBackend(std::shared_ptr<ComputeBackend> backend);
    ComputeBackend* computeBackend() const { return backend_.get(); }

    /// Runtime model swap (recreates all neurons, preserving count).
    void setNeuronModel(NeuronModelType model);
    NeuronModelType neuronModel() const { return current_model_; }

    /// Plasticity.
    void setPlasticityRule(std::shared_ptr<PlasticityRule> rule);
    PlasticityRule* plasticityRule() const { return plasticity_.get(); }

    /// Current simulation time for this region.
    double currentTime() const { return current_time_; }
    void setCurrentTime(double t) { current_time_ = t; }

    /// Firing rate (Hz), averaged over last 100ms of recorded spike times.
    double firingRate() const;

    /// Number of neurons that spiked within the last 100ms.
    uint32_t activeNeuronCount() const;

    /// Neuron ID range base for this region.
    uint32_t baseNeuronId() const { return base_neuron_id_; }
    void setBaseNeuronId(uint32_t id) { base_neuron_id_ = id; }

    /// Record a spike time for firing rate tracking.
    void recordSpikeTime(double t);

private:
    uint32_t id_;
    std::string name_;
    uint32_t neuron_count_;
    uint32_t base_neuron_id_ = 0;

    std::vector<std::unique_ptr<Neuron>> neurons_;
    std::vector<Synapse> internal_synapses_;
    std::unordered_map<uint32_t, std::vector<Synapse>> projections_;

    std::shared_ptr<ComputeBackend> backend_;
    std::shared_ptr<PlasticityRule> plasticity_;
    NeuronModelType current_model_ = NeuronModelType::Izhikevich;
    double current_time_ = 0.0;

    /// Recent spike times for firing rate calculation.
    std::vector<double> recent_spike_times_;

    /// Factory method to create a neuron of the given model type.
    std::unique_ptr<Neuron> createNeuron(NeuronModelType model, uint32_t neuron_id);
};

} // namespace biobrain
