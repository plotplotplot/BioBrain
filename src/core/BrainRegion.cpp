#include "core/BrainRegion.h"
#include "core/IzhikevichNeuron.h"
#include "core/HodgkinHuxleyNeuron.h"
#include "core/AdExNeuron.h"
#include "core/LIFNeuron.h"
#include <algorithm>

namespace biobrain {

BrainRegion::BrainRegion(uint32_t id, const std::string& name, uint32_t neuron_count)
    : id_(id), name_(name), neuron_count_(neuron_count) {}

void BrainRegion::initializeNeurons(NeuronModelType model) {
    current_model_ = model;
    neurons_.clear();
    neurons_.reserve(neuron_count_);
    for (uint32_t i = 0; i < neuron_count_; ++i) {
        neurons_.push_back(createNeuron(model, base_neuron_id_ + i));
    }
}

void BrainRegion::addProjection(uint32_t target_region_id, std::vector<Synapse> synapses) {
    auto& existing = projections_[target_region_id];
    existing.insert(existing.end(),
                    std::make_move_iterator(synapses.begin()),
                    std::make_move_iterator(synapses.end()));
}

void BrainRegion::setComputeBackend(std::shared_ptr<ComputeBackend> backend) {
    backend_ = std::move(backend);
}

void BrainRegion::setNeuronModel(NeuronModelType model) {
    current_model_ = model;
    neurons_.clear();
    neurons_.reserve(neuron_count_);
    for (uint32_t i = 0; i < neuron_count_; ++i) {
        neurons_.push_back(createNeuron(model, base_neuron_id_ + i));
    }
}

void BrainRegion::setPlasticityRule(std::shared_ptr<PlasticityRule> rule) {
    plasticity_ = std::move(rule);
}

double BrainRegion::firingRate() const {
    std::lock_guard lock(spike_times_mutex_);
    if (neuron_count_ == 0) return 0.0;
    double window_start = current_time_ - 100.0;
    size_t count = 0;
    for (const auto& t : recent_spike_times_) {
        if (t >= window_start) ++count;
    }
    return static_cast<double>(count) / (static_cast<double>(neuron_count_) * 0.1);
}

uint32_t BrainRegion::activeNeuronCount() const {
    std::lock_guard lock(spike_times_mutex_);
    double window_start = current_time_ - 100.0;
    size_t count = 0;
    for (const auto& t : recent_spike_times_) {
        if (t >= window_start) ++count;
    }
    return static_cast<uint32_t>(std::min(count, static_cast<size_t>(neuron_count_)));
}

void BrainRegion::recordSpikeTime(double t) {
    std::lock_guard lock(spike_times_mutex_);
    recent_spike_times_.push_back(t);
    if (recent_spike_times_.size() > 100000) {
        double cutoff = current_time_ - 200.0;
        recent_spike_times_.erase(
            std::remove_if(recent_spike_times_.begin(), recent_spike_times_.end(),
                           [cutoff](double spike_t) { return spike_t < cutoff; }),
            recent_spike_times_.end());
    }
}

std::unique_ptr<Neuron> BrainRegion::createNeuron(NeuronModelType model, uint32_t neuron_id) {
    std::unique_ptr<Neuron> neuron;
    switch (model) {
        case NeuronModelType::Izhikevich:
            neuron = IzhikevichNeuron::create(IzhikevichNeuron::Type::RegularSpiking);
            break;
        case NeuronModelType::HodgkinHuxley:
            neuron = std::make_unique<HodgkinHuxleyNeuron>();
            break;
        case NeuronModelType::AdEx:
            neuron = std::make_unique<AdExNeuron>();
            break;
        case NeuronModelType::LIF:
            neuron = std::make_unique<LIFNeuron>();
            break;
    }
    neuron->id = neuron_id;
    return neuron;
}

void BrainRegion::injectCurrent(uint32_t local_index, double current_nA) {
    if (injected_currents_.empty()) {
        injected_currents_.resize(neurons_.size(), 0.0);
    }
    if (local_index < injected_currents_.size()) {
        injected_currents_[local_index] += current_nA;
    }
}

void BrainRegion::buildSynapseIndex() {
    pre_synapse_index_.clear();
    post_synapse_index_.clear();
    for (uint32_t i = 0; i < internal_synapses_.size(); ++i) {
        pre_synapse_index_[internal_synapses_[i].pre_id].push_back(i);
        post_synapse_index_[internal_synapses_[i].post_id].push_back(i);
    }
    index_built_ = true;
}

static const std::vector<uint32_t> kEmpty;

const std::vector<uint32_t>& BrainRegion::getSynapsesForPreNeuron(uint32_t neuron_id) const {
    auto it = pre_synapse_index_.find(neuron_id);
    return (it != pre_synapse_index_.end()) ? it->second : kEmpty;
}

const std::vector<uint32_t>& BrainRegion::getPostSynapsesForNeuron(uint32_t neuron_id) const {
    auto it = post_synapse_index_.find(neuron_id);
    return (it != post_synapse_index_.end()) ? it->second : kEmpty;
}

} // namespace biobrain
