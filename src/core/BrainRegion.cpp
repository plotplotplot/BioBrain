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
    if (neuron_count_ == 0) return 0.0;

    double window_start = current_time_ - 100.0; // last 100ms
    size_t count = 0;
    for (const auto& t : recent_spike_times_) {
        if (t >= window_start) {
            ++count;
        }
    }
    // Hz = spikes / (neuron_count * window_in_seconds)
    // window = 100ms = 0.1s
    return static_cast<double>(count) / (static_cast<double>(neuron_count_) * 0.1);
}

uint32_t BrainRegion::activeNeuronCount() const {
    double window_start = current_time_ - 100.0;
    // Count unique neurons that spiked -- approximate by counting recent spikes
    // (a neuron may spike multiple times, but this is a simple estimate)
    size_t count = 0;
    for (const auto& t : recent_spike_times_) {
        if (t >= window_start) {
            ++count;
        }
    }
    // Cap at neuron_count_ since we can't have more active neurons than total
    return static_cast<uint32_t>(std::min(count, static_cast<size_t>(neuron_count_)));
}

void BrainRegion::recordSpikeTime(double t) {
    recent_spike_times_.push_back(t);

    // Prune old spike times (older than 200ms) to prevent unbounded growth.
    // We keep a generous window so firingRate() has clean data.
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

} // namespace biobrain
