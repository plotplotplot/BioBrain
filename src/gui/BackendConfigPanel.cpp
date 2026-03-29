#include "BackendConfigPanel.h"
#include "NeuronParamEditor.h"
#include "core/BrainRegion.h"

#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>

using biobrain::NeuronModelType;

BackendConfigPanel::BackendConfigPanel(QWidget* parent)
    : QWidget(parent)
{
    setupUI();
}

void BackendConfigPanel::setupUI()
{
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(8, 8, 8, 8);

    // --- Model Selection ---
    auto* modelGroup = new QGroupBox("Neuron Model", this);
    auto* modelLayout = new QFormLayout(modelGroup);

    neuronModelCombo_ = new QComboBox(this);
    neuronModelCombo_->addItems({"Izhikevich", "Hodgkin-Huxley", "AdEx", "LIF"});
    modelLayout->addRow("Model:", neuronModelCombo_);

    computeBackendCombo_ = new QComboBox(this);
    computeBackendCombo_->addItems({
        "CPU Event-Driven", "Metal GPU Batch", "Hybrid (auto)"
    });
    modelLayout->addRow("Compute:", computeBackendCombo_);

    plasticityCombo_ = new QComboBox(this);
    plasticityCombo_->addItems({
        "STDP + Dopamine (3-factor)",
        "STDP (Hebbian only)",
        "Full Neuromodulatory",
        "None"
    });
    modelLayout->addRow("Plasticity:", plasticityCombo_);

    mainLayout->addWidget(modelGroup);

    // --- Synapse Types ---
    auto* synGroup = new QGroupBox("Synapse Types", this);
    auto* synLayout = new QVBoxLayout(synGroup);

    ampaCheck_  = new QCheckBox("AMPA",   this);
    nmdaCheck_  = new QCheckBox("NMDA",   this);
    gabaACheck_ = new QCheckBox("GABA-A", this);
    gabaBCheck_ = new QCheckBox("GABA-B", this);

    ampaCheck_->setChecked(true);
    nmdaCheck_->setChecked(true);
    gabaACheck_->setChecked(true);
    gabaBCheck_->setChecked(false);

    for (auto* cb : {ampaCheck_, nmdaCheck_, gabaACheck_, gabaBCheck_})
        synLayout->addWidget(cb);

    mainLayout->addWidget(synGroup);

    // --- Myelination ---
    auto* myelinGroup = new QGroupBox("Myelination", this);
    auto* myelinLayout = new QVBoxLayout(myelinGroup);

    myelinationSlider_ = new QSlider(Qt::Horizontal, this);
    myelinationSlider_->setRange(0, 100);
    myelinationSlider_->setValue(0);

    myelinationLabel_ = new QLabel("0.00", this);
    myelinationLabel_->setAlignment(Qt::AlignCenter);

    myelinLayout->addWidget(myelinationSlider_);
    myelinLayout->addWidget(myelinationLabel_);
    mainLayout->addWidget(myelinGroup);

    // --- Neuron Parameters ---
    paramEditor_ = new NeuronParamEditor(this);
    mainLayout->addWidget(paramEditor_);

    mainLayout->addStretch();

    // --- Connections ---
    connect(neuronModelCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &BackendConfigPanel::onNeuronModelChanged);
    connect(computeBackendCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &BackendConfigPanel::onComputeBackendChanged);
    connect(plasticityCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &BackendConfigPanel::onPlasticityChanged);

    for (auto* cb : {ampaCheck_, nmdaCheck_, gabaACheck_, gabaBCheck_})
        connect(cb, &QCheckBox::toggled, this, &BackendConfigPanel::onSynapseToggled);

    connect(myelinationSlider_, &QSlider::valueChanged,
            this, &BackendConfigPanel::onMyelinationChanged);
}

void BackendConfigPanel::setRegion(std::shared_ptr<biobrain::BrainRegion> region)
{
    current_region_ = std::move(region);
    updateFromRegion();
}

void BackendConfigPanel::updateFromRegion()
{
    if (!current_region_) return;

    // Set neuron model combo to match region's current model
    neuronModelCombo_->blockSignals(true);
    switch (current_region_->neuronModel()) {
        case NeuronModelType::Izhikevich:   neuronModelCombo_->setCurrentIndex(0); break;
        case NeuronModelType::HodgkinHuxley: neuronModelCombo_->setCurrentIndex(1); break;
        case NeuronModelType::AdEx:          neuronModelCombo_->setCurrentIndex(2); break;
        case NeuronModelType::LIF:           neuronModelCombo_->setCurrentIndex(3); break;
    }
    neuronModelCombo_->blockSignals(false);

    // Update parameter editor based on current model
    onNeuronModelChanged(neuronModelCombo_->currentIndex());
}

void BackendConfigPanel::onNeuronModelChanged(int index)
{
    if (current_region_) {
        auto model = static_cast<NeuronModelType>(index);
        current_region_->setNeuronModel(model);
    }

    // Set parameter editor to match model type
    std::vector<std::pair<std::string, double>> params;

    switch (index) {
        case 0: // Izhikevich
            params = {{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
            break;
        case 1: // Hodgkin-Huxley
            params = {
                {"g_Na", 120.0}, {"g_K", 36.0}, {"g_L", 0.3},
                {"E_Na", 50.0},  {"E_K", -77.0}, {"E_L", -54.4},
                {"C_m", 1.0}
            };
            break;
        case 2: // AdEx
            params = {
                {"C", 281.0}, {"g_L", 30.0}, {"E_L", -70.6},
                {"V_T", -50.4}, {"Delta_T", 2.0},
                {"a", 4.0}, {"b", 0.0805}, {"tau_w", 144.0}
            };
            break;
        case 3: // LIF
            params = {
                {"V_rest", -65.0}, {"V_thresh", -50.0},
                {"V_reset", -65.0}, {"tau_m", 10.0}, {"R_m", 10.0}
            };
            break;
    }

    paramEditor_->setParameters(params);

    if (current_region_)
        emit configChanged(current_region_->id());
}

void BackendConfigPanel::onComputeBackendChanged(int /*index*/)
{
    // Backend swapping would be wired to actual ComputeBackend factory
    if (current_region_)
        emit configChanged(current_region_->id());
}

void BackendConfigPanel::onPlasticityChanged(int /*index*/)
{
    if (current_region_)
        emit configChanged(current_region_->id());
}

void BackendConfigPanel::onSynapseToggled()
{
    if (current_region_)
        emit configChanged(current_region_->id());
}

void BackendConfigPanel::onMyelinationChanged(int value)
{
    double ratio = value / 100.0;
    myelinationLabel_->setText(QString::number(ratio, 'f', 2));

    if (current_region_)
        emit configChanged(current_region_->id());
}
