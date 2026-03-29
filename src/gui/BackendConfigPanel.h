#pragma once

#include <QWidget>
#include <QComboBox>
#include <QCheckBox>
#include <QSlider>
#include <QLabel>
#include <QGroupBox>
#include <memory>

namespace biobrain { class BrainRegion; }
class NeuronParamEditor;

class BackendConfigPanel : public QWidget {
    Q_OBJECT
public:
    explicit BackendConfigPanel(QWidget* parent = nullptr);

    void setRegion(std::shared_ptr<biobrain::BrainRegion> region);

signals:
    void configChanged(uint32_t region_id);

private slots:
    void onNeuronModelChanged(int index);
    void onComputeBackendChanged(int index);
    void onPlasticityChanged(int index);
    void onSynapseToggled();
    void onMyelinationChanged(int value);

private:
    void setupUI();
    void updateFromRegion();

    std::shared_ptr<biobrain::BrainRegion> current_region_;

    QComboBox* neuronModelCombo_    = nullptr;
    QComboBox* computeBackendCombo_ = nullptr;
    QComboBox* plasticityCombo_     = nullptr;

    QCheckBox* ampaCheck_  = nullptr;
    QCheckBox* nmdaCheck_  = nullptr;
    QCheckBox* gabaACheck_ = nullptr;
    QCheckBox* gabaBCheck_ = nullptr;

    QSlider* myelinationSlider_ = nullptr;
    QLabel*  myelinationLabel_  = nullptr;

    NeuronParamEditor* paramEditor_ = nullptr;
};
