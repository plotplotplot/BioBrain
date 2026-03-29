#pragma once

#include <QGroupBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <vector>
#include <string>
#include <utility>

class NeuronParamEditor : public QGroupBox {
    Q_OBJECT
public:
    explicit NeuronParamEditor(QWidget* parent = nullptr);

    /// Set parameter names and values for current model.
    void setParameters(const std::vector<std::pair<std::string, double>>& params);

    /// Get current parameter values.
    std::vector<double> getValues() const;

signals:
    void parameterChanged();

private:
    QFormLayout* layout_;
    std::vector<QDoubleSpinBox*> spinboxes_;
};
