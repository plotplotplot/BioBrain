#include "NeuronParamEditor.h"

#include <cmath>

NeuronParamEditor::NeuronParamEditor(QWidget* parent)
    : QGroupBox("Neuron Parameters", parent)
    , layout_(new QFormLayout(this))
{
    layout_->setContentsMargins(8, 20, 8, 8);
}

void NeuronParamEditor::setParameters(
    const std::vector<std::pair<std::string, double>>& params)
{
    // Remove old spinboxes
    for (auto* sb : spinboxes_) {
        layout_->removeRow(sb);
    }
    spinboxes_.clear();

    // Create new rows
    for (const auto& [name, value] : params) {
        auto* spin = new QDoubleSpinBox(this);
        spin->setDecimals(4);
        spin->setRange(-1e6, 1e6);
        spin->setSingleStep(std::abs(value) > 1.0 ? 1.0 : 0.01);
        spin->setValue(value);
        spin->setMinimumWidth(100);

        connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, &NeuronParamEditor::parameterChanged);

        layout_->addRow(QString::fromStdString(name) + ":", spin);
        spinboxes_.push_back(spin);
    }
}

std::vector<double> NeuronParamEditor::getValues() const
{
    std::vector<double> vals;
    vals.reserve(spinboxes_.size());
    for (const auto* sb : spinboxes_)
        vals.push_back(sb->value());
    return vals;
}
