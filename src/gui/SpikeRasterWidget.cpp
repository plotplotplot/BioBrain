#include "SpikeRasterWidget.h"

#include <QPainter>
#include <QPaintEvent>
#include <algorithm>

SpikeRasterWidget::SpikeRasterWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumSize(400, 200);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void SpikeRasterWidget::addSpikes(const std::vector<uint32_t>& neuron_ids,
                                   const std::vector<double>& times)
{
    for (size_t i = 0; i < neuron_ids.size() && i < times.size(); ++i) {
        spikes_.push_back({neuron_ids[i], times[i]});
        if (times[i] > current_time_)
            current_time_ = times[i];
    }

    // Evict old spikes beyond ring buffer limit
    while (spikes_.size() > MAX_SPIKES)
        spikes_.pop_front();

    update();
}

void SpikeRasterWidget::paintEvent(QPaintEvent* /*event*/)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);

    const int w = width();
    const int h = height();

    // Dark background
    p.fillRect(rect(), QColor(0x0a, 0x0a, 0x1a));

    if (spikes_.empty()) {
        p.setPen(QColor("#555"));
        p.drawText(rect(), Qt::AlignCenter, "No spike data");
        return;
    }

    // Determine visible time range
    double t_max = current_time_;
    double t_min = t_max - time_window_;

    // Determine visible neuron ID range from recent spikes
    uint32_t id_min = UINT32_MAX;
    uint32_t id_max = 0;
    for (const auto& sp : spikes_) {
        if (sp.time >= t_min && sp.time <= t_max) {
            id_min = std::min(id_min, sp.neuron_id);
            id_max = std::max(id_max, sp.neuron_id);
        }
    }

    if (id_min > id_max) return; // no visible spikes

    double id_range = static_cast<double>(id_max - id_min + 1);

    // Margins for axis labels
    constexpr int marginL = 50;
    constexpr int marginR = 10;
    constexpr int marginT = 10;
    constexpr int marginB = 24;
    int plotW = w - marginL - marginR;
    int plotH = h - marginT - marginB;

    if (plotW <= 0 || plotH <= 0) return;

    // Draw axis lines
    p.setPen(QColor("#333"));
    p.drawLine(marginL, marginT, marginL, marginT + plotH);
    p.drawLine(marginL, marginT + plotH, marginL + plotW, marginT + plotH);

    // Axis labels
    p.setPen(QColor("#666"));
    QFont smallFont = font();
    smallFont.setPointSize(8);
    p.setFont(smallFont);
    p.drawText(marginL, h - 2, QString("%1 ms").arg(t_min, 0, 'f', 1));
    p.drawText(marginL + plotW - 50, h - 2, QString("%1 ms").arg(t_max, 0, 'f', 1));
    p.save();
    p.translate(10, marginT + plotH / 2);
    p.rotate(-90);
    p.drawText(0, 0, "Neuron");
    p.restore();

    // Draw spikes
    // Excitatory (even neuron IDs): magenta #ff44aa
    // Inhibitory (odd neuron IDs): blue #44aaff
    // (Heuristic: real excitatory/inhibitory would come from neuron type)
    const QColor excColor(0xff, 0x44, 0xaa);
    const QColor inhColor(0x44, 0xaa, 0xff);

    for (const auto& sp : spikes_) {
        if (sp.time < t_min || sp.time > t_max) continue;

        double xf = (sp.time - t_min) / time_window_;
        double yf = (sp.neuron_id - id_min) / id_range;

        int x = marginL + static_cast<int>(xf * plotW);
        int y = marginT + static_cast<int>((1.0 - yf) * plotH);

        // Simple heuristic: use neuron_id parity for exc/inh distinction
        p.setPen(Qt::NoPen);
        p.setBrush((sp.neuron_id % 5 == 0) ? inhColor : excColor);
        p.drawRect(x, y, 2, 2);
    }
}
