#pragma once

#include <QWidget>
#include <deque>
#include <vector>
#include <cstdint>

class SpikeRasterWidget : public QWidget {
    Q_OBJECT
public:
    explicit SpikeRasterWidget(QWidget* parent = nullptr);

    void setRegionId(uint32_t id) { region_id_ = id; update(); }

    /// Add spikes to display (called from simulation callback via queued connection).
    void addSpikes(const std::vector<uint32_t>& neuron_ids,
                   const std::vector<double>& times);

    /// Time window to display (ms).
    void setTimeWindow(double ms) { time_window_ = ms; update(); }

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    uint32_t region_id_ = 0;
    double time_window_ = 100.0; // show last 100ms
    double current_time_ = 0.0;

    struct SpikePoint {
        uint32_t neuron_id;
        double time;
    };
    std::deque<SpikePoint> spikes_;

    static constexpr size_t MAX_SPIKES = 50000;
};
