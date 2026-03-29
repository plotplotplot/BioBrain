#pragma once

#include <QWidget>
#include <QColor>
#include <QPointF>
#include <QString>
#include <vector>
#include <memory>

namespace biobrain { class BrainRegion; }

class ActivityMapWidget : public QWidget {
    Q_OBJECT
public:
    explicit ActivityMapWidget(QWidget* parent = nullptr);

    void setRegions(const std::vector<std::shared_ptr<biobrain::BrainRegion>>& regions);
    void updateActivity();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    struct RegionViz {
        QString name;
        uint32_t id;
        QColor color;
        double activity;   // 0-1 normalized firing rate
        QPointF position;  // relative position (0-1) in widget
        double radius;     // base radius in pixels
    };

    std::vector<RegionViz> region_vizs_;
    std::vector<std::shared_ptr<biobrain::BrainRegion>> regions_;
};
