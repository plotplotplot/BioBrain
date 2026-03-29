#include "ActivityMapWidget.h"
#include "core/BrainRegion.h"

#include <QPainter>
#include <QRadialGradient>
#include <cmath>
#include <algorithm>

// Rough brain topology positions (relative 0-1) and colors for known regions.
// Order: Retina, LGN, V1, V2/V4, IT, VTA, Striatum, Motor
static const QColor kRegionColors[] = {
    QColor("#ff6b6b"), QColor("#ffa544"), QColor("#ffd93d"),
    QColor("#6bcb77"), QColor("#4d96ff"), QColor("#9b59b6"),
    QColor("#e84393"), QColor("#00cec9"), QColor("#fd79a8"),
};
static constexpr int kColorCount = sizeof(kRegionColors) / sizeof(kRegionColors[0]);

// Default layout positions for up to 9 regions (rough brain topology)
static const QPointF kDefaultPositions[] = {
    {0.10, 0.40}, // Retina (far left)
    {0.22, 0.35}, // LGN
    {0.38, 0.30}, // V1
    {0.52, 0.28}, // V2/V4
    {0.65, 0.32}, // IT
    {0.45, 0.70}, // VTA (ventral)
    {0.60, 0.65}, // Striatum
    {0.80, 0.45}, // Motor (far right)
    {0.80, 0.65}, // extra
};
static constexpr int kMaxDefaultPositions = sizeof(kDefaultPositions) / sizeof(kDefaultPositions[0]);

ActivityMapWidget::ActivityMapWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumSize(300, 150);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void ActivityMapWidget::setRegions(
    const std::vector<std::shared_ptr<biobrain::BrainRegion>>& regions)
{
    regions_ = regions;
    region_vizs_.clear();

    // Find max neuron count for radius scaling
    size_t maxNeurons = 1;
    for (const auto& r : regions_)
        maxNeurons = std::max(maxNeurons, r->neurons().size());

    for (size_t i = 0; i < regions_.size(); ++i) {
        const auto& r = regions_[i];
        RegionViz viz;
        viz.name = QString::fromStdString(r->name());
        viz.id = r->id();
        viz.color = kRegionColors[i % kColorCount];
        viz.activity = 0.0;
        viz.position = (i < static_cast<size_t>(kMaxDefaultPositions))
                           ? kDefaultPositions[i]
                           : QPointF(0.5, 0.5);
        // Radius proportional to sqrt(neuron count), range 20-60 px
        double frac = std::sqrt(static_cast<double>(r->neurons().size()))
                    / std::sqrt(static_cast<double>(maxNeurons));
        viz.radius = 20.0 + frac * 40.0;

        region_vizs_.push_back(viz);
    }

    update();
}

void ActivityMapWidget::updateActivity()
{
    // Normalize firing rates: assume max ~200 Hz maps to activity 1.0
    constexpr double kMaxRate = 200.0;

    for (size_t i = 0; i < region_vizs_.size() && i < regions_.size(); ++i) {
        double rate = regions_[i]->firingRate();
        region_vizs_[i].activity = std::clamp(rate / kMaxRate, 0.0, 1.0);
    }

    update();
}

void ActivityMapWidget::paintEvent(QPaintEvent* /*event*/)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const int w = width();
    const int h = height();

    // Background
    p.fillRect(rect(), QColor(0x0a, 0x0a, 0x1a));

    if (region_vizs_.empty()) {
        p.setPen(QColor("#555"));
        p.drawText(rect(), Qt::AlignCenter, "No regions loaded");
        return;
    }

    // Draw connections between sequential regions (rough axonal projections)
    p.setPen(QPen(QColor(60, 60, 80), 1.5));
    for (size_t i = 1; i < region_vizs_.size(); ++i) {
        QPointF from(region_vizs_[i - 1].position.x() * w,
                     region_vizs_[i - 1].position.y() * h);
        QPointF to(region_vizs_[i].position.x() * w,
                   region_vizs_[i].position.y() * h);
        p.drawLine(from, to);
    }

    // Draw each region as a glowing circle
    for (const auto& viz : region_vizs_) {
        QPointF center(viz.position.x() * w, viz.position.y() * h);
        double r = viz.radius;

        // Glow radius scales with activity
        double glowRadius = r * (1.0 + viz.activity * 1.5);

        // Radial gradient glow
        QRadialGradient grad(center, glowRadius);
        QColor glowColor = viz.color;
        glowColor.setAlphaF(0.15 + viz.activity * 0.6);
        grad.setColorAt(0.0, glowColor);
        glowColor.setAlphaF(0.0);
        grad.setColorAt(1.0, glowColor);

        p.setPen(Qt::NoPen);
        p.setBrush(grad);
        p.drawEllipse(center, glowRadius, glowRadius);

        // Solid core circle
        QColor coreColor = viz.color;
        coreColor.setAlphaF(0.5 + viz.activity * 0.5);
        p.setBrush(coreColor);
        p.setPen(QPen(viz.color.lighter(140), 1.5));
        p.drawEllipse(center, r * 0.6, r * 0.6);

        // Label
        p.setPen(QColor("#ccc"));
        QFont f = font();
        f.setPointSize(9);
        p.setFont(f);
        QRectF textRect(center.x() - 50, center.y() + r * 0.7, 100, 20);
        p.drawText(textRect, Qt::AlignHCenter | Qt::AlignTop, viz.name);
    }
}
