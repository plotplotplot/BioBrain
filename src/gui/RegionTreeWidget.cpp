#include "RegionTreeWidget.h"
#include "core/BrainRegion.h"

#include <QHeaderView>

// Palette of distinct region colors
static const QColor kRegionPalette[] = {
    QColor("#ff6b6b"), // red
    QColor("#ffa544"), // orange
    QColor("#ffd93d"), // yellow
    QColor("#6bcb77"), // green
    QColor("#4d96ff"), // blue
    QColor("#9b59b6"), // purple
    QColor("#e84393"), // pink
    QColor("#00cec9"), // teal
    QColor("#fd79a8"), // salmon
};
static constexpr int kPaletteSize = sizeof(kRegionPalette) / sizeof(kRegionPalette[0]);

RegionTreeWidget::RegionTreeWidget(QWidget* parent)
    : QTreeWidget(parent)
{
    setHeaderLabels({"Region", "Neurons", "Rate (Hz)", "Backend"});
    header()->setStretchLastSection(true);
    header()->setSectionResizeMode(QHeaderView::ResizeToContents);
    setRootIsDecorated(false);
    setAlternatingRowColors(false);
    setIndentation(0);

    connect(this, &QTreeWidget::itemClicked,
            this, &RegionTreeWidget::onItemClicked);
}

QColor RegionTreeWidget::colorForRegion(uint32_t id)
{
    return kRegionPalette[id % kPaletteSize];
}

void RegionTreeWidget::setRegions(
    const std::vector<std::shared_ptr<biobrain::BrainRegion>>& regions)
{
    regions_ = regions;
    clear();

    for (const auto& region : regions_) {
        auto* item = new QTreeWidgetItem(this);
        item->setData(0, Qt::UserRole, region->id());
        item->setText(0, QString::fromStdString(region->name()));
        item->setText(1, QString::number(region->neurons().size()));
        item->setText(2, "0.0");
        item->setText(3, region->computeBackend()
                             ? QString(region->computeBackend()->name())
                             : "none");

        // Color-coded left border via decoration
        QColor c = colorForRegion(region->id());
        item->setForeground(0, QBrush(c));
    }
}

void RegionTreeWidget::updateStats()
{
    for (int i = 0; i < topLevelItemCount() && i < static_cast<int>(regions_.size()); ++i) {
        auto* item = topLevelItem(i);
        const auto& region = regions_[static_cast<size_t>(i)];

        item->setText(2, QString::number(region->firingRate(), 'f', 1));
        item->setText(1, QString::number(region->neurons().size()));

        if (region->computeBackend())
            item->setText(3, QString(region->computeBackend()->name()));
    }
}

void RegionTreeWidget::onItemClicked(QTreeWidgetItem* item, int /*column*/)
{
    if (!item) return;
    bool ok = false;
    uint32_t id = item->data(0, Qt::UserRole).toUInt(&ok);
    if (ok)
        emit regionSelected(id);
}
