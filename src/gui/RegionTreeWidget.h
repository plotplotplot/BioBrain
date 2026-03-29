#pragma once

#include <QTreeWidget>
#include <memory>
#include <vector>

namespace biobrain { class BrainRegion; }

class RegionTreeWidget : public QTreeWidget {
    Q_OBJECT
public:
    explicit RegionTreeWidget(QWidget* parent = nullptr);

    void setRegions(const std::vector<std::shared_ptr<biobrain::BrainRegion>>& regions);
    void updateStats();

signals:
    void regionSelected(uint32_t region_id);

private slots:
    void onItemClicked(QTreeWidgetItem* item, int column);

private:
    static QColor colorForRegion(uint32_t id);

    std::vector<std::shared_ptr<biobrain::BrainRegion>> regions_;
};
