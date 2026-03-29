#include "MainWindow.h"
#include "RegionTreeWidget.h"
#include "SpikeRasterWidget.h"
#include "ActivityMapWidget.h"
#include "WebcamWidget.h"
#include "BackendConfigPanel.h"

#include "core/Simulation.h"
#include "core/BrainRegion.h"

#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QDockWidget>
#include <QAction>
#include <QIcon>
#include <QStyle>
#include <QApplication>
#include <QSplitter>
#include <QFileDialog>
#include <QMessageBox>

MainWindow::MainWindow(std::shared_ptr<biobrain::Simulation> sim,
                       QWidget* parent)
    : QMainWindow(parent)
    , simulation_(std::move(sim))
{
    setWindowTitle("BioBrain - Neural Simulation Dashboard");
    resize(1600, 1000);

    applyDarkTheme();
    setupMenus();
    setupToolbar();
    setupDocks();
    setupStatusBar();

    // Wire up spike callback so raster widget receives live data
    if (simulation_) {
        simulation_->setSpikeCallback(
            [this](uint32_t /*region_id*/,
                   const std::vector<uint32_t>& neuron_ids,
                   const std::vector<double>& times) {
                // Qt queued invocation from simulation thread
                QMetaObject::invokeMethod(spikeRaster_, [this, neuron_ids, times]() {
                    spikeRaster_->addSpikes(neuron_ids, times);
                }, Qt::QueuedConnection);
            });

        // Populate widgets with initial region data
        regionTree_->setRegions(simulation_->regions());
        activityMap_->setRegions(simulation_->regions());
    }

    // 60 Hz status bar refresh
    statusTimer_ = new QTimer(this);
    connect(statusTimer_, &QTimer::timeout, this, &MainWindow::updateStatusBar);
    statusTimer_->start(16); // ~60 Hz
}

MainWindow::~MainWindow() = default;

// ---------------------------------------------------------------------------
// Menus
// ---------------------------------------------------------------------------
void MainWindow::setupMenus()
{
    // --- File ---
    QMenu* fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(tr("&New Config"), this, []() {}, QKeySequence::New);
    fileMenu->addAction(tr("&Open Config..."), this, []() {
        QFileDialog::getOpenFileName(nullptr, "Open Config", QString(), "JSON (*.json)");
    }, QKeySequence::Open);
    fileMenu->addAction(tr("&Save Config"), this, []() {}, QKeySequence::Save);
    fileMenu->addSeparator();
    fileMenu->addAction(tr("&Quit"), qApp, &QApplication::quit, QKeySequence::Quit);

    // --- Simulation ---
    QMenu* simMenu = menuBar()->addMenu(tr("&Simulation"));
    simMenu->addAction(tr("&Run"),   this, &MainWindow::onRun,   QKeySequence(Qt::Key_F5));
    simMenu->addAction(tr("&Pause"), this, &MainWindow::onPause, QKeySequence(Qt::Key_F6));
    simMenu->addAction(tr("S&top"),  this, &MainWindow::onStop,  QKeySequence(Qt::Key_F7));
    simMenu->addSeparator();
    simMenu->addAction(tr("&Reset"), this, [this]() {
        onStop();
    });

    // --- View (dock toggles added after docks are created) ---
    menuBar()->addMenu(tr("&View"));
}

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------
void MainWindow::setupToolbar()
{
    QToolBar* tb = addToolBar(tr("Simulation"));
    tb->setMovable(false);
    tb->setIconSize(QSize(20, 20));

    QAction* runAct = tb->addAction(tr("Run"));
    runAct->setToolTip("Start simulation (F5)");
    // Green icon placeholder via palette-colored text
    connect(runAct, &QAction::triggered, this, &MainWindow::onRun);

    QAction* pauseAct = tb->addAction(tr("Pause"));
    pauseAct->setToolTip("Pause simulation (F6)");
    connect(pauseAct, &QAction::triggered, this, &MainWindow::onPause);

    QAction* stopAct = tb->addAction(tr("Stop"));
    stopAct->setToolTip("Stop simulation (F7)");
    connect(stopAct, &QAction::triggered, this, &MainWindow::onStop);

    // Style the toolbar buttons
    tb->setStyleSheet(
        "QToolBar { background: #12122a; border-bottom: 1px solid #333; spacing: 6px; padding: 4px; }"
        "QToolButton { background: #2a2a4a; color: #e0e0e0; border: 1px solid #444; "
        "              border-radius: 3px; padding: 4px 14px; font-weight: bold; }"
        "QToolButton:hover { background: #3a3a5a; }"
    );
}

// ---------------------------------------------------------------------------
// Dock Widgets
// ---------------------------------------------------------------------------
void MainWindow::setupDocks()
{
    QMenu* viewMenu = menuBar()->actions().back()->menu(); // "View" menu

    // --- Region Tree (left) ---
    {
        auto* dock = new QDockWidget(tr("Brain Regions"), this);
        dock->setObjectName("RegionTreeDock");
        regionTree_ = new RegionTreeWidget(dock);
        dock->setWidget(regionTree_);
        addDockWidget(Qt::LeftDockWidgetArea, dock);
        viewMenu->addAction(dock->toggleViewAction());

        connect(regionTree_, &RegionTreeWidget::regionSelected,
                this, &MainWindow::onRegionSelected);
    }

    // --- Spike Raster (center) ---
    {
        spikeRaster_ = new SpikeRasterWidget(this);
        setCentralWidget(spikeRaster_);
    }

    // --- Activity Map (bottom) ---
    {
        auto* dock = new QDockWidget(tr("Activity Map"), this);
        dock->setObjectName("ActivityMapDock");
        activityMap_ = new ActivityMapWidget(dock);
        dock->setWidget(activityMap_);
        addDockWidget(Qt::BottomDockWidgetArea, dock);
        viewMenu->addAction(dock->toggleViewAction());
    }

    // --- Webcam (bottom, tabbed with activity map) ---
    {
        auto* dock = new QDockWidget(tr("Webcam Feed"), this);
        dock->setObjectName("WebcamDock");
        webcamView_ = new WebcamWidget(dock);
        dock->setWidget(webcamView_);
        addDockWidget(Qt::BottomDockWidgetArea, dock);
        viewMenu->addAction(dock->toggleViewAction());
    }

    // --- Config Panel (right) ---
    {
        auto* dock = new QDockWidget(tr("Backend Config"), this);
        dock->setObjectName("ConfigDock");
        configPanel_ = new BackendConfigPanel(dock);
        dock->setWidget(configPanel_);
        addDockWidget(Qt::RightDockWidgetArea, dock);
        viewMenu->addAction(dock->toggleViewAction());
    }
}

// ---------------------------------------------------------------------------
// Status Bar
// ---------------------------------------------------------------------------
void MainWindow::setupStatusBar()
{
    timeLabel_      = new QLabel("Time: 0.000 s");
    activeLabel_    = new QLabel("Active: 0");
    gpuLabel_       = new QLabel("GPU: idle");
    spikeRateLabel_ = new QLabel("Spikes/s: 0");

    for (auto* lbl : {timeLabel_, activeLabel_, gpuLabel_, spikeRateLabel_}) {
        lbl->setStyleSheet("QLabel { color: #888; padding: 0 12px; }");
        statusBar()->addPermanentWidget(lbl);
    }
}

void MainWindow::updateStatusBar()
{
    if (!simulation_) return;

    double t = simulation_->currentTime();
    timeLabel_->setText(QString("Time: %1 s").arg(t / 1000.0, 0, 'f', 3));
    activeLabel_->setText(QString("Active: %1").arg(simulation_->totalActiveNeurons()));
    spikeRateLabel_->setText(QString("Spikes/s: %1").arg(
        static_cast<int>(simulation_->spikesPerSecond())));

    gpuLabel_->setText(simulation_->isRunning() ? "GPU: active" : "GPU: idle");

    // Refresh sub-widgets
    regionTree_->updateStats();
    activityMap_->updateActivity();
}

// ---------------------------------------------------------------------------
// Slots
// ---------------------------------------------------------------------------
void MainWindow::onRun()
{
    if (!simulation_) return;
    if (simulation_->isPaused())
        simulation_->resume();
    else if (!simulation_->isRunning())
        simulation_->start();
}

void MainWindow::onPause()
{
    if (simulation_ && simulation_->isRunning())
        simulation_->pause();
}

void MainWindow::onStop()
{
    if (simulation_)
        simulation_->stop();
}

void MainWindow::onRegionSelected(uint32_t region_id)
{
    spikeRaster_->setRegionId(region_id);

    if (simulation_) {
        biobrain::BrainRegion* region = simulation_->getRegion(region_id);
        if (region) {
            // Find the shared_ptr for this region
            for (auto& r : simulation_->regions()) {
                if (r->id() == region_id) {
                    configPanel_->setRegion(r);
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dark Theme
// ---------------------------------------------------------------------------
void MainWindow::applyDarkTheme()
{
    setStyleSheet(R"(
        QMainWindow { background: #1a1a2e; color: #e0e0e0; }
        QDockWidget { color: #e0e0e0; }
        QDockWidget::title {
            background: #12122a; padding: 6px; border: 1px solid #333;
        }
        QTreeWidget { background: #12122a; color: #e0e0e0; border: none; }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background: #1a1a3a; color: #fff; border: 1px solid #444;
            padding: 2px 6px;
        }
        QPushButton {
            background: #2a2a4a; color: #e0e0e0; border: 1px solid #444;
            padding: 4px 12px;
        }
        QPushButton:hover { background: #3a3a5a; }
        QSlider::groove:horizontal {
            background: #333; height: 6px; border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #4af; width: 14px; margin: -4px 0; border-radius: 7px;
        }
        QGroupBox {
            border: 1px solid #333; border-radius: 4px;
            margin-top: 8px; padding-top: 16px; color: #888;
        }
        QLabel { color: #aaa; }
        QStatusBar { background: #0d0d1a; color: #888; }
        QMenuBar { background: #12122a; color: #e0e0e0; }
        QMenuBar::item:selected { background: #2a2a4a; }
        QMenu { background: #1a1a3a; color: #e0e0e0; border: 1px solid #444; }
        QMenu::item:selected { background: #3a3a5a; }
        QHeaderView::section {
            background: #1a1a3a; color: #aaa; border: 1px solid #333;
            padding: 4px;
        }
        QCheckBox { color: #aaa; }
        QCheckBox::indicator {
            width: 14px; height: 14px; border: 1px solid #555;
            border-radius: 2px; background: #1a1a3a;
        }
        QCheckBox::indicator:checked { background: #4af; }
    )");
}
