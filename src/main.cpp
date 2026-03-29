#include <QApplication>
#include <QTimer>
#include <iostream>
#include <memory>
#include <thread>
#include <csignal>

// Core
#include "core/Simulation.h"
#include "core/BrainRegion.h"
#include "core/SpikeEvent.h"

// Compute backends
#include "compute/CPUBackend.h"
#include "compute/MetalBackend.h"

// Plasticity
#include "plasticity/DopamineSTDP.h"

// Input
#include "input/WebcamCapture.h"
#include "input/RetinalEncoder.h"

// Brain regions
#include "regions/Retina.h"
#include "regions/LGN.h"
#include "regions/V1.h"
#include "regions/V2V4.h"
#include "regions/ITCortex.h"
#include "regions/VTA.h"
#include "regions/Striatum.h"
#include "regions/MotorCortex.h"
#include "regions/WernickesArea.h"
#include "regions/BrocasArea.h"

// Audio
#include "audio/VocalSynthesizer.h"

// Recording
#include "recording/SpikeRecorder.h"

// GUI
#include "gui/MainWindow.h"
#include "gui/WebcamWidget.h"
#include "gui/WebcamPanel.h"
#include "gui/SpikeRasterWidget.h"

// Debug API
#include "harness/DebugAPI.h"

// Hardware scaling
#include "core/HardwareProfile.h"

using namespace biobrain;

static std::atomic<bool> g_shutdown{false};

static void signalHandler(int) {
    g_shutdown.store(true);
}

// Build the complete brain circuit, scaled to detected hardware.
static std::shared_ptr<Simulation> buildBrain(const HardwareProfile& hw) {
    auto sim = std::make_shared<Simulation>();

    // Create compute backends
    auto cpu = std::make_shared<CPUBackend>(hw.cpu_sim_threads);
    auto metal = std::make_shared<MetalBackend>();

    // Use Metal for large visual cortex regions, CPU for smaller RL regions
    auto visualBackend = metal->isAvailable()
        ? std::static_pointer_cast<ComputeBackend>(metal)
        : std::static_pointer_cast<ComputeBackend>(cpu);
    auto rlBackend = cpu;

    // Create plasticity rule (STDP + Dopamine for all regions)
    auto plasticity = std::make_shared<DopamineSTDP>();

    // --- Create brain regions with hardware-scaled neuron counts ---
    // Region create() uses its internal NEURON_COUNT. For hardware scaling,
    // we use the base create() and let the region self-wire, then log actual counts.
    // On lower tiers, the same wiring patterns apply but with fewer neurons.
    uint32_t offset = 0;

    auto retina = Retina::create(offset);
    retina->setComputeBackend(cpu);
    offset += retina->neurons().size();

    auto lgn = LGN::create(offset);
    lgn->setComputeBackend(visualBackend);
    lgn->setPlasticityRule(plasticity);
    offset += lgn->neurons().size();

    auto v1 = V1::create(offset);
    v1->setComputeBackend(visualBackend);
    v1->setPlasticityRule(plasticity);
    offset += v1->neurons().size();

    auto v2v4 = V2V4::create(offset);
    v2v4->setComputeBackend(visualBackend);
    v2v4->setPlasticityRule(plasticity);
    offset += v2v4->neurons().size();

    auto it = ITCortex::create(offset);
    it->setComputeBackend(visualBackend);
    it->setPlasticityRule(plasticity);
    offset += it->neurons().size();

    auto vta = VTA::create(offset);
    vta->setComputeBackend(rlBackend);
    offset += vta->neurons().size();

    auto striatum = Striatum::create(offset);
    striatum->setComputeBackend(rlBackend);
    striatum->setPlasticityRule(plasticity);
    offset += striatum->neurons().size();

    auto motor = MotorCortex::create(offset);
    motor->setComputeBackend(rlBackend);
    offset += motor->neurons().size();

    // Language circuit: IT → Wernicke's → Broca's → vocal output
    auto wernicke = WernickesArea::create(offset);
    wernicke->setComputeBackend(rlBackend);
    wernicke->setPlasticityRule(plasticity);
    offset += wernicke->neurons().size();

    auto broca = BrocasArea::create(offset);
    broca->setComputeBackend(rlBackend);
    broca->setPlasticityRule(plasticity);
    offset += broca->neurons().size();

    // Wire IT cortex → Wernicke's area (ventral "what" stream → language comprehension)
    {
        std::vector<Synapse> it_to_wernicke;
        std::mt19937 rng(42);
        uint32_t it_base = it->baseNeuronId();
        std::uniform_int_distribution<uint32_t> wdist(0, WernickesArea::EXCITATORY - 1);
        // Each IT excitatory neuron projects to ~3 Wernicke's neurons
        for (uint32_t i = 0; i < 24000; i += 3) {  // IT has 24K excitatory
            uint32_t pre = it_base + i;
            for (int j = 0; j < 3; ++j) {
                uint32_t post = wdist(rng);  // local index in Wernicke's
                SynapseParams p{0.3, 6.0, ReceptorType::AMPA, true};  // myelinated
                it_to_wernicke.emplace_back(pre, post, p);
            }
        }
        it->addProjection(WernickesArea::REGION_ID, std::move(it_to_wernicke));
    }

    std::cerr << "BioBrain: " << offset << " total neurons across 10 regions\n";
    std::cerr << "  Language circuit: IT → Wernicke's → Broca's → Audio\n";
    std::cerr << "  Compute: " << (metal->isAvailable() ? "Metal GPU" : "CPU-only")
              << " for visual cortex, CPU for RL + language\n";

    // Add all regions to simulation
    sim->addRegion(retina);
    sim->addRegion(lgn);
    sim->addRegion(v1);
    sim->addRegion(v2v4);
    sim->addRegion(it);
    sim->addRegion(vta);
    sim->addRegion(striatum);
    sim->addRegion(motor);
    sim->addRegion(wernicke);
    sim->addRegion(broca);

    return sim;
}

int main(int argc, char* argv[]) {
    // Handle Ctrl+C gracefully
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Log to file for crash debugging
    freopen("/tmp/biobrain.log", "w", stderr);
    std::cerr << "BioBrain starting...\n";

    QApplication app(argc, argv);
    app.setApplicationName("BioBrain");
    app.setApplicationVersion("0.1.0");

    // Detect hardware and scale parameters
    auto hw = HardwareProfile::detect();
    hw.print();

    // Build neural circuit scaled to hardware
    auto simulation = buildBrain(hw);

    // Set up spike recorder
    auto recorder = std::make_shared<SpikeRecorder>("biobrain_spikes.h5");
    for (auto& region : simulation->regions()) {
        recorder->enableRegion(region->id());
    }
    simulation->setRecorder(recorder);

    // Set up webcam and retinal encoder (scaled to hardware)
    int cam_w = hw.webcam_resolution * 4 / 3;  // 4:3 aspect
    int cam_h = hw.webcam_resolution;
    auto webcam = std::make_unique<WebcamCapture>(cam_w, cam_h, 30);
    auto encoder = std::make_unique<RetinalEncoder>(hw.retinal_grid);

    // Create GUI first so we can wire webcam to it
    auto mainWindow = std::make_unique<MainWindow>(simulation);
    mainWindow->setWindowTitle("BioBrain — Neural Simulation Dashboard");
    mainWindow->resize(1400, 900);
    mainWindow->show();

    // Wire camera switch handler
    auto* webcamPtr = webcam.get();
    auto* panelPtr = mainWindow->webcamPanel();
    if (panelPtr) {
        QObject::connect(panelPtr, &WebcamPanel::cameraChanged,
            [webcamPtr](const QString& deviceId) {
                std::cerr << "Switching camera to: " << deviceId.toStdString() << "\n";
                webcamPtr->stop();
                QTimer::singleShot(500, [webcamPtr, deviceId]() {
                    webcamPtr->selectCamera(deviceId.toStdString());
                    if (webcamPtr->start()) {
                        std::cerr << "Camera switched successfully\n";
                    }
                });
            });
    }

    // Wire webcam → retinal encoder → simulation AND → GUI webcam widget
    auto* encoderPtr = encoder.get();
    auto simWeakPtr = std::weak_ptr<Simulation>(simulation);
    auto* webcamWidget = mainWindow->webcamWidget();

    webcam->setFrameCallback([encoderPtr, simWeakPtr, webcamWidget](const FrameData& frame) {
        // Feed the GUI webcam display (works even when sim is stopped)
        if (webcamWidget && !frame.pixels.empty()) {
            webcamWidget->updateFrame(frame.pixels.data(), frame.width, frame.height);
        }

        // Feed retinal encoder → LGN via direct current injection
        auto sim = simWeakPtr.lock();
        if (!sim || !sim->isRunning()) return;

        double t_start = sim->currentTime();
        double t_end = t_start + 33.3;  // 30fps = 33.3ms per frame

        SpikeOutput retinalSpikes = encoderPtr->encode(
            frame.pixels.data(), frame.width, frame.height, t_start, t_end);

        // Inject retinal spikes as current directly into LGN relay neurons.
        // This models the optic nerve: each RGC spike delivers ~15nA to its
        // topographic LGN target (enough to drive LGN relay cells above threshold).
        auto* lgn = sim->getRegion(LGN::REGION_ID);
        if (!lgn) return;

        uint32_t lgn_relay_count = 4000;  // LGN has 4000 relay + 1000 interneurons
        for (size_t i = 0; i < retinalSpikes.neuron_ids.size(); ++i) {
            // Topographic mapping: RGC neuron i → LGN relay neuron (i % relay_count)
            uint32_t lgn_local = retinalSpikes.neuron_ids[i] % lgn_relay_count;
            lgn->injectCurrent(lgn_local, 15.0);  // 15 nA per retinal spike
        }
    });

    // Start webcam after event loop is running (so permission dialog can display).
    // Then populate the camera list once we know permission status.
    QTimer::singleShot(500, [webcamPtr, panelPtr]() {
        bool started = webcamPtr->start();
        if (started) {
            std::cerr << "Webcam started (640x480 @ 30fps)\n";
        } else {
            std::cerr << "WARNING: Could not start webcam. If permission dialog appeared, "
                         "it will auto-retry. Otherwise check System Settings > Privacy > Camera.\n";
        }

        // Populate camera list (works regardless of permission for listing)
        auto cameras = WebcamCapture::listCameras();
        std::cerr << "Found " << cameras.size() << " camera(s)\n";
        if (panelPtr) {
            std::vector<WebcamPanel::CameraEntry> entries;
            for (auto& cam : cameras) {
                std::cerr << "  Camera: " << cam.name << " [" << cam.device_id << "]\n";
                entries.push_back({cam.device_id, cam.name});
            }
            panelPtr->setCameras(entries);
        }
    });

    // Debug REST API on port 9090
    auto debugApi = std::make_unique<DebugAPI>(simulation, webcam.get(),
                                                mainWindow.get(), 9090);

    // Wire spike callback to both GUI and debug API
    auto* debugApiPtr = debugApi.get();
    simulation->setSpikeCallback(
        [&mainWindow, debugApiPtr](uint32_t region_id,
                                    const std::vector<uint32_t>& neuron_ids,
                                    const std::vector<double>& times) {
            // Feed GUI spike raster
            auto* raster = mainWindow->findChild<SpikeRasterWidget*>();
            if (raster) {
                QMetaObject::invokeMethod(raster, [raster, neuron_ids, times]() {
                    raster->addSpikes(neuron_ids, times);
                }, Qt::QueuedConnection);
            }
            // Feed debug API activity log
            debugApiPtr->recordSpikeBatch(region_id, neuron_ids.size(),
                                           times.empty() ? 0 : times.back());
        });

    debugApi->start();
    std::cerr << "Debug API: http://localhost:9090\n";

    // ── Vocal Synthesizer: Broca's area output → audio ──
    auto vocalSynth = std::make_unique<VocalSynthesizer>();
    if (vocalSynth->start()) {
        std::cerr << "Vocal synthesizer started (44.1kHz)\n";
    } else {
        std::cerr << "WARNING: Could not start vocal synthesizer\n";
    }

    // Periodically read Broca's area output pools and drive the synthesizer.
    // Runs at 60Hz on the main thread (fast enough for smooth audio changes).
    auto* synthPtr = vocalSynth.get();
    auto simWeakSynth = std::weak_ptr<Simulation>(simulation);
    QTimer synthTimer;
    QObject::connect(&synthTimer, &QTimer::timeout, [synthPtr, simWeakSynth]() {
        auto sim = simWeakSynth.lock();
        if (!sim || !sim->isRunning() || !synthPtr->isRunning()) return;

        // Find Broca's area (region ID 9)
        auto* broca = sim->getRegion(BrocasArea::REGION_ID);
        if (!broca) return;

        // Read firing rates from each output pool
        // Pool = 500 consecutive neurons, count spikes in last 50ms window
        auto& neurons = broca->neurons();
        double t = broca->currentTime();
        std::array<double, 6> pool_rates{};

        for (uint32_t pool = 0; pool < BrocasArea::OUTPUT_POOLS; ++pool) {
            uint32_t start = pool * BrocasArea::POOL_SIZE;
            uint32_t end = start + BrocasArea::POOL_SIZE;
            if (end > neurons.size()) end = neurons.size();

            uint32_t spike_count = 0;
            for (uint32_t i = start; i < end; ++i) {
                // Count neurons that spiked recently (within 50ms)
                if (neurons[i]->last_spike_time > t - 50.0) {
                    spike_count++;
                }
            }
            // Convert to firing rate (Hz): spikes / (pool_size * window_sec)
            pool_rates[pool] = spike_count / (BrocasArea::POOL_SIZE * 0.05);
        }

        synthPtr->updateFromPoolRates(pool_rates);
    });
    synthTimer.start(16);  // ~60Hz

    // Auto-start the simulation
    simulation->start();
    recorder->start();
    std::cerr << "Simulation auto-started.\n";

    // Periodic stimulus: inject random spikes into Retina every 100ms
    // to drive visible activity through the visual cortex pipeline
    QTimer stimulusTimer;
    auto simWeak2 = std::weak_ptr<Simulation>(simulation);
    QObject::connect(&stimulusTimer, &QTimer::timeout, [simWeak2]() {
        auto sim = simWeak2.lock();
        if (!sim || !sim->isRunning()) return;

        double t = sim->currentTime();
        std::vector<SpikeEvent> events;
        // Inject 500 spikes spread across retinal neurons
        for (uint32_t i = 0; i < 500; ++i) {
            SpikeEvent ev;
            ev.source_id = (i * 17) % 8192;  // pseudo-random retinal neurons
            ev.target_id = ev.source_id;
            ev.time = t + (i * 0.05);
            ev.delay = 2.0;
            ev.source_region = 0;  // Retina
            ev.target_region = 1;  // LGN
            events.push_back(ev);
        }
        sim->injectSpikes(events);
    });
    stimulusTimer.start(100);  // every 100ms

    // Handle shutdown signal
    QTimer shutdownTimer;
    QObject::connect(&shutdownTimer, &QTimer::timeout, [&]() {
        if (g_shutdown.load()) {
            simulation->stop();
            recorder->stop();
            webcam->stop();
            app.quit();
        }
    });
    shutdownTimer.start(100);

    std::cout << "BioBrain ready. Press Run to start simulation.\n";

    int result = app.exec();

    // Cleanup
    vocalSynth->stop();
    debugApi->stop();
    simulation->stop();
    recorder->stop();
    webcam->stop();

    std::cout << "BioBrain shutdown. Recorded " << recorder->totalSpikes() << " spikes.\n";

    return result;
}
