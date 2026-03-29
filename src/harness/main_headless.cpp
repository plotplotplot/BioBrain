// BioBrain Headless Mode
// Runs the full simulation with REST API + web dashboard, no Qt GUI required.
// Usage: ./BioBrainHeadless [--port 9090]

#include "harness/RestHarness.h"
#include "harness/DebugAPI.h"
#include "core/Simulation.h"
#include "core/BrainRegion.h"
#include "core/SpikeEvent.h"
#include "core/HardwareProfile.h"
#include "core/IzhikevichNeuron.h"
#include "compute/CPUBackend.h"
#include "plasticity/DopamineSTDP.h"
#include "recording/SpikeRecorder.h"
#include "input/RetinalEncoder.h"

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

#include <iostream>
#include <memory>
#include <csignal>
#include <atomic>
#include <thread>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

using namespace biobrain;

static std::atomic<bool> g_shutdown{false};
static void signalHandler(int) { g_shutdown.store(true); }

static std::shared_ptr<Simulation> buildBrain(const HardwareProfile& hw) {
    auto sim = std::make_shared<Simulation>();
    auto cpu = std::make_shared<CPUBackend>(hw.cpu_sim_threads);
    auto plasticity = std::make_shared<DopamineSTDP>();

    uint32_t offset = 0;
    auto add = [&](auto region, bool with_plasticity = true) {
        region->setComputeBackend(cpu);
        if (with_plasticity) region->setPlasticityRule(plasticity);
        offset += region->neurons().size();
        sim->addRegion(region);
    };

    add(Retina::create(offset), false);
    add(LGN::create(offset));
    add(V1::create(offset));
    add(V2V4::create(offset));
    add(ITCortex::create(offset));
    add(VTA::create(offset), false);
    add(Striatum::create(offset));
    add(MotorCortex::create(offset));
    add(WernickesArea::create(offset));
    add(BrocasArea::create(offset));

    fprintf(stderr, "Brain: %u neurons across %zu regions\n",
            offset, sim->regions().size());
    return sim;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Parse args
    int port = 9090;
    bool kill_existing = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--kill") == 0 || std::strcmp(argv[i], "-k") == 0) {
            kill_existing = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: BioBrainHeadless [options]\n");
            fprintf(stderr, "  --port N    Listen on port N (default: 9090)\n");
            fprintf(stderr, "  --kill, -k  Kill existing BioBrain processes on startup\n");
            fprintf(stderr, "  --help, -h  Show this help\n");
            return 0;
        }
    }

    // Kill existing BioBrain processes if requested
    if (kill_existing) {
        fprintf(stderr, "Killing existing BioBrain processes...\n");
        #ifdef __APPLE__
        system("pkill -f BioBrain 2>/dev/null; sleep 1");
        #else
        system("pkill -f BioBrain 2>/dev/null; sleep 1");
        #endif
    }

    // Check if port is already in use, auto-increment if so
    {
        int test_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (test_fd >= 0) {
            int opt = 1;
            setsockopt(test_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = INADDR_ANY;
            addr.sin_port = htons(port);
            while (bind(test_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
                fprintf(stderr, "Port %d in use, trying %d...\n", port, port + 1);
                port++;
                addr.sin_port = htons(port);
                if (port > 9100) {
                    fprintf(stderr, "ERROR: No available port in range 9090-9100\n");
                    close(test_fd);
                    return 1;
                }
            }
            close(test_fd);
        }
    }

    fprintf(stderr,
        "\n"
        "╔══════════════════════════════════════════════╗\n"
        "║   BioBrain Headless — Web Dashboard Mode     ║\n"
        "╚══════════════════════════════════════════════╝\n\n");

    // Detect hardware
    auto hw = HardwareProfile::detect();
    hw.print();

    // Build brain
    auto sim = buildBrain(hw);

    // Recorder
    auto recorder = std::make_shared<SpikeRecorder>("biobrain_spikes.h5");
    for (auto& r : sim->regions()) recorder->enableRegion(r->id());
    sim->setRecorder(recorder);

    // Debug API (serves web dashboard at /)
    auto debugApi = std::make_unique<DebugAPI>(sim, nullptr, nullptr, port);

    // Wire spike callback to debug API
    auto* apiPtr = debugApi.get();
    sim->setSpikeCallback([apiPtr](uint32_t region_id,
                                    const std::vector<uint32_t>& neuron_ids,
                                    const std::vector<double>& times) {
        apiPtr->recordSpikeBatch(region_id, neuron_ids.size(),
                                  times.empty() ? 0 : times.back());
    });

    debugApi->start();

    // Auto-start simulation
    sim->start();
    recorder->start();

    fprintf(stderr, "Web dashboard: http://localhost:%d\n", port);
    fprintf(stderr, "REST API:      http://localhost:%d/api/sim/status\n", port);
    fprintf(stderr, "\nPress Ctrl+C to stop.\n\n");

    // Wait for shutdown
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    fprintf(stderr, "\nShutting down...\n");
    sim->stop();
    recorder->stop();
    debugApi->stop();
    fprintf(stderr, "Recorded %zu spikes.\n", recorder->totalSpikes());

    return 0;
}
