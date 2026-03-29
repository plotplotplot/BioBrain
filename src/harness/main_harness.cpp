// BioBrain REST Test Harness
// Lightweight HTTP server for exploring the simulation without Qt6.
// Run: ./BioBrainHarness
// Then: curl http://localhost:8080/api/status

#include "harness/RestHarness.h"
#include "core/Simulation.h"
#include "core/BrainRegion.h"
#include "core/SpikeEvent.h"
#include "core/IzhikevichNeuron.h"
#include "core/HodgkinHuxleyNeuron.h"
#include "core/AdExNeuron.h"
#include "core/LIFNeuron.h"
#include "core/Synapse.h"
#include "core/SpikeRouter.h"
#include "plasticity/STDP.h"
#include "plasticity/DopamineSTDP.h"
#include "compute/CPUBackend.h"
#include "recording/SpikeRecorder.h"
#include "regions/Retina.h"
#include "regions/LGN.h"
#include "regions/V1.h"
#include "regions/V2V4.h"
#include "regions/ITCortex.h"
#include "regions/VTA.h"
#include "regions/Striatum.h"
#include "regions/MotorCortex.h"

#include <iostream>
#include <sstream>
#include <memory>
#include <chrono>
#include <csignal>
#include <atomic>
#include <mutex>
#include <deque>
#include <cmath>

using namespace biobrain;

static std::atomic<bool> g_shutdown{false};
static void signalHandler(int) { g_shutdown.store(true); }

// ─── Spike log for the REST API ────────────────────────────────────────────────
struct SpikeLog {
    std::mutex mutex;
    struct Entry {
        uint32_t region_id;
        uint32_t neuron_count;
        double time;
    };
    std::deque<Entry> entries;
    static constexpr size_t MAX_ENTRIES = 10000;

    void add(uint32_t region_id, uint32_t count, double time) {
        std::lock_guard lock(mutex);
        entries.push_back({region_id, count, time});
        while (entries.size() > MAX_ENTRIES) entries.pop_front();
    }

    std::string toJson(size_t limit = 100) {
        std::lock_guard lock(mutex);
        std::ostringstream ss;
        ss << "[";
        size_t start = entries.size() > limit ? entries.size() - limit : 0;
        bool first = true;
        for (size_t i = start; i < entries.size(); ++i) {
            if (!first) ss << ",";
            ss << "{\"region\":" << entries[i].region_id
               << ",\"spikes\":" << entries[i].neuron_count
               << ",\"time\":" << entries[i].time << "}";
            first = false;
        }
        ss << "]";
        return ss.str();
    }
};

// ─── JSON helpers ──────────────────────────────────────────────────────────────
static std::string jsonString(const std::string& key, const std::string& val) {
    return "\"" + key + "\":\"" + val + "\"";
}
static std::string jsonNum(const std::string& key, double val) {
    std::ostringstream ss;
    ss << "\"" << key << "\":" << val;
    return ss.str();
}
static std::string jsonInt(const std::string& key, int64_t val) {
    return "\"" + key + "\":" + std::to_string(val);
}
static std::string jsonBool(const std::string& key, bool val) {
    return "\"" + key + "\":" + (val ? "true" : "false");
}

// ─── Build a small test brain (subset for fast testing) ────────────────────────
static std::shared_ptr<Simulation> buildTestBrain() {
    auto sim = std::make_shared<Simulation>();
    auto cpu = std::make_shared<CPUBackend>(4);
    auto plasticity = std::make_shared<DopamineSTDP>();

    uint32_t offset = 0;

    auto retina = Retina::create(offset);
    retina->setComputeBackend(cpu);
    offset += Retina::NEURON_COUNT;

    auto lgn = LGN::create(offset);
    lgn->setComputeBackend(cpu);
    lgn->setPlasticityRule(plasticity);
    offset += LGN::NEURON_COUNT;

    auto v1 = V1::create(offset);
    v1->setComputeBackend(cpu);
    v1->setPlasticityRule(plasticity);
    offset += V1::NEURON_COUNT;

    auto v2v4 = V2V4::create(offset);
    v2v4->setComputeBackend(cpu);
    v2v4->setPlasticityRule(plasticity);
    offset += V2V4::NEURON_COUNT;

    auto it = ITCortex::create(offset);
    it->setComputeBackend(cpu);
    it->setPlasticityRule(plasticity);
    offset += ITCortex::NEURON_COUNT;

    auto vta = VTA::create(offset);
    vta->setComputeBackend(cpu);
    offset += VTA::NEURON_COUNT;

    auto striatum = Striatum::create(offset);
    striatum->setComputeBackend(cpu);
    striatum->setPlasticityRule(plasticity);
    offset += Striatum::NEURON_COUNT;

    auto motor = MotorCortex::create(offset);
    motor->setComputeBackend(cpu);
    offset += MotorCortex::NEURON_COUNT;

    sim->addRegion(retina);
    sim->addRegion(lgn);
    sim->addRegion(v1);
    sim->addRegion(v2v4);
    sim->addRegion(it);
    sim->addRegion(vta);
    sim->addRegion(striatum);
    sim->addRegion(motor);

    std::cout << "Test brain: " << offset << " neurons across 8 regions\n";
    return sim;
}

// ─── Standalone neuron tests ───────────────────────────────────────────────────
static std::string testSingleNeuron(const std::string& model, double current,
                                     double duration_ms) {
    std::unique_ptr<Neuron> neuron;

    if (model == "izhikevich" || model == "iz") {
        neuron = IzhikevichNeuron::create(IzhikevichNeuron::Type::RegularSpiking);
    } else if (model == "hh" || model == "hodgkin-huxley") {
        neuron = std::make_unique<HodgkinHuxleyNeuron>();
    } else if (model == "adex") {
        neuron = std::make_unique<AdExNeuron>();
    } else if (model == "lif") {
        neuron = std::make_unique<LIFNeuron>();
    } else {
        return "{\"error\":\"Unknown model. Use: izhikevich, hh, adex, lif\"}";
    }

    double dt = 0.1;
    int steps = static_cast<int>(duration_ms / dt);
    std::vector<double> voltages;
    std::vector<double> times;
    std::vector<double> spike_times;
    voltages.reserve(steps);
    times.reserve(steps);

    for (int i = 0; i < steps; ++i) {
        double t = i * dt;
        bool spiked = neuron->step(dt, current);
        voltages.push_back(neuron->voltage());
        times.push_back(t);
        if (spiked) spike_times.push_back(t);
    }

    // Compute firing rate
    double rate = (spike_times.size() / duration_ms) * 1000.0;

    // Subsample voltage trace for JSON (max 500 points)
    int stride = std::max(1, steps / 500);

    std::ostringstream ss;
    ss << "{" << jsonString("model", model) << ","
       << jsonNum("current_nA", current) << ","
       << jsonNum("duration_ms", duration_ms) << ","
       << jsonInt("total_spikes", spike_times.size()) << ","
       << jsonNum("firing_rate_Hz", rate) << ","
       << "\"spike_times\":[";
    for (size_t i = 0; i < spike_times.size(); ++i) {
        if (i) ss << ",";
        ss << spike_times[i];
    }
    ss << "],\"voltage_trace\":{\"dt\":" << (dt * stride) << ",\"values\":[";
    bool first = true;
    for (int i = 0; i < steps; i += stride) {
        if (!first) ss << ",";
        ss << std::round(voltages[i] * 100) / 100;
        first = false;
    }
    ss << "]}}";
    return ss.str();
}

// ─── Test synapse ──────────────────────────────────────────────────────────────
static std::string testSynapse(const std::string& receptor_str) {
    ReceptorType receptor = ReceptorType::AMPA;
    if (receptor_str == "nmda") receptor = ReceptorType::NMDA;
    else if (receptor_str == "gaba_a") receptor = ReceptorType::GABA_A;
    else if (receptor_str == "gaba_b") receptor = ReceptorType::GABA_B;

    SynapseParams sp{0.5, 1.0, receptor, true};
    Synapse syn(0, 1, sp);

    double dt = 0.1;
    int steps = 500;  // 50ms
    std::vector<double> conductances;
    std::vector<double> currents;

    // Deliver a spike at t=5ms
    for (int i = 0; i < steps; ++i) {
        double t = i * dt;
        if (std::abs(t - 5.0) < dt / 2) {
            syn.deliverSpike(t);
        }
        double I = syn.computeCurrent(-65.0, dt);  // resting potential
        conductances.push_back(syn.conductance());
        currents.push_back(I);
    }

    // Subsample
    int stride = std::max(1, steps / 200);
    std::ostringstream ss;
    ss << "{" << jsonString("receptor", receptor_str) << ","
       << jsonNum("weight", 0.5) << ","
       << jsonNum("spike_time_ms", 5.0) << ","
       << "\"conductance\":[";
    bool first = true;
    for (int i = 0; i < steps; i += stride) {
        if (!first) ss << ",";
        ss << conductances[i];
        first = false;
    }
    ss << "],\"current\":[";
    first = true;
    for (int i = 0; i < steps; i += stride) {
        if (!first) ss << ",";
        ss << currents[i];
        first = false;
    }
    ss << "],\"dt\":" << (dt * stride) << "}";
    return ss.str();
}

// ─── Test STDP ─────────────────────────────────────────────────────────────────
static std::string testSTDP() {
    std::ostringstream ss;
    ss << "{\"stdp_curve\":[";

    bool first = true;
    for (double delta = -50.0; delta <= 50.0; delta += 1.0) {
        STDP stdp;
        if (delta > 0) {
            // Pre before post
            stdp.onPreSpike(0.0, 0);
            stdp.onPostSpike(delta, 0);
        } else {
            // Post before pre
            stdp.onPostSpike(0.0, 0);
            stdp.onPreSpike(-delta, 0);
        }
        double dw = stdp.computeWeightChange(0.1, 0.0);

        if (!first) ss << ",";
        ss << "{\"delta_t\":" << delta << ",\"dw\":" << dw << "}";
        first = false;
    }
    ss << "]}";
    return ss.str();
}

// ─── Test spike router ─────────────────────────────────────────────────────────
static std::string testSpikeRouter() {
    SpikeRouter router;

    // Submit 10 spikes with different delays
    for (int i = 0; i < 10; ++i) {
        SpikeEvent ev;
        ev.source_id = i;
        ev.target_id = i + 100;
        ev.time = 0.0;
        ev.delay = i * 2.0;  // delays: 0, 2, 4, ..., 18 ms
        ev.source_region = 0;
        ev.target_region = 1;
        router.submitSpike(ev);
    }

    std::ostringstream ss;
    ss << "{\"test\":\"spike_router\","
       << "\"pending_after_submit\":" << router.pendingCount() << ","
       << "\"events_by_time\":[";

    bool first = true;
    for (double t = 0; t <= 20; t += 2.0) {
        auto events = router.getEventsUntil(t);
        if (!events.empty()) {
            if (!first) ss << ",";
            ss << "{\"time\":" << t << ",\"delivered\":" << events.size()
               << ",\"source_ids\":[";
            for (size_t j = 0; j < events.size(); ++j) {
                if (j) ss << ",";
                ss << events[j].source_id;
            }
            ss << "]}";
            first = false;
        }
    }

    ss << "],\"pending_after_drain\":" << router.pendingCount() << "}";
    return ss.str();
}

// ─── Main ──────────────────────────────────────────────────────────────────────
int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "╔══════════════════════════════════════════╗\n"
              << "║     BioBrain REST Test Harness v0.1      ║\n"
              << "╚══════════════════════════════════════════╝\n\n";

    // Build brain
    std::cout << "Building brain...\n";
    auto sim = buildTestBrain();

    // Spike log
    auto spikeLog = std::make_shared<SpikeLog>();
    sim->setSpikeCallback([spikeLog](uint32_t region_id,
                                      const std::vector<uint32_t>& neuron_ids,
                                      const std::vector<double>& times) {
        spikeLog->add(region_id, neuron_ids.size(),
                      times.empty() ? 0 : times.back());
    });

    // REST server
    RestHarness server(8080);

    // ── GET /api/status ──
    server.route("/api/status", [&](const std::string&, const std::string&) {
        std::ostringstream ss;
        ss << "{" << jsonBool("running", sim->isRunning()) << ","
           << jsonBool("paused", sim->isPaused()) << ","
           << jsonNum("sim_time_ms", sim->currentTime()) << ","
           << jsonInt("total_active_neurons", sim->totalActiveNeurons()) << ","
           << jsonNum("spikes_per_second", sim->spikesPerSecond()) << ","
           << jsonInt("region_count", sim->regions().size()) << "}";
        return ss.str();
    });

    // ── GET /api/regions ──
    server.route("/api/regions", [&](const std::string&, const std::string&) {
        std::ostringstream ss;
        ss << "[";
        bool first = true;
        for (auto& r : sim->regions()) {
            if (!first) ss << ",";
            ss << "{" << jsonInt("id", r->id()) << ","
               << jsonString("name", r->name()) << ","
               << jsonInt("neuron_count", r->neurons().size()) << ","
               << jsonNum("firing_rate_Hz", r->firingRate()) << ","
               << jsonInt("active_neurons", r->activeNeuronCount()) << ","
               << jsonString("model", r->neuronModel() == NeuronModelType::Izhikevich ? "Izhikevich"
                   : r->neuronModel() == NeuronModelType::HodgkinHuxley ? "Hodgkin-Huxley"
                   : r->neuronModel() == NeuronModelType::AdEx ? "AdEx" : "LIF") << ","
               << jsonString("backend", r->computeBackend() ? r->computeBackend()->name() : "none")
               << "}";
            first = false;
        }
        ss << "]";
        return ss.str();
    });

    // ── POST /api/start ──
    server.route("/api/start", [&](const std::string&, const std::string&) {
        sim->start();
        return "{\"status\":\"started\"}";
    });

    // ── POST /api/pause ──
    server.route("/api/pause", [&](const std::string&, const std::string&) {
        sim->pause();
        return "{\"status\":\"paused\"}";
    });

    // ── POST /api/stop ──
    server.route("/api/stop", [&](const std::string&, const std::string&) {
        sim->stop();
        return "{\"status\":\"stopped\"}";
    });

    // ── GET /api/spikes ──
    server.route("/api/spikes", [&](const std::string&, const std::string&) {
        return spikeLog->toJson(200);
    });

    // ── POST /api/inject — inject a burst of spikes into a region ──
    server.route("/api/inject", [&](const std::string&, const std::string& body) {
        // Simple: inject 100 spikes into Retina to test propagation
        uint32_t region = 0;  // Retina
        uint32_t count = 100;

        // Parse simple params from body: region=N&count=M
        if (body.find("region=") != std::string::npos) {
            auto pos = body.find("region=");
            region = std::stoi(body.substr(pos + 7));
        }
        if (body.find("count=") != std::string::npos) {
            auto pos = body.find("count=");
            count = std::stoi(body.substr(pos + 6));
        }

        double t = sim->currentTime();
        std::vector<SpikeEvent> events;
        for (uint32_t i = 0; i < count; ++i) {
            SpikeEvent ev;
            ev.source_id = i;
            ev.target_id = i;
            ev.time = t + (i * 0.1);  // spread over 10ms
            ev.delay = 2.0;
            ev.source_region = region;
            ev.target_region = (region < 7) ? region + 1 : region;
            events.push_back(ev);
        }
        sim->injectSpikes(events);

        std::ostringstream ss;
        ss << "{\"injected\":" << count
           << ",\"source_region\":" << region
           << ",\"at_time\":" << t << "}";
        return ss.str();
    });

    // ── GET /api/test/neuron?model=izhikevich&current=10&duration=100 ──
    server.route("/api/test/neuron", [](const std::string& path, const std::string&) {
        std::string model = "izhikevich";
        double current = 10.0;
        double duration = 100.0;

        auto qpos = path.find('?');
        if (qpos != std::string::npos) {
            std::string query = path.substr(qpos + 1);
            if (auto p = query.find("model="); p != std::string::npos) {
                auto end = query.find('&', p);
                model = query.substr(p + 6, end == std::string::npos ? end : end - p - 6);
            }
            if (auto p = query.find("current="); p != std::string::npos) {
                current = std::stod(query.substr(p + 8));
            }
            if (auto p = query.find("duration="); p != std::string::npos) {
                duration = std::stod(query.substr(p + 9));
            }
        }
        return testSingleNeuron(model, current, duration);
    });

    // ── GET /api/test/synapse?receptor=ampa ──
    server.route("/api/test/synapse", [](const std::string& path, const std::string&) {
        std::string receptor = "ampa";
        auto qpos = path.find("receptor=");
        if (qpos != std::string::npos) {
            auto end = path.find('&', qpos);
            receptor = path.substr(qpos + 9, end == std::string::npos ? end : end - qpos - 9);
        }
        return testSynapse(receptor);
    });

    // ── GET /api/test/stdp ──
    server.route("/api/test/stdp", [](const std::string&, const std::string&) {
        return testSTDP();
    });

    // ── GET /api/test/router ──
    server.route("/api/test/router", [](const std::string&, const std::string&) {
        return testSpikeRouter();
    });

    // ── GET / — index page ──
    server.route("/", [](const std::string& path, const std::string&) {
        if (path != "/" && path != "/index.html") return std::string("{\"error\":\"not found\"}");
        return std::string(
            "<!DOCTYPE html><html><head><title>BioBrain Harness</title>"
            "<style>body{background:#1a1a2e;color:#e0e0e0;font-family:monospace;padding:20px;}"
            "a{color:#4af;} h1{color:#f4a;} pre{background:#12122a;padding:12px;border-radius:8px;overflow-x:auto;}"
            ".endpoint{margin:8px 0;} code{color:#4fa;}</style></head><body>"
            "<h1>BioBrain REST Test Harness</h1>"
            "<h2>Simulation Control</h2>"
            "<div class='endpoint'><code>GET  /api/status</code> — simulation state</div>"
            "<div class='endpoint'><code>GET  /api/regions</code> — all brain regions</div>"
            "<div class='endpoint'><code>POST /api/start</code> — start simulation</div>"
            "<div class='endpoint'><code>POST /api/pause</code> — pause simulation</div>"
            "<div class='endpoint'><code>POST /api/stop</code> — stop simulation</div>"
            "<div class='endpoint'><code>GET  /api/spikes</code> — recent spike log</div>"
            "<div class='endpoint'><code>POST /api/inject</code> — inject spikes (body: region=0&count=100)</div>"
            "<h2>Component Tests</h2>"
            "<div class='endpoint'><code>GET  /api/test/neuron?model=izhikevich&current=10&duration=100</code></div>"
            "<div class='endpoint'><code>GET  /api/test/neuron?model=hh&current=10&duration=100</code></div>"
            "<div class='endpoint'><code>GET  /api/test/neuron?model=adex&current=500&duration=100</code></div>"
            "<div class='endpoint'><code>GET  /api/test/neuron?model=lif&current=20&duration=100</code></div>"
            "<div class='endpoint'><code>GET  /api/test/synapse?receptor=ampa</code></div>"
            "<div class='endpoint'><code>GET  /api/test/synapse?receptor=nmda</code></div>"
            "<div class='endpoint'><code>GET  /api/test/synapse?receptor=gaba_a</code></div>"
            "<div class='endpoint'><code>GET  /api/test/stdp</code> — STDP timing curve</div>"
            "<div class='endpoint'><code>GET  /api/test/router</code> — spike router test</div>"
            "</body></html>"
        );
    });

    std::cout << "\nEndpoints:\n"
              << "  GET  http://localhost:8080/              — web index\n"
              << "  GET  http://localhost:8080/api/status    — simulation status\n"
              << "  GET  http://localhost:8080/api/regions   — brain regions\n"
              << "  POST http://localhost:8080/api/start     — start sim\n"
              << "  GET  http://localhost:8080/api/test/neuron?model=izhikevich&current=10&duration=100\n"
              << "  GET  http://localhost:8080/api/test/stdp — STDP curve\n"
              << "  GET  http://localhost:8080/api/test/router — spike router\n"
              << "\nPress Ctrl+C to stop.\n\n";

    // Run server (blocks)
    server.startAsync();

    // Wait for shutdown
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\nShutting down...\n";
    sim->stop();
    server.stop();
    return 0;
}
