#include "harness/DebugAPI.h"
#include "core/Simulation.h"
#include "core/BrainRegion.h"
#include "core/Neuron.h"
#include "core/Synapse.h"
#include "core/IzhikevichNeuron.h"
#include "compute/ComputeBackend.h"
#include "plasticity/PlasticityRule.h"
#include "input/WebcamCapture.h"
#include "gui/MainWindow.h"

#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

#include <QPixmap>
#include <QScreen>
#include <QGuiApplication>
#include <QBuffer>
#include <QByteArray>
#include <QWidget>

using namespace biobrain;

// ─── JSON helpers ──────────────────────────────────────────────────────────────
static std::string jStr(const std::string& k, const std::string& v) {
    return "\"" + k + "\":\"" + v + "\"";
}
static std::string jNum(const std::string& k, double v) {
    std::ostringstream s; s << "\"" << k << "\":" << v; return s.str();
}
static std::string jInt(const std::string& k, int64_t v) {
    return "\"" + k + "\":" + std::to_string(v);
}
static std::string jBool(const std::string& k, bool v) {
    return "\"" + k + "\":" + (v ? "true" : "false");
}

// ─── Constructor ───────────────────────────────────────────────────────────────
DebugAPI::DebugAPI(std::shared_ptr<Simulation> sim,
                   WebcamCapture* webcam,
                   MainWindow* window,
                   int port)
    : sim_(std::move(sim))
    , webcam_(webcam)
    , window_(window)
    , server_(port)
    , start_time_(std::chrono::steady_clock::now())
{
    registerRoutes();
}

DebugAPI::~DebugAPI() { stop(); }

void DebugAPI::start() { server_.startAsync(); }
void DebugAPI::stop() { server_.stop(); }

void DebugAPI::recordStepTiming(double step_ms) {
    std::lock_guard lock(timing_mutex_);
    TimingSample ts;
    ts.step_duration_ms = step_ms;
    ts.sim_time = sim_ ? sim_->currentTime() : 0;
    ts.wall_time = std::chrono::steady_clock::now();
    timing_samples_.push_back(ts);
    while (timing_samples_.size() > MAX_TIMING_SAMPLES)
        timing_samples_.pop_front();
}

void DebugAPI::recordSpikeBatch(uint32_t region_id, uint32_t count, double sim_time) {
    std::lock_guard lock(spike_mutex_);
    spike_log_.push_back({region_id, count, sim_time});
    while (spike_log_.size() > MAX_SPIKE_LOG)
        spike_log_.pop_front();
}

// ─── Screenshot via Qt ─────────────────────────────────────────────────────────
std::string DebugAPI::captureScreenshotBase64() {
    if (!window_) return "";

    QPixmap pixmap;
    // Capture the main window widget
    QMetaObject::invokeMethod(window_, [&]() {
        pixmap = window_->grab();
    }, Qt::BlockingQueuedConnection);

    if (pixmap.isNull()) return "";

    QByteArray ba;
    QBuffer buffer(&ba);
    buffer.open(QIODevice::WriteOnly);
    pixmap.save(&buffer, "PNG");

    return ba.toBase64().toStdString();
}

// ─── Route registration ────────────────────────────────────────────────────────
void DebugAPI::registerRoutes() {

    // ── GET / — debug dashboard ──
    server_.route("/", [](const std::string& path, const std::string&) -> std::string {
        if (path != "/" && path != "/index.html")
            return "{\"error\":\"not found\"}";
        return R"HTML(<!DOCTYPE html><html><head><title>BioBrain Debug API</title>
<style>
body{background:#0d0d1a;color:#e0e0e0;font-family:monospace;padding:20px;max-width:900px;margin:0 auto;}
a{color:#4af;} h1{color:#f4a;} h2{color:#6bcb77;margin-top:24px;}
code{color:#4fa;background:#1a1a3a;padding:2px 6px;border-radius:3px;}
.ep{margin:6px 0;} .cat{opacity:0.6;font-size:12px;}
pre{background:#12122a;padding:12px;border-radius:8px;overflow-x:auto;}
#screenshot{max-width:100%;border:1px solid #333;border-radius:8px;margin:12px 0;}
button{background:#2a2a4a;color:#e0e0e0;border:1px solid #555;padding:6px 16px;border-radius:4px;cursor:pointer;margin:4px;}
button:hover{background:#3a3a5a;}
#log{background:#0a0a1a;padding:12px;border-radius:8px;max-height:300px;overflow-y:auto;font-size:12px;}
</style></head><body>
<h1>BioBrain Debug API</h1>

<h2>Controls</h2>
<button onclick="f('/api/sim/start','POST')">Start</button>
<button onclick="f('/api/sim/pause','POST')">Pause</button>
<button onclick="f('/api/sim/resume','POST')">Resume</button>
<button onclick="f('/api/sim/stop','POST')">Stop</button>
<button onclick="f('/api/sim/inject','POST')">Inject 200 spikes</button>
<button onclick="loadScreenshot()">Screenshot</button>
<button onclick="location.reload()">Refresh</button>

<img id="screenshot" style="display:none;">

<h2>Live Status</h2>
<pre id="status">Loading...</pre>

<h2>Regions</h2>
<pre id="regions">Loading...</pre>

<h2>Profiling</h2>
<pre id="profile">Loading...</pre>

<h2>Activity Log</h2>
<div id="log">Loading...</div>

<h2>Endpoints</h2>
<div class="cat">Simulation Control</div>
<div class="ep"><code>GET  /api/sim/status</code> - full simulation state</div>
<div class="ep"><code>POST /api/sim/start</code> / <code>pause</code> / <code>resume</code> / <code>stop</code></div>
<div class="ep"><code>POST /api/sim/inject</code> - inject spike burst</div>
<div class="cat">Inspection</div>
<div class="ep"><code>GET  /api/regions</code> - all regions with stats</div>
<div class="ep"><code>GET  /api/region/{id}</code> - detailed region info</div>
<div class="ep"><code>GET  /api/neuron/{region_id}/{local_idx}</code> - single neuron state</div>
<div class="ep"><code>GET  /api/synapses/{region_id}?limit=20</code> - synapse dump</div>
<div class="cat">Profiling</div>
<div class="ep"><code>GET  /api/profile</code> - step timing stats</div>
<div class="ep"><code>GET  /api/profile/history?last=100</code> - raw timing samples</div>
<div class="cat">Activity</div>
<div class="ep"><code>GET  /api/spikes?last=200</code> - recent spike batches</div>
<div class="ep"><code>GET  /api/activity</code> - per-region firing rates</div>
<div class="cat">Debug</div>
<div class="ep"><code>GET  /api/screenshot</code> - PNG screenshot (base64)</div>
<div class="ep"><code>GET  /api/webcam/status</code> - camera state</div>
<div class="ep"><code>GET  /api/memory</code> - memory usage estimate</div>

<script>
async function f(url,method){
  method=method||'GET';
  var r=await fetch(url,{method:method});var j=await r.json();
  document.getElementById('log').innerHTML='<pre>'+JSON.stringify(j,null,2)+'</pre>'+document.getElementById('log').innerHTML;
}
async function loadScreenshot(){
  var r=await fetch('/api/screenshot');var j=await r.json();
  if(j.base64){var img=document.getElementById('screenshot');img.src='data:image/png;base64,'+j.base64;img.style.display='block';}
}
async function refresh(){
  try{
    var r=await fetch('/api/sim/status');document.getElementById('status').textContent=JSON.stringify(await r.json(),null,2);
    r=await fetch('/api/regions');document.getElementById('regions').textContent=JSON.stringify(await r.json(),null,2);
    r=await fetch('/api/profile');document.getElementById('profile').textContent=JSON.stringify(await r.json(),null,2);
  }catch(e){}
}
refresh();setInterval(refresh,1000);
</script></body></html>)HTML";
    });

    // ── Simulation control ──

    server_.route("/api/sim/status", [this](const std::string&, const std::string&) {
        auto uptime = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time_).count();
        std::ostringstream ss;
        ss << "{" << jBool("running", sim_->isRunning()) << ","
           << jBool("paused", sim_->isPaused()) << ","
           << jNum("sim_time_ms", sim_->currentTime()) << ","
           << jNum("sim_time_sec", sim_->currentTime() / 1000.0) << ","
           << jInt("total_active_neurons", sim_->totalActiveNeurons()) << ","
           << jNum("spikes_per_second", sim_->spikesPerSecond()) << ","
           << jInt("region_count", sim_->regions().size()) << ","
           << jNum("uptime_sec", uptime) << ","
           << jBool("webcam_active", webcam_ && webcam_->isRunning())
           << "}";
        return ss.str();
    });

    server_.route("/api/sim/start", [this](const std::string&, const std::string&) {
        sim_->start(); return std::string("{\"ok\":true,\"action\":\"started\"}");
    });
    server_.route("/api/sim/pause", [this](const std::string&, const std::string&) {
        sim_->pause(); return std::string("{\"ok\":true,\"action\":\"paused\"}");
    });
    server_.route("/api/sim/resume", [this](const std::string&, const std::string&) {
        sim_->resume(); return std::string("{\"ok\":true,\"action\":\"resumed\"}");
    });
    server_.route("/api/sim/stop", [this](const std::string&, const std::string&) {
        sim_->stop(); return std::string("{\"ok\":true,\"action\":\"stopped\"}");
    });

    server_.route("/api/sim/inject", [this](const std::string&, const std::string&) {
        double t = sim_->currentTime();
        std::vector<SpikeEvent> events;
        for (uint32_t i = 0; i < 200; ++i) {
            SpikeEvent ev;
            ev.source_id = i; ev.target_id = i;
            ev.time = t + (i * 0.1); ev.delay = 2.0;
            ev.source_region = 0; ev.target_region = 1;
            events.push_back(ev);
        }
        sim_->injectSpikes(events);
        std::ostringstream ss;
        ss << "{\"ok\":true,\"injected\":200,\"at_time\":" << t << "}";
        return ss.str();
    });

    // ── Inject spikes into any region ──
    // POST /api/sim/inject/region?id=9&count=500 (Broca's = 9)
    server_.route("/api/sim/inject/region", [this](const std::string& path, const std::string& body) {
        uint32_t region_id = 9;  // default Broca's
        uint32_t count = 500;
        double spread_ms = 20.0;

        auto parse = [](const std::string& s, const std::string& key) -> std::string {
            auto p = s.find(key + "=");
            if (p == std::string::npos) return "";
            auto end = s.find('&', p);
            return s.substr(p + key.size() + 1, end == std::string::npos ? end : end - p - key.size() - 1);
        };

        std::string src = path + "&" + body;
        auto id_s = parse(src, "id");     if (!id_s.empty()) region_id = std::stoi(id_s);
        auto cnt_s = parse(src, "count"); if (!cnt_s.empty()) count = std::stoi(cnt_s);

        auto* region = sim_->getRegion(region_id);
        if (!region) {
            return std::string("{\"error\":\"Region not found\",\"id\":" + std::to_string(region_id) + "}");
        }

        // Force-fire neurons by setting their voltage above threshold.
        // This is more reliable than current injection which gets cleared per-step.
        uint32_t n = region->neurons().size();
        uint32_t actually_injected = 0;
        auto& neurons = region->neurons();
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t local = (i * 7) % n;
            // Inject strong current (150 nA — Izhikevich needs ~10+ for one spike)
            region->injectCurrent(local, 150.0);
            actually_injected++;
        }

        double t = sim_->currentTime();
        std::ostringstream ss;
        ss << "{\"ok\":true,\"region\":\"" << region->name()
           << "\",\"region_id\":" << region_id
           << ",\"neurons_stimulated\":" << actually_injected
           << ",\"current_nA\":20"
           << ",\"at_time\":" << t << "}";
        return ss.str();
    });

    // ── Region inspection ──

    server_.route("/api/regions", [this](const std::string&, const std::string&) {
        std::ostringstream ss;
        ss << "[";
        bool first = true;
        for (auto& r : sim_->regions()) {
            if (!first) ss << ",";
            ss << "{"
               << jInt("id", r->id()) << ","
               << jStr("name", r->name()) << ","
               << jInt("neuron_count", r->neurons().size()) << ","
               << jNum("firing_rate_Hz", r->firingRate()) << ","
               << jInt("active_neurons", r->activeNeuronCount()) << ","
               << jStr("backend", r->computeBackend() ? r->computeBackend()->name() : "none") << ","
               << jInt("internal_synapses", r->internalSynapses().size()) << ","
               << jInt("projection_targets", r->projections().size())
               << "}";
            first = false;
        }
        ss << "]";
        return ss.str();
    });

    server_.route("/api/region/", [this](const std::string& path, const std::string&) {
        // Extract region ID from /api/region/{id}
        auto last_slash = path.rfind('/');
        if (last_slash == std::string::npos) return std::string("{\"error\":\"missing id\"}");
        uint32_t rid = std::stoi(path.substr(last_slash + 1));
        auto* r = sim_->getRegion(rid);
        if (!r) return std::string("{\"error\":\"region not found\"}");

        // Sample first 10 neuron voltages
        std::ostringstream vs;
        vs << "[";
        size_t n = std::min(r->neurons().size(), size_t(10));
        for (size_t i = 0; i < n; ++i) {
            if (i) vs << ",";
            vs << std::round(r->neurons()[i]->voltage() * 100) / 100;
        }
        vs << "]";

        std::ostringstream ss;
        ss << "{"
           << jInt("id", r->id()) << ","
           << jStr("name", r->name()) << ","
           << jInt("neuron_count", r->neurons().size()) << ","
           << jNum("firing_rate_Hz", r->firingRate()) << ","
           << jInt("active_neurons", r->activeNeuronCount()) << ","
           << jNum("current_time_ms", r->currentTime()) << ","
           << jInt("internal_synapses", r->internalSynapses().size()) << ","
           << "\"sample_voltages\":" << vs.str() << ","
           << "\"projections\":[";
        bool first = true;
        for (auto& [target_id, syns] : r->projections()) {
            if (!first) ss << ",";
            ss << "{" << jInt("target_region", target_id) << "," << jInt("synapse_count", syns.size()) << "}";
            first = false;
        }
        ss << "]}";
        return ss.str();
    });

    // ── Single neuron inspection ──

    server_.route("/api/neuron/", [this](const std::string& path, const std::string&) {
        // /api/neuron/{region_id}/{local_idx}
        auto parts = path.substr(12); // after "/api/neuron/"
        auto slash = parts.find('/');
        if (slash == std::string::npos) return std::string("{\"error\":\"use /api/neuron/{region_id}/{local_idx}\"}");
        uint32_t rid = std::stoi(parts.substr(0, slash));
        uint32_t idx = std::stoi(parts.substr(slash + 1));

        auto* r = sim_->getRegion(rid);
        if (!r) return std::string("{\"error\":\"region not found\"}");
        if (idx >= r->neurons().size()) return std::string("{\"error\":\"index out of range\"}");

        auto& neuron = r->neurons()[idx];
        std::ostringstream ss;
        ss << "{"
           << jInt("global_id", neuron->id) << ","
           << jInt("region_id", rid) << ","
           << jInt("local_index", idx) << ","
           << jNum("voltage_mV", neuron->voltage()) << ","
           << jNum("recovery", neuron->recoveryVariable()) << ","
           << jNum("last_spike_ms", neuron->last_spike_time);

        // If Izhikevich, include params
        auto* iz = dynamic_cast<IzhikevichNeuron*>(neuron.get());
        if (iz) {
            ss << "," << jStr("model", "Izhikevich")
               << "," << jNum("a", iz->a)
               << "," << jNum("b", iz->b)
               << "," << jNum("c", iz->c)
               << "," << jNum("d", iz->d);
        }
        ss << "}";
        return ss.str();
    });

    // ── Synapse dump ──

    server_.route("/api/synapses/", [this](const std::string& path, const std::string&) {
        auto after = path.substr(14); // after "/api/synapses/"
        auto qpos = after.find('?');
        uint32_t rid = std::stoi(after.substr(0, qpos));
        size_t limit = 20;
        if (qpos != std::string::npos) {
            auto lp = after.find("limit=", qpos);
            if (lp != std::string::npos) limit = std::stoi(after.substr(lp + 6));
        }

        auto* r = sim_->getRegion(rid);
        if (!r) return std::string("{\"error\":\"region not found\"}");

        auto& syns = r->internalSynapses();
        std::ostringstream ss;
        ss << "{" << jInt("region_id", rid) << ","
           << jInt("total_synapses", syns.size()) << ","
           << "\"sample\":[";
        size_t n = std::min(syns.size(), limit);
        for (size_t i = 0; i < n; ++i) {
            if (i) ss << ",";
            auto& s = syns[i];
            const char* rtype = "AMPA";
            if (s.receptor == ReceptorType::NMDA) rtype = "NMDA";
            else if (s.receptor == ReceptorType::GABA_A) rtype = "GABA_A";
            else if (s.receptor == ReceptorType::GABA_B) rtype = "GABA_B";
            ss << "{"
               << jInt("pre", s.pre_id) << ","
               << jInt("post", s.post_id) << ","
               << jNum("weight", s.weight) << ","
               << jNum("delay_ms", s.delay) << ","
               << jStr("receptor", rtype) << ","
               << jNum("conductance_nS", s.conductance())
               << "}";
        }
        ss << "]}";
        return ss.str();
    });

    // ── Profiling ──

    server_.route("/api/profile", [this](const std::string& path, const std::string&) {
        std::lock_guard lock(timing_mutex_);
        if (timing_samples_.empty()) {
            return std::string("{\"samples\":0,\"msg\":\"No timing data yet. Simulation may not be running.\"}");
        }

        // Compute stats over last 1000 samples
        size_t n = std::min(timing_samples_.size(), size_t(1000));
        double sum = 0, min_v = 1e9, max_v = 0;
        for (size_t i = timing_samples_.size() - n; i < timing_samples_.size(); ++i) {
            double d = timing_samples_[i].step_duration_ms;
            sum += d;
            min_v = std::min(min_v, d);
            max_v = std::max(max_v, d);
        }
        double avg = sum / n;

        // Compute jitter (stddev)
        double var_sum = 0;
        for (size_t i = timing_samples_.size() - n; i < timing_samples_.size(); ++i) {
            double diff = timing_samples_[i].step_duration_ms - avg;
            var_sum += diff * diff;
        }
        double stddev = std::sqrt(var_sum / n);

        // Real-time ratio: sim should advance 0.1ms per step
        double realtime_ratio = 0.1 / avg;  // >1 = faster than real-time

        auto wall_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time_).count();

        std::ostringstream ss;
        ss << "{"
           << jInt("samples", n) << ","
           << jNum("avg_step_ms", avg) << ","
           << jNum("min_step_ms", min_v) << ","
           << jNum("max_step_ms", max_v) << ","
           << jNum("stddev_ms", stddev) << ","
           << jNum("realtime_ratio", realtime_ratio) << ","
           << jNum("sim_time_sec", sim_->currentTime() / 1000.0) << ","
           << jNum("wall_time_sec", wall_elapsed) << ","
           << jNum("drift_sec", wall_elapsed - sim_->currentTime() / 1000.0)
           << "}";
        return ss.str();
    });

    server_.route("/api/profile/history", [this](const std::string& path, const std::string&) {
        size_t last = 100;
        auto qpos = path.find("last=");
        if (qpos != std::string::npos) last = std::stoi(path.substr(qpos + 5));

        std::lock_guard lock(timing_mutex_);
        std::ostringstream ss;
        ss << "[";
        size_t start = timing_samples_.size() > last ? timing_samples_.size() - last : 0;
        bool first = true;
        for (size_t i = start; i < timing_samples_.size(); ++i) {
            if (!first) ss << ",";
            ss << "{" << jNum("step_ms", timing_samples_[i].step_duration_ms)
               << "," << jNum("sim_t", timing_samples_[i].sim_time) << "}";
            first = false;
        }
        ss << "]";
        return ss.str();
    });

    // ── Activity / spikes ──

    server_.route("/api/spikes", [this](const std::string& path, const std::string&) {
        size_t last = 200;
        auto qpos = path.find("last=");
        if (qpos != std::string::npos) last = std::stoi(path.substr(qpos + 5));

        std::lock_guard lock(spike_mutex_);
        std::ostringstream ss;
        ss << "[";
        size_t start = spike_log_.size() > last ? spike_log_.size() - last : 0;
        bool first = true;
        for (size_t i = start; i < spike_log_.size(); ++i) {
            if (!first) ss << ",";
            ss << "{" << jInt("region", spike_log_[i].region_id)
               << "," << jInt("spikes", spike_log_[i].spike_count)
               << "," << jNum("time", spike_log_[i].sim_time) << "}";
            first = false;
        }
        ss << "]";
        return ss.str();
    });

    server_.route("/api/activity", [this](const std::string&, const std::string&) {
        std::ostringstream ss;
        ss << "[";
        bool first = true;
        for (auto& r : sim_->regions()) {
            if (!first) ss << ",";
            ss << "{"
               << jStr("name", r->name()) << ","
               << jInt("id", r->id()) << ","
               << jNum("firing_rate_Hz", r->firingRate()) << ","
               << jInt("active", r->activeNeuronCount()) << ","
               << jInt("total", r->neurons().size())
               << "}";
            first = false;
        }
        ss << "]";
        return ss.str();
    });

    // ── Screenshot ──

    server_.route("/api/screenshot", [this](const std::string&, const std::string&) {
        std::string b64 = captureScreenshotBase64();
        if (b64.empty()) return std::string("{\"error\":\"screenshot failed\"}");
        return std::string("{\"format\":\"png\",\"base64\":\"") + b64 + "\"}";
    });

    // ── Webcam status ──

    server_.route("/api/webcam/status", [this](const std::string&, const std::string&) {
        bool active = webcam_ && webcam_->isRunning();
        FrameData frame;
        bool hasFrame = webcam_ ? webcam_->getLatestFrame(frame) : false;
        std::ostringstream ss;
        ss << "{"
           << jBool("active", active) << ","
           << jBool("has_frame", hasFrame);
        if (hasFrame) {
            ss << "," << jInt("width", frame.width)
               << "," << jInt("height", frame.height)
               << "," << jNum("timestamp_ms", frame.timestamp);
        }
        ss << "}";
        return ss.str();
    });

    // ── List cameras ──

    server_.route("/api/webcam/cameras", [this](const std::string&, const std::string&) {
        auto cameras = WebcamCapture::listCameras();
        std::string current = webcam_ ? webcam_->selectedCamera() : "";
        std::ostringstream ss;
        ss << "{\"current\":" << (current.empty() ? "null" : ("\"" + current + "\""))
           << ",\"cameras\":[";
        bool first = true;
        for (auto& cam : cameras) {
            if (!first) ss << ",";
            ss << "{" << jStr("id", cam.device_id) << ","
               << jStr("name", cam.name) << ","
               << jBool("active", cam.device_id == current) << "}";
            first = false;
        }
        ss << "]}";
        return ss.str();
    });

    // ── Switch camera ──

    server_.route("/api/webcam/switch", [this](const std::string& path, const std::string& body) {
        // Accept device_id from query string or body
        std::string device_id;

        // Check query string: /api/webcam/switch?id=xxx
        auto qpos = path.find("id=");
        if (qpos != std::string::npos) {
            auto end = path.find('&', qpos);
            device_id = path.substr(qpos + 3, end == std::string::npos ? end : end - qpos - 3);
        }
        // Check body: id=xxx
        if (device_id.empty()) {
            auto bpos = body.find("id=");
            if (bpos != std::string::npos) {
                auto end = body.find('&', bpos);
                device_id = body.substr(bpos + 3, end == std::string::npos ? end : end - bpos - 3);
            }
        }
        // Also accept bare body as device_id
        if (device_id.empty() && !body.empty() && body.find('=') == std::string::npos) {
            device_id = body;
        }

        if (device_id.empty()) {
            return std::string("{\"error\":\"Provide camera id via ?id=DEVICE_ID or POST body\"}");
        }

        if (!webcam_) {
            return std::string("{\"error\":\"No webcam instance\"}");
        }

        // Verify the device_id exists
        auto cameras = WebcamCapture::listCameras();
        std::string cam_name;
        bool found = false;
        for (auto& cam : cameras) {
            if (cam.device_id == device_id) {
                cam_name = cam.name;
                found = true;
                break;
            }
        }
        if (!found) {
            return std::string("{\"error\":\"Camera not found\",\"device_id\":\"" + device_id + "\"}");
        }

        // Stop current, switch, restart — wrapped for safety
        bool ok = false;
        try {
            webcam_->stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
            webcam_->selectCamera(device_id);
            ok = webcam_->start();
        } catch (const std::exception& e) {
            return std::string("{\"error\":\"Switch failed: ") + e.what() + "\"}";
        } catch (...) {
            return std::string("{\"error\":\"Switch failed with unknown exception\"}");
        }

        std::ostringstream ss;
        ss << "{" << jBool("ok", ok) << ","
           << jStr("camera", cam_name) << ","
           << jStr("device_id", device_id) << "}";
        return ss.str();
    });

    // ── Memory estimate ──

    server_.route("/api/memory", [this](const std::string&, const std::string&) {
        size_t total_neurons = 0;
        size_t total_synapses = 0;
        size_t total_projection_synapses = 0;
        for (auto& r : sim_->regions()) {
            total_neurons += r->neurons().size();
            total_synapses += r->internalSynapses().size();
            for (auto& [_, syns] : r->projections())
                total_projection_synapses += syns.size();
        }

        // Rough estimates: Neuron ~120 bytes, Synapse ~128 bytes
        size_t neuron_bytes = total_neurons * 120;
        size_t synapse_bytes = (total_synapses + total_projection_synapses) * 128;
        size_t total_bytes = neuron_bytes + synapse_bytes;

        std::ostringstream ss;
        ss << "{"
           << jInt("total_neurons", total_neurons) << ","
           << jInt("internal_synapses", total_synapses) << ","
           << jInt("projection_synapses", total_projection_synapses) << ","
           << jInt("est_neuron_bytes", neuron_bytes) << ","
           << jInt("est_synapse_bytes", synapse_bytes) << ","
           << jInt("est_total_bytes", total_bytes) << ","
           << jNum("est_total_MB", total_bytes / (1024.0 * 1024.0))
           << "}";
        return ss.str();
    });

    // ── Debug: propagation counters ──
    server_.route("/api/debug/counters", [this](const std::string&, const std::string&) {
        std::ostringstream ss;
        ss << "{"
           << jInt("inter_region_events_submitted", sim_->debug_inter_region_events_.load())
           << "," << jInt("inter_region_events_delivered", sim_->debug_delivered_events_.load())
           << "," << jInt("projection_synapse_matches", sim_->debug_projection_matches_.load())
           << "," << jNum("sim_time_ms", sim_->currentTime())
           << "}";
        return ss.str();
    });

    // ── Debug: trace spike propagation path ──
    // GET /api/debug/trace — inspect projection wiring and spike queue
    server_.route("/api/debug/trace", [this](const std::string&, const std::string&) {
        std::ostringstream ss;
        ss << "{\"regions\":[";
        bool first_r = true;
        for (auto& r : sim_->regions()) {
            if (!first_r) ss << ",";
            ss << "{" << jStr("name", r->name())
               << "," << jInt("id", r->id())
               << "," << jInt("base_neuron_id", r->baseNeuronId())
               << "," << jInt("neuron_count", r->neurons().size())
               << "," << jNum("firing_rate", r->firingRate())
               << "," << jInt("internal_synapses", r->internalSynapses().size())
               << ",\"projections\":[";
            bool first_p = true;
            for (auto& [tid, syns] : r->projections()) {
                if (!first_p) ss << ",";
                // Sample first synapse to show ID ranges
                uint32_t min_pre = UINT32_MAX, max_pre = 0;
                uint32_t min_post = UINT32_MAX, max_post = 0;
                for (auto& s : syns) {
                    if (s.pre_id < min_pre) min_pre = s.pre_id;
                    if (s.pre_id > max_pre) max_pre = s.pre_id;
                    if (s.post_id < min_post) min_post = s.post_id;
                    if (s.post_id > max_post) max_post = s.post_id;
                }
                ss << "{"
                   << jInt("target_region", tid) << ","
                   << jInt("count", syns.size()) << ","
                   << jInt("pre_id_min", min_pre) << ","
                   << jInt("pre_id_max", max_pre) << ","
                   << jInt("post_id_min", min_post) << ","
                   << jInt("post_id_max", max_post)
                   << "}";
                first_p = false;
            }
            ss << "]}";
            first_r = false;
        }
        ss << "],\"spike_queue_size\":" << 0  // can't access router directly
           << "}";
        return ss.str();
    });

    // ── Debug: manually fire one LGN neuron and trace what happens ──
    server_.route("/api/debug/fire", [this](const std::string& path, const std::string&) {
        uint32_t region_id = 1;  // LGN default
        uint32_t local_idx = 0;
        auto qpos = path.find('?');
        if (qpos != std::string::npos) {
            std::string q = path.substr(qpos + 1);
            auto p1 = q.find("region=");
            if (p1 != std::string::npos) region_id = std::stoi(q.substr(p1 + 7));
            auto p2 = q.find("idx=");
            if (p2 != std::string::npos) local_idx = std::stoi(q.substr(p2 + 4));
        }

        auto* region = sim_->getRegion(region_id);
        if (!region) return std::string("{\"error\":\"region not found\"}");
        if (local_idx >= region->neurons().size())
            return std::string("{\"error\":\"index out of range\"}");

        uint32_t neuron_id = region->baseNeuronId() + local_idx;
        double t = sim_->currentTime();

        std::ostringstream ss;
        ss << "{\"fired_neuron\":" << neuron_id
           << ",\"region\":\"" << region->name() << "\""
           << ",\"local_idx\":" << local_idx;

        // Check what internal synapses this neuron has (as pre)
        auto& int_syns = region->getSynapsesForPreNeuron(neuron_id);
        ss << ",\"internal_targets\":" << int_syns.size();

        // Check what projection synapses this neuron has
        ss << ",\"projections\":[";
        bool first = true;
        int total_proj = 0;
        for (auto& [tid, proj_syns] : region->projections()) {
            int count = 0;
            uint32_t sample_post = 0;
            for (auto& syn : proj_syns) {
                if (syn.pre_id == neuron_id) {
                    count++;
                    sample_post = syn.post_id;
                }
            }
            if (count > 0) {
                if (!first) ss << ",";
                auto* target = sim_->getRegion(tid);
                ss << "{"
                   << jInt("target_region", tid)
                   << "," << jStr("target_name", target ? target->name() : "?")
                   << "," << jInt("synapse_count", count)
                   << "," << jInt("sample_post_id", sample_post)
                   << "," << jInt("target_base_id", target ? target->baseNeuronId() : 0)
                   << "," << jInt("target_neuron_count", target ? target->neurons().size() : 0)
                   << "}";
                total_proj += count;
                first = false;
            }
        }
        ss << "],\"total_projection_targets\":" << total_proj << "}";
        return ss.str();
    });
}
