// VocalSynthesizer_linux.cpp — PulseAudio-based vocal synthesis for Linux
// This file is the Linux counterpart of VocalSynthesizer.mm (macOS/CoreAudio).
// Selected by CMake based on target platform; both implement the same interface.
// The synthesis algorithm (Rosenberg glottal pulse + formant resonators) is
// identical to the macOS version.

#include "audio/VocalSynthesizer.h"

#include <pulse/simple.h>
#include <pulse/error.h>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>

// ─── PulseAudio implementation ────────────────────────────────────────────────

struct VocalSynthesizer::Impl {
    pa_simple* pa_handle = nullptr;
    std::thread audio_thread;
    VocalSynthesizer* owner = nullptr;
};

// ─── Synthesizer implementation ───────────────────────────────────────────────

VocalSynthesizer::VocalSynthesizer()
    : impl_(std::make_unique<Impl>())
{
    impl_->owner = this;
}

VocalSynthesizer::~VocalSynthesizer() {
    stop();
}

bool VocalSynthesizer::start() {
    if (running_.load()) return true;

    // Configure PulseAudio sample spec: 44100 Hz, mono, float32
    pa_sample_spec spec{};
    spec.format = PA_SAMPLE_FLOAT32LE;
    spec.channels = 1;
    spec.rate = static_cast<uint32_t>(SAMPLE_RATE);

    // Buffer attributes for low latency
    pa_buffer_attr attr{};
    attr.maxlength = static_cast<uint32_t>(-1);    // server default
    attr.tlength   = static_cast<uint32_t>(SAMPLE_RATE * sizeof(float) / 20);  // ~50ms
    attr.prebuf    = static_cast<uint32_t>(-1);     // server default
    attr.minreq    = static_cast<uint32_t>(-1);     // server default
    attr.fragsize  = static_cast<uint32_t>(-1);     // not used for playback

    int error = 0;
    impl_->pa_handle = pa_simple_new(
        nullptr,                    // default server
        "BioBrain",                 // application name
        PA_STREAM_PLAYBACK,
        nullptr,                    // default device
        "Vocal Synthesizer",        // stream description
        &spec,
        nullptr,                    // default channel map
        &attr,
        &error
    );

    if (!impl_->pa_handle) {
        fprintf(stderr, "BioBrain Audio: Failed to open PulseAudio: %s\n",
                pa_strerror(error));
        return false;
    }

    running_.store(true);

    // Launch audio thread that generates samples and writes to PulseAudio
    static constexpr int BUFFER_FRAMES = 512;

    impl_->audio_thread = std::thread([this]() {
        float buffer[BUFFER_FRAMES];

        while (running_.load(std::memory_order_acquire)) {
            // Generate a block of samples
            renderCallback(this, buffer, BUFFER_FRAMES);

            // Write to PulseAudio (blocks until buffer space is available)
            int error = 0;
            if (pa_simple_write(impl_->pa_handle, buffer,
                                BUFFER_FRAMES * sizeof(float), &error) < 0) {
                fprintf(stderr, "BioBrain Audio: pa_simple_write failed: %s\n",
                        pa_strerror(error));
                break;
            }
        }

        // Drain remaining audio on clean shutdown
        if (impl_->pa_handle) {
            int error = 0;
            pa_simple_drain(impl_->pa_handle, &error);
        }
    });

    fprintf(stderr, "BioBrain Audio: Vocal synthesizer started (%.0f Hz, mono, PulseAudio)\n",
            SAMPLE_RATE);
    return true;
}

void VocalSynthesizer::stop() {
    if (!running_.load()) return;
    running_.store(false);

    // Wait for audio thread to finish
    if (impl_->audio_thread.joinable()) {
        impl_->audio_thread.join();
    }

    // Close PulseAudio connection
    if (impl_->pa_handle) {
        pa_simple_free(impl_->pa_handle);
        impl_->pa_handle = nullptr;
    }

    fprintf(stderr, "BioBrain Audio: Vocal synthesizer stopped\n");
}

// ─── Parameter setters (shared with macOS) ────────────────────────────────────

void VocalSynthesizer::updateFromPoolRates(const std::array<double, 6>& rates) {
    // Map pool firing rates (0-100 Hz) to synthesis parameters
    auto map = [](double rate, double min, double max) {
        double t = std::clamp(rate / 100.0, 0.0, 1.0);
        return min + t * (max - min);
    };

    setF1(map(rates[0], 250.0, 900.0));
    setF2(map(rates[1], 700.0, 2500.0));
    setF0(map(rates[2], 80.0, 300.0));
    setAmplitude(std::clamp(rates[3] / 100.0, 0.0, 1.0));
    setTilt(std::clamp(rates[4] / 100.0, 0.0, 1.0));
    setNoise(std::clamp(rates[5] / 100.0, 0.0, 1.0));
}

void VocalSynthesizer::setF1(double hz)      { f1_.store(std::clamp(hz, 250.0, 900.0)); }
void VocalSynthesizer::setF2(double hz)      { f2_.store(std::clamp(hz, 700.0, 2500.0)); }
void VocalSynthesizer::setF0(double hz)      { f0_.store(std::clamp(hz, 80.0, 300.0)); }
void VocalSynthesizer::setAmplitude(double a) { amplitude_.store(std::clamp(a, 0.0, 1.0)); }
void VocalSynthesizer::setTilt(double t)     { tilt_.store(std::clamp(t, 0.0, 1.0)); }
void VocalSynthesizer::setNoise(double n)    { noise_.store(std::clamp(n, 0.0, 1.0)); }

// ─── Audio synthesis (identical to macOS VocalSynthesizer.mm) ─────────────────

float VocalSynthesizer::generateSample() {
    double f0  = f0_.load();
    double f1  = f1_.load();
    double f2  = f2_.load();
    double amp = amplitude_.load();
    double tlt = tilt_.load();
    double nse = noise_.load();
    double vol = volume_.load();

    if (amp < 0.001) return 0.0f;  // silence

    // ── Glottal source: band-limited sawtooth approximation ──
    // Phase advances at F0
    phase_ += f0 / SAMPLE_RATE;
    if (phase_ >= 1.0) phase_ -= 1.0;

    // Rosenberg glottal pulse (more natural than raw sawtooth)
    // Open phase: 0 to 0.6, closed phase: 0.6 to 1.0
    double glottal;
    if (phase_ < 0.6) {
        double t = phase_ / 0.6;
        glottal = 3.0 * t * t - 2.0 * t * t * t;  // smooth opening
    } else {
        double t = (phase_ - 0.6) / 0.4;
        glottal = (1.0 - t);  // abrupt closing
    }
    glottal = glottal * 2.0 - 1.0;  // center around zero

    // Apply spectral tilt (low-pass filter, simulates breathiness)
    glottal *= (1.0 - tlt * 0.5);

    // ── Aspiration noise ──
    double noise_sample = (static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0);

    // Mix glottal + noise
    double source = glottal * (1.0 - nse) + noise_sample * nse;

    // ── Vocal tract formant filter (2 resonators in series) ──
    // Simple 2-pole resonator: H(z) = 1 / (1 - 2r*cos(w)*z^-1 + r^2*z^-2)
    auto resonator = [](double input, double freq, double bw,
                        double state[2], double sr) -> double {
        double w = 2.0 * M_PI * freq / sr;
        double r = std::exp(-M_PI * bw / sr);
        double a1 = -2.0 * r * std::cos(w);
        double a2 = r * r;
        double output = input - a1 * state[0] - a2 * state[1];
        state[1] = state[0];
        state[0] = output;
        return output * (1.0 - r);  // normalize gain
    };

    // F1 bandwidth ~80 Hz, F2 bandwidth ~120 Hz (typical values)
    double filtered = resonator(source, f1, 80.0, formant1_state_, SAMPLE_RATE);
    filtered = resonator(filtered, f2, 120.0, formant2_state_, SAMPLE_RATE);

    // Apply amplitude and master volume
    double output = filtered * amp * vol;

    // Soft clip to prevent harsh distortion
    output = std::tanh(output * 2.0) * 0.5;

    return static_cast<float>(output);
}

void VocalSynthesizer::renderCallback(void* userData, float* buffer, uint32_t numFrames) {
    auto* synth = static_cast<VocalSynthesizer*>(userData);
    for (uint32_t i = 0; i < numFrames; ++i) {
        buffer[i] = synth->generateSample();
    }
}
