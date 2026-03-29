#pragma once

#include <vector>
#include <random>
#include <cstdint>

struct SpikeOutput {
    std::vector<uint32_t> neuron_ids;   // which RGCs fired
    std::vector<double> spike_times;     // when they fired (ms)
};

class RetinalEncoder {
public:
    // grid_size: spatial resolution (e.g., 64 = 64x64 grid)
    // Total neurons = grid_size^2 * 2 (ON + OFF center cells)
    explicit RetinalEncoder(int grid_size = 64);

    // Encode a frame into spikes for the given time window
    // frame_pixels: RGB row-major, frame_w x frame_h
    // t_start, t_end: time window in ms (typically 33.3ms for 30fps)
    SpikeOutput encode(const uint8_t* frame_pixels, int frame_w, int frame_h,
                       double t_start, double t_end);

    int totalNeurons() const;  // grid_size^2 * 2
    int gridSize() const;

private:
    int grid_size_;
    int total_neurons_;

    // Previous frame luminance for temporal differencing
    std::vector<double> prev_luminance_;
    bool has_prev_frame_ = false;

    // Center-surround filter kernels (Difference of Gaussians)
    std::vector<double> center_kernel_;   // sigma_center = 1.0 pixels
    std::vector<double> surround_kernel_; // sigma_surround = 3.0 pixels
    int kernel_size_;

    // Random number generator for Poisson spike generation
    std::mt19937 rng_;

    // Convert RGB to grayscale luminance (0-1)
    void rgbToLuminance(const uint8_t* rgb, int w, int h, std::vector<double>& lum);

    // Downsample luminance to grid_size x grid_size
    void downsample(const std::vector<double>& input, int in_w, int in_h,
                    std::vector<double>& output);

    // Apply Difference-of-Gaussians center-surround filter
    // Returns positive values for ON-center, negative for OFF-center
    void applyCenterSurround(const std::vector<double>& luminance,
                             std::vector<double>& on_response,
                             std::vector<double>& off_response);

    // Generate Poisson spikes from firing rates
    // rate: firing rate in Hz (0-200), t_start/t_end in ms
    void generatePoissonSpikes(const std::vector<double>& rates,
                               uint32_t id_offset,
                               double t_start, double t_end,
                               SpikeOutput& output);

    // Build Gaussian kernel
    static std::vector<double> makeGaussianKernel(double sigma, int size);
};
