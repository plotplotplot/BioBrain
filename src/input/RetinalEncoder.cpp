#include "RetinalEncoder.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

RetinalEncoder::RetinalEncoder(int grid_size)
    : grid_size_(grid_size)
    , total_neurons_(grid_size * grid_size * 2)  // ON + OFF cells
    , rng_(std::random_device{}())
{
    // Kernel size should be large enough to cover ~3*sigma_surround on each side
    // sigma_surround = 3.0, so 3*3 = 9 -> diameter ~19, use odd size
    kernel_size_ = 19;

    center_kernel_  = makeGaussianKernel(1.0, kernel_size_);
    surround_kernel_ = makeGaussianKernel(3.0, kernel_size_);
}

// ---------------------------------------------------------------------------
// Public accessors
// ---------------------------------------------------------------------------

int RetinalEncoder::totalNeurons() const { return total_neurons_; }
int RetinalEncoder::gridSize() const { return grid_size_; }

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

SpikeOutput RetinalEncoder::encode(const uint8_t* frame_pixels, int frame_w, int frame_h,
                                   double t_start, double t_end)
{
    SpikeOutput output;

    // 1. RGB -> luminance (full resolution)
    std::vector<double> luminance;
    rgbToLuminance(frame_pixels, frame_w, frame_h, luminance);

    // 2. Downsample to grid_size x grid_size
    std::vector<double> downsampled;
    downsample(luminance, frame_w, frame_h, downsampled);

    // 3. Apply center-surround (DoG) filter
    std::vector<double> on_response, off_response;
    applyCenterSurround(downsampled, on_response, off_response);

    // 4. Rate coding: map response magnitude to firing rate [0, 200] Hz
    constexpr double max_rate = 200.0;  // Hz, biological max for RGCs

    // Find maximum response for normalization
    double max_on  = *std::max_element(on_response.begin(), on_response.end());
    double max_off = *std::max_element(off_response.begin(), off_response.end());
    double max_resp = std::max(max_on, max_off);

    if (max_resp < 1e-10) {
        // Frame is uniform -- no spikes
        return output;
    }

    std::vector<double> on_rates(grid_size_ * grid_size_);
    std::vector<double> off_rates(grid_size_ * grid_size_);

    for (int i = 0; i < grid_size_ * grid_size_; ++i) {
        on_rates[i]  = std::clamp(max_rate * on_response[i]  / max_resp, 0.0, max_rate);
        off_rates[i] = std::clamp(max_rate * off_response[i] / max_resp, 0.0, max_rate);
    }

    // 5. Poisson spike generation
    //    ON cells:  IDs 0 .. grid_size^2 - 1
    //    OFF cells: IDs grid_size^2 .. 2*grid_size^2 - 1
    generatePoissonSpikes(on_rates,  0,                            t_start, t_end, output);
    generatePoissonSpikes(off_rates, static_cast<uint32_t>(grid_size_ * grid_size_), t_start, t_end, output);

    // Store current luminance for potential future temporal differencing
    prev_luminance_ = std::move(downsampled);
    has_prev_frame_ = true;

    return output;
}

// ---------------------------------------------------------------------------
// RGB -> Luminance  (BT.709)
// ---------------------------------------------------------------------------

void RetinalEncoder::rgbToLuminance(const uint8_t* rgb, int w, int h,
                                    std::vector<double>& lum)
{
    lum.resize(static_cast<size_t>(w) * h);
    for (int i = 0; i < w * h; ++i) {
        double r = rgb[i * 3 + 0] / 255.0;
        double g = rgb[i * 3 + 1] / 255.0;
        double b = rgb[i * 3 + 2] / 255.0;
        lum[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
}

// ---------------------------------------------------------------------------
// Downsample via area averaging
// ---------------------------------------------------------------------------

void RetinalEncoder::downsample(const std::vector<double>& input, int in_w, int in_h,
                                std::vector<double>& output)
{
    output.assign(static_cast<size_t>(grid_size_) * grid_size_, 0.0);

    double scale_x = static_cast<double>(in_w) / grid_size_;
    double scale_y = static_cast<double>(in_h) / grid_size_;

    for (int gy = 0; gy < grid_size_; ++gy) {
        int y0 = static_cast<int>(gy * scale_y);
        int y1 = static_cast<int>((gy + 1) * scale_y);
        y1 = std::min(y1, in_h);
        if (y1 <= y0) y1 = y0 + 1;

        for (int gx = 0; gx < grid_size_; ++gx) {
            int x0 = static_cast<int>(gx * scale_x);
            int x1 = static_cast<int>((gx + 1) * scale_x);
            x1 = std::min(x1, in_w);
            if (x1 <= x0) x1 = x0 + 1;

            double sum = 0.0;
            int count = 0;
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    sum += input[y * in_w + x];
                    ++count;
                }
            }
            output[gy * grid_size_ + gx] = (count > 0) ? sum / count : 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Center-surround (Difference of Gaussians)
// ---------------------------------------------------------------------------

void RetinalEncoder::applyCenterSurround(const std::vector<double>& luminance,
                                         std::vector<double>& on_response,
                                         std::vector<double>& off_response)
{
    int n = grid_size_;
    int half = kernel_size_ / 2;

    // Apply center Gaussian blur
    std::vector<double> center_filtered(n * n, 0.0);
    std::vector<double> surround_filtered(n * n, 0.0);

    auto convolve = [&](const std::vector<double>& kernel, std::vector<double>& result) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                double sum = 0.0;
                double weight_sum = 0.0;
                for (int ky = -half; ky <= half; ++ky) {
                    for (int kx = -half; kx <= half; ++kx) {
                        int sy = y + ky;
                        int sx = x + kx;
                        // Clamp to border
                        sy = std::clamp(sy, 0, n - 1);
                        sx = std::clamp(sx, 0, n - 1);
                        double w = kernel[(ky + half) * kernel_size_ + (kx + half)];
                        sum += luminance[sy * n + sx] * w;
                        weight_sum += w;
                    }
                }
                result[y * n + x] = (weight_sum > 0.0) ? sum / weight_sum : 0.0;
            }
        }
    };

    convolve(center_kernel_,  center_filtered);
    convolve(surround_kernel_, surround_filtered);

    // ON response = center - surround  (bright center, dark surround)
    // OFF response = surround - center (dark center, bright surround)
    // Rectify: clamp negatives to 0
    on_response.resize(n * n);
    off_response.resize(n * n);

    for (int i = 0; i < n * n; ++i) {
        double diff = center_filtered[i] - surround_filtered[i];
        on_response[i]  = std::max(0.0,  diff);
        off_response[i] = std::max(0.0, -diff);
    }
}

// ---------------------------------------------------------------------------
// Poisson spike generation
// ---------------------------------------------------------------------------

void RetinalEncoder::generatePoissonSpikes(const std::vector<double>& rates,
                                           uint32_t id_offset,
                                           double t_start, double t_end,
                                           SpikeOutput& output)
{
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int num_neurons = static_cast<int>(rates.size());

    // Iterate 1ms bins within [t_start, t_end)
    for (double t = t_start; t < t_end; t += 1.0) {
        for (int i = 0; i < num_neurons; ++i) {
            if (rates[i] <= 0.0) continue;

            // Probability of spike in a 1ms bin = rate_Hz * 0.001
            double prob = rates[i] * 0.001;
            if (unif(rng_) < prob) {
                output.neuron_ids.push_back(id_offset + static_cast<uint32_t>(i));
                output.spike_times.push_back(t);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gaussian kernel builder
// ---------------------------------------------------------------------------

std::vector<double> RetinalEncoder::makeGaussianKernel(double sigma, int size)
{
    std::vector<double> kernel(static_cast<size_t>(size) * size);
    int half = size / 2;
    double sum = 0.0;
    double two_sigma_sq = 2.0 * sigma * sigma;

    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            double val = std::exp(-(x * x + y * y) / two_sigma_sq);
            kernel[(y + half) * size + (x + half)] = val;
            sum += val;
        }
    }

    // Normalize
    for (double& v : kernel) {
        v /= sum;
    }

    return kernel;
}
