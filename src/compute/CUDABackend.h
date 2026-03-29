#pragma once

#include "ComputeBackend.h"
#include <memory>

class CUDABackend : public ComputeBackend {
public:
    CUDABackend();
    ~CUDABackend() override;

    UpdateResult updateNeurons(BrainRegion& region, double dt,
                               std::span<const double> I_syn) override;
    const char* name() const override { return "CUDA GPU Batch"; }
    bool isAvailable() const { return available_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool available_ = false;
};
