#pragma once

#include "ComputeBackend.h"

// Forward declare Objective-C types to keep header pure C++
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLComputePipelineState;
@protocol MTLBuffer;
#else
typedef void* id;
#endif

class MetalBackend : public ComputeBackend {
public:
    MetalBackend();
    ~MetalBackend() override;

    UpdateResult updateNeurons(BrainRegion& region, double dt,
                               std::span<const double> I_syn) override;

    const char* name() const override { return "Metal GPU Batch"; }

    bool isAvailable() const { return available_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool available_ = false;
};
