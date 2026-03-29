#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "compute/MetalBackend.h"
#include "core/BrainRegion.h"
#include "core/Neuron.h"
#include "core/IzhikevichNeuron.h"

// Metal GPU state layout matching izhikevich.metal
struct MetalIzhikevichState {
    float v, u, a, b, c, d, I_syn;
    uint32_t spiked;
};

struct MetalBackend::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLComputePipelineState> izhikevichPipeline = nil;
    // Additional pipelines for HH, AdEx can be added here

    bool setup() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) return false;

            commandQueue = [device newCommandQueue];
            if (!commandQueue) return false;

            // Load shader library
            NSString* libPath = [[NSBundle mainBundle] pathForResource:@"BioBrain"
                                                               ofType:@"metallib"];
            if (!libPath) {
                // Try current directory
                NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
                libPath = [cwd stringByAppendingPathComponent:@"BioBrain.metallib"];
            }

            NSError* error = nil;
            id<MTLLibrary> library = nil;

            if (libPath && [[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
                NSURL* url = [NSURL fileURLWithPath:libPath];
                library = [device newLibraryWithURL:url error:&error];
            }

            if (!library) {
                // Shader library not found — Metal not available for compute
                return false;
            }

            // Create compute pipeline for Izhikevich kernel
            id<MTLFunction> izhFunc = [library newFunctionWithName:@"izhikevich_step"];
            if (izhFunc) {
                izhikevichPipeline = [device newComputePipelineStateWithFunction:izhFunc
                                                                          error:&error];
            }

            return (izhikevichPipeline != nil);
        }
    }
};

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {
    available_ = impl_->setup();
}

MetalBackend::~MetalBackend() = default;

UpdateResult MetalBackend::updateNeurons(BrainRegion& region, double dt,
                                          std::span<const double> I_syn) {
    UpdateResult result;

    if (!available_ || !impl_->izhikevichPipeline) {
        return result;  // Fallback: no spikes if Metal unavailable
    }

    @autoreleasepool {
        auto& neurons = region.neurons();
        size_t count = neurons.size();
        if (count == 0) return result;

        // Upload neuron state to Metal buffer
        size_t bufSize = count * sizeof(MetalIzhikevichState);
        id<MTLBuffer> neuronBuffer = [impl_->device newBufferWithLength:bufSize
                                                                options:MTLResourceStorageModeShared];

        auto* states = static_cast<MetalIzhikevichState*>([neuronBuffer contents]);

        for (size_t i = 0; i < count; ++i) {
            auto* n = dynamic_cast<IzhikevichNeuron*>(neurons[i].get());
            if (!n) continue;

            states[i].v = static_cast<float>(n->voltage());
            states[i].u = static_cast<float>(n->recoveryVariable());
            states[i].a = static_cast<float>(n->paramA());
            states[i].b = static_cast<float>(n->paramB());
            states[i].c = static_cast<float>(n->paramC());
            states[i].d = static_cast<float>(n->paramD());
            states[i].I_syn = (i < I_syn.size()) ? static_cast<float>(I_syn[i]) : 0.0f;
            states[i].spiked = 0;
        }

        // dt buffer
        float dt_f = static_cast<float>(dt);
        id<MTLBuffer> dtBuffer = [impl_->device newBufferWithBytes:&dt_f
                                                            length:sizeof(float)
                                                           options:MTLResourceStorageModeShared];

        // Encode and dispatch
        id<MTLCommandBuffer> cmdBuffer = [impl_->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:impl_->izhikevichPipeline];
        [encoder setBuffer:neuronBuffer offset:0 atIndex:0];
        [encoder setBuffer:dtBuffer offset:0 atIndex:1];

        NSUInteger threadGroupSize = impl_->izhikevichPipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > count) threadGroupSize = count;

        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // Read back results
        double sim_time = region.currentTime();
        for (size_t i = 0; i < count; ++i) {
            if (states[i].spiked) {
                result.spiked_neuron_ids.push_back(neurons[i]->id);
                result.spike_times.push_back(sim_time);
            }
            // Write state back to CPU neuron objects
            auto* n = dynamic_cast<IzhikevichNeuron*>(neurons[i].get());
            if (n) {
                n->setVoltage(states[i].v);
                n->setRecovery(states[i].u);
                if (states[i].spiked) {
                    n->last_spike_time = sim_time;
                }
            }
        }
    }

    return result;
}
