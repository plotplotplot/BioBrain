#pragma once

#include "ComputeBackend.h"
#include <thread>
#include <vector>

class CPUBackend : public ComputeBackend {
public:
    explicit CPUBackend(int num_threads = 6);
    ~CPUBackend() override = default;

    UpdateResult updateNeurons(BrainRegion& region, double dt,
                               std::span<const double> I_syn) override;

    const char* name() const override { return "CPU Event-Driven"; }

private:
    int num_threads_;
};
