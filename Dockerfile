# BioBrain Dockerfile
# Builds the neural simulation with CUDA GPU support on Ubuntu.
#
# Usage:
#   docker build -t biobrain .
#
#   # Headless REST harness (no GPU required):
#   docker run -p 9090:9090 biobrain
#
#   # With NVIDIA GPU:
#   docker run --gpus all -p 9090:9090 biobrain
#
#   # Full GUI with X11 forwarding + webcam:
#   docker run --gpus all -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     --device /dev/video0 -p 9090:9090 \
#     biobrain ./BioBrain

# ── Stage 1: Build ──────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ make \
    qt6-base-dev libqt6multimedia6 qt6-multimedia-dev \
    libpulse-dev libhdf5-dev libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89;90" \
    && cmake --build build -j$(nproc)

# ── Stage 2: Runtime ────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libqt6widgets6 libqt6multimedia6 libqt6gui6 \
    libpulse0 libhdf5-cpp-103-1 libv4l-0 \
    libgl1 libglib2.0-0 libfontconfig1 libxkbcommon0 \
    libdbus-1-3 libxcb-xinerama0 libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built binaries
COPY --from=builder /build/build/BioBrain /app/BioBrain
COPY --from=builder /build/build/BioBrainHarness /app/BioBrainHarness

# Copy Metal shader lib if it was built (won't exist on Linux, that's fine)
COPY --from=builder /build/build/BioBrain.metallib /app/ 2>/dev/null || true

EXPOSE 9090

# Default: run headless REST harness
CMD ["./BioBrainHarness"]
