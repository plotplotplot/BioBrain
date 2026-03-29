// WebcamCapture_linux.cpp — V4L2-based webcam capture for Linux
// This file is the Linux counterpart of WebcamCapture.mm (macOS/AVFoundation).
// Selected by CMake based on target platform; both implement the same interface.

#include "input/WebcamCapture.h"

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <algorithm>

// ---------------------------------------------------------------------------
// V4L2 memory-mapped buffer descriptor
// ---------------------------------------------------------------------------
struct V4L2Buffer {
    void*  start  = nullptr;
    size_t length = 0;
};

static constexpr int NUM_BUFFERS = 4;

// ---------------------------------------------------------------------------
// Pimpl struct holding V4L2 state
// ---------------------------------------------------------------------------
struct WebcamCapture::Impl {
    int fd = -1;
    V4L2Buffer buffers[NUM_BUFFERS] = {};
    int buffer_count = 0;
    std::thread capture_thread;
    std::chrono::steady_clock::time_point capture_start;
};

// ---------------------------------------------------------------------------
// Helper: xioctl with EINTR retry
// ---------------------------------------------------------------------------
static int xioctl(int fd, unsigned long request, void* arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

// ---------------------------------------------------------------------------
// YUYV (YUV 4:2:2) → RGB conversion
// ---------------------------------------------------------------------------
static void yuyv_to_rgb(const uint8_t* yuyv, uint8_t* rgb, int width, int height) {
    int pixel_count = width * height;
    for (int i = 0; i < pixel_count / 2; ++i) {
        int y0 = yuyv[i * 4 + 0];
        int u  = yuyv[i * 4 + 1];
        int y1 = yuyv[i * 4 + 2];
        int v  = yuyv[i * 4 + 3];

        int c0 = y0 - 16;
        int c1 = y1 - 16;
        int d  = u - 128;
        int e  = v - 128;

        auto clamp8 = [](int val) -> uint8_t {
            return static_cast<uint8_t>(std::clamp(val, 0, 255));
        };

        rgb[i * 6 + 0] = clamp8((298 * c0 + 409 * e + 128) >> 8);
        rgb[i * 6 + 1] = clamp8((298 * c0 - 100 * d - 208 * e + 128) >> 8);
        rgb[i * 6 + 2] = clamp8((298 * c0 + 516 * d + 128) >> 8);

        rgb[i * 6 + 3] = clamp8((298 * c1 + 409 * e + 128) >> 8);
        rgb[i * 6 + 4] = clamp8((298 * c1 - 100 * d - 208 * e + 128) >> 8);
        rgb[i * 6 + 5] = clamp8((298 * c1 + 516 * d + 128) >> 8);
    }
}

// ---------------------------------------------------------------------------
// WebcamCapture implementation
// ---------------------------------------------------------------------------

std::vector<CameraInfo> WebcamCapture::listCameras() {
    std::vector<CameraInfo> result;

    DIR* dir = opendir("/dev");
    if (!dir) {
        fprintf(stderr, "BioBrain: Cannot open /dev to enumerate cameras\n");
        return result;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        // Match /dev/video*
        if (strncmp(entry->d_name, "video", 5) != 0) {
            continue;
        }

        std::string path = std::string("/dev/") + entry->d_name;
        int fd = open(path.c_str(), O_RDWR | O_NONBLOCK);
        if (fd < 0) {
            continue;
        }

        struct v4l2_capability cap{};
        if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
            // Only include devices that support video capture
            if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) {
                CameraInfo info;
                info.device_id = path;
                info.name = reinterpret_cast<const char*>(cap.card);
                result.push_back(info);
            }
        }

        close(fd);
    }

    closedir(dir);

    // Sort by device path for consistent ordering
    std::sort(result.begin(), result.end(),
              [](const CameraInfo& a, const CameraInfo& b) {
                  return a.device_id < b.device_id;
              });

    return result;
}

void WebcamCapture::selectCamera(const std::string& device_id) {
    selected_device_id_ = device_id;
}

WebcamCapture::WebcamCapture(int width, int height, int fps)
    : impl_(std::make_unique<Impl>())
    , target_width_(width)
    , target_height_(height)
    , target_fps_(fps)
{
}

WebcamCapture::~WebcamCapture() {
    stop();
}

bool WebcamCapture::start() {
    if (running_.load()) {
        return true;  // Already running
    }

    // --- Determine device path ---
    std::string device_path = selected_device_id_;
    if (device_path.empty()) {
        device_path = "/dev/video0";  // Linux default
    }

    // --- Open device ---
    impl_->fd = open(device_path.c_str(), O_RDWR);
    if (impl_->fd < 0) {
        fprintf(stderr, "BioBrain: Failed to open camera device %s: %s\n",
                device_path.c_str(), strerror(errno));
        return false;
    }

    // --- Verify capabilities ---
    struct v4l2_capability cap{};
    if (xioctl(impl_->fd, VIDIOC_QUERYCAP, &cap) < 0) {
        fprintf(stderr, "BioBrain: VIDIOC_QUERYCAP failed on %s\n", device_path.c_str());
        close(impl_->fd);
        impl_->fd = -1;
        return false;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "BioBrain: %s does not support video capture\n", device_path.c_str());
        close(impl_->fd);
        impl_->fd = -1;
        return false;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "BioBrain: %s does not support streaming I/O\n", device_path.c_str());
        close(impl_->fd);
        impl_->fd = -1;
        return false;
    }

    fprintf(stderr, "BioBrain: Using camera: %s (%s)\n",
            reinterpret_cast<const char*>(cap.card), device_path.c_str());

    // --- Set pixel format (prefer YUYV, fallback to MJPEG) ---
    struct v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = static_cast<uint32_t>(target_width_);
    fmt.fmt.pix.height = static_cast<uint32_t>(target_height_);
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (xioctl(impl_->fd, VIDIOC_S_FMT, &fmt) < 0) {
        fprintf(stderr, "BioBrain: YUYV format not supported, trying MJPEG\n");
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        if (xioctl(impl_->fd, VIDIOC_S_FMT, &fmt) < 0) {
            fprintf(stderr, "BioBrain: Failed to set pixel format: %s\n", strerror(errno));
            close(impl_->fd);
            impl_->fd = -1;
            return false;
        }
    }

    // The driver may adjust the resolution — use actual values
    int actual_width = static_cast<int>(fmt.fmt.pix.width);
    int actual_height = static_cast<int>(fmt.fmt.pix.height);
    uint32_t pixel_format = fmt.fmt.pix.pixelformat;

    fprintf(stderr, "BioBrain: Capture format: %dx%d, pixfmt=%.4s\n",
            actual_width, actual_height,
            reinterpret_cast<const char*>(&pixel_format));

    // --- Set frame rate (best effort) ---
    struct v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(target_fps_);
    if (xioctl(impl_->fd, VIDIOC_S_PARM, &parm) == 0) {
        double actual_fps = static_cast<double>(parm.parm.capture.timeperframe.denominator) /
                            static_cast<double>(parm.parm.capture.timeperframe.numerator);
        fprintf(stderr, "BioBrain: Set frame rate to %.1f fps\n", actual_fps);
    } else {
        fprintf(stderr, "BioBrain: Could not set frame rate, using device default\n");
    }

    // --- Request memory-mapped buffers ---
    struct v4l2_requestbuffers req{};
    req.count = NUM_BUFFERS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(impl_->fd, VIDIOC_REQBUFS, &req) < 0) {
        fprintf(stderr, "BioBrain: VIDIOC_REQBUFS failed: %s\n", strerror(errno));
        close(impl_->fd);
        impl_->fd = -1;
        return false;
    }

    if (req.count < 2) {
        fprintf(stderr, "BioBrain: Insufficient buffer memory on %s\n", device_path.c_str());
        close(impl_->fd);
        impl_->fd = -1;
        return false;
    }

    impl_->buffer_count = static_cast<int>(req.count);

    // --- Map buffers ---
    for (int i = 0; i < impl_->buffer_count; ++i) {
        struct v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = static_cast<uint32_t>(i);

        if (xioctl(impl_->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            fprintf(stderr, "BioBrain: VIDIOC_QUERYBUF failed for buffer %d\n", i);
            close(impl_->fd);
            impl_->fd = -1;
            return false;
        }

        impl_->buffers[i].length = buf.length;
        impl_->buffers[i].start = mmap(nullptr, buf.length,
                                        PROT_READ | PROT_WRITE,
                                        MAP_SHARED,
                                        impl_->fd, buf.m.offset);

        if (impl_->buffers[i].start == MAP_FAILED) {
            fprintf(stderr, "BioBrain: mmap failed for buffer %d: %s\n", i, strerror(errno));
            // Unmap previously mapped buffers
            for (int j = 0; j < i; ++j) {
                munmap(impl_->buffers[j].start, impl_->buffers[j].length);
            }
            close(impl_->fd);
            impl_->fd = -1;
            return false;
        }
    }

    // --- Queue buffers ---
    for (int i = 0; i < impl_->buffer_count; ++i) {
        struct v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = static_cast<uint32_t>(i);

        if (xioctl(impl_->fd, VIDIOC_QBUF, &buf) < 0) {
            fprintf(stderr, "BioBrain: VIDIOC_QBUF failed for buffer %d\n", i);
            for (int j = 0; j < impl_->buffer_count; ++j) {
                munmap(impl_->buffers[j].start, impl_->buffers[j].length);
            }
            close(impl_->fd);
            impl_->fd = -1;
            return false;
        }
    }

    // --- Start streaming ---
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(impl_->fd, VIDIOC_STREAMON, &type) < 0) {
        fprintf(stderr, "BioBrain: VIDIOC_STREAMON failed: %s\n", strerror(errno));
        for (int i = 0; i < impl_->buffer_count; ++i) {
            munmap(impl_->buffers[i].start, impl_->buffers[i].length);
        }
        close(impl_->fd);
        impl_->fd = -1;
        return false;
    }

    running_.store(true, std::memory_order_release);
    impl_->capture_start = std::chrono::steady_clock::now();

    // --- Launch capture thread ---
    impl_->capture_thread = std::thread([this, actual_width, actual_height, pixel_format]() {
        bool first_frame = true;

        while (running_.load(std::memory_order_acquire)) {
            // Use select() to wait for a frame with timeout
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(impl_->fd, &fds);

            struct timeval tv;
            tv.tv_sec = 2;
            tv.tv_usec = 0;

            int r = select(impl_->fd + 1, &fds, nullptr, nullptr, &tv);
            if (r < 0) {
                if (errno == EINTR) continue;
                fprintf(stderr, "BioBrain: select() error on capture fd: %s\n", strerror(errno));
                break;
            }
            if (r == 0) {
                fprintf(stderr, "BioBrain: Capture timeout (no frame in 2s)\n");
                continue;
            }

            // Dequeue buffer
            struct v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;

            if (xioctl(impl_->fd, VIDIOC_DQBUF, &buf) < 0) {
                if (errno == EAGAIN) continue;
                fprintf(stderr, "BioBrain: VIDIOC_DQBUF failed: %s\n", strerror(errno));
                break;
            }

            // Convert to RGB
            FrameData frame;
            frame.width = actual_width;
            frame.height = actual_height;
            frame.pixels.resize(static_cast<size_t>(actual_width * actual_height * 3));

            auto now = std::chrono::steady_clock::now();
            frame.timestamp = std::chrono::duration<double, std::milli>(
                now - impl_->capture_start).count();

            if (pixel_format == V4L2_PIX_FMT_YUYV) {
                yuyv_to_rgb(static_cast<const uint8_t*>(impl_->buffers[buf.index].start),
                            frame.pixels.data(), actual_width, actual_height);
            } else {
                // For MJPEG or unsupported formats, zero-fill as fallback.
                // A production implementation would decode MJPEG here.
                std::memset(frame.pixels.data(), 0, frame.pixels.size());
                if (first_frame) {
                    fprintf(stderr, "BioBrain: Warning: MJPEG decode not implemented, "
                            "frames will be blank. Use a YUYV-capable camera.\n");
                }
            }

            if (first_frame) {
                fprintf(stderr, "BioBrain: First webcam frame received (%dx%d)\n",
                        actual_width, actual_height);
                first_frame = false;
            }

            // Re-queue the buffer before delivering (minimize latency)
            if (xioctl(impl_->fd, VIDIOC_QBUF, &buf) < 0) {
                fprintf(stderr, "BioBrain: VIDIOC_QBUF re-queue failed: %s\n", strerror(errno));
                break;
            }

            deliverFrame(std::move(frame));
        }
    });

    fprintf(stderr, "BioBrain: Webcam capture started successfully (%dx%d @ %d fps)\n",
            actual_width, actual_height, target_fps_);
    return true;
}

void WebcamCapture::stop() {
    if (!running_.load()) {
        return;
    }

    // 1. Signal the capture thread to stop
    running_.store(false, std::memory_order_release);

    // 2. Wait for capture thread to finish
    if (impl_->capture_thread.joinable()) {
        impl_->capture_thread.join();
    }

    // 3. Stop streaming
    if (impl_->fd >= 0) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(impl_->fd, VIDIOC_STREAMOFF, &type);
    }

    // 4. Unmap buffers
    for (int i = 0; i < impl_->buffer_count; ++i) {
        if (impl_->buffers[i].start && impl_->buffers[i].start != MAP_FAILED) {
            munmap(impl_->buffers[i].start, impl_->buffers[i].length);
            impl_->buffers[i].start = nullptr;
            impl_->buffers[i].length = 0;
        }
    }
    impl_->buffer_count = 0;

    // 5. Close device
    if (impl_->fd >= 0) {
        close(impl_->fd);
        impl_->fd = -1;
    }

    fprintf(stderr, "BioBrain: Webcam capture stopped\n");
}

bool WebcamCapture::isRunning() const {
    return running_.load(std::memory_order_acquire);
}

void WebcamCapture::deliverFrame(FrameData frame) {
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = std::move(frame);
    }
    has_new_frame_.store(true, std::memory_order_release);

    // Fire callback if set
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (frame_callback_) {
            frame_callback_(latest_frame_);
        }
    }
}

bool WebcamCapture::getLatestFrame(FrameData& out) {
    if (!has_new_frame_.load(std::memory_order_acquire)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(frame_mutex_);
    out = latest_frame_;
    has_new_frame_.store(false, std::memory_order_release);
    return true;
}

void WebcamCapture::setFrameCallback(std::function<void(const FrameData&)> callback) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    frame_callback_ = std::move(callback);
}
