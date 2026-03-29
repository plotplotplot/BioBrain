#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>

#include "WebcamCapture.h"
#include <chrono>

// ---------------------------------------------------------------------------
// Objective-C delegate that receives sample buffers from AVCaptureVideoDataOutput
// ---------------------------------------------------------------------------
@interface BioBrainFrameDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property (nonatomic, assign) WebcamCapture* owner;
@property (nonatomic, assign) std::chrono::steady_clock::time_point captureStart;
@end

@implementation BioBrainFrameDelegate

- (void)captureOutput:(AVCaptureOutput*)output
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection*)connection
{
    @autoreleasepool {
        if (!self.owner || !self.owner->isRunning()) {
            return;
        }

        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        if (!imageBuffer) {
            return;
        }

        CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

        size_t width  = CVPixelBufferGetWidth(imageBuffer);
        size_t height = CVPixelBufferGetHeight(imageBuffer);
        size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
        uint8_t* baseAddress = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(imageBuffer));

        // The pixel format is kCVPixelFormatType_32BGRA.
        // Convert BGRA -> RGB, row-major.
        FrameData frame;
        frame.width  = static_cast<int>(width);
        frame.height = static_cast<int>(height);
        frame.pixels.resize(width * height * 3);

        auto now = std::chrono::steady_clock::now();
        frame.timestamp = std::chrono::duration<double, std::milli>(now - self.captureStart).count();

        for (size_t y = 0; y < height; ++y) {
            uint8_t* srcRow = baseAddress + y * bytesPerRow;
            uint8_t* dstRow = frame.pixels.data() + y * width * 3;
            for (size_t x = 0; x < width; ++x) {
                uint8_t b = srcRow[x * 4 + 0];
                uint8_t g = srcRow[x * 4 + 1];
                uint8_t r = srcRow[x * 4 + 2];
                dstRow[x * 3 + 0] = r;
                dstRow[x * 3 + 1] = g;
                dstRow[x * 3 + 2] = b;
            }
        }

        CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

        // Deliver the frame to the C++ owner via its public delivery method
        self.owner->deliverFrame(std::move(frame));
    }
}

- (void)captureOutput:(AVCaptureOutput*)output
  didDropSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection*)connection
{
    // Dropped frame -- no action needed
}

@end

// ---------------------------------------------------------------------------
// Pimpl struct holding Objective-C objects
// ---------------------------------------------------------------------------
struct WebcamCapture::Impl {
    AVCaptureSession*           session    = nil;
    AVCaptureDeviceInput*       input      = nil;
    AVCaptureVideoDataOutput*   output     = nil;
    BioBrainFrameDelegate*      delegate   = nil;
    dispatch_queue_t            queue      = nil;
};

// ---------------------------------------------------------------------------
// WebcamCapture implementation
// ---------------------------------------------------------------------------

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
    @autoreleasepool {
        if (running_.load()) {
            return true;  // Already running
        }

        // --- Find camera device ---
        AVCaptureDevice* device = nil;

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
        // macOS 14+ uses AVCaptureDeviceDiscoverySession
        AVCaptureDeviceDiscoverySession* discovery =
            [AVCaptureDeviceDiscoverySession
                discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera,
                                                  AVCaptureDeviceTypeExternal]
                                      mediaType:AVMediaTypeVideo
                                       position:AVCaptureDevicePositionUnspecified];
        if (discovery.devices.count > 0) {
            device = discovery.devices.firstObject;
        }
#else
        device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
#endif

        if (!device) {
            return false;  // No camera available
        }

        // --- Configure capture session ---
        impl_->session = [[AVCaptureSession alloc] init];

        // Set session preset for requested resolution
        if (target_width_ <= 352 && target_height_ <= 288) {
            impl_->session.sessionPreset = AVCaptureSessionPresetLow;
        } else if (target_width_ <= 640 && target_height_ <= 480) {
            impl_->session.sessionPreset = AVCaptureSessionPresetMedium;
        } else if (target_width_ <= 1280 && target_height_ <= 720) {
            impl_->session.sessionPreset = AVCaptureSessionPresetHigh;
        } else {
            impl_->session.sessionPreset = AVCaptureSessionPresetHigh;
        }

        // --- Input ---
        NSError* error = nil;
        impl_->input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
        if (!impl_->input || error) {
            return false;
        }
        if (![impl_->session canAddInput:impl_->input]) {
            return false;
        }
        [impl_->session addInput:impl_->input];

        // --- Configure frame rate ---
        [device lockForConfiguration:&error];
        if (!error) {
            CMTime frameDuration = CMTimeMake(1, target_fps_);
            device.activeVideoMinFrameDuration = frameDuration;
            device.activeVideoMaxFrameDuration = frameDuration;
            [device unlockForConfiguration];
        }

        // --- Output ---
        impl_->output = [[AVCaptureVideoDataOutput alloc] init];
        impl_->output.alwaysDiscardsLateVideoFrames = YES;

        // Request BGRA pixel format for easy conversion
        impl_->output.videoSettings = @{
            (__bridge NSString*)kCVPixelBufferPixelFormatTypeKey :
                @(kCVPixelFormatType_32BGRA)
        };

        // Delegate and dispatch queue
        impl_->delegate = [[BioBrainFrameDelegate alloc] init];
        impl_->delegate.owner = this;
        impl_->delegate.captureStart = std::chrono::steady_clock::now();

        impl_->queue = dispatch_queue_create("com.biobrain.webcam", DISPATCH_QUEUE_SERIAL);
        [impl_->output setSampleBufferDelegate:impl_->delegate queue:impl_->queue];

        if (![impl_->session canAddOutput:impl_->output]) {
            return false;
        }
        [impl_->session addOutput:impl_->output];

        // --- Start ---
        [impl_->session startRunning];
        running_.store(true, std::memory_order_release);
        return true;
    }
}

void WebcamCapture::stop() {
    @autoreleasepool {
        if (!running_.load()) {
            return;
        }
        running_.store(false, std::memory_order_release);

        if (impl_->session && [impl_->session isRunning]) {
            [impl_->session stopRunning];
        }

        // Remove inputs/outputs
        if (impl_->input) {
            [impl_->session removeInput:impl_->input];
            impl_->input = nil;
        }
        if (impl_->output) {
            [impl_->session removeOutput:impl_->output];
            impl_->output = nil;
        }

        impl_->delegate.owner = nullptr;
        impl_->delegate = nil;
        impl_->session = nil;
        impl_->queue = nil;
    }
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
