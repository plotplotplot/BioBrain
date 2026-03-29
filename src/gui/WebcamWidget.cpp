#include "WebcamWidget.h"

#include <QPixmap>
#include <QTimer>

WebcamWidget::WebcamWidget(QWidget* parent)
    : QLabel(parent)
{
    setMinimumSize(320, 240);
    setAlignment(Qt::AlignCenter);
    setStyleSheet("QLabel { background: #0a0a1a; color: #555; border: 1px solid #333; }");
    setText("No camera feed");

    // Refresh display at ~30 Hz
    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &WebcamWidget::refreshDisplay);
    timer->start(33); // ~30 fps
}

void WebcamWidget::updateFrame(const uint8_t* rgb_data, int width, int height)
{
    if (!rgb_data || width <= 0 || height <= 0) return;

    std::lock_guard<std::mutex> lock(frame_mutex_);
    // Deep copy the frame data
    current_frame_ = QImage(rgb_data, width, height,
                            width * 3, QImage::Format_RGB888).copy();
    has_new_frame_ = true;
}

void WebcamWidget::refreshDisplay()
{
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!has_new_frame_) return;
    has_new_frame_ = false;

    QPixmap pix = QPixmap::fromImage(current_frame_);
    setPixmap(pix.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
