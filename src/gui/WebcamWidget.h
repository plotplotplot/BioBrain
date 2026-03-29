#pragma once

#include <QLabel>
#include <QImage>
#include <mutex>
#include <cstdint>

class WebcamWidget : public QLabel {
    Q_OBJECT
public:
    explicit WebcamWidget(QWidget* parent = nullptr);

    /// Update with new frame data (thread-safe, called from capture thread).
    void updateFrame(const uint8_t* rgb_data, int width, int height);

public slots:
    void refreshDisplay();

private:
    QImage current_frame_;
    std::mutex frame_mutex_;
    bool has_new_frame_ = false;
};
