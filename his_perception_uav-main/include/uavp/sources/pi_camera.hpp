// Camera Pi qua libcamera (thay picamera2 cua ban Python). Mo theo muc sensor_mode cua file K: kich thuoc anh, che do
// cam bien, lat 180 do, fps co dinh, net khoa (LensPosition). Dinh dang BGR888 cua libcamera = byte [R, G, B] -> anh
// RGB dung quy uoc pipeline.
// Luong libcamera giu request moi nhat (cac request cu duoc tra lai ngay); next_frame() chep request moi nhat vao
// slot cua FramePool (mot lan chep duy nhat, chi cho khung duoc dung) roi tra request cho camera.
// Thoi diem khung: SensorTimestamp (CLOCK_BOOTTIME) doi ve CLOCK_MONOTONIC (dong ho chung voi LD06).
#pragma once

#include <memory>

#include "uavp/common/config.hpp"
#include "uavp/sources/frame_source.hpp"

namespace uavp {

class PiCamera final : public IFrameSource {
public:
    PiCamera(const CaptureConfig& capture, int buffer_count, int sensor_bit_depth);
    ~PiCamera() override;
    PiCamera(const PiCamera&) = delete;
    PiCamera& operator=(const PiCamera&) = delete;

    [[nodiscard]] bool next_frame(cv::Mat& image_rgb, FrameMeta& meta, double timeout_s) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace uavp
