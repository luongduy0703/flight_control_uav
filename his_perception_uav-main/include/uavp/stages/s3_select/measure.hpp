// s3 select / measure: mask -> W, L tren mat phang cach camera D (undistort contour). Giong measure.py.
#pragma once

#include <optional>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"
#include "uavp/stages/s2_segment/mask_shape.hpp"

namespace uavp {

// --- measure --------------------------------------------------------------------------------------------------
// Diem anh -> diem tren mat phang cach camera dist_m (met, goc o truc quang): undistort roi nhan khoang cach.
void points_on_plane_m(std::span<const cv::Point> points_px, const CameraModel& camera, double dist_m,
                       std::vector<cv::Point2f>& out_m);

struct SizeMm {
    double W_mm{kNaN};
    double L_mm{kNaN};
    double yaw_deg{kNaN};
};
[[nodiscard]] SizeMm rect_size_mm(std::span<const cv::Point2f> points_m);

// Observation tu dau ra s2 (nullopt = khong co mask).
[[nodiscard]] Observation measure(std::optional<SegObject> obj, float conf, uint32_t frame_idx,
                                  const CameraModel& camera, double D_m);

}  // namespace uavp
