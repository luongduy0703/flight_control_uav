// s2 segment / mask_shape: mask -> contour, tam (trong tam), minAreaRect (yaw). Giong mask_shape.py.
#pragma once

#include <memory>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/types.hpp"

namespace uavp {

// --- Hinh dang mask -------------------------------------------------------------------------------------------
// Contour ngoai lon nhat cua mask (toa do anh goc); false neu mask rong.
[[nodiscard]] bool largest_contour_px(const SegMask& seg, std::vector<cv::Point>& contour_px);

struct RectSize {
    double short_side{0.0};
    double long_side{0.0};
    double yaw_deg{0.0};  // huong canh dai so voi truc x, [0, 180)
};

// cv::minAreaRect cua cac diem (cung don vi voi diem).
[[nodiscard]] RectSize min_area_rect(std::span<const cv::Point2f> points);

// Tam (trong tam mask), yaw (minAreaRect contour tren anh). false neu mask rong.
[[nodiscard]] bool seg_object(SegMask seg, SegObject& out);

}  // namespace uavp
