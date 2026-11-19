#include "uavp/stages/s3_select/measure.hpp"

#include <opencv2/calib3d.hpp>

namespace uavp {

// --- measure --------------------------------------------------------------------------------------------------
void points_on_plane_m(std::span<const cv::Point> points_px, const CameraModel& camera, double dist_m,
                       std::vector<cv::Point2f>& out_m) {
    std::vector<cv::Point2d> px(points_px.begin(), points_px.end()), normalized;
    cv::undistortPoints(px, normalized, camera.K_px, camera.dist_coeffs);
    out_m.clear();
    for (const auto& p : normalized)
        out_m.emplace_back(static_cast<float>(p.x * dist_m), static_cast<float>(p.y * dist_m));
}

SizeMm rect_size_mm(std::span<const cv::Point2f> points_m) {
    const RectSize r = min_area_rect(points_m);
    return {1000.0 * r.short_side, 1000.0 * r.long_side, r.yaw_deg};
}

Observation measure(std::optional<SegObject> obj, float conf, uint32_t frame_idx, const CameraModel& camera,
                    double D_m) {
    constexpr std::size_t kMinContourPoints = 3;
    Observation o;
    o.frame_idx = frame_idx;
    o.conf = conf;
    if (!obj) return o;
    o.area_px = obj->area_px;
    std::vector<cv::Point> contour;
    if (largest_contour_px(obj->mask, contour) && contour.size() >= kMinContourPoints) {
        std::vector<cv::Point2f> floor_m;
        points_on_plane_m(contour, camera, D_m, floor_m);
        const SizeMm s = rect_size_mm(floor_m);
        o.W_mm = s.W_mm;
        o.L_mm = s.L_mm;
    }
    o.seg = std::move(obj);
    return o;
}

}  // namespace uavp
