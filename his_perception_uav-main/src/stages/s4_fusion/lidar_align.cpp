#include "uavp/stages/s4_fusion/lidar_align.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/calib3d.hpp>

#include "uavp/stages/s4_fusion/lidar.hpp"

namespace uavp {

namespace {

bool inside(const cv::Point2f& p, int w, int h) noexcept {
    return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0.0F && p.y >= 0.0F && p.x < static_cast<float>(w) &&
           p.y < static_cast<float>(h);
}

// Tia qua pixel (co meo) -> (x_n, y_n, 1).
cv::Vec3d pixel_ray(cv::Point2d px, const CameraModel& camera) {
    const std::vector<cv::Point2d> in{px};
    std::vector<cv::Point2d> out;
    cv::undistortPoints(in, out, cv::Mat(camera.K_px), camera.dist_coeffs);
    return {out[0].x, out[0].y, 1.0};
}

}  // namespace

LidarAlign lidar_alignment(std::span<const cv::Point3f> points_C_mm, cv::Point2d center_px, const CameraModel& camera,
                           int width_px, int height_px, double max_gap_px, double e_max_mm) {
    LidarAlign out;
    std::vector<cv::Point2f> uv;
    project_px(points_C_mm, camera, uv);
    double best = std::numeric_limits<double>::infinity();
    cv::Point2d best_px;
    cv::Vec3d best_C;
    const auto vis = [&](std::size_t i) { return points_C_mm[i].z > 0.0F && inside(uv[i], width_px, height_px); };
    const auto consider = [&](cv::Point2d foot, cv::Vec3d p_C) {
        const double e = cv::norm(center_px - foot);
        if (e < best) best = e, best_px = foot, best_C = p_C;
    };
    for (std::size_t i = 0; i < uv.size(); ++i) {
        if (!vis(i)) continue;
        ++out.n_visible;
        const cv::Point3f& p = points_C_mm[i];
        consider(cv::Point2d(uv[i]), cv::Vec3d(p.x, p.y, p.z));
        if (i + 1 >= uv.size() || !vis(i + 1)) continue;
        const cv::Point2d a(uv[i]), b(uv[i + 1]), ab = b - a;
        const double len2 = ab.dot(ab);
        if (len2 <= 0.0 || len2 > max_gap_px * max_gap_px) continue;
        const double s = (center_px - a).dot(ab) / len2;
        if (s <= 0.0 || s >= 1.0) continue;  // 2 dau mut da xet o tren
        const cv::Point3f& q = points_C_mm[i + 1];
        consider(a + s * ab, cv::Vec3d(p.x + s * (q.x - p.x), p.y + s * (q.y - p.y), p.z + s * (q.z - p.z)));
    }
    if (out.n_visible == 0) return out;
    out.valid = true;
    out.e_px = best;
    out.foot_px = best_px;
    out.foot_C_mm = best_C;
    out.d_mm = best_C[2];
    out.e_mm = best * out.d_mm / camera.K_px(0, 0);
    out.ok = out.e_mm <= e_max_mm;
    out.delta_C_mm = pixel_ray(center_px, camera) * out.d_mm - best_C;
    return out;
}

}  // namespace uavp
