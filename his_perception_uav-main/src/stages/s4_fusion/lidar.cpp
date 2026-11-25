// LiDAR -> he camera, chieu len anh, loc trong mask (giong lidar.py, fuse.py buoc 1-2).
#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "uavp/stages/s4_fusion/lidar.hpp"

namespace uavp {

double lidar_angle_rad(double angle_ld06_deg, bool clockwise, double offset_deg) noexcept {
    const double a = clockwise ? -angle_ld06_deg : angle_ld06_deg;
    return (py_mod(a + offset_deg + 180.0, 360.0) - 180.0) * CV_PI / 180.0;
}

void scan_to_camera_mm(const Scan& scan, const Extrinsics& ext, const FusionConfig& cfg,
                       std::vector<cv::Point3f>& points_C_mm, std::vector<int64_t>& t_ns) {
    const double deg_min = cfg.scan_angle_min_deg, deg_max = cfg.scan_angle_max_deg;
    const auto& R = ext.R_CL;
    for (std::size_t i = 0; i < scan.size(); ++i) {
        const double raw_mm = scan.range_mm[i];
        // loc re truoc (tam, cuong do), luong giac sau
        if (raw_mm <= 0.0 || raw_mm < cfg.r_min_mm || raw_mm > cfg.r_max_mm || scan.intensity[i] < cfg.intensity_min)
            continue;
        const double theta = lidar_angle_rad(scan.angle_deg[i], ext.angle_clockwise, ext.angle_offset_deg);
        const double theta_deg = theta * 180.0 / CV_PI;
        if (theta_deg < deg_min || theta_deg > deg_max) continue;
        const double r = ext.range_scale * raw_mm + ext.range_offset_mm;
        const double xl = r * std::cos(theta), yl = r * std::sin(theta);  // mat quet z_L = 0
        const double zc = R(2, 0) * xl + R(2, 1) * yl + ext.t_CL_mm[2];
        if (zc <= 0.0) continue;
        const double xc = R(0, 0) * xl + R(0, 1) * yl + ext.t_CL_mm[0];
        const double yc = R(1, 0) * xl + R(1, 1) * yl + ext.t_CL_mm[1];
        points_C_mm.emplace_back(static_cast<float>(xc), static_cast<float>(yc), static_cast<float>(zc));
        t_ns.push_back(scan.t_ns[i]);
    }
}

void eroded_mask(const SegMask& seg, int erode_px, cv::Mat& out) {
    if (erode_px <= 0) {
        out = seg.mask;
        return;
    }
    const cv::Mat kernel = cv::Mat::ones(2 * erode_px + 1, 2 * erode_px + 1, CV_8U);
    cv::erode(seg.mask, out, kernel);
}

void project_px(std::span<const cv::Point3f> points_C_mm, const CameraModel& camera, std::vector<cv::Point2f>& uv_px) {
    uv_px.clear();
    if (points_C_mm.empty()) return;
    const std::vector<cv::Point3f> pts(points_C_mm.begin(), points_C_mm.end());
    cv::projectPoints(pts, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera.K_px, camera.dist_coeffs, uv_px);
}

double box_max_norm_radius(double x0_px, double y0_px, double w_px, double h_px, const CameraModel& camera) {
    const std::vector<cv::Point2d> corners{
        {x0_px, y0_px}, {x0_px + w_px, y0_px}, {x0_px, y0_px + h_px}, {x0_px + w_px, y0_px + h_px}};
    std::vector<cv::Point2d> normalized;
    cv::undistortPoints(corners, normalized, camera.K_px, camera.dist_coeffs);
    double best = 0.0;
    for (const auto& p : normalized) best = std::max(best, std::hypot(p.x, p.y));
    return best;
}

void visible_px(std::span<const cv::Point3f> points_C_mm, const CameraModel& camera, int width_px, int height_px,
                std::vector<cv::Point2f>& uv_px) {
    const double max_r = box_max_norm_radius(0, 0, width_px - 1, height_px - 1, camera);
    std::vector<cv::Point3f> front;
    for (const auto& p : points_C_mm)
        if (p.z > 0.f && std::hypot(p.x, p.y) / p.z <= max_r) front.push_back(p);
    project_px(front, camera, uv_px);
    const auto out = std::remove_if(uv_px.begin(), uv_px.end(), [&](const cv::Point2f& uv) {
        return uv.x < 0.f || uv.x >= static_cast<float>(width_px) || uv.y < 0.f || uv.y >= static_cast<float>(height_px);
    });
    uv_px.erase(out, uv_px.end());
}

void in_mask(std::span<const cv::Point3f> points_C_mm, const cv::Mat& mask, const SegMask& seg,
             const CameraModel& camera, std::vector<uint8_t>& keep) {
    keep.assign(points_C_mm.size(), 0);
    if (points_C_mm.empty()) return;
    std::vector<cv::Point2f> uv;
    project_px(points_C_mm, camera, uv);
    // meo (k1, k2, k3) gap lai o ban kinh lon: bo diem ngoai tam nhin cua ROI truoc khi tin phep chieu
    const double roi_r = box_max_norm_radius(seg.x0_px - 1, seg.y0_px - 1, mask.cols + 2, mask.rows + 2, camera);
    for (std::size_t i = 0; i < points_C_mm.size(); ++i) {
        const cv::Point3f& p = points_C_mm[i];
        const int col = round_half_even(uv[i].x) - seg.x0_px;  // np.round: nua ve so chan
        const int row = round_half_even(uv[i].y) - seg.y0_px;
        if (col < 0 || col >= mask.cols || row < 0 || row >= mask.rows) continue;
        if (std::hypot(static_cast<double>(p.x), static_cast<double>(p.y)) / p.z > roi_r) continue;
        keep[i] = mask.ptr<uint8_t>(row)[col] != 0 ? 1 : 0;
    }
}

}  // namespace uavp
