#include "uavp/tracking/world_pose.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>

namespace uavp {

BodyCamera load_body_camera(const std::filesystem::path& path) {
    const YAML::Node raw = YAML::LoadFile(path.string());
    if (!raw["R_BC"] || !raw["t_BC_m"]) throw std::runtime_error("Thieu R_BC / t_BC_m trong " + path.string());
    BodyCamera bc;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            bc.R_BC(r, c) = raw["R_BC"][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)].as<double>();
    const auto t = raw["t_BC_m"].as<std::vector<double>>();
    bc.t_BC_mm = {1000.0 * t.at(0), 1000.0 * t.at(1), 1000.0 * t.at(2)};
    return bc;
}

cv::Matx33d rotation_ned_from_body(double roll_rad, double pitch_rad, double yaw_rad) noexcept {
    const double cr = std::cos(roll_rad), sr = std::sin(roll_rad);
    const double cp = std::cos(pitch_rad), sp = std::sin(pitch_rad);
    const double cy = std::cos(yaw_rad), sy = std::sin(yaw_rad);
    return {cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
            -sp,     cp * sr,                cp * cr};
}

CamPose camera_pose(const FcPose& fc, const BodyCamera& bc) noexcept {
    const cv::Matx33d R_WB = rotation_ned_from_body(fc.roll_rad, fc.pitch_rad, fc.yaw_rad);
    cv::Vec3d t_WB_mm{0.0, 0.0, 0.0};
    if (fc.has_position) t_WB_mm = 1000.0 * fc.pos_ned_m;
    if (fc.has_range) t_WB_mm[2] = -1000.0 * fc.range_m * std::cos(fc.roll_rad) * std::cos(fc.pitch_rad);
    CamPose c;
    c.R_WC = R_WB * bc.R_BC;
    c.t_WC_mm = R_WB * bc.t_BC_mm + t_WB_mm;
    c.has_position = fc.has_position || fc.has_range;
    return c;
}

std::optional<cv::Point2d> project_point_px(const cv::Point3d& p_C_mm, const CameraModel& camera) {
    if (p_C_mm.z <= 0.0) return std::nullopt;
    const std::vector<cv::Point3d> pts{p_C_mm};
    std::vector<cv::Point2d> uv;
    cv::projectPoints(pts, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera.K_px, camera.dist_coeffs, uv);
    return uv[0];
}

std::optional<cv::Point2d> rotate_pixel(cv::Point2d uv_px, const cv::Matx33d& R_new_old, const CameraModel& camera) {
    const std::vector<cv::Point2d> in{uv_px};
    std::vector<cv::Point2d> norm;
    cv::undistortPoints(in, norm, camera.K_px, camera.dist_coeffs);
    const cv::Vec3d ray = R_new_old * cv::Vec3d(norm[0].x, norm[0].y, 1.0);
    return project_point_px({ray[0], ray[1], ray[2]}, camera);
}

cv::Matx33d rotation_homography(const cv::Matx33d& R_new_old, const CameraModel& camera) noexcept {
    return camera.K_px * R_new_old * camera.K_px.inv();
}

}  // namespace uavp
