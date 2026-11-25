// He the gioi W = NED cua EKF Pixhawk (mm). Than B = FRD cua FC. Camera C = OpenCV (x phai, y xuong, z truc quang).
//   p_W = R_WB (R_BC p_C + t_BC) + t_WB,   R_WB = Rz(yaw) Ry(pitch) Rx(roll) (ZYX, quy uoc MAVLink ATTITUDE)
//   t_WB: x, y tu LOCAL_POSITION_NED (neu co, khong thi 0); z = -do cao: rangefinder huong xuong x cos(roll) cos(pitch)
//         (san phang), khong co rangefinder thi z cua LOCAL_POSITION_NED.
// Sau khi khoa vat: p_W = T_WC(t0) p_C(t0); moi khung p_C(t) = T_CW(t) p_W -> chieu len anh = tam ROI du doan.
// Chi co goc quay (khong vi tri, khong rangefinder): anh dich theo homography H = K R(t <- t0) K^-1.
#pragma once

#include <filesystem>
#include <optional>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"
#include "uavp/sources/mavlink.hpp"

namespace uavp {

// Camera -> than FC (configs/calib/T_BC.yaml).
struct BodyCamera {
    cv::Matx33d R_BC{cv::Matx33d::eye()};
    cv::Vec3d t_BC_mm{0.0, 0.0, 0.0};
};
[[nodiscard]] BodyCamera load_body_camera(const std::filesystem::path& path);

[[nodiscard]] cv::Matx33d rotation_ned_from_body(double roll_rad, double pitch_rad, double yaw_rad) noexcept;

// Tu the camera trong W tu tu the FC. has_position = co vi tri ngang hoac do cao (rangefinder).
[[nodiscard]] CamPose camera_pose(const FcPose& fc, const BodyCamera& bc) noexcept;

[[nodiscard]] inline cv::Point3d to_world(const CamPose& c, const cv::Point3d& p_C_mm) noexcept {
    const cv::Vec3d p = c.R_WC * cv::Vec3d(p_C_mm.x, p_C_mm.y, p_C_mm.z) + c.t_WC_mm;
    return {p[0], p[1], p[2]};
}
[[nodiscard]] inline cv::Point3d to_camera(const CamPose& c, const cv::Point3d& p_W_mm) noexcept {
    const cv::Vec3d p = c.R_WC.t() * (cv::Vec3d(p_W_mm.x, p_W_mm.y, p_W_mm.z) - c.t_WC_mm);
    return {p[0], p[1], p[2]};
}

// Chieu diem he camera (mm) len anh co meo. nullopt neu sau camera.
[[nodiscard]] std::optional<cv::Point2d> project_point_px(const cv::Point3d& p_C_mm, const CameraModel& camera);
// Diem anh qua phep quay camera thuan tuy R_new_old (khu meo -> quay tia -> chieu lai co meo).
[[nodiscard]] std::optional<cv::Point2d> rotate_pixel(cv::Point2d uv_px, const cv::Matx33d& R_new_old,
                                                      const CameraModel& camera);
// Homography quay thuan tuy H = K R K^-1 (bo qua meo: dung de doi mask nho quanh tam).
[[nodiscard]] cv::Matx33d rotation_homography(const cv::Matx33d& R_new_old, const CameraModel& camera) noexcept;

}  // namespace uavp
