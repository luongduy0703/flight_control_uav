// Kiem tra sai so s4 bang ArUco dan tren mat tren vat (giong src/uavp/aruco_check.py). Chay mot lan cuoi phien.
//   1. Tim marker (tu dien + id trong config) -> 4 goc (px). 2. solvePnP IPPE_SQUARE -> R_CM, t_CM (mm).
//   3. Tia qua tam vat cat mat phang marker -> p_obj ArUco; d ArUco = Z.
#pragma once

#include <array>

#include <opencv2/core.hpp>

#include "uavp/common/common.hpp"
#include "uavp/common/config.hpp"

namespace uavp {

struct ArucoPose {
    cv::Matx33d R_CM;                       // he marker -> he camera
    cv::Vec3d t_CM_mm;                      // tam marker
    std::array<cv::Point2f, 4> corners_px{};  // TL, TR, BR, BL
    double reproj_px{0.0};                  // RMS chieu lai 4 goc
    double tilt_deg{0.0};                   // goc phap tuyen marker - truc quang
    int marker_id{0};
};

[[nodiscard]] bool find_marker_px(const cv::Mat& image_rgb, const ArucoConfig& cfg, std::array<cv::Point2f, 4>& corners);
[[nodiscard]] ArucoPose marker_pose(const std::array<cv::Point2f, 4>& corners_px, int marker_id, double side_mm,
                                    const CameraModel& camera);
// Diem (mm): tia camera qua center_px cat mat phang marker.
[[nodiscard]] cv::Point3d ray_plane_mm(cv::Point2d center_px, const ArucoPose& pose, const CameraModel& camera);
// (pose, p_obj ArUco tai tam vat); false neu khong thay marker.
[[nodiscard]] bool check_aruco(const cv::Mat& image_rgb, cv::Point2d center_px, const ArucoConfig& cfg,
                               const CameraModel& camera, ArucoPose& pose, cv::Point3d& p_aruco_mm);

}  // namespace uavp
