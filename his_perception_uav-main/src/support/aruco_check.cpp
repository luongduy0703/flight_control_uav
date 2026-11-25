#include "uavp/support/aruco_check.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
#include <opencv2/objdetect/aruco_detector.hpp>
#define UAVP_ARUCO_DETECTOR_API 1
#else
#include <opencv2/aruco.hpp>
#endif

namespace uavp {
namespace {

// Ten tu dien trong config -> hang so OpenCV (bang tra, chi cac tu dien hay dung).
constexpr std::array<std::pair<std::string_view, int>, 8> kDictionaries{{
    {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
    {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
    {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
    {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
    {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
    {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
    {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
    {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
}};

int dictionary_id(const std::string& name) {
    for (const auto& [n, id] : kDictionaries)
        if (n == name) return id;
    throw std::invalid_argument("aruco.dictionary khong ho tro: " + name);
}

void detect(const cv::Mat& gray, const std::string& dict_name, std::vector<std::vector<cv::Point2f>>& corners,
            std::vector<int>& ids) {
#ifdef UAVP_ARUCO_DETECTOR_API
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;  // goc duoi pixel
    const cv::aruco::ArucoDetector detector(cv::aruco::getPredefinedDictionary(dictionary_id(dict_name)), params);
    detector.detectMarkers(gray, corners, ids);
#else
    const auto params = cv::aruco::DetectorParameters::create();
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    const auto dictionary = cv::aruco::getPredefinedDictionary(dictionary_id(dict_name));
    cv::aruco::detectMarkers(gray, dictionary, corners, ids, params);
#endif
}

}  // namespace

bool find_marker_px(const cv::Mat& image_rgb, const ArucoConfig& cfg, std::array<cv::Point2f, 4>& corners) {
    cv::Mat gray;
    cv::cvtColor(image_rgb, gray, cv::COLOR_RGB2GRAY);  // anh pipeline RGB
    std::vector<std::vector<cv::Point2f>> all;
    std::vector<int> ids;
    detect(gray, cfg.dictionary, all, ids);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != cfg.marker_id) continue;
        for (std::size_t k = 0; k < 4; ++k) corners[k] = all[i][k];
        return true;
    }
    return false;
}

ArucoPose marker_pose(const std::array<cv::Point2f, 4>& corners_px, int marker_id, double side_mm,
                      const CameraModel& camera) {
    const double h = side_mm / 2.0;  // TL, TR, BR, BL nhu cv::aruco va SOLVEPNP_IPPE_SQUARE
    const std::vector<cv::Point3d> object{{-h, h, 0.0}, {h, h, 0.0}, {h, -h, 0.0}, {-h, -h, 0.0}};
    const std::vector<cv::Point2d> image(corners_px.begin(), corners_px.end());
    cv::Vec3d rvec, tvec;
    if (!cv::solvePnP(object, image, camera.K_px, camera.dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE))
        throw std::runtime_error("solvePnP ArUco that bai");
    ArucoPose pose;
    cv::Rodrigues(rvec, pose.R_CM);
    pose.t_CM_mm = tvec;
    pose.corners_px = corners_px;
    pose.marker_id = marker_id;
    std::vector<cv::Point2d> uv;
    cv::projectPoints(object, rvec, tvec, camera.K_px, camera.dist_coeffs, uv);
    double sq = 0.0;
    for (std::size_t i = 0; i < 4; ++i) sq += (uv[i].x - image[i].x) * (uv[i].x - image[i].x) +
                                              (uv[i].y - image[i].y) * (uv[i].y - image[i].y);
    pose.reproj_px = std::sqrt(sq / 4.0);
    pose.tilt_deg = std::acos(std::min(1.0, std::abs(pose.R_CM(2, 2)))) * 180.0 / CV_PI;
    return pose;
}

cv::Point3d ray_plane_mm(cv::Point2d center_px, const ArucoPose& pose, const CameraModel& camera) {
    std::vector<cv::Point2d> in{center_px}, out;
    cv::undistortPoints(in, out, camera.K_px, camera.dist_coeffs);
    const cv::Vec3d ray(out[0].x, out[0].y, 1.0);
    const cv::Vec3d normal(pose.R_CM(0, 2), pose.R_CM(1, 2), pose.R_CM(2, 2));
    const double denom = normal.dot(ray);
    if (std::abs(denom) < 1e-9) return {kNaN, kNaN, kNaN};  // tia song song mat phang
    const double s = normal.dot(pose.t_CM_mm) / denom;
    return {ray[0] * s, ray[1] * s, ray[2] * s};
}

bool check_aruco(const cv::Mat& image_rgb, cv::Point2d center_px, const ArucoConfig& cfg, const CameraModel& camera,
                 ArucoPose& pose, cv::Point3d& p_aruco_mm) {
    std::array<cv::Point2f, 4> corners{};
    if (!find_marker_px(image_rgb, cfg, corners)) return false;
    pose = marker_pose(corners, cfg.marker_id, cfg.side_mm, camera);
    p_aruco_mm = ray_plane_mm(center_px, pose, camera);
    return true;
}

}  // namespace uavp
