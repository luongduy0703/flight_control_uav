// ArUco diem cat canh (planA): he S cua marker, vi tri marker trong W, camera trong S, dong log MARKER.
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <optional>
#include <string>

#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
#include <opencv2/objdetect/aruco_detector.hpp>
#else
#include <opencv2/aruco.hpp>
#endif

#include "uavp/support/launch_marker.hpp"

using namespace uavp;

namespace {

constexpr double kDeg = CV_PI / 180.0;

// R_WM khi marker nam phang tren san, mui ten (x_S) huong psi so voi Bac.
cv::Matx33d marker_on_floor(double psi_rad) {
    const double c = std::cos(psi_rad), s = std::sin(psi_rad);
    // cot: x_M = y_S = (-s, c, 0), y_M = x_S = (c, s, 0), z_M = -z_S = (0, 0, -1) (len tren)
    return {-s, c, 0.0,
            c, s, 0.0,
            0.0, 0.0, -1.0};
}

ArucoPose pose_from(const cv::Matx33d& R_WM, const cv::Vec3d& t_WM, const CamPose& cam) {
    ArucoPose p;
    p.R_CM = cam.R_WC.t() * R_WM;
    p.t_CM_mm = cam.R_WC.t() * (t_WM - cam.t_WC_mm);
    p.reproj_px = 0.3;
    p.tilt_deg = 0.0;
    p.marker_id = 1;
    return p;
}

// Camera nhin thang xuong, canh tren anh huong theo yaw: R_WC = R_WB(yaw) * R_BC danh nghia (configs/calib/T_BC.yaml).
CamPose camera_down(double yaw_rad, const cv::Vec3d& t_WC_mm) {
    BodyCamera bc;
    bc.R_BC = {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    FcPose fc;
    fc.yaw_rad = yaw_rad;
    fc.pos_ned_m = t_WC_mm / 1000.0;
    fc.has_position = true;
    return camera_pose(fc, bc);
}

}  // namespace

TEST(LaunchMarker, MarkerInWorldAndCameraInMarkerFrame) {
    const double psi = 30.0 * kDeg, yaw = 40.0 * kDeg;
    const cv::Vec3d t_WM{1000.0, 2000.0, 50.0};
    const CamPose cam = camera_down(yaw, {1100.0, 2000.0, -750.0});  // 10 cm Bac, 80 cm tren marker
    const MarkerObservation o = marker_observation(pose_from(marker_on_floor(psi), t_WM, cam), cam, yaw, 42);
    ASSERT_TRUE(o.has_world);
    EXPECT_EQ(o.t_ns, 42);
    EXPECT_NEAR(o.p_M_W_mm[0], 1000.0, 1e-6);
    EXPECT_NEAR(o.p_M_W_mm[1], 2000.0, 1e-6);
    EXPECT_NEAR(o.p_M_W_mm[2], 50.0, 1e-6);
    EXPECT_NEAR(o.yaw_M_W_deg, 30.0, 1e-6);
    EXPECT_NEAR(o.yaw_fc_deg, 40.0, 1e-6);
    EXPECT_NEAR(o.yaw_rel_deg, 10.0, 1e-6);
    EXPECT_NEAR(o.tilt_W_deg, 0.0, 1e-6);
    // Camera lech 100 mm ve Bac; trong S (mui ten 30 do): x = 100 cos30, y = -100 sin30; cao 800 mm -> z = -800.
    EXPECT_NEAR(o.p_cam_S_mm[0], 100.0 * std::cos(psi), 1e-6);
    EXPECT_NEAR(o.p_cam_S_mm[1], -100.0 * std::sin(psi), 1e-6);
    EXPECT_NEAR(o.p_cam_S_mm[2], -800.0, 1e-6);
}

TEST(LaunchMarker, WithoutFcOnlyCameraFrame) {
    const CamPose cam = camera_down(0.0, {0.0, 0.0, -500.0});
    const MarkerObservation o = marker_observation(pose_from(marker_on_floor(0.0), {0.0, 0.0, 0.0}, cam), std::nullopt,
                                                   std::nan(""), 1);
    EXPECT_FALSE(o.has_world);
    EXPECT_TRUE(std::isnan(o.yaw_M_W_deg));
    EXPECT_TRUE(std::isnan(o.p_M_W_mm[0]));
    EXPECT_NEAR(o.p_cam_S_mm[2], -500.0, 1e-6);
    const std::string line = marker_line(o);
    EXPECT_EQ(line.rfind("MARKER seen=1 t_ns=1 ", 0), 0U);
    EXPECT_NE(line.find(" world=0 "), std::string::npos);
}

TEST(LaunchMarker, ObjectArucoUsesItsOwnTagAndMarker) {
    const CamPose cam = camera_down(0.0, {0.0, 0.0, -500.0});
    const MarkerObservation o = marker_observation(pose_from(marker_on_floor(0.0), {0.0, 0.0, 0.0}, cam), cam, 0.0, 7);
    const std::string line = marker_line(o, "OBJARUCO");
    EXPECT_EQ(line.rfind("OBJARUCO seen=1 t_ns=7 ", 0), 0U);
    EXPECT_EQ(line.find("MARKER seen="), std::string::npos);  // planA: khong lan voi ArUco diem cat canh

    LaunchMarkerConfig launch;
    launch.enabled = true;
    launch.marker_id = 1;
    launch.side_mm = 100.0;
    launch.period_s = 0.3;
    launch.track_object = true;
    const ArucoConfig aruco{"DICT_4X4_50", 0, 36.5, true};
    const LaunchMarkerConfig obj = object_marker_config(launch, aruco);
    EXPECT_TRUE(obj.enabled);
    EXPECT_EQ(obj.marker_id, 0);
    EXPECT_DOUBLE_EQ(obj.side_mm, 36.5);
    EXPECT_DOUBLE_EQ(obj.period_s, 0.3);  // chu ky / nguong lay tu launch_marker
    launch.track_object = false;
    EXPECT_FALSE(object_marker_config(launch, aruco).enabled);
}

TEST(LaunchMarker, WrapsYawAcross180) {
    const CamPose cam = camera_down(-170.0 * kDeg, {0.0, 0.0, -600.0});
    const MarkerObservation o =
        marker_observation(pose_from(marker_on_floor(175.0 * kDeg), {0.0, 0.0, 0.0}, cam), cam, -170.0 * kDeg, 1);
    EXPECT_NEAR(o.yaw_M_W_deg, 175.0, 1e-6);
    EXPECT_NEAR(o.yaw_rel_deg, 15.0, 1e-6);  // -170 - 175 = -345 -> 15
}

// Anh tong hop: marker 200 px o tam anh, camera ly tuong (khong meo) -> tim + solvePnP + he S.
TEST(LaunchMarker, DetectsSyntheticMarkerImage) {
    constexpr int kSidePx = 200;
    constexpr double kSideMm = 100.0, kF = 800.0;
    cv::Mat marker;
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
    cv::aruco::generateImageMarker(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50), 1, kSidePx, marker, 1);
#else
    cv::aruco::drawMarker(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50), 1, kSidePx, marker, 1);
#endif
    cv::Mat gray(1080, 1920, CV_8UC1, cv::Scalar(255));
    const int x0 = 960 - kSidePx / 2, y0 = 540 - kSidePx / 2;
    marker.copyTo(gray(cv::Rect(x0, y0, kSidePx, kSidePx)));
    cv::Mat rgb;
    cv::cvtColor(gray, rgb, cv::COLOR_GRAY2RGB);
    CameraModel camera;
    camera.K_px = {kF, 0.0, 959.5, 0.0, kF, 539.5, 0.0, 0.0, 1.0};  // tam marker dung tam anh
    camera.dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    const ArucoConfig cfg{"DICT_4X4_50", 1, kSideMm, true};
    std::array<cv::Point2f, 4> corners{};
    ASSERT_TRUE(find_marker_px(rgb, cfg, corners));
    const ArucoConfig other{"DICT_4X4_50", 0, kSideMm, true};
    EXPECT_FALSE(find_marker_px(rgb, other, corners));  // marker vat (id 0) khong bi nham
    ASSERT_TRUE(find_marker_px(rgb, cfg, corners));
    const ArucoPose pose = marker_pose(corners, 1, kSideMm, camera);
    const MarkerObservation o = marker_observation(pose, std::nullopt, std::nan(""), 0);
    const double z_mm = kF * kSideMm / kSidePx;  // 400 mm
    EXPECT_NEAR(o.p_cam_S_mm[2], -z_mm, 0.01 * z_mm);
    EXPECT_NEAR(o.p_cam_S_mm[0], 0.0, 2.0);
    EXPECT_NEAR(o.p_cam_S_mm[1], 0.0, 2.0);
    EXPECT_LT(o.reproj_px, 1.0);
    // Canh tren anh = canh tren marker = mui ten x_S; truc y_S (phai mui ten) = x_C; x_S = -y_C.
    const cv::Matx33d R_SM(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
    const cv::Matx33d R_SC = R_SM * pose.R_CM.t();
    EXPECT_NEAR(R_SC(1, 0), 1.0, 0.01);   // x_C -> y_S
    EXPECT_NEAR(R_SC(0, 1), -1.0, 0.01);  // y_C -> -x_S
}

// Vung KHONG chon vat: thay marker o khung nay -> tam 4 goc + ban kinh = exclude_radius_mm o do sau marker.
TEST(LaunchMarker, ExclusionZoneFromSeenMarker) {
    constexpr int kSidePx = 200;
    constexpr double kSideMm = 100.0, kF = 800.0;
    cv::Mat marker;
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
    cv::aruco::generateImageMarker(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50), 1, kSidePx, marker, 1);
#else
    cv::aruco::drawMarker(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50), 1, kSidePx, marker, 1);
#endif
    cv::Mat gray(1080, 1920, CV_8UC1, cv::Scalar(255));
    marker.copyTo(gray(cv::Rect(600, 700, kSidePx, kSidePx)));  // tam (699.5, 799.5)
    cv::Mat rgb;
    cv::cvtColor(gray, rgb, cv::COLOR_GRAY2RGB);
    CameraModel camera;
    camera.K_px = {kF, 0.0, 959.5, 0.0, kF, 539.5, 0.0, 0.0, 1.0};
    camera.dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    LaunchMarkerConfig cfg;
    cfg.enabled = true;
    cfg.dictionary = "DICT_4X4_50";
    cfg.side_mm = kSideMm;
    cfg.max_tilt_deg = 30.0;
    cfg.exclude_radius_mm = 150.0;
    const BodyCamera body;
    const auto csv = std::filesystem::temp_directory_path() / "uavp_test_launch_marker.csv";
    LaunchMarkerTracker tr(cfg, camera, body, csv);
    EXPECT_FALSE(tr.zone_px(1, nullptr));  // chua thay: khong loai gi
    tr.update(rgb, 1'000'000'000, nullptr);
    ASSERT_EQ(tr.seen(), 1);
    const auto z = tr.zone_px(1'000'000'000, nullptr);
    ASSERT_TRUE(z);
    EXPECT_NEAR(z->first.x, 699.5, 1.0);
    EXPECT_NEAR(z->first.y, 799.5, 1.0);
    // marker 100 mm = 200 px -> 150 mm ~ 300 px (lech nhe: marker khong o truc quang)
    EXPECT_NEAR(z->second, 300.0, 30.0);
    EXPECT_FALSE(tr.zone_px(2'000'000'000, nullptr));  // khung khac, khong co FC de chieu
    std::filesystem::remove(csv);
}
