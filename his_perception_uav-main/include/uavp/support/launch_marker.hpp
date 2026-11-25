// ArUco diem cat canh (planA): marker dan duoi chan UAV. KHONG dua vao EKF Pixhawk. Dung de:
//   - planA: lan dau thay -> he local so toa do vat (EKF) voi thuoc; giu vi tri hover theo marker (dong MARKER).
//   - s3: vung KHONG chon vat quanh marker (zone_px) -> to giay marker khong bao gio bi chon lam vat gap.
//
// He S cua marker (in tren to giay, tools planA/make_launch_marker.py):
//   goc = tam marker tren san, x = mui ten "+x" (phia canh tren marker), y = ben phai mui ten (nhin tu tren xuong),
//   z = huong xuong (san = 0, mat vat cao h -> z = -h). Cung chieu NED: x "Bac", y "Dong", z xuong.
// He M cua cv::aruco (SOLVEPNP_IPPE_SQUARE): x phai, y len (canh tren), z ra khoi giay (len tren).
//   => p_S = R_SM p_M, R_SM = [[0,1,0],[1,0,0],[0,0,-1]].
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"
#include "uavp/sources/mavlink.hpp"
#include "uavp/support/aruco_check.hpp"
#include "uavp/tracking/world_pose.hpp"

namespace uavp {

// Mot lan thay marker.
struct MarkerObservation {
    int64_t t_ns{0};
    double reproj_px{0.0};
    double tilt_cam_deg{0.0};          // phap tuyen marker - truc quang
    cv::Vec3d p_cam_S_mm{0, 0, 0};     // tam quang camera trong he S (z am: camera tren san)
    bool has_world{false};             // co tu the FC luc chup
    cv::Vec3d p_M_W_mm{0, 0, 0};       // tam marker trong W (NED EKF)
    double yaw_M_W_deg{0.0};           // huong truc x_S (mui ten) trong W, so voi Bac, (-180, 180]
    double tilt_W_deg{0.0};            // phap tuyen marker - phuong dung (W)
    double yaw_fc_deg{0.0};            // yaw UAV (FC) luc chup
    double yaw_rel_deg{0.0};           // mui UAV so voi mui ten marker = yaw_fc - yaw_M_W, (-180, 180]
};

// yaw_fc_rad: NaN khi khong co FC.

// Toan thuan (test duoc khong can anh): tu the marker trong camera (+ tu the camera trong W neu co).
[[nodiscard]] MarkerObservation marker_observation(const ArucoPose& pose, const std::optional<CamPose>& cam,
                                                   double yaw_fc_rad, int64_t t_ns);

// Dong log cho planA (pi/perception_process.py: MARKER_MARK / OBJARUCO_MARK). tag: "MARKER" (diem cat canh) |
// "OBJARUCO" (ArUco tren vat).
[[nodiscard]] std::string marker_line(const MarkerObservation& o, const char* tag = "MARKER");

class LaunchMarkerTracker {
public:
    // tag: ten dong log (marker_line). Dung chung cho ArUco tren vat: cfg voi dictionary / marker_id / side_mm cua
    // muc aruco (object_marker_config).
    LaunchMarkerTracker(const LaunchMarkerConfig& cfg, const CameraModel& camera, const BodyCamera& body_camera,
                        const std::filesystem::path& csv_path, std::string tag = "MARKER");
    // Goi moi khung (tu gioi han theo period_s). fc: buffer tu the Pixhawk (nullptr: chi do trong he camera).
    void update(const cv::Mat& image_rgb, int64_t t_ns, const FcPoseBuffer* fc);
    [[nodiscard]] int seen() const noexcept { return seen_; }
    // Lan do HOP LE moi tu lan goi truoc (moi lan do tra ve dung mot lan), nullopt neu chua co lan moi.
    [[nodiscard]] std::optional<MarkerObservation> take_new() noexcept {
        if (!fresh_) return std::nullopt;
        fresh_ = false;
        return last_;
    }
    // Vi tri marker tren anh luc t_ns (vung KHONG chon vat): thay o chinh khung nay -> tam 4 goc; khong -> chieu tam
    // marker trong W (median cac lan thay) bang tu the FC luc t_ns. Ban kinh = exclude_radius_mm o do sau marker.
    // nullopt: chua tung thay marker / exclude_radius_mm = 0 / khong chieu duoc.
    [[nodiscard]] std::optional<std::pair<cv::Point2d, double>> zone_px(int64_t t_ns, const FcPoseBuffer* fc) const;

private:
    LaunchMarkerConfig cfg_;
    ArucoConfig aruco_;
    const CameraModel& camera_;
    const BodyCamera& body_camera_;
    std::ofstream csv_;
    int64_t last_try_ns_{0};
    bool visible_{false};
    int seen_{0};
    int rejected_{0};
    std::string tag_;
    int64_t last_seen_t_ns_{-1};
    cv::Point2d last_center_px_{0.0, 0.0};
    double last_depth_mm_{0.0};
    std::vector<cv::Vec3d> world_mm_;  // tam marker trong W moi lan thay (gioi han kMaxWorldSamples)
    MarkerObservation last_{};         // lan do hop le gan nhat (take_new)
    bool fresh_{false};
};

// Cau hinh bam ArUco tren vat: chu ky / nguong cua launch_marker, marker cua muc aruco.
[[nodiscard]] LaunchMarkerConfig object_marker_config(const LaunchMarkerConfig& launch, const ArucoConfig& aruco);

}  // namespace uavp
