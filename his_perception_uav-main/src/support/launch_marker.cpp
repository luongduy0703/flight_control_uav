#include "uavp/support/launch_marker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

#include "uavp/common/common.hpp"
#include "uavp/common/log.hpp"

namespace uavp {
namespace {

constexpr const char* kModule = "uavp.marker";
constexpr double kRadToDeg = 180.0 / CV_PI;
constexpr std::size_t kMaxWorldSamples = 50;  // tam marker trong W: 50 lan thay dau (marker co dinh tren san)
constexpr const char* kCsvHeader =
    "t_ns,reproj_px,tilt_cam_deg,cam_x_S_mm,cam_y_S_mm,cam_z_S_mm,world,M_n_W_mm,M_e_W_mm,M_d_W_mm,yaw_M_W_deg,"
    "tilt_W_deg,yaw_fc_deg,yaw_rel_deg\n";

// He S (mui ten x, phai y, xuong z) tu he M cua cv::aruco (x phai, y canh tren, z ra khoi giay).
const cv::Matx33d kR_SM(0.0, 1.0, 0.0,
                        1.0, 0.0, 0.0,
                        0.0, 0.0, -1.0);

double wrap_deg(double a) {
    if (!std::isfinite(a)) return a;
    a = std::fmod(a + 180.0, 360.0);
    if (a <= 0.0) a += 360.0;
    return a - 180.0;  // (-180, 180]
}

}  // namespace

MarkerObservation marker_observation(const ArucoPose& pose, const std::optional<CamPose>& cam, double yaw_fc_rad,
                                     int64_t t_ns) {
    MarkerObservation o;
    o.t_ns = t_ns;
    o.reproj_px = pose.reproj_px;
    o.tilt_cam_deg = pose.tilt_deg;
    const cv::Vec3d p_cam_M = -(pose.R_CM.t() * pose.t_CM_mm);  // tam quang camera trong he M
    o.p_cam_S_mm = kR_SM * p_cam_M;
    o.yaw_fc_deg = std::isfinite(yaw_fc_rad) ? wrap_deg(yaw_fc_rad * kRadToDeg) : kNaN;
    o.p_M_W_mm = {kNaN, kNaN, kNaN};
    o.yaw_M_W_deg = o.tilt_W_deg = o.yaw_rel_deg = kNaN;
    if (!cam) return o;
    const cv::Matx33d R_WM = cam->R_WC * pose.R_CM;
    // Truc x_S = truc y_M (cot 1); phap tuyen marker z_M (cot 2) phai huong LEN = -z_W (NED).
    o.yaw_M_W_deg = wrap_deg(std::atan2(R_WM(1, 1), R_WM(0, 1)) * kRadToDeg);
    o.tilt_W_deg = std::acos(std::clamp(-R_WM(2, 2), -1.0, 1.0)) * kRadToDeg;
    o.yaw_rel_deg = wrap_deg(o.yaw_fc_deg - o.yaw_M_W_deg);
    o.has_world = cam->has_position;
    if (o.has_world) o.p_M_W_mm = cam->R_WC * pose.t_CM_mm + cam->t_WC_mm;
    return o;
}

std::string marker_line(const MarkerObservation& o, const char* tag) {
    char b[512];
    std::snprintf(b, sizeof(b),
                  "%s seen=1 t_ns=%lld reproj_px=%.2f tilt_cam_deg=%.1f p_cam_S_mm=%.1f,%.1f,%.1f world=%d "
                  "p_M_W_mm=%.1f,%.1f,%.1f yaw_M_W_deg=%.2f tilt_W_deg=%.1f yaw_fc_deg=%.2f yaw_rel_deg=%.2f",
                  tag, static_cast<long long>(o.t_ns), o.reproj_px, o.tilt_cam_deg, o.p_cam_S_mm[0], o.p_cam_S_mm[1],
                  o.p_cam_S_mm[2], o.has_world ? 1 : 0, o.p_M_W_mm[0], o.p_M_W_mm[1], o.p_M_W_mm[2], o.yaw_M_W_deg,
                  o.tilt_W_deg, o.yaw_fc_deg, o.yaw_rel_deg);
    return b;
}

LaunchMarkerTracker::LaunchMarkerTracker(const LaunchMarkerConfig& cfg, const CameraModel& camera,
                                         const BodyCamera& body_camera, const std::filesystem::path& csv_path,
                                         std::string tag)
    : cfg_(cfg), aruco_{cfg.dictionary, cfg.marker_id, cfg.side_mm, true}, camera_(camera), body_camera_(body_camera),
      csv_(csv_path), tag_(std::move(tag)) {
    std::array<cv::Point2f, 4> unused{};
    const cv::Mat blank(16, 16, CV_8UC3, cv::Scalar(255, 255, 255));
    (void)find_marker_px(blank, aruco_, unused);  // tu dien sai -> loi ngay luc khoi dong, khong phai giua chung
    if (!csv_) throw std::runtime_error("Khong ghi duoc " + csv_path.string());
    csv_ << kCsvHeader;
    csv_.flush();
    UAVP_LOG_INFO(kModule, "ArUco %s: %s id %d, canh %.1f mm, moi %.2f s (chi danh gia, khong dieu khien) -> %s",
                  tag_.c_str(), cfg.dictionary.c_str(), cfg.marker_id, cfg.side_mm, cfg.period_s, csv_path.c_str());
}

void LaunchMarkerTracker::update(const cv::Mat& image_rgb, int64_t t_ns, const FcPoseBuffer* fc) {
    if (static_cast<double>(t_ns - last_try_ns_) / 1e9 < cfg_.period_s) return;
    last_try_ns_ = t_ns;
    std::array<cv::Point2f, 4> corners{};
    MarkerObservation o;
    const char* lost_reason = nullptr;
    try {
        if (!find_marker_px(image_rgb, aruco_, corners)) {
            lost_reason = "khong_thay";
        } else {
            const ArucoPose pose = marker_pose(corners, cfg_.marker_id, cfg_.side_mm, camera_);
            std::optional<CamPose> cam;
            double yaw_fc = kNaN;
            FcPose p;
            if (fc != nullptr && fc->pose_at(t_ns, p)) {
                cam = camera_pose(p, body_camera_);
                yaw_fc = p.yaw_rad;
            }
            o = marker_observation(pose, cam, yaw_fc, t_ns);
            const double tilt = std::isfinite(o.tilt_W_deg) ? o.tilt_W_deg : o.tilt_cam_deg;
            if (o.reproj_px > cfg_.max_reproj_px) lost_reason = "reproj";
            else if (tilt > cfg_.max_tilt_deg) lost_reason = "nghieng";
        }
    } catch (const std::exception& e) {
        UAVP_LOG_WARN(kModule, "ArUco diem cat canh: %s", e.what());
        lost_reason = "loi";
    }
    if (lost_reason != nullptr) {
        if (lost_reason[0] != 'k') ++rejected_;
        if (visible_)
            UAVP_LOG_INFO(kModule, "%s seen=0 n=%d rejected=%d reason=%s", tag_.c_str(), seen_, rejected_, lost_reason);
        visible_ = false;
        return;
    }
    visible_ = true;
    ++seen_;
    last_seen_t_ns_ = t_ns;
    last_center_px_ = (cv::Point2d(corners[0]) + cv::Point2d(corners[1]) + cv::Point2d(corners[2]) +
                       cv::Point2d(corners[3])) / 4.0;
    last_depth_mm_ = -o.p_cam_S_mm[2];  // camera cao bao nhieu so voi mat marker
    if (o.has_world && world_mm_.size() < kMaxWorldSamples) world_mm_.push_back(o.p_M_W_mm);
    last_ = o;
    fresh_ = true;
    UAVP_LOG_INFO(kModule, "%s", marker_line(o, tag_.c_str()).c_str());
    char row[384];
    std::snprintf(row, sizeof(row), "%lld,%.3f,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.2f,%.3f,%.2f,%.3f,%.3f\n",
                  static_cast<long long>(o.t_ns), o.reproj_px, o.tilt_cam_deg, o.p_cam_S_mm[0], o.p_cam_S_mm[1],
                  o.p_cam_S_mm[2], o.has_world ? 1 : 0, o.p_M_W_mm[0], o.p_M_W_mm[1], o.p_M_W_mm[2], o.yaw_M_W_deg,
                  o.tilt_W_deg, o.yaw_fc_deg, o.yaw_rel_deg);
    csv_ << row;
    csv_.flush();
}

std::optional<std::pair<cv::Point2d, double>> LaunchMarkerTracker::zone_px(int64_t t_ns, const FcPoseBuffer* fc) const {
    if (seen_ == 0 || cfg_.exclude_radius_mm <= 0.0) return std::nullopt;
    const double f = camera_.K_px(0, 0);
    if (t_ns == last_seen_t_ns_ && last_depth_mm_ > 0.0)  // thay ngay o khung nay
        return std::make_pair(last_center_px_, cfg_.exclude_radius_mm * f / last_depth_mm_);
    FcPose p;
    if (world_mm_.empty() || fc == nullptr || !fc->pose_at(t_ns, p)) return std::nullopt;
    const CamPose cam = camera_pose(p, body_camera_);
    if (!cam.has_position) return std::nullopt;
    std::vector<double> c[3];
    for (const auto& w : world_mm_)
        for (int k = 0; k < 3; ++k) c[k].push_back(w[k]);
    const cv::Point3d m_W(median_inplace(std::span<double>(c[0])), median_inplace(std::span<double>(c[1])),
                          median_inplace(std::span<double>(c[2])));
    const cv::Point3d m_C = to_camera(cam, m_W);
    const std::optional<cv::Point2d> uv = project_point_px(m_C, camera_);
    if (!uv || m_C.z <= 0.0) return std::nullopt;
    return std::make_pair(*uv, cfg_.exclude_radius_mm * f / m_C.z);
}

LaunchMarkerConfig object_marker_config(const LaunchMarkerConfig& launch, const ArucoConfig& aruco) {
    LaunchMarkerConfig c = launch;
    c.enabled = launch.track_object;
    c.dictionary = aruco.dictionary;
    c.marker_id = aruco.marker_id;
    c.side_mm = aruco.side_mm;
    return c;
}

}  // namespace uavp
