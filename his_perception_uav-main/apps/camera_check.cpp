// uavp_camera_check: camera + LiDAR LD06 + Pixhawk, KHONG model. Kiem tra he toa do LiDAR - camera - than UAV - the gioi W.
//   ./build/uavp_camera_check --cfg configs/run/real_time.yaml --set sources.mavlink.use_rangefinder=false aruco.side_mm=36.5
//       [--seconds 0] [--origin-s 2] [--no-lidar] [--fc-height-m 0.711] [--fc-above-rangefinder-m 0.09]
// Do cao Pixhawk tren san khi dat goc, theo thu tu uu tien: LiDAR (diem san, T_CL) > rangefinder > --fc-height-m.
// LiDAR (T_CL): diem san (xa nhat, |x_C| < 300 mm) va diem mat vat (|x_C - x_ArUco| < 15 mm, z gan ArUco). Dan ArUco
//   CANH duong quet LiDAR (o den / trang lam LD06 do lech) -> mat vat do LiDAR so voi do sau ArUco (kiem T_CL).
//   May tinh: python tools/remote_viewer.py --host <ip-pi>   (SPACE = luu 1 mau, q = thoat)
// Goc W (UAV_FRAMES_PLAN.md muc 2b): mat san ngay duoi Pixhawk luc BAT DAU chay (giu UAV yen --origin-s giay dau):
//   o = vi tri EKF cua Pixhawk + (0, 0, range * cos(roll) cos(pitch) + fc_above_rangefinder_m). Truc W = NED cua EKF.
//   Them he "theo mui luc dat goc" (truoc / phai / xuong, xoay -yaw0) de do doi chieu bang thuoc.
// Moi khung:
//   - 2 mui ten theo T_BC: "MUI +x_B", "PHAI +y_B" -> doi chieu voi thuc te (dua tay vao phia mui: tay theo mui ten MUI);
//   - tu the Pixhawk luc chup (noi suy theo SensorTimestamp), vi tri UAV trong W;
//   - thay ArUco: tam marker p_C (PnP, khong LiDAR) -> p_B = R_BC p_C + t_BC -> p_W = T_WB(t) p_B - o.
//     Vat dung yen: p_W phai dung yen khi nghieng / xoay / doi UAV.
// Ket qua runs/<ngay-gio>_camera_check/:
//   aruco_track.csv  moi khung thay marker: tu the FC, R_CM, t_CM (mm), p_B, p_W (EKF) va p_W (goc W)
//   snapshots.csv    cac mau luu bang SPACE (giu yen UAV truoc khi bam);  snap_NNN.jpg: anh tung mau
//   summary.json     goc W, trung binh / do lech chuan p_W
// ArUco DIEM CAT CANH (planA, id 1, --launch-id / --launch-side-mm; -1 = tat): do moi --launch-every khung.
// KHONG lam goc W (goc W van tu EKF nhu tren). Marker chi dung de:
//   1. Dat UAV: hien camera lech tam marker + mui UAV lech mui ten.
//   2. Ground truth: thuoc do tam ArUco vat (id 0) tu tam marker (--gt X Y Z m: x mui ten, y phai, z = -cao).
//      Doi sang he "theo mui luc dat goc" bang VI TRI DAT UAV (camera so voi marker do bang anh luc dat goc + t_BC),
//      KHONG qua EKF -> so voi VAT theo mui (camera + T_BC + EKF) = sai so toan chuoi.
//   3. Sai so EKF: marker nam yen; vi tri marker trong W (camera + EKF) luc dat goc = moc. Moi lan thay lai:
//      lech so voi moc = EKF troi (vd mang UAV ve lai cho cat canh).
//   launch_track.csv: moi lan thay marker cat canh (camera lech, marker theo mui, EKF troi).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/json_writer.hpp"
#include "uavp/common/log.hpp"
#include "uavp/common/run_files.hpp"
#include "uavp/sources/ld06.hpp"
#include "uavp/sources/mavlink.hpp"
#include "uavp/sources/pi_camera.hpp"
#include "uavp/stages/s4_fusion/lidar.hpp"
#include "uavp/support/aruco_check.hpp"
#include "uavp/support/cpu_affinity.hpp"
#include "uavp/support/remote_view.hpp"
#include "uavp/tracking/world_pose.hpp"

using namespace uavp;

namespace {

constexpr const char* kModule = "uavp.camera_check";
constexpr double kRadToDeg = 180.0 / CV_PI;
constexpr double kArrowDepthMm = 1000.0;  // mui ten ve tai do sau 1 m duoi camera
constexpr double kArrowLenMm = 200.0;
constexpr int kOriginMinSamples = 10;
const cv::Scalar kYellow(255, 255, 0), kCyan(0, 255, 255), kGreen(0, 255, 0), kRed(255, 60, 60);  // anh RGB
const cv::Scalar kOrange(255, 120, 0);
constexpr float kFloorHalfMm = 300.0F;  // diem san: |x_C| < 300 mm (giua anh, theo duong quet)
constexpr float kFloorBandMm = 30.0F;   // diem san: z trong 30 mm quanh phan vi 90 (xa nhat = san)
constexpr float kTopHalfMm = 15.0F;     // mat vat: |x_C - x_ArUco| < 15 mm
constexpr float kTopBandMm = 60.0F;     // mat vat: |z_C - z_ArUco| < 60 mm (bo san, bo khung)
constexpr int kMinPoints = 3;

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

struct Args {
    std::string cfg;
    std::vector<std::string> overrides;
    double seconds{0.0};               // 0 = toi khi q / Ctrl+C
    double origin_s{2.0};              // 0 = khong dat goc (p_W theo goc EKF)
    double fc_above_rangefinder_m{0.09};  // Pixhawk cao hon mat MTF-01 (EKF2_RNG_POS_Z)
    double fc_height_m{0.0};  // >0: Pixhawk cao hon san (thuoc) khi dat goc, dung khi khong co LiDAR / rangefinder
    bool lidar{true};
    int launch_id{1};               // ArUco diem cat canh (cung tu dien voi muc aruco); -1 = tat
    double launch_side_mm{100.0};   // canh VIEN DEN do bang thuoc
    int launch_every{3};            // do marker cat canh moi N khung (ArUco 1920x1080 ton CPU)
    std::optional<cv::Vec3d> gt_m;  // tam ArUco vat (id 0) do thuoc tu tam marker cat canh (m): x mui ten, y phai, z = -cao
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) a.cfg = argv[++i];
        else if (std::strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) a.seconds = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--origin-s") == 0 && i + 1 < argc) a.origin_s = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--fc-above-rangefinder-m") == 0 && i + 1 < argc)
            a.fc_above_rangefinder_m = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--fc-height-m") == 0 && i + 1 < argc) a.fc_height_m = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--no-lidar") == 0) a.lidar = false;
        else if (std::strcmp(argv[i], "--launch-id") == 0 && i + 1 < argc) a.launch_id = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--launch-side-mm") == 0 && i + 1 < argc) a.launch_side_mm = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--launch-every") == 0 && i + 1 < argc)
            a.launch_every = std::max(1, std::stoi(argv[++i]));
        else if (std::strcmp(argv[i], "--gt") == 0 && i + 3 < argc) {
            const double x = std::stod(argv[++i]), y = std::stod(argv[++i]), z = std::stod(argv[++i]);
            a.gt_m = cv::Vec3d(x, y, z);
        }
        else if (std::strcmp(argv[i], "--set") == 0)
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.overrides.emplace_back(argv[++i]);
    }
    if (a.cfg.empty())
        throw std::invalid_argument("Dung: uavp_camera_check --cfg configs/run/real_time.yaml [--seconds 0] [--origin-s 2] [--fc-height-m 0.74] "
                                    "[--launch-id 1] [--launch-side-mm 100] [--gt X Y Z] [--set ...]");
    if (a.launch_side_mm <= 0) throw std::invalid_argument("--launch-side-mm phai > 0");
    return a;
}

// Goc W: trung binh vai giay dau (UAV dung yen).
struct WorldOrigin {
    bool set{false};
    int n{0};
    cv::Vec3d sum_pos_m{0, 0, 0};
    double sum_ground_m{0.0}, sum_sin{0.0}, sum_cos{0.0};
    cv::Vec3d o_ned_m{0, 0, 0};  // goc W trong he EKF
    double yaw0_rad{0.0};
    double ground_m{0.0};        // Pixhawk cao hon san luc dat goc

    const char* ground_src{"?"};  // nguon do cao: lidar / rangefinder / thuoc
    // h_m: Pixhawk cao hon san (m) tai mau nay.
    void add(const FcPose& p, double h_m, const char* src) {
        sum_pos_m += p.pos_ned_m;
        sum_ground_m += h_m;
        ground_src = src;
        sum_sin += std::sin(p.yaw_rad);
        sum_cos += std::cos(p.yaw_rad);
        ++n;
    }
    void finish() {
        ground_m = sum_ground_m / n;
        o_ned_m = sum_pos_m / n + cv::Vec3d(0.0, 0.0, ground_m);
        yaw0_rad = std::atan2(sum_sin, sum_cos);
        set = true;
    }
    // EKF NED (mm) -> W (mm, NED, goc tai san duoi Pixhawk luc dat goc).
    [[nodiscard]] cv::Vec3d to_w_mm(const cv::Vec3d& ekf_mm) const { return ekf_mm - 1000.0 * o_ned_m; }
    // W -> theo mui luc dat goc: (truoc, phai, xuong).
    [[nodiscard]] cv::Vec3d to_nose(const cv::Vec3d& w) const {
        const double c = std::cos(yaw0_rad), s = std::sin(yaw0_rad);
        return {c * w[0] + s * w[1], -s * w[0] + c * w[1], w[2]};
    }
};

// Diem LiDAR cua vong quet moi nhat (he C, mm) + san + mat vat.
struct LidarView {
    std::size_t scan_count{0};
    std::vector<cv::Point3f> pts;
    std::vector<int64_t> t_ns;
    int floor_n{0};
    cv::Point3d floor_C{kNaN, kNaN, kNaN};
    int top_n{0};
    cv::Point3d top_C{kNaN, kNaN, kNaN};
    cv::Vec3d top_w_mm{kNaN, kNaN, kNaN};   // mat vat trong W (neu da dat goc)
    double floor_ekf_d_mm{kNaN};            // san (z NED EKF) tai diem LiDAR
};

cv::Point3d median_point(std::vector<cv::Point3f> v) {
    if (v.empty()) return {kNaN, kNaN, kNaN};
    const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    cv::Point3d m;
    std::nth_element(v.begin(), mid, v.end(), [](const auto& a, const auto& b) { return a.x < b.x; });
    m.x = mid->x;
    std::nth_element(v.begin(), mid, v.end(), [](const auto& a, const auto& b) { return a.y < b.y; });
    m.y = mid->y;
    std::nth_element(v.begin(), mid, v.end(), [](const auto& a, const auto& b) { return a.z < b.z; });
    m.z = mid->z;
    return m;
}

void find_floor(LidarView& lv) {
    std::vector<cv::Point3f> c;
    for (const auto& p : lv.pts)
        if (std::abs(p.x) < kFloorHalfMm) c.push_back(p);
    lv.floor_n = 0;
    lv.floor_C = {kNaN, kNaN, kNaN};
    if (static_cast<int>(c.size()) < kMinPoints) return;
    std::vector<float> z;
    for (const auto& p : c) z.push_back(p.z);
    const auto k = z.begin() + static_cast<std::ptrdiff_t>(z.size() * 9 / 10);
    std::nth_element(z.begin(), k, z.end());
    const float z90 = *k;
    std::vector<cv::Point3f> sel;
    for (const auto& p : c)
        if (std::abs(p.z - z90) < kFloorBandMm) sel.push_back(p);
    if (static_cast<int>(sel.size()) < kMinPoints) return;
    lv.floor_n = static_cast<int>(sel.size());
    lv.floor_C = median_point(sel);
}

void find_top(LidarView& lv, const cv::Vec3d& aruco_C) {
    std::vector<cv::Point3f> sel;
    for (const auto& p : lv.pts)
        if (std::abs(p.x - aruco_C[0]) < kTopHalfMm && std::abs(p.z - aruco_C[2]) < kTopBandMm) sel.push_back(p);
    lv.top_n = static_cast<int>(sel.size());
    lv.top_C = lv.top_n >= kMinPoints ? median_point(sel) : cv::Point3d(kNaN, kNaN, kNaN);
}

struct Marker {
    bool found{false};
    ArucoPose pose;
    cv::Point3d p_B_mm{kNaN, kNaN, kNaN};
    cv::Vec3d p_ekf_mm{kNaN, kNaN, kNaN};  // T_WB(t) p_B, goc EKF
    cv::Vec3d p_w_mm{kNaN, kNaN, kNaN};    // goc W (neu da dat)
    cv::Vec3d p_nose_mm{kNaN, kNaN, kNaN};
    cv::Vec3d err_mm{kNaN, kNaN, kNaN};    // VAT theo mui - ground truth theo mui (neu co --gt + marker luc dat goc)
};

// He S cua marker cat canh (mui ten x, phai y, xuong z) tu he M cua cv::aruco (x phai, y canh tren, z len).
const cv::Matx33d kR_SM(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0);

double wrap_rad(double a) { return std::atan2(std::sin(a), std::cos(a)); }

// Mot lan thay ArUco diem cat canh.
struct LaunchObs {
    bool found{false};
    ArucoPose pose;
    cv::Vec3d cam_S_mm{kNaN, kNaN, kNaN};  // tam quang camera trong he marker (z am: camera tren san)
    double yaw_rel_deg{kNaN};              // mui UAV - mui ten marker
    bool world{false};
    cv::Vec3d M_ekf_mm{kNaN, kNaN, kNaN};  // tam marker trong EKF
    double yaw_M_rad{kNaN};                // huong mui ten trong EKF (so voi Bac)
    cv::Vec3d M_nose_mm{kNaN, kNaN, kNaN}; // tam marker theo mui (camera + EKF), sau khi dat goc
    cv::Vec3d drift_mm{kNaN, kNaN, kNaN};  // M_nose - moc luc dat goc = EKF troi
};

// Marker cat canh luc dat goc (UAV nam yen tren marker). KHONG phai goc W.
struct LaunchRef {
    bool set{false};
    int n{0}, n_yaw{0}, n_world{0};
    cv::Vec3d sum_cam_S{0, 0, 0}, sum_M_ekf{0, 0, 0};
    double sum_sin{0.0}, sum_cos{0.0};
    cv::Vec3d cam_S_mm{kNaN, kNaN, kNaN};        // camera so voi tam marker (he marker), trung binh
    double yaw_rel_rad{0.0};                     // mui UAV - mui ten (0 neu khong co FC)
    cv::Vec3d M_place_mm{kNaN, kNaN, kNaN};      // tam marker theo mui, tu VI TRI DAT UAV (anh + t_BC), khong EKF
    cv::Vec3d M_nose0_mm{kNaN, kNaN, kNaN};      // tam marker theo mui qua camera + EKF luc dat goc = moc EKF troi
    void add(const LaunchObs& o) {
        sum_cam_S += o.cam_S_mm;
        ++n;
        if (std::isfinite(o.yaw_rel_deg)) {
            sum_sin += std::sin(o.yaw_rel_deg / kRadToDeg);
            sum_cos += std::cos(o.yaw_rel_deg / kRadToDeg);
            ++n_yaw;
        }
        if (o.world) {
            sum_M_ekf += o.M_ekf_mm;
            ++n_world;
        }
    }
    // Vec (x mui ten, y phai) cua he marker -> he theo mui: mui ten lech mui UAV -yaw_rel.
    [[nodiscard]] cv::Vec3d marker_to_nose(const cv::Vec3d& v) const {
        const double c = std::cos(-yaw_rel_rad), s = std::sin(-yaw_rel_rad);
        return {c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]};
    }
    void finish(const WorldOrigin& w, const BodyCamera& bc) {
        cam_S_mm = sum_cam_S / n;
        yaw_rel_rad = n_yaw > 0 ? std::atan2(sum_sin, sum_cos) : 0.0;
        // Camera trong he theo mui = t_BC (UAV nam ngang); marker = camera - (camera so voi marker, xoay sang he mui).
        const cv::Vec3d cam_in_marker = marker_to_nose({cam_S_mm[0], cam_S_mm[1], 0.0});
        M_place_mm = {bc.t_BC_mm[0] - cam_in_marker[0], bc.t_BC_mm[1] - cam_in_marker[1], 0.0};  // marker tren san
        if (n_world > 0 && w.set) M_nose0_mm = w.to_nose(w.to_w_mm(sum_M_ekf / n_world));
        set = true;
    }
    // Thuoc (m, he marker) -> theo mui (mm).
    [[nodiscard]] cv::Vec3d gt_nose_mm(const cv::Vec3d& gt_m) const {
        const cv::Vec3d g = marker_to_nose(1000.0 * gt_m);
        return {M_place_mm[0] + g[0], M_place_mm[1] + g[1], g[2]};
    }
};

LaunchObs find_launch(const cv::Mat& image, const ArucoConfig& cfg, const CameraModel& camera, const BodyCamera& bc,
                      const std::optional<FcPose>& fc) {
    LaunchObs o;
    std::array<cv::Point2f, 4> corners{};
    if (!find_marker_px(image, cfg, corners)) return o;
    o.found = true;
    o.pose = marker_pose(corners, cfg.marker_id, cfg.side_mm, camera);
    o.cam_S_mm = kR_SM * (-(o.pose.R_CM.t() * o.pose.t_CM_mm));
    if (!fc) return o;
    const CamPose cp = camera_pose(*fc, bc);
    const cv::Matx33d R_WM = cp.R_WC * o.pose.R_CM;
    o.yaw_M_rad = std::atan2(R_WM(1, 1), R_WM(0, 1));  // truc x_S = y_M (cot 1)
    o.yaw_rel_deg = wrap_rad(fc->yaw_rad - o.yaw_M_rad) * kRadToDeg;
    if (!fc->has_position) return o;
    const cv::Vec3d t = o.pose.t_CM_mm;
    const cv::Point3d m = to_world(cp, {t[0], t[1], t[2]});
    o.world = true;
    o.M_ekf_mm = {m.x, m.y, m.z};
    return o;
}

// Thong ke (Welford) cho summary.
struct RunningStats {
    int n{0};
    cv::Vec3d mean{0, 0, 0}, m2{0, 0, 0};
    void add(const cv::Vec3d& x) {
        if (!std::isfinite(x[0])) return;
        ++n;
        const cv::Vec3d d = x - mean;
        mean += d / n;
        const cv::Vec3d d2 = x - mean;
        for (int k = 0; k < 3; ++k) m2[k] += d[k] * d2[k];
    }
    [[nodiscard]] cv::Vec3d stddev() const {
        if (n < 2) return {kNaN, kNaN, kNaN};
        return {std::sqrt(m2[0] / (n - 1)), std::sqrt(m2[1] / (n - 1)), std::sqrt(m2[2] / (n - 1))};
    }
};

std::string pose_line(const std::optional<FcPose>& p, const WorldOrigin& o) {
    if (!p) return "FC: khong co tu the (bridge / router?)";
    char b[256];
    std::snprintf(b, sizeof(b), "roll %.1f  pitch %.1f  yaw %.1f deg", p->roll_rad * kRadToDeg, p->pitch_rad * kRadToDeg,
                  p->yaw_rad * kRadToDeg);
    std::string s = b;
    if (p->has_position && o.set) {
        const cv::Vec3d u = o.to_w_mm(1000.0 * p->pos_ned_m);
        std::snprintf(b, sizeof(b), " | UAV trong W (%.0f, %.0f, %.0f) mm", u[0], u[1], u[2]);
        s += b;
    } else if (p->has_position) {
        std::snprintf(b, sizeof(b), " | NED EKF (%.3f, %.3f, %.3f) m", p->pos_ned_m[0], p->pos_ned_m[1], p->pos_ned_m[2]);
        s += b;
    }
    if (p->has_range) {
        std::snprintf(b, sizeof(b), " | range %.2f m", p->range_m);
        s += b;
    }
    return s;
}

void put(cv::Mat& img, const std::string& text, int row, const cv::Scalar& color) {
    const cv::Point org(16, img.rows - 24 - 52 * (row + 1));  // tu duoi len; dong duoi cung danh cho chu cua viewer
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 0), 6, cv::LINE_AA);
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv::LINE_AA);
}

// Mui ten tu diem duoi tam anh theo huong mot truc cua than (trong he camera: R_BC^T * e).
void body_arrow(cv::Mat& img, const BodyCamera& bc, const CameraModel& camera, const cv::Vec3d& axis_B,
                const std::string& label, const cv::Scalar& color) {
    const cv::Vec3d d_C = bc.R_BC.t() * axis_B;
    const cv::Point3d base(0.0, 0.0, kArrowDepthMm);
    const cv::Point3d tip(kArrowLenMm * d_C[0], kArrowLenMm * d_C[1], kArrowDepthMm + kArrowLenMm * d_C[2]);
    const auto a = project_point_px(base, camera);
    const auto b = project_point_px(tip, camera);
    if (!a || !b) return;
    cv::arrowedLine(img, *a, *b, color, 6, cv::LINE_AA, 0, 0.2);
    cv::putText(img, label, *b + cv::Point2d(12, 12), cv::FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv::LINE_AA);
}

Marker find_marker(const cv::Mat& image, const ArucoConfig& aruco, const CameraModel& camera, const BodyCamera& bc,
                   const std::optional<FcPose>& fc, const WorldOrigin& origin) {
    Marker m;
    std::array<cv::Point2f, 4> corners{};
    if (!find_marker_px(image, aruco, corners)) return m;
    m.found = true;
    m.pose = marker_pose(corners, aruco.marker_id, aruco.side_mm, camera);
    const cv::Vec3d p_C = m.pose.t_CM_mm;
    const cv::Vec3d p_B = bc.R_BC * p_C + bc.t_BC_mm;
    m.p_B_mm = {p_B[0], p_B[1], p_B[2]};
    if (fc && fc->has_position) {
        const cv::Point3d p = to_world(camera_pose(*fc, bc), {p_C[0], p_C[1], p_C[2]});
        m.p_ekf_mm = {p.x, p.y, p.z};
        if (origin.set) {
            m.p_w_mm = origin.to_w_mm(m.p_ekf_mm);
            m.p_nose_mm = origin.to_nose(m.p_w_mm);
        }
    }
    return m;
}

std::string csv_header() {
    return "t_ns,roll_deg,pitch_deg,yaw_deg,has_pos,n_m,e_m,d_m,has_range,range_m,"
           "R_CM00,R_CM01,R_CM02,R_CM10,R_CM11,R_CM12,R_CM20,R_CM21,R_CM22,tC_x_mm,tC_y_mm,tC_z_mm,reproj_px,"
           "pB_x_mm,pB_y_mm,pB_z_mm,pEKF_n_mm,pEKF_e_mm,pEKF_d_mm,pW_n_mm,pW_e_mm,pW_d_mm,"
           "nose_fwd_mm,nose_right_mm,nose_down_mm,err_fwd_mm,err_right_mm,err_down_mm\n";
}

std::string csv_row(int64_t t_ns, const std::optional<FcPose>& fc, const Marker& m) {
    char b[1280];
    const FcPose p = fc.value_or(FcPose{});
    const double nan = kNaN;
    const auto& R = m.pose.R_CM;
    const auto& t = m.pose.t_CM_mm;
    std::snprintf(b, sizeof(b),
                  "%lld,%.4f,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%d,%.3f,"
                  "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f,%.3f,"
                  "%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
                  static_cast<long long>(t_ns), fc ? p.roll_rad * kRadToDeg : nan, fc ? p.pitch_rad * kRadToDeg : nan,
                  fc ? p.yaw_rad * kRadToDeg : nan, p.has_position ? 1 : 0, p.pos_ned_m[0], p.pos_ned_m[1],
                  p.pos_ned_m[2], p.has_range ? 1 : 0, p.range_m, R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2),
                  R(2, 0), R(2, 1), R(2, 2), t[0], t[1], t[2], m.pose.reproj_px, m.p_B_mm.x, m.p_B_mm.y, m.p_B_mm.z,
                  m.p_ekf_mm[0], m.p_ekf_mm[1], m.p_ekf_mm[2], m.p_w_mm[0], m.p_w_mm[1], m.p_w_mm[2], m.p_nose_mm[0],
                  m.p_nose_mm[1], m.p_nose_mm[2], m.err_mm[0], m.err_mm[1], m.err_mm[2]);
    return b;
}

void write_summary(const std::filesystem::path& dir, int frames, int marker_frames, int snaps, const RunningStats& w,
                   const BodyCamera& bc, const WorldOrigin& o, const RunningStats& lidar, const LaunchRef& lr,
                   const RunningStats& nose_stats, const RunningStats& drift_stats, const std::optional<cv::Vec3d>& gt_m) {
    const cv::Vec3d sd = w.stddev();
    JsonWriter j;
    j.begin_object().key("frames").value(frames).key("marker_frames").value(marker_frames).key("snapshots").value(snaps);
    j.key("t_BC_mm").begin_array().num(bc.t_BC_mm[0]).num(bc.t_BC_mm[1]).num(bc.t_BC_mm[2]).end_array();
    j.key("origin_set").value(o.set).key("origin_ekf_ned_m");
    j.begin_array().num(o.o_ned_m[0]).num(o.o_ned_m[1]).num(o.o_ned_m[2]).end_array();
    j.key("origin_yaw_deg").num(o.yaw0_rad * kRadToDeg).key("fc_above_ground_m").num(o.ground_m);
    j.key("fc_above_ground_source").value(o.ground_src);
    const cv::Vec3d ls = lidar.stddev();
    j.key("lidar_scans_with_top").value(lidar.n);
    j.key("lidar_top_minus_aruco_mm").begin_array().num(lidar.mean[0]).num(ls[0]).end_array();
    j.key("lidar_floor_zC_mm").num(lidar.mean[1]);
    j.key("lidar_top_height_W_mm").begin_array().num(lidar.mean[2]).num(ls[2]).end_array();
    j.key("p_frame").value(o.set ? "W (goc dat luc bat dau)" : "EKF").key("p_samples").value(w.n);
    j.key("p_mean_mm").begin_array().num(w.mean[0]).num(w.mean[1]).num(w.mean[2]).end_array();
    j.key("p_std_mm").begin_array().num(sd[0]).num(sd[1]).num(sd[2]).end_array();
    // Marker cat canh (KHONG phai goc W): dat UAV, ground truth theo mui, EKF troi
    j.key("launch_set").value(lr.set).key("launch_samples").value(lr.n);
    j.key("launch_cam_S_mm").begin_array().num(lr.cam_S_mm[0]).num(lr.cam_S_mm[1]).num(lr.cam_S_mm[2]).end_array();
    j.key("launch_yaw_rel_deg").num(lr.yaw_rel_rad * kRadToDeg);
    j.key("launch_M_place_nose_mm").begin_array().num(lr.M_place_mm[0]).num(lr.M_place_mm[1]).num(lr.M_place_mm[2]).end_array();
    j.key("launch_M_ekf_nose0_mm").begin_array().num(lr.M_nose0_mm[0]).num(lr.M_nose0_mm[1]).num(lr.M_nose0_mm[2]).end_array();
    const cv::Vec3d dsd = drift_stats.stddev();
    j.key("ekf_drift_samples").value(drift_stats.n);
    j.key("ekf_drift_mean_mm").begin_array().num(drift_stats.mean[0]).num(drift_stats.mean[1]).num(drift_stats.mean[2]).end_array();
    j.key("ekf_drift_std_mm").begin_array().num(dsd[0]).num(dsd[1]).num(dsd[2]).end_array();
    const cv::Vec3d nsd = nose_stats.stddev();
    j.key("object_nose_samples").value(nose_stats.n);
    j.key("object_nose_mean_mm").begin_array().num(nose_stats.mean[0]).num(nose_stats.mean[1]).num(nose_stats.mean[2]).end_array();
    j.key("object_nose_std_mm").begin_array().num(nsd[0]).num(nsd[1]).num(nsd[2]).end_array();
    j.key("gt_nose_mm");
    if (gt_m && lr.set && nose_stats.n > 0) {
        const cv::Vec3d g = lr.gt_nose_mm(*gt_m), e = nose_stats.mean - g;
        j.begin_array().num(g[0]).num(g[1]).num(g[2]).end_array();
        j.key("error_nose_mm").begin_array().num(e[0]).num(e[1]).num(e[2]).end_array();
        j.key("error_2d_mm").num(std::hypot(e[0], e[1])).key("error_3d_mm").num(std::sqrt(e.dot(e)));
    } else {
        j.null();
    }
    j.end_object();
    write_text(dir / "summary.json", j.str());
}

int run_main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const YAML::Node raw = load_raw(args.cfg, args.overrides);
    const Config cfg = config_from_raw(raw);
    const std::string cores = pin_process_to_cores(cfg.cpu.cores);
    const SourcesConfig src = sources_config(raw["sources"]);
    const MavlinkConfig mav = mavlink_config(raw["sources"]["mavlink"]);
    const BodyCamera bc = load_body_camera(lock_config(raw["lock_track"]).calib_T_BC);
    const CaptureConfig cap = load_capture(raw["calib_K"].as<std::string>(), src);
    const ArucoConfig aruco = aruco_config(raw["aruco"]);
    const ArucoConfig launch_aruco{aruco.dictionary, args.launch_id, args.launch_side_mm, true};
    if (args.launch_id >= 0 && args.launch_id == aruco.marker_id)
        throw std::invalid_argument("--launch-id trung aruco.marker_id (marker vat): 2 marker phai khac id");
    const RemoteConfig rcfg{raw["remote"]["port"].as<int>(), raw["remote"]["preview_width_px"].as<int>(),
                            raw["remote"]["jpeg_quality"].as<int>(), raw["remote"]["max_fps"].as<double>(),
                            raw["remote"]["wait_fps"].as<double>(), raw["remote"]["enabled"].as<bool>()};
    auto logger = std::make_unique<Logger>(Level::kInfo);
    set_logger(logger.get());
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    const std::filesystem::path dir = make_run_dir(cfg.runs_root, "camera_check");
    logger->open_file(dir / "run.log");
    UAVP_LOG_INFO(kModule, "camera_check: CPU %s | T_BC t = (%.0f, %.0f, %.0f) mm | ArUco %s id %d canh %.1f mm | "
                  "use_rangefinder %s", cores.c_str(), bc.t_BC_mm[0], bc.t_BC_mm[1], bc.t_BC_mm[2],
                  aruco.dictionary.c_str(), aruco.marker_id, aruco.side_mm, mav.use_rangefinder ? "true" : "false");
    if (args.launch_id >= 0)
        UAVP_LOG_INFO(kModule, "ArUco diem cat canh: id %d canh %.1f mm, moi %d khung | thuoc tu tam marker: %s", args.launch_id,
                      args.launch_side_mm, args.launch_every,
                      args.gt_m ? cv::format("x %.3f y %.3f z %.3f m", (*args.gt_m)[0], (*args.gt_m)[1],
                                             (*args.gt_m)[2]).c_str()
                                : "chua nhap (--gt X Y Z)");
    if (mav.use_rangefinder)
        UAVP_LOG_WARN(kModule, "use_rangefinder = true: z cua p_W lay tu rangefinder (bo qua offset MTF-01). "
                      "Nen chay voi --set sources.mavlink.use_rangefinder=false");
    if (!args.lidar && !mav.use_rangefinder && args.fc_height_m <= 0)
        UAVP_LOG_WARN(kModule, "Khong LiDAR, khong rangefinder, khong --fc-height-m: KHONG dat duoc goc W");

    std::unique_ptr<IFcSource> fc;
    if (mav.enabled) {
        try {
            fc = open_fc_source(mav);
        } catch (const std::exception& e) {
            UAVP_LOG_WARN(kModule, "Khong mo duoc nguon MAVLink: %s -> chay khong co Pixhawk", e.what());
        }
    }
    std::unique_ptr<Ld06Reader> lidar;
    if (args.lidar) {
        try {
            lidar = std::make_unique<Ld06Reader>(src.lidar_port, src.lidar_baudrate,
                                                 static_cast<std::size_t>(src.lidar_buffer_scans),
                                                 src.lidar_extra_latency_ms);
            UAVP_LOG_INFO(kModule, "LiDAR: %s (T_CL tu config)", src.lidar_port.c_str());
        } catch (const std::exception& e) {
            UAVP_LOG_WARN(kModule, "Khong mo duoc LiDAR: %s -> chay khong LiDAR", e.what());
        }
    }
    std::unique_ptr<RemoteView> remote;
    if (rcfg.enabled) {
        remote = std::make_unique<RemoteView>(rcfg);
        remote->set_fps(rcfg.wait_fps);
        UAVP_LOG_INFO(kModule, "Cho may tinh ket noi: python tools/remote_viewer.py --host <ip-pi> --port %d "
                      "(SPACE = luu mau, q = thoat)", rcfg.port);
    }
    PiCamera camera(cap, src.camera_buffer_count, src.sensor_bit_depth);
    cv::Mat image(cap.height_px, cap.width_px, CV_8UC3);
    if (args.origin_s > 0)
        UAVP_LOG_INFO(kModule, "Dat goc W: GIU UAV YEN %.1f s dau (goc = san ngay duoi Pixhawk)", args.origin_s);

    std::ofstream track(dir / "aruco_track.csv");
    std::ofstream snaps_csv(dir / "snapshots.csv");
    std::ofstream lidar_csv(dir / "lidar_track.csv");
    std::ofstream launch_csv(dir / "launch_track.csv");
    launch_csv << "t_ns,cam_x_S_mm,cam_y_S_mm,cam_z_S_mm,yaw_rel_deg,reproj_px,world,M_n_ekf_mm,M_e_ekf_mm,M_d_ekf_mm,"
                  "yaw_M_deg,M_fwd_mm,M_right_mm,M_down_mm,drift_fwd_mm,drift_right_mm,drift_down_mm\n";
    track << csv_header();
    snaps_csv << "snap," << csv_header();
    lidar_csv << "t_ns,floor_n,floor_xC_mm,floor_yC_mm,floor_zC_mm,floor_ekf_d_mm,top_n,top_xC_mm,top_yC_mm,top_zC_mm,"
                 "aruco_zC_mm,top_minus_aruco_mm,top_W_n_mm,top_W_e_mm,top_W_d_mm\n";
    RunningStats p_stats;
    RunningStats lidar_stats;  // (mat vat LiDAR - ArUco, z san C, do cao mat vat tren san W)
    LidarView lv;
    WorldOrigin origin;
    LaunchRef launch_ref;      // marker cat canh luc dat goc (KHONG phai goc W)
    LaunchObs launch;          // lan thay gan nhat (giu de hien thi)
    int64_t launch_seen_ns = 0;
    RunningStats nose_stats;   // VAT theo mui (sau khi dat goc)
    RunningStats drift_stats;  // EKF troi do bang marker cat canh
    int frames = 0, marker_frames = 0, snaps = 0;
    const int64_t t_start = mono_ns();
    int64_t t_log = t_start;
    const int64_t t_end = args.seconds > 0 ? t_start + static_cast<int64_t>(args.seconds * 1e9) : INT64_MAX;
    const int64_t origin_end = t_start + static_cast<int64_t>(args.origin_s * 1e9);

    while (!g_stop && mono_ns() < t_end) {
        FrameMeta meta;
        if (!camera.next_frame(image, meta, src.frame_timeout_s)) throw std::runtime_error("Camera: khong co khung moi");
        ++frames;
        std::optional<FcPose> pose;
        if (fc) {
            FcPose p;
            if (fc->poses().pose_at(meta.t_ns, p)) pose = p;
        }
        bool new_scan = false;
        if (lidar && lidar->buffer().count() != lv.scan_count) {
            Scan scan;
            if (lidar->buffer().latest_scan(scan)) {
                lv.scan_count = lidar->buffer().count();
                lv.pts.clear();
                lv.t_ns.clear();
                scan_to_camera_mm(scan, cfg.fusion.extrinsics, cfg.fusion, lv.pts, lv.t_ns);
                find_floor(lv);
                lv.floor_ekf_d_mm = kNaN;
                if (pose && pose->has_position && lv.floor_n > 0)
                    lv.floor_ekf_d_mm = to_world(camera_pose(*pose, bc), lv.floor_C).z;
                new_scan = true;
            }
        }
        std::optional<double> ground_m;
        const char* ground_src = "";
        if (pose && pose->has_position) {
            if (std::isfinite(lv.floor_ekf_d_mm)) {
                ground_m = lv.floor_ekf_d_mm / 1000.0 - pose->pos_ned_m[2];
                ground_src = "LiDAR";
            } else if (pose->has_range) {
                ground_m = pose->range_m * std::cos(pose->roll_rad) * std::cos(pose->pitch_rad) +
                           args.fc_above_rangefinder_m;
                ground_src = "rangefinder";
            } else if (args.fc_height_m > 0) {
                ground_m = args.fc_height_m;
                ground_src = "--fc-height-m";
            }
        }
        if (args.launch_id >= 0 && (frames % args.launch_every == 0 || (!origin.set && args.origin_s > 0))) {
            LaunchObs lo = find_launch(image, launch_aruco, cfg.camera, bc, pose);
            if (lo.found) {
                if (origin.set && lo.world) {
                    lo.M_nose_mm = origin.to_nose(origin.to_w_mm(lo.M_ekf_mm));
                    if (launch_ref.set && std::isfinite(launch_ref.M_nose0_mm[0])) {
                        lo.drift_mm = lo.M_nose_mm - launch_ref.M_nose0_mm;
                        drift_stats.add(lo.drift_mm);
                    }
                }
                launch = lo;
                launch_seen_ns = meta.t_ns;
                char b[384];
                std::snprintf(b, sizeof(b), "%lld,%.1f,%.1f,%.1f,%.2f,%.3f,%d,%.1f,%.1f,%.1f,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
                              static_cast<long long>(meta.t_ns), lo.cam_S_mm[0], lo.cam_S_mm[1], lo.cam_S_mm[2],
                              lo.yaw_rel_deg, lo.pose.reproj_px, lo.world ? 1 : 0, lo.M_ekf_mm[0], lo.M_ekf_mm[1],
                              lo.M_ekf_mm[2], lo.yaw_M_rad * kRadToDeg, lo.M_nose_mm[0], lo.M_nose_mm[1], lo.M_nose_mm[2],
                              lo.drift_mm[0], lo.drift_mm[1], lo.drift_mm[2]);
                launch_csv << b;
                if (args.origin_s > 0 && !origin.set) launch_ref.add(lo);
            }
        }
        if (args.origin_s > 0 && !origin.set && pose && pose->has_position && ground_m) {
            origin.add(*pose, *ground_m, ground_src);
            if (meta.t_ns >= origin_end && origin.n >= kOriginMinSamples) {
                origin.finish();
                p_stats = {};  // bo mau goc EKF truoc khi dat goc (khong tron 2 he)
                lidar_stats = {};
                UAVP_LOG_INFO(kModule, "GOC W: EKF (%.3f, %.3f, %.3f) m, Pixhawk cao %.3f m so voi san (%s), yaw0 %.1f deg "
                              "(%d mau)", origin.o_ned_m[0], origin.o_ned_m[1], origin.o_ned_m[2], origin.ground_m,
                              origin.ground_src, origin.yaw0_rad * kRadToDeg, origin.n);
                if (args.launch_id >= 0 && launch_ref.n >= 3) {
                    launch_ref.finish(origin, bc);
                    UAVP_LOG_INFO(kModule, "MARKER CAT CANH luc dat goc (%d mau; KHONG phai goc W): camera lech tam marker "
                                  "x %.0f y %.0f mm, mui UAV lech %+.1f deg | tam marker theo mui: dat UAV (%.0f, %.0f) mm, "
                                  "qua EKF (%.0f, %.0f, %.0f) mm = moc EKF troi", launch_ref.n, launch_ref.cam_S_mm[0],
                                  launch_ref.cam_S_mm[1], launch_ref.yaw_rel_rad * kRadToDeg, launch_ref.M_place_mm[0],
                                  launch_ref.M_place_mm[1], launch_ref.M_nose0_mm[0], launch_ref.M_nose0_mm[1],
                                  launch_ref.M_nose0_mm[2]);
                    if (args.gt_m) {
                        const cv::Vec3d g = launch_ref.gt_nose_mm(*args.gt_m);
                        UAVP_LOG_INFO(kModule, "Ground truth vat (thuoc) theo mui: truoc %.0f, phai %.0f mm, cao %.0f mm",
                                      g[0], g[1], -g[2]);
                    }
                } else if (args.launch_id >= 0) {
                    UAVP_LOG_WARN(kModule, "Khong thay marker cat canh id %d luc dat goc (%d mau): khong co ground truth "
                                  "theo mui / moc EKF troi", args.launch_id, launch_ref.n);
                }
            }
        }
        Marker m = find_marker(image, aruco, cfg.camera, bc, pose, origin);
        if (m.found && launch_ref.set && args.gt_m && std::isfinite(m.p_nose_mm[0]))
            m.err_mm = m.p_nose_mm - launch_ref.gt_nose_mm(*args.gt_m);
        if (m.found) {
            ++marker_frames;
            p_stats.add(origin.set ? m.p_w_mm : m.p_ekf_mm);
            if (origin.set) nose_stats.add(m.p_nose_mm);
            track << csv_row(meta.t_ns, pose, m);
        }
        lv.top_n = 0;
        lv.top_C = {kNaN, kNaN, kNaN};
        lv.top_w_mm = {kNaN, kNaN, kNaN};
        if (m.found && !lv.pts.empty()) {
            find_top(lv, m.pose.t_CM_mm);
            if (lv.top_n > 0 && pose && pose->has_position && origin.set)
                lv.top_w_mm = origin.to_w_mm(cv::Vec3d(to_world(camera_pose(*pose, bc), lv.top_C)));
        }
        const double top_minus_aruco = lv.top_n > 0 ? lv.top_C.z - m.pose.t_CM_mm[2] : kNaN;
        if (new_scan && lidar) {
            char b[512];
            std::snprintf(b, sizeof(b), "%lld,%d,%.1f,%.1f,%.1f,%.1f,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
                          static_cast<long long>(meta.t_ns), lv.floor_n, lv.floor_C.x, lv.floor_C.y, lv.floor_C.z,
                          lv.floor_ekf_d_mm, lv.top_n, lv.top_C.x, lv.top_C.y, lv.top_C.z,
                          m.found ? m.pose.t_CM_mm[2] : kNaN, top_minus_aruco, lv.top_w_mm[0], lv.top_w_mm[1],
                          lv.top_w_mm[2]);
            lidar_csv << b;
            if (std::isfinite(top_minus_aruco))
                lidar_stats.add({top_minus_aruco, lv.floor_C.z, -lv.top_w_mm[2]});
        }

        const int key = remote ? remote->poll_key() : -1;
        if (key == 'q') break;
        const bool snap = key == 's';
        const bool show = remote && remote->connected() && (snap || remote->due());
        if (!show && !snap && mono_ns() - t_log < kNsPerS) continue;

        cv::Mat canvas = image.clone();
        if (!lv.pts.empty()) {
            std::vector<cv::Point2f> uv;
            project_px(lv.pts, cfg.camera, uv);
            for (const auto& q : uv)
                if (q.x >= 0 && q.y >= 0 && q.x < static_cast<float>(canvas.cols) && q.y < static_cast<float>(canvas.rows))
                    cv::circle(canvas, q, 4, kOrange, cv::FILLED, cv::LINE_AA);
        }
        std::string lidar_line;
        if (lidar) {
            char b[256];
            if (lv.pts.empty()) {
                lidar_line = "LiDAR: chua co vong quet";
            } else {
                std::snprintf(b, sizeof(b), "LiDAR: san z_C %.0f mm (%d d) | mat vat z_C %.0f (%d d), lech ArUco %+.0f | "
                              "cao tren san (W) %.0f mm", lv.floor_C.z, lv.floor_n, lv.top_C.z, lv.top_n, top_minus_aruco,
                              -lv.top_w_mm[2]);
                lidar_line = b;
            }
        }
        body_arrow(canvas, bc, cfg.camera, {1, 0, 0}, "MUI +x_B", kYellow);
        body_arrow(canvas, bc, cfg.camera, {0, 1, 0}, "PHAI +y_B", kCyan);
        std::string marker_line = "ArUco: khong thay";
        std::string world_line, nose_line;
        if (m.found) {
            std::vector<cv::Point> poly(m.pose.corners_px.begin(), m.pose.corners_px.end());
            cv::polylines(canvas, poly, true, kGreen, 4, cv::LINE_AA);
            char b[256];
            std::snprintf(b, sizeof(b), "ArUco p_C (%.0f, %.0f, %.0f) | p_B (%.0f, %.0f, %.0f) mm | reproj %.2f px",
                          m.pose.t_CM_mm[0], m.pose.t_CM_mm[1], m.pose.t_CM_mm[2], m.p_B_mm.x, m.p_B_mm.y, m.p_B_mm.z,
                          m.pose.reproj_px);
            marker_line = b;
            const cv::Vec3d sd = p_stats.stddev();
            if (origin.set && std::isfinite(m.p_w_mm[0])) {
                std::snprintf(b, sizeof(b), "VAT trong W (%.0f, %.0f, %.0f) mm NED | %d mau, lech chuan (%.0f, %.0f, %.0f)",
                              m.p_w_mm[0], m.p_w_mm[1], m.p_w_mm[2], p_stats.n, sd[0], sd[1], sd[2]);
                world_line = b;
                std::snprintf(b, sizeof(b), "theo mui luc dat goc: truoc %.0f, phai %.0f mm, cao tren san %.0f mm",
                              m.p_nose_mm[0], m.p_nose_mm[1], -m.p_nose_mm[2]);
                nose_line = b;
            } else if (std::isfinite(m.p_ekf_mm[0])) {
                std::snprintf(b, sizeof(b), "VAT (goc EKF) (%.0f, %.0f, %.0f) mm | lech chuan (%.0f, %.0f, %.0f)",
                              m.p_ekf_mm[0], m.p_ekf_mm[1], m.p_ekf_mm[2], sd[0], sd[1], sd[2]);
                world_line = b;
            }
        }
        std::string launch_line, a_line;
        if (args.launch_id >= 0) {
            char b[256];
            const bool fresh = launch.found && meta.t_ns - launch_seen_ns < kNsPerS / 2;
            if (!fresh) {
                launch_line = "Marker cat canh: khong thay";
            } else if (launch_ref.set && std::isfinite(launch.drift_mm[0])) {
                std::snprintf(b, sizeof(b), "Marker cat canh: EKF troi (%+.0f, %+.0f, %+.0f) mm so voi luc dat goc",
                              launch.drift_mm[0], launch.drift_mm[1], launch.drift_mm[2]);
                launch_line = b;
            } else {
                std::snprintf(b, sizeof(b), "Marker cat canh: camera lech x %.0f y %.0f mm, cao %.0f | mui lech %+.1f deg",
                              launch.cam_S_mm[0], launch.cam_S_mm[1], -launch.cam_S_mm[2], launch.yaw_rel_deg);
                launch_line = b;
            }
            if (m.found && std::isfinite(m.err_mm[0])) {
                std::snprintf(b, sizeof(b), "VAT - thuoc (theo mui): (%+.0f, %+.0f, %+.0f) mm = %.0f mm", m.err_mm[0],
                              m.err_mm[1], m.err_mm[2], std::sqrt(m.err_mm.dot(m.err_mm)));
                a_line = b;
            } else if (m.found && !args.gt_m) {
                a_line = "VAT - thuoc: them --gt X Y Z (tam ArUco vat tu tam marker cat canh, m)";
            }
        }
        if (!launch_line.empty()) put(canvas, launch_line, 7, launch.found ? kCyan : kRed);
        if (!a_line.empty()) put(canvas, a_line, 6, kCyan);
        if (!lidar_line.empty()) put(canvas, lidar_line, 4, kOrange);
        if (args.origin_s > 0 && !origin.set) put(canvas, "DANG DAT GOC W: giu UAV yen...", 5, kRed);
        put(canvas, pose_line(pose, origin), 0, kYellow);
        put(canvas, marker_line, 1, m.found ? kGreen : kRed);
        if (!world_line.empty()) put(canvas, world_line, 2, kGreen);
        if (!nose_line.empty()) put(canvas, nose_line, 3, kGreen);

        if (snap) {
            ++snaps;
            char name[32];
            std::snprintf(name, sizeof(name), "snap_%03d.jpg", snaps);
            write_image_rgb(dir / name, canvas, kJpegQuality);
            snaps_csv << snaps << ',' << csv_row(meta.t_ns, pose, m);
            snaps_csv.flush();
            UAVP_LOG_INFO(kModule, "Mau %d: %s | %s | %s | %s | %s | %s | %s", snaps, pose_line(pose, origin).c_str(),
                          marker_line.c_str(), world_line.c_str(), nose_line.c_str(), a_line.c_str(), launch_line.c_str(),
                          lidar_line.c_str());
        }
        if (show) remote->send_frame(canvas, meta.t_ns, {}, {}, snap);
        if (mono_ns() - t_log >= kNsPerS) {
            t_log = mono_ns();
            UAVP_LOG_INFO(kModule, "%s | %s | %s | %s | %s | %s", pose_line(pose, origin).c_str(), marker_line.c_str(),
                          world_line.c_str(), a_line.c_str(), launch_line.c_str(), lidar_line.c_str());
        }
    }
    track.flush();
    write_summary(dir, frames, marker_frames, snaps, p_stats, bc, origin, lidar_stats, launch_ref, nose_stats, drift_stats,
                  args.gt_m);
    if (drift_stats.n > 0)
        UAVP_LOG_INFO(kModule, "EKF troi (marker cat canh, %d lan thay lai): trung binh (%+.0f, %+.0f, %+.0f) mm",
                      drift_stats.n, drift_stats.mean[0], drift_stats.mean[1], drift_stats.mean[2]);
    if (nose_stats.n > 0 && launch_ref.set && args.gt_m) {
        const cv::Vec3d e = nose_stats.mean - launch_ref.gt_nose_mm(*args.gt_m);
        UAVP_LOG_INFO(kModule, "SAI SO VAT so voi thuoc (theo mui, %d khung): (%+.0f, %+.0f, %+.0f) mm | 2D %.0f | 3D %.0f mm",
                      nose_stats.n, e[0], e[1], e[2], std::hypot(e[0], e[1]), std::sqrt(e.dot(e)));
    }
    const cv::Vec3d sd = p_stats.stddev();
    if (lidar_stats.n > 0) {
        const cv::Vec3d ls = lidar_stats.stddev();
        UAVP_LOG_INFO(kModule, "LiDAR (%d vong): mat vat - ArUco %+.1f +- %.1f mm | san z_C %.0f mm | mat vat cao tren san "
                      "(W) %.0f +- %.0f mm", lidar_stats.n, lidar_stats.mean[0], ls[0], lidar_stats.mean[1],
                      lidar_stats.mean[2], ls[2]);
    }
    UAVP_LOG_INFO(kModule, "Tong: %d khung, %d khung thay ArUco, %d mau | vat (%s) trung binh (%.0f, %.0f, %.0f) mm, "
                  "lech chuan (%.0f, %.0f, %.0f) mm", frames, marker_frames, snaps, origin.set ? "W" : "goc EKF",
                  p_stats.mean[0], p_stats.mean[1], p_stats.mean[2], sd[0], sd[1], sd[2]);
    UAVP_LOG_INFO(kModule, "Ket qua: %s", dir.c_str());
    set_logger(nullptr);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& e) {
        set_logger(nullptr);
        std::cerr << "LOI: " << e.what() << '\n';
        return 1;
    }
}
