// uavp_sensor_check: test nhanh 3 nguon tren Pi (khong chay model) va ket hop chung.
//   ./build/uavp_sensor_check --cfg configs/run/real_time.yaml --seconds 10 [--set sources.mavlink.source=serial]
// Moi 1 s in: camera (khung/s, tre tu SensorTimestamp), LiDAR (vong/s, diem/vong), Pixhawk (ATTITUDE/s, roll / pitch /
// yaw, vi tri NED, rangefinder, tre). Cuoi cung (runs/<ngay-gio>_sensor_check/):
//   overlay.jpg       khung cuoi + diem LiDAR chieu len anh (mau theo khoang cach) + tu the FC
//   lidar_world.csv   diem LiDAR 1 s cuoi: he camera (mm) va he the gioi NED (mm, = T_WC(t) p_C theo tu the luc quet)
//   summary.json      tan so, do tre, tu the cuoi, diem mat dat duoi tam anh trong he camera + he the gioi
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/log.hpp"
#include "uavp/common/run_files.hpp"
#include "uavp/sources/frame_source.hpp"
#include "uavp/sources/ld06.hpp"
#include "uavp/sources/mavlink.hpp"
#include "uavp/sources/pi_camera.hpp"
#include "uavp/stages/s4_fusion/lidar.hpp"
#include "uavp/support/cpu_affinity.hpp"
#include "uavp/support/draw.hpp"
#include "uavp/tracking/world_pose.hpp"

using namespace uavp;

namespace {

constexpr const char* kModule = "uavp.sensor_check";
constexpr double kRadToDeg = 180.0 / CV_PI;
constexpr double kAxisConeDeg = 3.0;   // diem LiDAR trong 3 do quanh truc quang -> do sau tam anh
constexpr int64_t kWorldWindowNs = kNsPerS;  // lidar_world.csv: diem 1 s cuoi

struct Args {
    std::string cfg;
    std::vector<std::string> overrides;
    double seconds{10.0};
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) a.cfg = argv[++i];
        else if (std::strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) a.seconds = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--set") == 0)
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.overrides.emplace_back(argv[++i]);
    }
    if (a.cfg.empty()) throw std::invalid_argument("Dung: uavp_sensor_check --cfg configs/run/real_time.yaml [--seconds 10]");
    return a;
}

struct Counters {
    int frames{0};
    std::size_t scans{0}, att{0}, pos{0}, rng{0};
    double cam_age_ms_sum{0.0};
};

struct Stats {
    double cam_fps{0}, cam_age_ms{0}, lidar_hz{0}, att_hz{0}, pos_hz{0}, rng_hz{0};
    std::size_t points_per_scan{0};
};

Stats rates(const Counters& now, const Counters& last, double dt_s, std::size_t pts) {
    Stats s;
    const int frames = now.frames - last.frames;
    s.cam_fps = frames / dt_s;
    s.cam_age_ms = frames > 0 ? (now.cam_age_ms_sum - last.cam_age_ms_sum) / frames : 0.0;
    s.lidar_hz = static_cast<double>(now.scans - last.scans) / dt_s;
    s.att_hz = static_cast<double>(now.att - last.att) / dt_s;
    s.pos_hz = static_cast<double>(now.pos - last.pos) / dt_s;
    s.rng_hz = static_cast<double>(now.rng - last.rng) / dt_s;
    s.points_per_scan = pts;
    return s;
}

std::string pose_text(const std::optional<FcPose>& p) {
    if (!p) return "FC: khong co tu the";
    char b[200];
    std::snprintf(b, sizeof(b), "FC roll %.1f pitch %.1f yaw %.1f do", p->roll_rad * kRadToDeg, p->pitch_rad * kRadToDeg,
                  p->yaw_rad * kRadToDeg);
    std::string s = b;
    if (p->has_position) {
        std::snprintf(b, sizeof(b), " | NED (%.2f, %.2f, %.2f) m", p->pos_ned_m[0], p->pos_ned_m[1], p->pos_ned_m[2]);
        s += b;
    } else {
        s += " | NED: EKF chua co vi tri";
    }
    if (p->has_range) {
        std::snprintf(b, sizeof(b), " | range %.2f m", p->range_m);
        s += b;
    }
    return s;
}

std::optional<FcPose> fc_pose(const IFcSource* fc, int64_t t_ns) {
    FcPose p;
    if (fc == nullptr || !fc->poses().pose_at(t_ns, p)) return std::nullopt;
    return p;
}

// Diem LiDAR (vong quet moi trong 1 s cuoi) -> he camera + he the gioi theo tu the FC luc quet.
struct WorldPoints {
    std::vector<cv::Point3f> p_C;
    std::vector<cv::Point3d> p_W;  // nan neu khong co tu the
    std::vector<int64_t> t_ns;
};

void lidar_to_world(const ScanBuffer& lidar, const IFcSource* fc, const Config& cfg, const BodyCamera& bc,
                    WorldPoints& out) {
    std::vector<Scan> scans;
    std::size_t next = 0;
    bool truncated = false;
    lidar.collected_since(mono_ns() - kWorldWindowNs, scans, next, truncated);
    for (const auto& s : scans) {
        std::vector<cv::Point3f> pts;
        std::vector<int64_t> t;
        scan_to_camera_mm(s, cfg.fusion.extrinsics, cfg.fusion, pts, t);
        for (std::size_t i = 0; i < pts.size(); ++i) {
            const auto p = fc_pose(fc, t[i]);
            out.p_C.push_back(pts[i]);
            out.t_ns.push_back(t[i]);
            out.p_W.push_back(p ? to_world(camera_pose(*p, bc), cv::Point3d(pts[i].x, pts[i].y, pts[i].z))
                                : cv::Point3d(kNaN, kNaN, kNaN));
        }
    }
}

void write_world_csv(const std::filesystem::path& path, const WorldPoints& w) {
    std::string text = "t_ns,X_C_mm,Y_C_mm,Z_C_mm,X_W_mm,Y_W_mm,Z_W_mm\n";
    char line[192];
    for (std::size_t i = 0; i < w.p_C.size(); ++i) {
        std::snprintf(line, sizeof(line), "%lld,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n", static_cast<long long>(w.t_ns[i]),
                      static_cast<double>(w.p_C[i].x), static_cast<double>(w.p_C[i].y), static_cast<double>(w.p_C[i].z),
                      w.p_W[i].x, w.p_W[i].y, w.p_W[i].z);
        text += line;
    }
    write_text(path, text);
}

// Mat dat duoi tam anh: median z cua diem LiDAR gan truc quang (<= 3 do) -> p_C = (0, 0, d) -> p_W.
struct GroundPoint {
    cv::Point3d p_C{kNaN, kNaN, kNaN};
    cv::Point3d p_W{kNaN, kNaN, kNaN};
    int n{0};
};

GroundPoint ground_under_center(const WorldPoints& w, const IFcSource* fc, const BodyCamera& bc) {
    GroundPoint g;
    std::vector<float> z;
    int64_t t_last = 0;
    const double tan_cone = std::tan(kAxisConeDeg * CV_PI / 180.0);
    for (std::size_t i = 0; i < w.p_C.size(); ++i) {
        const cv::Point3f& p = w.p_C[i];
        if (std::hypot(static_cast<double>(p.x), static_cast<double>(p.y)) > tan_cone * p.z) continue;
        z.push_back(p.z);
        t_last = std::max(t_last, w.t_ns[i]);
    }
    g.n = static_cast<int>(z.size());
    if (z.empty()) return g;
    g.p_C = {0.0, 0.0, median_inplace(std::span<float>(z))};
    if (const auto p = fc_pose(fc, t_last)) g.p_W = to_world(camera_pose(*p, bc), g.p_C);
    return g;
}

cv::Mat overlay(const cv::Mat& image, const WorldPoints& w, const Config& cfg, const std::vector<std::string>& lines) {
    std::vector<cv::Point2f> uv;
    project_px(w.p_C, cfg.camera, uv);
    cv::Mat out = image.clone();
    for (std::size_t i = 0; i < uv.size(); ++i) {
        if (w.p_C[i].z <= 0.f || uv[i].x < 0 || uv[i].y < 0 || uv[i].x >= static_cast<float>(image.cols) || uv[i].y >= static_cast<float>(image.rows)) continue;
        const double k = std::clamp((static_cast<double>(w.p_C[i].z) - 300.0) / 2000.0, 0.0, 1.0);  // gan do, xa lam
        cv::circle(out, uv[i], 3, cv::Scalar(255 * (1 - k), 80, 255 * k), -1);  // anh RGB
    }
    draw_center(out, cfg.camera.center_px());
    draw_title(out, lines);
    return out;
}

void write_summary(const std::filesystem::path& dir, const Stats& s, const std::optional<FcPose>& pose,
                   const GroundPoint& g, const std::string& fc_desc, std::size_t crc_lidar, std::size_t crc_fc) {
    JsonWriter w;
    w.begin_object().key("camera").begin_object().key("fps").num(s.cam_fps).key("age_ms").num(s.cam_age_ms).end_object();
    w.key("lidar").begin_object().key("scans_per_s").num(s.lidar_hz).key("points_per_scan").value(s.points_per_scan);
    w.key("crc_errors").value(crc_lidar).end_object();
    w.key("fc").begin_object().key("source").value(fc_desc).key("attitude_hz").num(s.att_hz).key("position_hz").num(s.pos_hz);
    w.key("range_hz").num(s.rng_hz).key("crc_errors").value(crc_fc).key("pose");
    if (!pose) w.null();
    else {
        w.begin_object().key("roll_deg").num(pose->roll_rad * kRadToDeg).key("pitch_deg").num(pose->pitch_rad * kRadToDeg);
        w.key("yaw_deg").num(pose->yaw_rad * kRadToDeg).key("has_position").value(pose->has_position).key("pos_ned_m");
        w.begin_array().num(pose->pos_ned_m[0]).num(pose->pos_ned_m[1]).num(pose->pos_ned_m[2]).end_array();
        w.key("has_range").value(pose->has_range).key("range_m").num(pose->range_m).end_object();
    }
    w.end_object().key("ground_under_center").begin_object().key("n_points").value(g.n).key("p_C_mm");
    w.begin_array().num(g.p_C.x).num(g.p_C.y).num(g.p_C.z).end_array().key("p_W_mm");
    w.begin_array().num(g.p_W.x).num(g.p_W.y).num(g.p_W.z).end_array().end_object().end_object();
    write_text(dir / "summary.json", w.str());
}

int run_main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const YAML::Node raw = load_raw(args.cfg, args.overrides);
    const Config cfg = config_from_raw(raw);
    const std::string cores = pin_process_to_cores(cfg.cpu.cores);
    const SourcesConfig src = sources_config(raw["sources"]);
    const MavlinkConfig mav = mavlink_config(raw["sources"]["mavlink"]);
    const BodyCamera bc = load_body_camera(lock_config(raw["lock_track"]).calib_T_BC);
    const CaptureConfig cap = load_capture(raw["calib_K"].as<std::string>());
    auto logger = std::make_unique<Logger>(Level::kInfo);
    set_logger(logger.get());
    UAVP_LOG_INFO(kModule, "Test cam bien %.0f s, CPU %s", args.seconds, cores.c_str());
    Ld06Reader lidar(src.lidar_port, src.lidar_baudrate, static_cast<std::size_t>(src.lidar_buffer_scans),
                     src.lidar_extra_latency_ms);
    std::unique_ptr<IFcSource> fc;
    if (mav.enabled) {
        try {
            fc = open_fc_source(mav);
        } catch (const std::exception& e) {
            UAVP_LOG_WARN(kModule, "Khong mo duoc nguon MAVLink: %s -> test tiep khong co Pixhawk", e.what());
        }
    }
    PiCamera camera(cap, src.camera_buffer_count, src.sensor_bit_depth);
    cv::Mat image(cap.height_px, cap.width_px, CV_8UC3);
    const std::filesystem::path dir = make_run_dir(cfg.runs_root, "sensor_check");
    logger->open_file(dir / "run.log");
    Counters c, last;
    Stats st;
    int64_t t_tick = mono_ns();
    const int64_t t_end = mono_ns() + static_cast<int64_t>(args.seconds * 1e9);
    while (mono_ns() < t_end) {
        FrameMeta meta;
        if (!camera.next_frame(image, meta, src.frame_timeout_s)) throw std::runtime_error("Camera: khong co khung moi");
        ++c.frames;
        c.cam_age_ms_sum += static_cast<double>(mono_ns() - meta.t_ns) / 1e6;
        if (static_cast<double>(mono_ns() - t_tick) / 1e9 < 1.0) continue;
        c.scans = lidar.buffer().count();
        c.att = fc ? fc->poses().n_attitude() : 0;
        c.pos = fc ? fc->poses().n_position() : 0;
        c.rng = fc ? fc->poses().n_range() : 0;
        Scan scan;
        const std::size_t pts = lidar.buffer().latest_scan(scan) ? scan.size() : 0;
        st = rates(c, last, static_cast<double>(mono_ns() - t_tick) / 1e9, pts);
        last = c;
        t_tick = mono_ns();
        UAVP_LOG_INFO(kModule, "camera %.1f khung/s (tre %.0f ms) | LiDAR %.1f vong/s, %zu diem/vong | ATTITUDE %.0f/s, "
                      "NED %.0f/s, range %.0f/s | %s", st.cam_fps, st.cam_age_ms, st.lidar_hz, st.points_per_scan,
                      st.att_hz, st.pos_hz, st.rng_hz, pose_text(fc_pose(fc.get(), meta.t_ns)).c_str());
    }
    WorldPoints wp;
    lidar_to_world(lidar.buffer(), fc.get(), cfg, bc, wp);
    const GroundPoint g = ground_under_center(wp, fc.get(), bc);
    const std::optional<FcPose> pose = fc_pose(fc.get(), mono_ns() - kNsPerMs * 20);
    UAVP_LOG_INFO(kModule, "Mat dat duoi tam anh (%d diem LiDAR): p_C = (%.0f, %.0f, %.0f) mm -> p_W = (%.0f, %.0f, %.0f) mm NED",
                  g.n, g.p_C.x, g.p_C.y, g.p_C.z, g.p_W.x, g.p_W.y, g.p_W.z);
    write_world_csv(dir / "lidar_world.csv", wp);
    write_image_rgb(dir / "overlay.jpg", overlay(image, wp, cfg, {pose_text(pose)}), kJpegQuality);
    write_summary(dir, st, pose, g, fc ? fc->describe() : "khong co", lidar.buffer().crc_errors(), fc ? fc->crc_errors() : 0);
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
