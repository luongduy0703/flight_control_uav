// Phan chung cua 2 chuong trinh thoi gian thuc TREN PI (apps/live_app.hpp). Chi chay tren cpu.cores (vd [2, 3]).
// Luong: camera (libcamera), LiDAR (LD06 serial), Pixhawk (MAVLink qua uavp_fc_bridge), log, theo doi phan cung,
// gui ve may tinh (remote); dong ho chung CLOCK_MONOTONIC.
//   kPerception (uavp_real_time_perception): bam SPACE (may tinh) / s -> s1 -> s3 -> s4 (LiDAR quanh cac khung co
//     mask cua pha chon vat), D camera - san co dinh (distance.fixed_m). Khong dung Pixhawk. Do thoi gian, phan cung,
//     sai so so voi ArUco.
//   kFull (uavp_real_time): lenh bat dau tu may tinh qua Pixhawk + dang hover -> s1 -> s3 -> khoa (p_W) -> bam (tu
//     the FC, optical flow, Kalman, segment theo LiDAR) -> s4 cuoi. D camera - san tu do cao optical flow.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <opencv2/core/version.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/log.hpp"
#include "uavp/common/run_files.hpp"
#include "uavp/model/ncnn_model.hpp"
#include "uavp/pipeline/session.hpp"
#include "uavp/pipeline/start_gate.hpp"
#include "uavp/sources/frame_source.hpp"
#include "uavp/sources/ld06.hpp"
#include "uavp/sources/mavlink.hpp"
#include "uavp/sources/pi_camera.hpp"
#include "uavp/stages/s3_select.hpp"
#include "uavp/stages/s2_segment/mask_shape.hpp"
#include "uavp/stages/s4_fusion.hpp"
#include "uavp/support/aruco_check.hpp"
#include "uavp/support/cpu_affinity.hpp"
#include "uavp/support/draw.hpp"
#include "uavp/support/hw_monitor.hpp"
#include "uavp/support/remote_view.hpp"
#include "uavp/support/terminal_keys.hpp"
#include "uavp/tracking/locked_tracker.hpp"
#include "live_app.hpp"

using namespace uavp;

namespace {

constexpr const char* kModule = "uavp.live";
constexpr double kHwStatusPeriodS = 1.0;  // gui CPU / nhiet / tan so ve may tinh moi 1 s
constexpr double kMaskPolyEpsPx = 2.0;    // rut gon duong vien mask gui ve may tinh (sai toi da 2 px anh goc)
constexpr const char* kWindow = "uavp real_time";
constexpr int kKeySpace = 32, kKeyEsc = 27;
constexpr double kKeyPollS = 0.5;
constexpr double kStatusPeriodS = 5.0;
constexpr double kAxisDrawMm = 40.0;

struct Args {
    std::string cfg;
    std::vector<std::string> overrides;
};

struct Live {
    IFrameSource& camera;
    const ScanBuffer& lidar;
    const FcPoseBuffer* fc;  // nullptr: khong co MAVLink
    const Config& cfg;
    AppMode mode;
    const StartConfig& start;     // kFull: dieu kien bat dau (lenh may tinh + hover)
    const DistanceConfig& distance;
    const SourcesConfig& src;
    const LockConfig& lock;
    const BodyCamera& body_camera;
    const DisplayConfig& display;
    const TerminalKeys& keys;
    HwMonitor& monitor;
    FramePool& pool;
    RemoteView* remote;      // nullptr: khong xem tu may tinh (remote.enabled = false)
    double remote_wait_fps;  // anh / s gui ve luc cho SPACE
    double remote_run_fps;   // anh / s gui ve luc chay
    Timings timings;
    int64_t started_ns{0};   // luc bat dau quet (lenh dung den sau luc nay moi tinh)
    int64_t hw_sent_ns{0};
    std::string timing_csv;  // timing.csv: moi khung pha 1 mot dong
    std::string track_csv;   // track.csv: moi khung pha 3 mot dong
    std::filesystem::path fc_log;  // fc_events.jsonl: trang thai Pixhawk o moi quyet dinh (rong = khong ghi)
};

constexpr const char* kTimingHeader = "frame,capture_ms,s1_ms,s2_ms,s3_ms,loop_ms,latency_ms\n";
constexpr const char* kTrackHeader =
    "frame,t_ns,pose_mode,method,n_good,ncc,u_kf,v_kf,u_pose,v_pose,sigma_px,segmented,pC_x_mm,pC_y_mm,pC_z_mm,step_ms,s2_ms,latency_ms\n";

// Bo logger toan cuc truoc khi logger bi huy (RAII).
struct LoggerGuard {
    explicit LoggerGuard(Logger* l) { set_logger(l); }
    ~LoggerGuard() { set_logger(nullptr); }
    LoggerGuard(const LoggerGuard&) = delete;
    LoggerGuard& operator=(const LoggerGuard&) = delete;
};

struct Scanned {
    Session session;
    std::vector<int> slot_of_track;  // track_id -> slot FramePool cua lan do hop le cuoi (-1: khong co)
    int frames{0};
    int64_t t_first_ns{0};
    double s1_s3_ms{0.0};
    cv::Mat last_image;              // ban sao khung cuoi pha chon vat: luu anh khi khong co vat selected
    int64_t last_t_ns{0};
};

// Ket qua pha khoa + bam.
struct LockRun {
    std::vector<MaskFrame> frames;   // moi khung co mask (pha 1 + pha 3) cho s4
    cv::Point3d p_W_lock_mm{kNaN, kNaN, kNaN};
    std::array<int, static_cast<std::size_t>(FlowMethod::kCount)> methods{};
    PoseMode last_mode{PoseMode::kNone};
    int ref_slot{-1};                // anh cua khung co mask moi nhat (s4 tham chieu, ArUco, final.jpg)
    int frames_tracked{0};
    int segments{0};
    double duration_s{0.0};
    bool has_world{false};
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) a.cfg = argv[++i];
        else if (std::strcmp(argv[i], "--set") == 0)
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.overrides.emplace_back(argv[++i]);
    }
    if (a.cfg.empty()) throw std::invalid_argument("Dung: <chuong trinh> --cfg configs/run/<file>.yaml [--set k=v ...]");
    return a;
}

std::optional<CamPose> pose_at(const Live& live, int64_t t_ns) {
    FcPose fc;
    if (live.fc == nullptr || !live.fc->pose_at(t_ns, fc)) return std::nullopt;
    return camera_pose(fc, live.body_camera);
}

int show_live(Live& live, const cv::Mat& image, const std::vector<std::string>& lines, const std::vector<Track>& tracks) {
    Scan scan;
    std::vector<cv::Point3f> pts;
    std::vector<int64_t> t;
    std::vector<cv::Point2f> uv;
    if (live.lidar.latest_scan(scan)) scan_to_camera_mm(scan, live.cfg.fusion.extrinsics, live.cfg.fusion, pts, t);
    visible_px(pts, live.cfg.camera, image.cols, image.rows, uv);
    std::vector<std::string> all(lines);
    all.push_back("LiDAR: " + std::to_string(uv.size()) + " diem trong anh (vong quet moi nhat)");
    cv::Mat bgr;
    cv::cvtColor(draw_live(image, uv, live.cfg.camera.center_px(), all, tracks), bgr, cv::COLOR_RGB2BGR);  // hien thi BGR
    cv::imshow(kWindow, bgr);
    return cv::waitKey(1) & 0xFF;
}

// Anh + vong quet LiDAR moi nhat (chieu len anh) -> may tinh (khi den luot max_fps); trang thai: moi khung.
void remote_frame(Live& live, const cv::Mat& image, int64_t t_ns, const std::string& status, bool force = false) {
    if (live.remote == nullptr || !live.remote->connected()) return;
    if (force || live.remote->due()) {
        Scan scan;
        std::vector<cv::Point3f> pts;
        std::vector<int64_t> t;
        std::vector<cv::Point2f> uv, uv_in;
        std::vector<float> z_in;
        if (live.lidar.latest_scan(scan)) scan_to_camera_mm(scan, live.cfg.fusion.extrinsics, live.cfg.fusion, pts, t);
        project_px(pts, live.cfg.camera, uv);
        for (std::size_t i = 0; i < uv.size(); ++i) {
            const bool in = uv[i].x >= 0.f && uv[i].y >= 0.f && uv[i].x < static_cast<float>(image.cols) &&
                            uv[i].y < static_cast<float>(image.rows) && std::abs(pts[i].x) < 2.f * pts[i].z;
            if (!in) continue;
            uv_in.push_back(uv[i]);
            z_in.push_back(pts[i].z);
        }
        live.remote->send_frame(image, t_ns, uv_in, z_in, force);
    }
    if (!status.empty()) live.remote->send_status(status);
    HwSample hw;
    if (static_cast<double>(mono_ns() - live.hw_sent_ns) / 1e9 >= kHwStatusPeriodS && live.monitor.latest(hw)) {
        live.hw_sent_ns = mono_ns();
        JsonWriter w;
        w.begin_object().key("phase").value("hw").key("cpu_total_pct").num(hw.cpu_total_pct);
        w.key("cpu_core_max_pct").num(hw.cpu_core_max_pct).key("proc_cpu_pct").num(hw.proc_cpu_pct);
        w.key("proc_rss_mb").num(hw.proc_rss_mb).key("mem_used_mb").num(hw.mem_used_mb).key("temp_c").num(hw.temp_c);
        w.key("freq_mhz_min").num(hw.freq_mhz_min).end_object();
        live.remote->send_status(w.str());
    }
}

int remote_key(Live& live) { return live.remote != nullptr ? live.remote->poll_key() : -1; }

// --- Log Pixhawk theo quyet dinh (fc_events.jsonl) -----------------------------------------------------------------
// Moi quyet dinh quan trong cua perception (cho / bat dau, s1, chon vat, loai vat, vat OK, khoa, s4, dung, ket thuc)
// ghi 1 dong JSON: su kien + chi tiet + trang thai Pixhawk luc do (HEARTBEAT / arm / che do, landed, van toc, do cao
// optical flow / rangefinder, tu the, vi tri NED, lenh may tinh). Ghi ngay (append): khong mat khi dung giua chung.
double round_to(double v, double scale) { return std::isfinite(v) ? std::round(v * scale) / scale : v; }

void write_fc_state(JsonWriter& w, const Live& live, int64_t now_ns) {
    if (live.fc == nullptr) {
        w.null();
        return;
    }
    const FcStatus s = live.fc->status(now_ns, live.start.hover_window_s);
    const auto age = [&](int64_t t) { return t > 0 ? static_cast<double>(now_ns - t) / 1e9 : kNaN; };
    const auto m4 = [](double v) { return round_to(v, 1e4); };  // m: 0.1 mm
    w.begin_object().key("heartbeat_age_s").num(age(s.heartbeat_ns)).key("armed").value(s.armed);
    w.key("custom_mode").value(static_cast<int64_t>(s.custom_mode)).key("landed_state").value(static_cast<int>(s.landed_state));
    w.key("vel_ned_m_s").begin_array();
    for (int k = 0; k < 3; ++k) w.value(round_to(s.vel_ned_m_s[k], 1e3));
    w.end_array().key("vel_age_s").num(age(s.vel_ns));
    w.key("height").begin_object().key("source").value(s.height_from_flow ? "optical_flow" : "rangefinder");
    w.key("median_m").value(m4(s.height_median_m)).key("min_m").value(m4(s.height_min_m));
    w.key("max_m").value(m4(s.height_max_m)).key("samples").value(s.height_samples).end_object();
    int64_t flow_t = 0;
    double flow_m = kNaN;
    int quality = 0;
    w.key("optical_flow");
    if (live.fc->latest_flow(flow_t, flow_m, quality))
        w.begin_object().key("ground_distance_m").value(m4(flow_m)).key("quality").value(quality).key("age_s").num(age(flow_t)).end_object();
    else
        w.null();
    FcPose p;
    const int64_t t_att = live.fc->latest_attitude_ns();
    w.key("attitude");
    if (t_att > 0 && live.fc->pose_at(t_att, p)) {
        constexpr double kDeg = 180.0 / 3.14159265358979323846;
        w.begin_object().key("roll_deg").num(p.roll_rad * kDeg).key("pitch_deg").num(p.pitch_rad * kDeg);
        w.key("yaw_deg").num(p.yaw_rad * kDeg).key("age_s").num(age(t_att)).end_object();
        w.key("pos_ned_m");
        if (p.has_position) w.begin_array().value(m4(p.pos_ned_m[0])).value(m4(p.pos_ned_m[1])).value(m4(p.pos_ned_m[2])).end_array();
        else w.null();
        w.key("rangefinder_m");
        if (p.has_range) w.value(m4(p.range_m));
        else w.null();
    } else {
        w.null();
    }
    w.key("command").begin_object().key("count").value(static_cast<int64_t>(s.command_count));
    w.key("start").value(s.command_start).key("age_s").num(age(s.command_ns)).end_object();
    w.end_object();
}

// JsonWriter in nhieu dong -> 1 dong (JSON Lines): bo xuong dong + thut dau dong.
std::string one_line(const std::string& json) {
    std::string out;
    out.reserve(json.size());
    bool skip_indent = false;
    for (const char c : json) {
        if (c == '\n') {
            skip_indent = true;
            continue;
        }
        if (skip_indent && c == ' ') continue;
        skip_indent = false;
        out += c;
    }
    return out;
}

// details(JsonWriter&): ghi cac khoa chi tiet cua su kien (ben trong object "details").
template <typename F>
void fc_event(const Live& live, std::string_view event, F&& details) {
    if (live.fc_log.empty()) return;
    const int64_t now = mono_ns();
    JsonWriter w;
    w.begin_object().key("event").value(event).key("t_mono_ns").value(now);
    w.key("t_run_s").num(live.started_ns > 0 ? static_cast<double>(now - live.started_ns) / 1e9 : kNaN);
    w.key("details").begin_object();
    details(w);
    w.end_object().key("fc");
    write_fc_state(w, live, now);
    w.end_object();
    std::ofstream(live.fc_log, std::ios::app) << one_line(w.str()) << '\n';
}

void fc_event(const Live& live, std::string_view event) {
    fc_event(live, event, [](JsonWriter&) {});
}

std::string box_json(JsonWriter& w, const Detection& d) {
    w.begin_array().num(d.x1).num(d.y1).num(d.x2).num(d.y2).end_array();
    return {};
}

// Trang thai pha chon vat (moi khung): thoi gian tung buoc, do tre tu luc chup, track cua nhom.
// Mask (duong vien lon nhat, rut gon con vai chuc dinh, toa do anh goc) + tam -> may tinh tu ve len anh.
// Gui so lieu thay vi ve + JPEG tren Pi: re (~0.2 ms, vai tram byte), anh van gui theo max_fps.
void mask_json(JsonWriter& w, const SegObject& obj) {
    std::vector<cv::Point> contour, poly;
    w.key("center_px").begin_array().num(obj.center_px.x).num(obj.center_px.y).end_array().key("mask");
    if (!largest_contour_px(obj.mask, contour)) {
        w.null();
        return;
    }
    cv::approxPolyDP(contour, poly, kMaskPolyEpsPx, true);
    w.begin_array();
    for (const auto& p : poly) w.begin_array().value(p.x).value(p.y).end_array();
    w.end_array();
}

std::string select_status(const Session& s, uint32_t idx, const StepTimes& t, double capture_ms, double loop_ms,
                          double latency_ms, const Config& cfg) {
    JsonWriter w;
    w.begin_object().key("phase").value("select").key("frame").value(static_cast<int64_t>(idx)).key("s1_ms");
    if (t.detected) w.num(t.detect_ms);
    else w.null();
    w.key("s2_ms").num(t.segment_ms).key("s3_ms").num(t.select_ms).key("capture_ms").num(capture_ms);
    w.key("loop_ms").num(loop_ms).key("latency_ms").num(latency_ms).key("n_det").value(s.detections.size());
    w.key("tracks").begin_array();
    for (const int i : s.group) {
        const Track& tr = s.tracks[static_cast<std::size_t>(i)];
        w.begin_object().key("id").value(tr.track_id).key("class").value(cfg.detector.classes.at(tr.class_id));
        w.key("state").value(name(tr.state)).key("n_obs").value(tr.observations.size()).key("W_mm").num(median_W_mm(tr));
        w.key("box");
        box_json(w, predicted_box(tr, idx));
        const SegObject* seg = last_seg(tr);
        if (seg != nullptr) {  // mask gan nhat (track OK / ABORT khong segment nua: van hien mask cuoi)
            mask_json(w, *seg);
            w.key("fresh").value(tr.last_mask_frame == idx);
        }
        w.end_object();
    }
    w.end_array().end_object();
    return w.str();
}

bool wait_remote(Live& live) {
    UAVP_LOG_INFO(kModule, ">>> May tinh: can vat theo anh + duong LiDAR, bam SPACE de bat dau quet vat, q de thoat <<<");
    live.remote->set_fps(live.remote_wait_fps);  // stream muot de can vat
    struct RestoreFps {  // ra khoi ham (SPACE / q / loi): ve max_fps
        Live& l;
        ~RestoreFps() { l.remote->set_fps(l.remote_run_fps); }
    } restore{live};
    const int slot = live.pool.acquire();
    for (;;) {
        FrameMeta meta;
        if (!live.camera.next_frame(live.pool.image(slot), meta, live.src.frame_timeout_s))
            throw std::runtime_error("Khong co khung moi sau frame_timeout_s");
        char st[160];
        std::snprintf(st, sizeof(st), "{\"phase\": \"wait\", \"fc_attitude\": %zu, \"lidar_crc\": %zu}",
                      live.fc ? live.fc->n_attitude() : 0U, live.lidar.crc_errors());
        remote_frame(live, live.pool.image(slot), meta.t_ns, st);
        int key = remote_key(live);
        if (key < 0 && live.keys.enabled()) key = live.keys.read_key(0.0).value_or(-1);
        if (key == 's' || key == kKeySpace || key == 'q') {
            live.pool.release(slot);
            return key != 'q';
        }
    }
}

// Ket qua cho bat dau: co quet khong + khoang cach camera -> san cho s3.
struct StartInfo {
    std::string distance_source;
    double D_m{kNaN};
    bool go{false};
};

// Gia tri that nhan tu Pixhawk luc cho (de thay du lieu co cap nhat khong): tuoi HEARTBEAT, che do, landed, van toc,
// do cao (nguon + so mau), roll / pitch, so mau ATTITUDE, optical flow.
struct FcSnapshot {
    double hb_age_s{kNaN}, att_age_s{kNaN}, vel_age_s{kNaN}, v_h{kNaN}, v_d{kNaN}, roll_deg{kNaN}, pitch_deg{kNaN};
    double flow_m{kNaN}, flow_age_s{kNaN};
    std::size_t n_att{0};
    uint32_t mode{0};
    int height_samples{0}, flow_quality{0};
    uint8_t landed{0};
    bool armed{false}, from_flow{false};
};

FcSnapshot fc_snapshot(const FcPoseBuffer& fc, const FcStatus& s, int64_t now) {
    const auto age = [&](int64_t t) { return t > 0 ? static_cast<double>(now - t) / 1e9 : kNaN; };
    FcSnapshot o;
    o.hb_age_s = age(s.heartbeat_ns);
    o.vel_age_s = age(s.vel_ns);
    if (s.vel_ns > 0) {
        o.v_h = std::hypot(s.vel_ned_m_s[0], s.vel_ned_m_s[1]);
        o.v_d = s.vel_ned_m_s[2];
    }
    o.mode = s.custom_mode;
    o.landed = s.landed_state;
    o.armed = s.armed;
    o.from_flow = s.height_from_flow;
    o.height_samples = s.height_samples;
    o.n_att = fc.n_attitude();
    const int64_t t_att = fc.latest_attitude_ns();
    FcPose p;
    if (t_att > 0 && fc.pose_at(t_att, p)) {
        constexpr double kDeg = 180.0 / 3.14159265358979323846;
        o.att_age_s = age(t_att);
        o.roll_deg = p.roll_rad * kDeg;
        o.pitch_deg = p.pitch_rad * kDeg;
    }
    int64_t ft = 0;
    if (fc.latest_flow(ft, o.flow_m, o.flow_quality)) o.flow_age_s = age(ft);
    return o;
}

std::string gate_json(const StartGate& g, const FcSnapshot& f) {
    JsonWriter w;
    w.begin_object().key("phase").value("wait_fc").key("commanded").value(g.commanded).key("armed").value(g.armed);
    w.key("mode_ok").value(g.mode_ok).key("in_air").value(g.in_air).key("still").value(g.still);
    w.key("height_stable").value(g.height_stable).key("height_m").num(g.height_m).key("reason").value(g.reason);
    w.key("hover_required").value(g.hover_required).key("fc").begin_object();
    w.key("hb_age_s").num(f.hb_age_s).key("mode").value(static_cast<int64_t>(f.mode)).key("landed").value(static_cast<int>(f.landed));
    w.key("n_att").value(f.n_att).key("att_age_s").num(f.att_age_s).key("roll_deg").num(f.roll_deg);
    w.key("pitch_deg").num(f.pitch_deg).key("v_h").num(f.v_h).key("v_d").num(f.v_d).key("vel_age_s").num(f.vel_age_s);
    w.key("height_src").value(f.height_samples > 0 ? (f.from_flow ? "flow" : "rangefinder") : "none");
    w.key("height_samples").value(f.height_samples).key("flow_m").num(f.flow_m).key("flow_q").value(f.flow_quality);
    w.key("flow_age_s").num(f.flow_age_s).end_object();
    w.end_object();
    return w.str();
}

std::string fc_line(const FcSnapshot& f) {
    char b[320];
    std::snprintf(b, sizeof(b),
                  "FC: HEARTBEAT tuoi %.1f s, %s, che do %u, landed %u | ATTITUDE %zu mau (tuoi %.2f s) roll %.1f pitch "
                  "%.1f | van toc %s | do cao %s %d mau | flow %.3f m q%d",
                  f.hb_age_s, f.armed ? "ARM" : "chua arm", f.mode, f.landed, f.n_att, f.att_age_s, f.roll_deg,
                  f.pitch_deg, std::isfinite(f.v_h) ? (std::to_string(f.v_h).substr(0, 4) + " m/s").c_str() : "CHUA CO",
                  f.height_samples > 0 ? (f.from_flow ? "flow" : "rangefinder") : "CHUA CO", f.height_samples, f.flow_m,
                  f.flow_quality);
    return b;
}

// Chay that: cho lenh bat dau tu may tinh (gcs_command.py start qua Pixhawk, hoac SPACE / s tren viewer neu
// start.accept_viewer_key) [+ Pi tu nhan dang hover neu start.require_hover]. Anh van gui ve may tinh (remote).
StartInfo wait_fc(Live& live) {
    if (live.start.require_hover)
        UAVP_LOG_INFO(kModule, ">>> Cho lenh bat dau tu may tinh + hover: da arm, che do bay, dang bay, van toc ~0, do cao "
                      "on dinh %.1f s <<<", live.start.hover_window_s);
    else
        UAVP_LOG_INFO(kModule, ">>> Cho lenh bat dau tu may tinh: gcs_command.py start%s (khong can hover) <<<",
                      live.start.accept_viewer_key ? " hoac SPACE tren remote_viewer" : "");
    const int64_t waiting_since = mono_ns();
    const int slot = live.pool.acquire();
    int64_t last_log = 0;
    for (;;) {
        FrameMeta meta;
        if (!live.camera.next_frame(live.pool.image(slot), meta, live.src.frame_timeout_s))
            throw std::runtime_error("Khong co khung moi sau frame_timeout_s");
        int key = remote_key(live);
        if (key < 0 && live.keys.enabled()) key = live.keys.read_key(0.0).value_or(-1);
        if (key == 'q') {
            live.pool.release(slot);
            fc_event(live, "quit_before_start");
            return {};
        }
        const int64_t now = mono_ns();
        const bool viewer_cmd = key == 's' || key == kKeySpace;
        const FcStatus fs = live.fc->status(now, live.start.hover_window_s);
        const StartGate g = check_start(fs, live.start, waiting_since, viewer_cmd);
        const FcSnapshot snap = fc_snapshot(*live.fc, fs, now);
        remote_frame(live, live.pool.image(slot), meta.t_ns, gate_json(g, snap));
        if (static_cast<double>(now - last_log) / 1e9 >= live.start.status_period_s) {
            last_log = now;
            UAVP_LOG_INFO(kModule, "Cho bat dau: %s", g.ok() ? "DU DIEU KIEN" : g.reason.c_str());
            UAVP_LOG_INFO(kModule, "  %s", fc_line(snap).c_str());
        }
        if (viewer_cmd && !live.start.accept_viewer_key)
            UAVP_LOG_WARN(kModule, "SPACE tren viewer khong duoc tinh (start.accept_viewer_key = false): dung gcs_command.py start");
        if (!g.ok()) continue;
        live.pool.release(slot);
        FcPose pose;
        const bool has_pose = live.fc->pose_at(now, pose);
        StartInfo info{"optical flow / rangefinder (FC)", kNaN, true};
        info.D_m = camera_floor_distance_m(g.height_m, has_pose ? pose.roll_rad : 0.0, has_pose ? pose.pitch_rad : 0.0,
                                           live.distance);
        UAVP_LOG_INFO(kModule, "BAT DAU: lenh %s%s (do cao cam bien %.3f m -> camera - san %.3f m)",
                      g.by_viewer ? "SPACE tren viewer" : "gcs_command.py qua Pixhawk", g.hover_required ? " + hover" : "",
                      g.height_m, info.D_m);
        fc_event(live, "start_gate", [&](JsonWriter& w) {
            w.key("command_from").value(g.by_viewer ? "viewer" : "pixhawk").key("hover_required").value(g.hover_required);
            w.key("armed").value(g.armed).key("mode_ok").value(g.mode_ok).key("in_air").value(g.in_air);
            w.key("still").value(g.still).key("height_stable").value(g.height_stable).key("hover").value(g.hover());
            w.key("sensor_height_m").value(round_to(g.height_m, 1e4)).key("D_fc_m").value(round_to(info.D_m, 1e4));
            w.key("camera_offset_m").value(live.distance.camera_offset_m);
        });
        return info;
    }
}

bool wait_key(Live& live);

StartInfo wait_for_start(Live& live) {
    live.monitor.set_phase(Phase::kWaitKey);
    fc_event(live, "wait_start", [&](JsonWriter& w) {
        w.key("require_hover").value(live.start.require_hover).key("accept_viewer_key").value(live.start.accept_viewer_key);
        w.key("distance_source").value(live.distance.source);
    });
    StartInfo info;
    if (live.mode == AppMode::kFull) info = wait_fc(live);
    else info.go = wait_key(live);
    live.started_ns = mono_ns();
    if (info.go && (live.distance.source == "fixed" || !std::isfinite(info.D_m))) {
        if (live.distance.source == "fc") UAVP_LOG_WARN(kModule, "Khong co do cao tu FC: dung distance.fixed_m");
        info.D_m = live.distance.fixed_m;
        info.distance_source = "co dinh (distance.fixed_m)";
    }
    if (info.go)
        fc_event(live, "start", [&](JsonWriter& w) {
            w.key("D_m").value(round_to(info.D_m, 1e4)).key("D_source").value(info.distance_source);
        });
    return info;
}

// Che do do thoi gian: bam SPACE (may tinh) / s (terminal hoac cua so).
bool wait_key(Live& live) {
    if (live.remote != nullptr) return wait_remote(live);
    if (live.display.show) {
        cv::namedWindow(kWindow, cv::WINDOW_NORMAL);
        cv::resizeWindow(kWindow, live.display.window_width_px, live.display.window_height_px);
        UAVP_LOG_INFO(kModule, "Xem truoc: bam s (hoac SPACE) de bat dau quet vat, q de thoat");
        const int slot = live.pool.acquire();
        for (;;) {
            FrameMeta meta;
            if (!live.camera.next_frame(live.pool.image(slot), meta, live.src.frame_timeout_s))
                throw std::runtime_error("Khong co khung moi sau frame_timeout_s");
            const int key = show_live(live, live.pool.image(slot), {"XEM TRUOC | s: bat dau quet | q: thoat"}, {});
            if (key == 's' || key == kKeySpace || key == 'q' || key == kKeyEsc) {
                live.pool.release(slot);
                return key == 's' || key == kKeySpace;
            }
        }
    }
    if (!live.keys.enabled()) {
        UAVP_LOG_WARN(kModule, "stdin khong phai terminal (ssh -t / khong chay nen): bat dau ngay");
        return true;
    }
    UAVP_LOG_INFO(kModule, ">>> Bam s de bat dau quet vat, q de thoat (khong can Enter) <<<");
    int64_t last_status = mono_ns();
    for (;;) {
        const auto key = live.keys.read_key(kKeyPollS);
        if (key == 's') return true;
        if (key == 'q') return false;
        if (static_cast<double>(mono_ns() - last_status) / 1e9 >= kStatusPeriodS) {
            last_status = mono_ns();
            Scan scan;
            const bool has = live.lidar.latest_scan(scan);
            UAVP_LOG_INFO(kModule, "Dang cho bam s | LiDAR: vong quet moi nhat %zu diem, %zu goi loi CRC | FC: %zu mau ATTITUDE",
                          has ? scan.size() : 0U, live.lidar.crc_errors(), live.fc ? live.fc->n_attitude() : 0U);
        }
    }
}

bool stop_requested(Live& live, const cv::Mat& image, const std::vector<Track>& tracks, const std::string& line) {
    if (remote_key(live) == 'q') {
        fc_event(live, "stop", [](JsonWriter& w) { w.key("by").value("viewer_q"); });
        return true;
    }
    if (live.mode == AppMode::kFull && live.fc != nullptr &&
        stop_commanded(live.fc->status(mono_ns(), live.start.hover_window_s), live.started_ns)) {
        UAVP_LOG_WARN(kModule, "Lenh DUNG tu may tinh: dung quet");
        fc_event(live, "stop", [](JsonWriter& w) { w.key("by").value("gcs_command_stop"); });
        return true;
    }
    if (!live.display.show) return live.keys.enabled() && live.keys.read_key(0.0) == 'q';
    return show_live(live, image, {line}, tracks) == 'q';
}

// Giu slot anh cua lan do hop le cuoi moi track; tra slot cua track ABORT (khong bao gio duoc chon).
void keep_frames(Live& live, Scanned& sc, int slot, uint32_t frame_idx) {
    auto& tracks = sc.session.tracks;
    sc.slot_of_track.resize(tracks.size(), -1);
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        int& held = sc.slot_of_track[i];
        const bool observed_now = !tracks[i].observations.empty() && tracks[i].observations.back().frame_idx == frame_idx;
        if (observed_now && held != slot) {
            live.pool.release(held);
            live.pool.retain(slot);
            held = slot;
        } else if (tracks[i].state == State::kAbort && held >= 0) {
            live.pool.release(held);
            held = -1;
        }
    }
}

void record_step_time(Live& live, uint32_t idx, double capture_ms, const StepTimes& t, double loop_ms,
                      double latency_ms) {
    live.timings.add("capture", capture_ms);
    if (t.detected) live.timings.add("s1_detect", t.detect_ms);  // khung khong chay s1: khong tinh vao median
    live.timings.add("s2_segment", t.segment_ms);
    live.timings.add("s3_select", t.select_ms);
    live.timings.add("loop", loop_ms);
    live.timings.add("latency_select", latency_ms);  // SensorTimestamp -> xong s3 cua khung
    char row[160];
    std::snprintf(row, sizeof(row), "%u,%.2f,%s,%.2f,%.2f,%.2f,%.2f\n", idx, capture_ms,
                  t.detected ? std::to_string(t.detect_ms).c_str() : "", t.segment_ms, t.select_ms, loop_ms, latency_ms);
    live.timing_csv += row;
}

// Su kien pha chon vat (fc_events.jsonl): s1 chay (khung 0 = bat dau; khung sau = mat vat / vat truoc bi loai),
// vat moi duoc chon (gan tam), vat chuyen OK / ABORT. `prev`: trang thai track o khung truoc.
void select_events(const Live& live, const Session& s, uint32_t idx, const StepTimes& t, std::vector<State>& prev) {
    if (t.detected)
        fc_event(live, "s1_detect", [&](JsonWriter& w) {
            w.key("frame").value(static_cast<int64_t>(idx)).key("n_det").value(s.detections.size());
            w.key("detect_ms").num(t.detect_ms).key("why").value(idx == 0 ? "bat_dau" : "mat_vat_hoac_chon_vat_moi");
        });
    for (std::size_t i = 0; i < s.tracks.size(); ++i) {
        const Track& tr = s.tracks[i];
        const bool is_new = i >= prev.size();
        if (!is_new && prev[i] == tr.state) continue;
        const SegObject* seg = last_seg(tr);
        const std::string_view event = is_new ? "target_selected" : tr.state == State::kOk ? "target_ok" : "target_abort";
        fc_event(live, event, [&](JsonWriter& w) {
            w.key("frame").value(static_cast<int64_t>(idx)).key("track_id").value(tr.track_id);
            w.key("class").value(live.cfg.detector.classes.at(tr.class_id)).key("state").value(name(tr.state));
            w.key("reason").value(name(tr.reason)).key("n_obs").value(tr.observations.size());
            w.key("W_mm").num(median_W_mm(tr)).key("conf").num(mean_conf(tr)).key("D_m").value(live.cfg.D_m);
            const cv::Point2d c = seg != nullptr ? seg->center_px : tr.detection.center_px();
            w.key("center_px").begin_array().num(c.x).num(c.y).end_array();
        });
    }
    prev.resize(s.tracks.size());
    for (std::size_t i = 0; i < s.tracks.size(); ++i) prev[i] = s.tracks[i].state;
}

// Pha 1: s1 -> s3 toi khi chon duoc vat (hoac het max_frames / bam q).
void run_until_selected(Live& live, Detector& det, Segmenter& seg, Scanned& sc) {
    live.monitor.set_phase(Phase::kS1S3);
    const int64_t start_all = mono_ns();
    std::vector<State> prev_states;
    for (uint32_t idx = 0; idx < static_cast<uint32_t>(live.cfg.max_frames); ++idx) {
        const int64_t start = mono_ns();
        const int slot = live.pool.acquire();
        FrameMeta meta;
        if (!live.camera.next_frame(live.pool.image(slot), meta, live.src.frame_timeout_s))
            throw std::runtime_error("Khong co khung moi sau frame_timeout_s");
        live.pool.t_ns(slot) = meta.t_ns;
        const double capture_ms = ms_since(start);
        if (idx == 0) sc.t_first_ns = meta.t_ns;
        const StepTimes t = session_step(sc.session, idx, meta.t_ns, live.pool.image(slot), live.cfg, det, seg);
        const double latency_ms = static_cast<double>(mono_ns() - meta.t_ns) / 1e6;
        select_events(live, sc.session, idx, t, prev_states);
        keep_frames(live, sc, slot, idx);
        remote_frame(live, live.pool.image(slot), meta.t_ns,
                     select_status(sc.session, idx, t, capture_ms, ms_since(start), latency_ms, live.cfg));
        char line[160];
        std::snprintf(line, sizeof(line), "khung %u | s1 %.0f ms | s2 %.0f ms | q: dung", idx, t.detect_ms, t.segment_ms);
        const bool stop = stop_requested(live, live.pool.image(slot), sc.session.tracks, line);
        if (stop || sc.session.done || idx + 1 == static_cast<uint32_t>(live.cfg.max_frames)) {
            live.pool.image(slot).copyTo(sc.last_image);  // chi khung cuoi (~2 ms)
            sc.last_t_ns = meta.t_ns;
        }
        live.pool.release(slot);
        record_step_time(live, idx, capture_ms, t, ms_since(start), latency_ms);
        sc.frames = static_cast<int>(idx) + 1;
        if (stop) UAVP_LOG_INFO(kModule, "Bam q: dung o khung %u", idx);
        if (stop || sc.session.done) break;
    }
    sc.s1_s3_ms = ms_since(start_all);
}

// --- s4 -----------------------------------------------------------------------------------------------------------
FusionResult run_fusion(Live& live, Fusion& fusion, const std::vector<MaskFrame>& frames, const char* label) {
    const int64_t start = mono_ns();
    const auto window_ns = static_cast<int64_t>(live.cfg.fusion.lidar_window_s * 1e9);
    int64_t t_lo = frames.front().t_ns, t_hi = frames.front().t_ns;
    for (const auto& f : frames) t_lo = std::min(t_lo, f.t_ns), t_hi = std::max(t_hi, f.t_ns);
    if (!live.lidar.wait_until(t_hi + window_ns, live.src.scan_timeout_s))
        UAVP_LOG_WARN(kModule, "s4: LiDAR chua phu toi khung co mask cuoi + cua so sau scan_timeout_s");
    std::vector<Scan> scans;
    std::size_t next = 0;
    bool truncated = false;
    live.lidar.collected_since(t_lo - window_ns, scans, next, truncated);
    if (truncated) UAVP_LOG_WARN(kModule, "Bo dem LiDAR khong phu het phien: tang sources.lidar_buffer_scans");
    const FusionResult r = fusion(frames, scans);
    live.timings.add("s4_fusion", ms_since(start));
    UAVP_LOG_INFO(kModule, "s4 (%s) %s | %d khung co mask, %d scan, %d diem, inlier %.0f%%, rms %.1f mm | %s | %.1f ms",
                  label, std::string(name(r.status)).c_str(), r.n_frames, r.n_scans, r.n_points, 100.0 * r.inlier_ratio,
                  r.rms_mm, r.pose_compensated ? "bu bang tu the FC" : "bu bang anh", live.timings.get("s4_fusion")->back());
    return r;
}

void log_fusion(const FusionResult& r) {
    UAVP_LOG_INFO(kModule, "s4 tam mat top p_obj = (%.1f, %.1f, %.1f) mm (he camera) | d = %.1f mm | dist = %.1f mm",
                  r.p_obj_mm.x, r.p_obj_mm.y, r.p_obj_mm.z, r.d_mm, r.dist_mm);
    UAVP_LOG_INFO(kModule, "s4 truc vat: x (%.3f, %.3f, %.3f) y (%.3f, %.3f, %.3f) z (%.3f, %.3f, %.3f) | nghieng %.1f do%s",
                  r.axis_x[0], r.axis_x[1], r.axis_x[2], r.axis_y[0], r.axis_y[1], r.axis_y[2], r.axis_z[0], r.axis_z[1],
                  r.axis_z[2], r.tilt_deg, r.normal_observable ? "" : " (CHUA do duoc nghieng vuong goc mat quet)");
    UAVP_LOG_INFO(kModule, "s4 W %.1f mm, L %.1f mm, yaw %.1f do | trai vuong goc mat quet %.1f mm | troi %.1f mm/s",
                  r.W_mm, r.L_mm, r.yaw_deg, r.spread_w_mm, r.drift_mm_s);
    if (r.n2_inliers > 0)
        UAVP_LOG_INFO(kModule, "s4 mat 2 song song: d %.1f mm (%d diem) | %s", r.d2_mm, r.n2_inliers,
                      r.ambiguous ? "TUONG DONG" : "it hon nhieu");
}

// Khung co mask cua track selected o pha 1 (+ tu the FC luc chup).
std::vector<MaskFrame> mask_frames(const Live& live, const Track& t) {
    std::vector<MaskFrame> out;
    for (const auto& o : t.observations)
        if (o.seg) out.push_back({*o.seg, pose_at(live, o.t_ns), o.t_ns, true});
    return out;
}

// --- Pha 2 + 3: khoa + bam ----------------------------------------------------------------------------------------
void lock_object(Live& live, LockedTracker& tracker, const Track& track, int slot, const FusionResult& r0, LockRun& run) {
    const MaskFrame& last = run.frames.back();
    const cv::Point3d p_C = std::isfinite(r0.p_obj_mm.z)
                                ? r0.p_obj_mm
                                : object_position_mm(last.obj.center_px, 1000.0 * live.cfg.D_m, live.cfg.camera);
    tracker.lock(live.pool.image(slot), last.obj, predicted_box(track, track.last_mask_frame), p_C, last.t_ns, last.pose);
    run.has_world = tracker.has_world();
    run.p_W_lock_mm = tracker.p_W_mm();
    run.ref_slot = slot;
    live.pool.retain(slot);
    if (run.has_world)
        UAVP_LOG_INFO(kModule, "KHOA vat: p_C = (%.1f, %.1f, %.1f) mm -> p_W = (%.1f, %.1f, %.1f) mm (NED EKF)%s", p_C.x,
                      p_C.y, p_C.z, run.p_W_lock_mm.x, run.p_W_lock_mm.y, run.p_W_lock_mm.z,
                      std::isfinite(r0.p_obj_mm.z) ? "" : " | chua co d LiDAR: dung D_m");
    else
        UAVP_LOG_WARN(kModule, "KHOA vat khong co vi tri FC (%s): bam bang %s, khong co p_W",
                      last.pose ? "chi co goc quay" : "khong co MAVLink",
                      last.pose ? "homography + optical flow" : "optical flow / NCC + Kalman");
}

void track_row(Live& live, int i, int64_t t_ns, const LockStep& st, bool segmented, double step_ms, double s2_ms,
               double latency_ms) {
    char row[352];
    std::snprintf(row, sizeof(row), "%d,%lld,%s,%s,%d,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.1f,%.1f,%.1f,%.2f,%.2f,%.2f\n", i,
                  static_cast<long long>(t_ns), std::string(name(st.mode)).c_str(),
                  std::string(name(st.meas.method)).c_str(), st.meas.n_good, st.meas.ncc, st.center_px.x, st.center_px.y,
                  st.pose_pred_px.x, st.pose_pred_px.y, st.kf_sigma_px, segmented ? 1 : 0, st.p_C_pose_mm.x,
                  st.p_C_pose_mm.y, st.p_C_pose_mm.z, step_ms, s2_ms, latency_ms);
    live.track_csv += row;
}

// Trang thai pha bam (moi khung): tam Kalman, ROI, che do tu the, flow / NCC, thoi gian, do tre.
std::string lock_status(int i, const LockStep& st, const LockedTracker& tr, bool segmented, double step_ms, double s2_ms,
                        double latency_ms, int segments) {
    JsonWriter w;
    w.begin_object().key("phase").value("lock").key("frame").value(i).key("pose_mode").value(name(st.mode));
    w.key("method").value(name(st.meas.method)).key("n_good").value(st.meas.n_good).key("center_px").begin_array();
    w.num(st.center_px.x).num(st.center_px.y).end_array().key("pose_pred_px").begin_array();
    w.num(st.pose_pred_px.x).num(st.pose_pred_px.y).end_array().key("roi");
    box_json(w, tr.roi_box());
    w.key("segmented").value(segmented).key("segments").value(segments).key("step_ms").num(step_ms);
    w.key("s2_ms").num(s2_ms).key("latency_ms").num(latency_ms).key("object").begin_object();
    mask_json(w, tr.object());  // mask moi (segment) hoac mask cu da doi theo flow / tu the
    w.end_object().end_object();
    return w.str();
}

// Khung co vong LiDAR moi: s2 tren ROI Kalman -> mask moi cho s4 va dat lai flow. true neu co mask.
bool segment_now(Live& live, Segmenter& seg, LockedTracker& tracker, int slot, int64_t t_ns,
                 const std::optional<CamPose>& pose, LockRun& run, double& s2_ms) {
    const int64_t start = mono_ns();
    SegObject obj;
    const bool ok = seg(live.pool.image(slot), tracker.roi_box(), obj);
    s2_ms = ms_since(start);
    live.timings.add("s2_lock", s2_ms);
    if (!ok) return false;
    tracker.on_segment(live.pool.image(slot), obj);
    run.frames.push_back({obj, pose, t_ns, true});
    ++run.segments;
    live.pool.release(run.ref_slot);  // anh tham chieu = khung co mask moi nhat
    live.pool.retain(slot);
    run.ref_slot = slot;
    return true;
}

bool converged(Live& live, Fusion& fusion, const LockRun& run) {
    if (!live.lock.stop_when_converged || run.segments < live.lock.min_segments) return false;
    const FusionResult r = run_fusion(live, fusion, run.frames, "kiem tra hoi tu");
    return r.status == FusionStatus::kOk && r.normal_observable && !r.ambiguous;
}

void run_lock(Live& live, Segmenter& seg, Fusion& fusion, LockedTracker& tracker, LockRun& run) {
    live.monitor.set_phase(Phase::kLock);
    const int64_t start_all = mono_ns();
    std::size_t lidar_seen = live.lidar.count();
    for (int i = 0; ms_since(start_all) < 1000.0 * live.lock.max_duration_s; ++i) {
        const int slot = live.pool.acquire();
        FrameMeta meta;
        if (!live.camera.next_frame(live.pool.image(slot), meta, live.src.frame_timeout_s))
            throw std::runtime_error("Khong co khung moi sau frame_timeout_s");
        const int64_t start = mono_ns();
        const std::optional<CamPose> pose = pose_at(live, meta.t_ns);
        const LockStep st = tracker.step(live.pool.image(slot), meta.t_ns, pose);
        const double step_ms = ms_since(start);
        live.timings.add("lock_step", step_ms);
        ++run.methods[static_cast<std::size_t>(st.meas.method)];
        run.last_mode = st.mode;
        double s2_ms = 0.0;
        bool segmented = false, check = false;
        if (const std::size_t n = live.lidar.count(); n > lidar_seen) {  // nhip LiDAR (10 Hz)
            lidar_seen = n;
            segmented = segment_now(live, seg, tracker, slot, meta.t_ns, pose, run, s2_ms);
            check = segmented;
        }
        if (!segmented && live.lock.warped_masks_for_lidar) run.frames.push_back({tracker.object(), pose, meta.t_ns, false});
        const double latency_ms = static_cast<double>(mono_ns() - meta.t_ns) / 1e6;  // SensorTimestamp -> tam ROI
        live.timings.add("latency_lock", latency_ms);
        track_row(live, i, meta.t_ns, st, segmented, step_ms, s2_ms, latency_ms);
        remote_frame(live, live.pool.image(slot), meta.t_ns,
                     lock_status(i, st, tracker, segmented, step_ms, s2_ms, latency_ms, run.segments));
        char line[160];
        std::snprintf(line, sizeof(line), "BAM %s/%s | tam (%.0f, %.0f) | segment %d | q: dung", std::string(name(st.mode)).c_str(),
                      std::string(name(st.meas.method)).c_str(), st.center_px.x, st.center_px.y, run.segments);
        const bool stop = stop_requested(live, live.pool.image(slot), {}, line);
        live.pool.release(slot);
        run.frames_tracked = i + 1;
        if (stop || (check && converged(live, fusion, run))) break;
    }
    run.duration_s = ms_since(start_all) / 1000.0;
    UAVP_LOG_INFO(kModule, "Bam: %d khung, %d lan segment (theo LiDAR), %.2f s | flow %d, NCC %d, khong do %d | tu the: %s",
                  run.frames_tracked, run.segments, run.duration_s, run.methods[1], run.methods[2], run.methods[0],
                  std::string(name(run.last_mode)).c_str());
}

// --- Ket qua --------------------------------------------------------------------------------------------------------
std::vector<std::string> run_aruco(JsonWriter& w, const cv::Mat& image, const SegObject& obj, const FusionResult& r,
                                   const ArucoConfig& aruco, const CameraModel& camera) {
    w.key("aruco");
    if (!aruco.check_aruco) {
        UAVP_LOG_INFO(kModule, "ArUco: check_aruco = false (vat khong co ArUco), bo qua");
        w.null();
        return {};
    }
    ArucoPose pose;
    cv::Point3d p;
    if (!check_aruco(image, obj.center_px, aruco, camera, pose, p)) {
        UAVP_LOG_WARN(kModule, "ArUco: khong thay marker %s id %d tren anh cua s4", aruco.dictionary.c_str(), aruco.marker_id);
        w.begin_object().key("found").value(false).end_object();
        return {"ArUco: khong thay marker"};
    }
    write_aruco_summary(w, pose, p, r);
    const double dist_a = std::sqrt(p.dot(p));
    UAVP_LOG_INFO(kModule, "ArUco id %d (canh %.1f mm): tam marker (%.1f, %.1f, %.1f) mm, nghieng %.1f do, chieu lai %.2f px",
                  pose.marker_id, aruco.side_mm, pose.t_CM_mm[0], pose.t_CM_mm[1], pose.t_CM_mm[2], pose.tilt_deg,
                  pose.reproj_px);
    UAVP_LOG_INFO(kModule, "Sai so s4 - ArUco: d %.2f mm (%.2f%%) | dist %.2f mm | nghieng s4 %.1f do - ArUco %.1f do",
                  r.d_mm - p.z, 100.0 * (r.d_mm - p.z) / p.z, r.dist_mm - dist_a, r.tilt_deg, pose.tilt_deg);
    char l1[96], l2[160];
    std::snprintf(l1, sizeof(l1), "ArUco: d = %.0f mm | dist = %.0f mm | nghieng %.1f do", p.z, dist_a, pose.tilt_deg);
    std::snprintf(l2, sizeof(l2), "Sai so s4 - ArUco: d %.2f mm (%.2f%%) | dist %.2f mm", r.d_mm - p.z,
                  100.0 * (r.d_mm - p.z) / p.z, r.dist_mm - dist_a);
    return {l1, l2};
}

std::vector<std::string> screen_lines(const Track& t, const SegObject& obj, const FusionResult& r, const LockRun& run,
                                      const cv::Point3d& p_W, const Config& cfg) {
    char b[6][192];
    std::snprintf(b[0], sizeof(b[0]), "selected #%d %s | tam (%.0f, %.0f) px", t.track_id,
                  cfg.detector.classes.at(t.class_id).c_str(), obj.center_px.x, obj.center_px.y);
    std::snprintf(b[1], sizeof(b[1]), "s4 %s: p_obj = (%.0f, %.0f, %.0f) mm | nghieng %.1f do%s",
                  std::string(name(r.status)).c_str(), r.p_obj_mm.x, r.p_obj_mm.y, r.p_obj_mm.z, r.tilt_deg,
                  r.normal_observable ? "" : " (1 huong)");
    std::snprintf(b[2], sizeof(b[2]), "d = %.0f mm | dist = %.0f mm | W %.0f | L %.0f mm | yaw %.0f do", r.d_mm,
                  r.dist_mm, r.W_mm, r.L_mm, r.yaw_deg);
    std::snprintf(b[3], sizeof(b[3]), "LiDAR: %d diem, %d khung co mask, %d scan, inlier %.0f%%", r.n_points, r.n_frames,
                  r.n_scans, 100.0 * r.inlier_ratio);
    std::snprintf(b[4], sizeof(b[4]), "p_W = (%.0f, %.0f, %.0f) mm (NED)", p_W.x, p_W.y, p_W.z);
    std::snprintf(b[5], sizeof(b[5]), "bam %d khung, %d segment, %.1f s", run.frames_tracked, run.segments, run.duration_s);
    return {b[0], b[1], b[2], b[3], b[4], b[5]};
}

void write_lock_summary(JsonWriter& w, const LockRun& run, const cv::Point3d& p_W_final, bool mavlink) {
    const auto p3 = [&](const cv::Point3d& p) {
        if (!std::isfinite(p.x)) w.null();
        else w.begin_array().num(p.x).num(p.y).num(p.z).end_array();
    };
    w.begin_object().key("mavlink").value(mavlink).key("has_world").value(run.has_world);
    w.key("pose_mode").value(name(run.last_mode)).key("p_W_lock_mm");
    p3(run.has_world ? run.p_W_lock_mm : cv::Point3d(kNaN, kNaN, kNaN));
    w.key("p_W_mm");
    p3(p_W_final);
    w.key("frames_tracked").value(run.frames_tracked).key("segments").value(run.segments);
    w.key("duration_s").num(run.duration_s).key("methods").begin_object();
    for (std::size_t k = 0; k < run.methods.size(); ++k) w.key(name(static_cast<FlowMethod>(k))).value(run.methods[k]);
    w.end_object().key("track_file").value("track.csv").end_object();
}

// frame.jpg, selected_mask.png, lidar_points.csv, final.jpg (+ 3 truc); display.show: hien man hinh cuoi.
cv::Mat save_and_show(const std::filesystem::path& dir, const cv::Mat& image, const SegObject& obj, const FusionResult& r,
                      const std::vector<std::string>& lines, const Config& cfg, bool show) {
    std::vector<cv::Point2f> uv;
    project_px(r.points_C_mm, cfg.camera, uv);
    write_image_rgb(dir / "frame.jpg", image, kJpegQuality);
    write_mask_png(dir / kSelectedMaskFile, obj.mask, image.rows, image.cols);
    write_points_csv(dir / "lidar_points.csv", r, uv);
    cv::Mat object_img = draw_object(image, obj, r);  // anh cuoi: mask + tam vat, khong chu / LiDAR
    draw_pose_axes(object_img, r, cfg.camera, kAxisDrawMm);
    write_image_rgb(dir / "object.jpg", object_img, kJpegQuality);
    cv::Mat final_img = draw_final(image, obj, uv, r.inliers, lines);
    draw_pose_axes(final_img, r, cfg.camera, kAxisDrawMm);
    write_image_rgb(dir / "final.jpg", final_img, kJpegQuality);
    UAVP_LOG_INFO(kModule, "Anh cuoi: %s (mask + tam vat), %s (them LiDAR + so do)", (dir / "object.jpg").c_str(),
                  (dir / "final.jpg").c_str());
    if (!show) return final_img;
    cv::Mat bgr;
    cv::cvtColor(final_img, bgr, cv::COLOR_RGB2BGR);  // hien thi BGR
    cv::imshow(kWindow, bgr);
    UAVP_LOG_INFO(kModule, "Man hinh cuoi: bam phim bat ky de dong");
    cv::waitKey(0);
    cv::destroyAllWindows();
    return final_img;
}

// Khong co vat selected: frame.jpg (khung cuoi) + final.jpg (mask cuoi moi track da xet + trang thai, W, ly do loai).
cv::Mat save_unselected(const std::filesystem::path& dir, const Scanned& sc, const Config& cfg) {
    std::vector<std::string> lines{"KHONG CO VAT SELECTED sau " + std::to_string(sc.frames) + " khung | D = " +
                                   std::to_string(cfg.D_m).substr(0, 5) + " m, W toi da " +
                                   std::to_string(static_cast<int>(cfg.decide.w_max_mm)) + " mm"};
    for (const Track& t : sc.session.tracks) {
        char l[160];
        const std::string_view reason = name(t.reason);
        std::snprintf(l, sizeof(l), "#%d %s %s | W %.1f mm | conf %.2f | %s", t.track_id,
                      cfg.detector.classes.at(t.class_id).c_str(), std::string(name(t.state)).c_str(), median_W_mm(t),
                      mean_conf(t), reason.empty() ? "-" : std::string(reason).c_str());
        lines.emplace_back(l);
    }
    for (const auto& l : lines) UAVP_LOG_INFO(kModule, "  %s", l.c_str());
    write_image_rgb(dir / "frame.jpg", sc.last_image, kJpegQuality);
    const cv::Mat img = draw_live(sc.last_image, {}, cfg.camera.center_px(), lines, sc.session.tracks);
    write_image_rgb(dir / "final.jpg", img, kJpegQuality);
    UAVP_LOG_INFO(kModule, "Anh cuoi (khong co vat selected): %s", (dir / "final.jpg").c_str());
    return img;
}

void check_lidar(const ScanBuffer& lidar, const SourcesConfig& src) {
    if (lidar.wait_first_scan(src.connect_timeout_s)) {
        UAVP_LOG_INFO(kModule, "LD06: da nhan vong quet dau tien");
        return;
    }
    const std::string hint = lidar.bytes_received() == 0
                                 ? "khong nhan duoc byte nao: LD06 chua co nguon 5 V / khong quay, sai cong"
                                 : "nhan " + std::to_string(lidar.bytes_received()) + " byte, " +
                                       std::to_string(lidar.crc_errors()) + " goi loi CRC: kiem tra baud (230400)";
    throw std::runtime_error("LD06 " + src.lidar_port + ": khong co vong quet sau connect_timeout_s (" + hint + ")");
}

// Mo nguon MAVLink; shm chua co (bridge vua khoi dong cung Pi, dang restart) -> thu lai toi timeout_s roi bao loi ro.
std::unique_ptr<IFcSource> open_fc_waiting(const MavlinkConfig& mav, double timeout_s) {
    const int64_t start = mono_ns();
    for (int attempt = 0;; ++attempt) {
        try {
            return open_fc_source(mav);
        } catch (const std::exception& e) {
            if (static_cast<double>(mono_ns() - start) / 1e9 >= timeout_s)
                throw std::runtime_error(std::string(e.what()) +
                                         " | sua: bash tools/install_fc_bridge.sh (chua cai) hoac sudo systemctl restart "
                                         "uavp-fc-bridge; xem journalctl -u uavp-fc-bridge -n 30");
            if (attempt == 0) UAVP_LOG_WARN(kModule, "Chua co du lieu Pixhawk (%s): cho bridge toi %.0f s", e.what(), timeout_s);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

// MAVLink: cho ATTITUDE dau tien; khong co -> chay khong tu the FC (canh bao, khong dung chuong trinh).
const FcPoseBuffer* check_mavlink(const IFcSource* fc, const SourcesConfig& src) {
    if (fc == nullptr) {
        UAVP_LOG_WARN(kModule, "MAVLink tat (sources.mavlink.enabled = false): khong co p_W, bam bang anh");
        return nullptr;
    }
    const int64_t start = mono_ns();
    while (fc->poses().n_attitude() == 0 && static_cast<double>(mono_ns() - start) / 1e9 < src.connect_timeout_s)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (fc->poses().n_attitude() > 0) {
        UAVP_LOG_INFO(kModule, "MAVLink: da nhan ATTITUDE tu Pixhawk (%s)", fc->describe().c_str());
        return &fc->poses();
    }
    UAVP_LOG_WARN(kModule, "MAVLink (%s): khong co ATTITUDE sau connect_timeout_s (%zu byte, %zu loi CRC): kiem tra "
                  "uavp_fc_bridge, SERIAL2_PROTOCOL / baud / SR2_EXTRA1. Chay khong co tu the FC", fc->describe().c_str(),
                  fc->bytes_received(), fc->crc_errors());
    return nullptr;
}

// Khung dau: net khop file K, SensorTimestamp (da doi ve monotonic) cung dong ho voi LiDAR.
void check_camera(IFrameSource& camera, FramePool& pool, const CaptureConfig& cap, const SourcesConfig& src) {
    const int slot = pool.acquire();
    FrameMeta meta;
    const bool ok = camera.next_frame(pool.image(slot), meta, src.frame_timeout_s);
    pool.release(slot);
    if (!ok) throw std::runtime_error("Khong co khung dau tien tu camera");
    const double age_s = static_cast<double>(mono_ns() - meta.t_ns) / 1e9;
    UAVP_LOG_INFO(kModule, "Camera khung dau: LensPosition %.3f, phoi sang %.0f us, gain %.2f, tre tu SensorTimestamp %.1f ms",
                  meta.lens_position, meta.exposure_us, meta.analogue_gain, age_s * 1e3);
    std::string errors;
    if (meta.lens_position < 0.0 || std::abs(meta.lens_position - cap.lens_position) > src.lens_position_tol)
        errors += "net LensPosition " + std::to_string(meta.lens_position) + " != " + std::to_string(cap.lens_position) + "; ";
    if (age_s < 0.0 || age_s > src.clock_skew_max_s)
        errors += "luc nhan - SensorTimestamp = " + std::to_string(age_s) + " s: khac dong ho; ";
    if (!errors.empty()) throw std::runtime_error("Camera khong khop file K / dong ho LiDAR: " + errors);
}

struct Models {
    std::unique_ptr<Detector> detector;
    std::unique_ptr<Segmenter> segmenter;
    std::unique_ptr<Fusion> fusion;
};

Models load_models(const Config& cfg) {
    const int64_t start = mono_ns();
    Models m{std::make_unique<Detector>(cfg.detector, std::make_unique<NcnnModel>(cfg.detector.model)),
             std::make_unique<Segmenter>(cfg.segmenter, std::make_unique<NcnnModel>(cfg.segmenter.model)),
             std::make_unique<Fusion>(cfg.fusion, cfg.camera)};
    pin_ncnn_threads(cfg.cpu.cores);
    UAVP_LOG_INFO(kModule, "Da nap model s1 %s (input %dx%d, %s), s2 %s | NCNN %d / %d luong (%.0f ms)",
                  cfg.detector.model.dir.c_str(), cfg.detector.input_w_px, cfg.detector.input_h_px,
                  cfg.detector.mode == DetectMode::kFull ? "ca anh" : "tile", cfg.segmenter.model.dir.c_str(),
                  cfg.detector.model.num_threads, cfg.segmenter.model.num_threads, ms_since(start));
    return m;
}

// Pha 2 -> 4 cho vat selected: s4 lan dau, khoa, bam, s4 cuoi. Ghi muc selected / fusion / lock / aruco vao w.
struct Final {
    FusionResult r;
    std::vector<std::string> lines;
    LockRun run;
};

// Su kien s4 (fc_events.jsonl): ket qua fusion + p_W (neu co) cung trang thai Pixhawk luc do.
void fusion_event(const Live& live, std::string_view event, const FusionResult& r, const cv::Point3d& p_W) {
    fc_event(live, event, [&](JsonWriter& w) {
        w.key("status").value(name(r.status)).key("d_mm").num(r.d_mm).key("p_obj_mm").begin_array();
        w.num(r.p_obj_mm.x).num(r.p_obj_mm.y).num(r.p_obj_mm.z).end_array().key("W_mm").num(r.W_mm).key("L_mm").num(r.L_mm);
        w.key("tilt_deg").num(r.tilt_deg).key("n_points").value(r.n_points).key("inlier_ratio").num(r.inlier_ratio);
        w.key("rms_mm").num(r.rms_mm).key("ambiguous").value(r.ambiguous).key("D_m").value(live.cfg.D_m);
        w.key("p_W_mm");
        if (std::isfinite(p_W.x)) w.begin_array().num(p_W.x).num(p_W.y).num(p_W.z).end_array();
        else w.null();
    });
}

Final lock_and_fuse(Live& live, Models& models, Scanned& sc, const Track& track, JsonWriter& w, const ArucoConfig& aruco) {
    Final f;
    LockRun& run = f.run;
    const int slot = sc.slot_of_track.at(static_cast<std::size_t>(track.track_id));
    write_track_summary(w, track, live.cfg);
    run.frames = mask_frames(live, track);
    if (run.frames.empty()) throw std::logic_error("Track selected khong co mask");  // OK can n_min lan do co mask
    live.monitor.set_phase(Phase::kS4);
    const FusionResult r0 = run_fusion(live, *models.fusion, run.frames, "lan dau, pha chon vat");
    fusion_event(live, "s4_first", r0, cv::Point3d(kNaN, kNaN, kNaN));
    LockedTracker tracker(live.lock, live.cfg.camera, live.body_camera);
    lock_object(live, tracker, track, slot, r0, run);
    fc_event(live, "lock", [&](JsonWriter& e) {
        e.key("track_id").value(track.track_id).key("has_world").value(run.has_world).key("p_W_lock_mm");
        if (run.has_world) e.begin_array().num(run.p_W_lock_mm.x).num(run.p_W_lock_mm.y).num(run.p_W_lock_mm.z).end_array();
        else e.null();
    });
    run_lock(live, *models.segmenter, *models.fusion, tracker, run);
    fc_event(live, "lock_end", [&](JsonWriter& e) {
        e.key("frames_tracked").value(run.frames_tracked).key("segments").value(run.segments);
        e.key("duration_s").num(run.duration_s).key("pose_mode").value(name(run.last_mode));
        e.key("flow").value(run.methods[1]).key("ncc").value(run.methods[2]).key("no_measure").value(run.methods[0]);
    });
    live.monitor.set_phase(Phase::kS4);
    f.r = run_fusion(live, *models.fusion, run.frames, "cuoi");
    log_fusion(f.r);
    const MaskFrame& ref = *std::max_element(run.frames.begin(), run.frames.end(),
                                             [](const MaskFrame& a, const MaskFrame& b) { return a.t_ns < b.t_ns; });
    const cv::Point3d p_W = ref.pose && ref.pose->has_position && std::isfinite(f.r.p_obj_mm.z)
                                ? to_world(*ref.pose, f.r.p_obj_mm)
                                : cv::Point3d(kNaN, kNaN, kNaN);
    fusion_event(live, "s4_final", f.r, p_W);
    if (std::isfinite(p_W.x))
        UAVP_LOG_INFO(kModule, "Tam vat he the gioi p_W = (%.1f, %.1f, %.1f) mm (NED EKF; luc khoa (%.1f, %.1f, %.1f))",
                      p_W.x, p_W.y, p_W.z, run.p_W_lock_mm.x, run.p_W_lock_mm.y, run.p_W_lock_mm.z);
    w.key("fusion");
    write_fusion_summary(w, f.r);
    w.key("lock");
    write_lock_summary(w, run, p_W, live.fc != nullptr);
    f.lines = screen_lines(track, ref.obj, f.r, run, p_W, live.cfg);
    for (auto& l : run_aruco(w, live.pool.image(run.ref_slot), ref.obj, f.r, aruco, live.cfg.camera))
        f.lines.push_back(std::move(l));
    return f;
}

// uavp_real_time_perception: s4 mot lan tren cac khung co mask cua pha chon vat (khong khoa / bam, khong Pixhawk).
Final fuse_only(Live& live, Models& models, Scanned& sc, const Track& track, JsonWriter& w, const ArucoConfig& aruco) {
    Final f;
    LockRun& run = f.run;
    write_track_summary(w, track, live.cfg);
    run.frames = mask_frames(live, track);
    if (run.frames.empty()) throw std::logic_error("Track selected khong co mask");
    run.ref_slot = sc.slot_of_track.at(static_cast<std::size_t>(track.track_id));
    live.monitor.set_phase(Phase::kS4);
    f.r = run_fusion(live, *models.fusion, run.frames, "pha chon vat");
    log_fusion(f.r);
    fusion_event(live, "s4_final", f.r, cv::Point3d(kNaN, kNaN, kNaN));
    const MaskFrame& ref = run.frames.back();
    w.key("fusion");
    write_fusion_summary(w, f.r);
    w.key("lock").null();
    f.lines = screen_lines(track, ref.obj, f.r, run, cv::Point3d(kNaN, kNaN, kNaN), live.cfg);
    for (auto& l : run_aruco(w, live.pool.image(run.ref_slot), ref.obj, f.r, aruco, live.cfg.camera))
        f.lines.push_back(std::move(l));
    return f;
}

void finish(Live& live, Models& models, Scanned& sc, const ArucoConfig& aruco,
            const std::vector<std::pair<std::string, std::string>>& device, const std::filesystem::path& dir) {
    const Track* track = selected_track(sc.session.tracks, live.cfg.camera.center_px());
    JsonWriter w;
    w.begin_object().key("frames").value(sc.frames).key("selected");
    Final f;
    if (track == nullptr) {
        UAVP_LOG_INFO(kModule, "s3: sau %d khung khong co vat selected -> khong khoa, khong s4", sc.frames);
        w.null().key("fusion").null().key("lock").null().key("aruco").null();
    } else {
        UAVP_LOG_INFO(kModule, "s3 selected #%d %s: tam (%.1f, %.1f) px, W %.2f mm", track->track_id,
                      live.cfg.detector.classes.at(track->class_id).c_str(), last_seg(*track)->center_px.x,
                      last_seg(*track)->center_px.y, median_W_mm(*track));
        f = live.mode == AppMode::kFull ? lock_and_fuse(live, models, sc, *track, w, aruco)
                                         : fuse_only(live, models, sc, *track, w, aruco);
    }
    fc_event(live, "finish", [&](JsonWriter& e) {
        e.key("frames").value(sc.frames).key("selected").value(track != nullptr);
        if (track != nullptr) e.key("track_id").value(track->track_id).key("W_mm").num(median_W_mm(*track));
    });
    live.monitor.set_phase(Phase::kSave);
    live.monitor.stop();
    UAVP_LOG_INFO(kModule, "LD06: %zu goi loi CRC", live.lidar.crc_errors());
    live.timings.log_summary();
    const double fps = sc.s1_s3_ms > 0.0 ? 1000.0 * sc.frames / sc.s1_s3_ms : 0.0;
    w.key("session").begin_object().key("s1_s3_ms").num(sc.s1_s3_ms).key("frames").value(sc.frames);
    w.key("lock_s").num(f.run.duration_s).key("fps_s1_s3").num(fps).end_object().key("timing_ms");
    live.timings.write_json(w);
    w.key("device");
    write_device_info(w, device);
    w.key("hw_usage");
    live.monitor.write_summary(w);
    w.end_object();
    write_text(dir / "result.json", w.str());
    write_text(dir / "timing.csv", live.timing_csv);
    write_text(dir / "track.csv", live.track_csv);
    live.monitor.write_csv(dir / "hw_usage.csv");
    if (track != nullptr) {
        const MaskFrame& ref = *std::max_element(f.run.frames.begin(), f.run.frames.end(),
                                                 [](const MaskFrame& a, const MaskFrame& b) { return a.t_ns < b.t_ns; });
        const cv::Mat final_img =
            save_and_show(dir, live.pool.image(f.run.ref_slot), ref.obj, f.r, f.lines, live.cfg, live.display.show);
        remote_frame(live, final_img, ref.t_ns, "{\"phase\": \"final\"}", true);
    } else if (!sc.last_image.empty()) {
        const cv::Mat img = save_unselected(dir, sc, live.cfg);
        remote_frame(live, img, sc.last_t_ns, "{\"phase\": \"final\"}", true);
    }
    if (live.remote != nullptr) {
        live.remote->send_result(w.str());
        std::this_thread::sleep_for(std::chrono::seconds(2));  // gui het anh cuoi + result.json truoc khi thoat
    }
}

int run_main(int argc, char** argv, AppMode mode) {
    const bool full = mode == AppMode::kFull;
    std::ios::sync_with_stdio(false);
    const Args args = parse_args(argc, argv);
    const YAML::Node raw = load_raw(args.cfg, args.overrides);
    Config cfg = config_from_raw(raw);  // D_m dat lai luc bat dau (distance)
    const std::string cores = pin_process_to_cores(cfg.cpu.cores);  // TRUOC moi luong (log, LD06, camera, NCNN)
    const StartConfig start = start_config(raw["start"]);
    const DistanceConfig distance = distance_config(raw["distance"]);
    const SourcesConfig src = sources_config(raw["sources"]);
    const MavlinkConfig mav = mavlink_config(raw["sources"]["mavlink"]);
    const LockConfig lock = lock_config(raw["lock_track"]);
    const BodyCamera body_camera = load_body_camera(lock.calib_T_BC);
    const CaptureConfig cap = load_capture(raw["calib_K"].as<std::string>());
    DisplayConfig display = display_config(raw["display"]);
    const MonitorConfig mon = monitor_config(raw["monitor"]);
    const ArucoConfig aruco = aruco_config(raw["aruco"]);
    const RemoteConfig rcfg{raw["remote"]["port"].as<int>(), raw["remote"]["preview_width_px"].as<int>(),
                            raw["remote"]["jpeg_quality"].as<int>(), raw["remote"]["max_fps"].as<double>(),
                            raw["remote"]["wait_fps"].as<double>(), raw["remote"]["enabled"].as<bool>()};
    auto logger = std::make_unique<Logger>(Level::kInfo);  // ring buffer lon: tren heap
    const LoggerGuard logger_guard(logger.get());
    UAVP_LOG_INFO(kModule, "CPU: chuong trinh chay tren %s", cores.c_str());
    if (display.show && std::getenv("DISPLAY") == nullptr && std::getenv("WAYLAND_DISPLAY") == nullptr) {
        display.show = false;  // SSH khong man hinh: imshow se loi GTK -> tat cua so, xem qua remote_viewer
        UAVP_LOG_WARN(kModule, "display.show = true nhung khong co man hinh (DISPLAY trong): tat cua so xem truoc");
    }
    const std::vector<std::pair<std::string, std::string>> device{
        {"ncnn", ncnn_version()}, {"opencv", CV_VERSION}, {"cpu_cores", cores},
        {"s1_model", cfg.detector.model.dir.string()}, {"s1_mode", cfg.detector.mode == DetectMode::kTile ? "tile" : "full"},
        {"s1_input", std::to_string(cfg.detector.input_w_px) + "x" + std::to_string(cfg.detector.input_h_px)},
        {"s1_threads", std::to_string(cfg.detector.model.num_threads)}, {"s2_model", cfg.segmenter.model.dir.string()},
        {"s2_threads", std::to_string(cfg.segmenter.model.num_threads)},
        {"s1_policy", "bat dau / mat vat"}, {"targets", std::to_string(kTargets)},
        {"app", full ? "real_time" : "real_time_perception"}, {"distance_source", full ? distance.source : "fixed"},
        {"display_show", display.show ? "true" : "false"}};
    Models models = load_models(cfg);  // loi model -> dung ngay, chua mo camera / LiDAR
    Ld06Reader lidar(src.lidar_port, src.lidar_baudrate, static_cast<std::size_t>(src.lidar_buffer_scans),
                     src.lidar_extra_latency_ms);
    // perception: khong dung Pixhawk (chi do thoi gian / sai so). full: bat buoc (lenh bat dau, tu the, do cao)
    const std::unique_ptr<IFcSource> mavlink = full && mav.enabled ? open_fc_waiting(mav, src.connect_timeout_s) : nullptr;
    PiCamera camera(cap, src.camera_buffer_count, src.sensor_bit_depth);
    FramePool pool(kTargets + 4, {cap.width_px, cap.height_px});
    check_lidar(lidar.buffer(), src);
    const FcPoseBuffer* fc = check_mavlink(mavlink.get(), src);
    if (full && fc == nullptr)
        throw std::runtime_error("uavp_real_time can Pixhawk (MAVLink): kiem tra uavp_fc_bridge, sources.mavlink.enabled");
    check_camera(camera, pool, cap, src);
    const std::filesystem::path dir = make_run_dir(cfg.runs_root, full ? "real_time" : "real_time_perception");
    logger->open_file(dir / "run.log");
    write_config(dir, raw, args.overrides);
    HwMonitor monitor(mon.period_s, mon.enabled);
    const TerminalKeys keys;
    const std::unique_ptr<RemoteView> remote = rcfg.enabled ? std::make_unique<RemoteView>(rcfg) : nullptr;
    const DistanceConfig dist_used = full ? distance : DistanceConfig{"fixed", distance.fixed_m, 0.0};
    Live live{camera, lidar.buffer(), fc, cfg, mode, start, dist_used, src, lock, body_camera, display, keys, monitor,
              pool, remote.get(), rcfg.wait_fps, rcfg.max_fps, Timings{}, 0, 0, kTimingHeader, kTrackHeader,
              fc != nullptr ? dir / "fc_events.jsonl" : std::filesystem::path{}};
    if (fc != nullptr) UAVP_LOG_INFO(kModule, "Log Pixhawk theo quyet dinh: %s", live.fc_log.c_str());
    const StartInfo info = wait_for_start(live);
    if (!info.go) {
        UAVP_LOG_INFO(kModule, "Thoat truoc khi quet");
        return 0;
    }
    cfg.D_m = info.D_m;  // s3: doi kich thuoc mask (px) -> mm de loai vat qua to cho tay gap
    UAVP_LOG_INFO(kModule, "s3: khoang cach camera - san D = %.3f m (%s)", cfg.D_m, info.distance_source.c_str());
    Scanned sc;
    run_until_selected(live, *models.detector, *models.segmenter, sc);
    finish(live, models, sc, aruco, device, dir);
    UAVP_LOG_INFO(kModule, "Ket qua: %s", dir.c_str());
    return 0;
}

}  // namespace

int run_live(int argc, char** argv, AppMode mode) {
    try {
        return run_main(argc, argv, mode);
    } catch (const std::exception& e) {  // loi luc khoi dong (config, model, camera, LiDAR, Pixhawk)
        std::cerr << "LOI: " << e.what() << '\n';
        return 1;
    }
}
