// uavp_fc_check: kiem tra Pixhawk -> Pi (khong can camera / model). Doc giong het uavp_real_time (shm cua
// uavp_fc_bridge, hoac serial truc tiep), in moi 1 s roi ket luan DAT / CANH BAO / LOI tung muc.
//   ./build/uavp_fc_check --cfg configs/run/real_time.yaml                      # 15 s, doc RAM (bridge dang chay)
//   ./build/uavp_fc_check --cfg configs/run/real_time.yaml --seconds 30         # trong luc do: gcs_command.py start / stop
//   sudo systemctl stop uavp-fc-bridge
//   ./build/uavp_fc_check --cfg configs/run/real_time.yaml --set sources.mavlink.source=serial   # mo thang /dev/ttyAMA0
// Moi giay: tan so tung loai tin, HEARTBEAT (arm, che do), landed, tu the, vi tri / van toc NED, do cao optical flow
// (ground_distance, quality) / rangefinder, lenh tu may tinh, dieu kien bat dau cua uavp_real_time (muc start).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "uavp/common/config.hpp"
#include "uavp/pipeline/start_gate.hpp"
#include "uavp/sources/mavlink.hpp"

using namespace uavp;

namespace {

constexpr double kDeg = 180.0 / 3.14159265358979323846;
constexpr double kMinAttitudeHz = 10.0;  // tu the noi suy cho moi khung camera
constexpr double kMaxAttitudeAgeS = 0.3;

struct Args {
    std::string cfg;
    int seconds{15};
    std::vector<std::string> overrides;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--cfg" && i + 1 < argc) a.cfg = argv[++i];
        else if (k == "--seconds" && i + 1 < argc) a.seconds = std::stoi(argv[++i]);
        else if (k == "--set")
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.overrides.emplace_back(argv[++i]);
    }
    if (a.cfg.empty() || a.seconds < 1)
        throw std::invalid_argument("Dung: uavp_fc_check --cfg configs/run/real_time.yaml [--seconds 15] [--set k=v ...]");
    return a;
}

const char* msg_name(uint32_t id) {
    switch (id) {
        case kMsgHeartbeat: return "HEARTBEAT";
        case kMsgAttitude: return "ATTITUDE";
        case kMsgLocalPositionNed: return "LOCAL_POSITION_NED";
        case kMsgDistanceSensor: return "DISTANCE_SENSOR";
        case kMsgRangefinder: return "RANGEFINDER";
        case kMsgCommandLong: return "COMMAND_LONG";
        case kMsgCommandAck: return "COMMAND_ACK";
        case kMsgVfrHud: return "VFR_HUD";
        case kMsgOpticalFlow: return "OPTICAL_FLOW";
        case kMsgExtendedSysState: return "EXTENDED_SYS_STATE";
        default: return "";
    }
}

const char* copter_mode(uint32_t m) {
    switch (m) {
        case 0: return "STABILIZE";
        case 2: return "ALT_HOLD";
        case 3: return "AUTO";
        case 4: return "GUIDED";
        case 5: return "LOITER";
        case 6: return "RTL";
        case 9: return "LAND";
        case 16: return "POSHOLD";
        default: return "?";
    }
}

const char* landed_name(uint8_t s) {
    switch (s) {
        case 1: return "duoi dat";
        case 2: return "DANG BAY";
        case 3: return "cat canh";
        case 4: return "ha canh";
        default: return "khong ro";
    }
}

double age_s(int64_t t, int64_t now) { return t > 0 ? static_cast<double>(now - t) / 1e9 : kNaN; }

uint64_t count_of(const std::vector<std::pair<uint32_t, uint64_t>>& c, uint32_t id) {
    for (const auto& [k, n] : c)
        if (k == id) return n;
    return 0;
}

struct Verdict {
    int fails{0}, warns{0};
    void line(const char* level, const std::string& item, const std::string& detail) {
        if (std::strcmp(level, "LOI") == 0) ++fails;
        if (std::strcmp(level, "CANH BAO") == 0) ++warns;
        std::printf("  [%-8s] %-34s %s\n", level, item.c_str(), detail.c_str());
    }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const YAML::Node raw = load_raw(args.cfg, args.overrides);
        const MavlinkConfig mav = mavlink_config(raw["sources"]["mavlink"]);
        const StartConfig start = start_config(raw["start"]);
        const DistanceConfig distance = distance_config(raw["distance"]);
        std::unique_ptr<IFcSource> fc;
        try {
            fc = open_fc_source(mav);
        } catch (const std::exception& e) {
            std::cerr << "LOI mo nguon MAVLink: " << e.what()
                      << "\n  shm: kiem tra 'systemctl status uavp-fc-bridge' (bridge phai chay; co mavlink-routerd: "
                         "sources.mavlink.link=udp + endpoint 127.0.0.1:14551 trong /etc/mavlink-router/main.conf)."
                         "\n  serial: CHI khi khong co mavlink-routerd: dung bridge ('sudo systemctl stop uavp-fc-bridge'), "
                         "them --set sources.mavlink.source=serial\n";
            return 2;
        }
        std::printf("uavp_fc_check: %s | %d s | start.require_hover=%s, accept_viewer_key=%s | distance.source=%s\n",
                    fc->describe().c_str(), args.seconds, start.require_hover ? "true" : "false",
                    start.accept_viewer_key ? "true" : "false", distance.source.c_str());
        std::printf("(Trong luc chay: thu 'gcs_command.py ... start' / 'stop' tren may tinh de kiem tra duong lenh)\n\n");
        const FcPoseBuffer& b = fc->poses();
        const int64_t t0 = mono_ns();
        std::vector<std::pair<uint32_t, uint64_t>> prev = b.msg_counts();
        std::map<uint32_t, double> max_hz;
        uint32_t commands_seen = 0, last_command_count = 0;
        bool saw_start = false, saw_stop = false;
        double flow_min = 1e9, flow_max = -1e9;
        for (int sec = 1; sec <= args.seconds; ++sec) {
            std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::nanoseconds(t0)) +
                                          std::chrono::seconds(sec));
            const int64_t now = mono_ns();
            const auto counts = b.msg_counts();
            const FcStatus s = b.status(now, start.hover_window_s);
            const auto hz = [&](uint32_t id) {
                const double r = static_cast<double>(count_of(counts, id) - count_of(prev, id));
                max_hz[id] = std::max(max_hz[id], r);
                return r;
            };
            for (const auto& [id, n] : counts) (void)hz(id);  // cap nhat max_hz moi loai
            std::printf("--- %2d s | %zu byte, %zu loi CRC ---\n", sec, fc->bytes_received(), fc->crc_errors());
            std::printf("  HEARTBEAT %4.0f/s  tuoi %5.2f s | %s | che do %u %s | %s\n", hz(kMsgHeartbeat),
                        age_s(s.heartbeat_ns, now), s.armed ? "DA ARM" : "chua arm", s.custom_mode,
                        copter_mode(s.custom_mode), landed_name(s.landed_state));
            FcPose p;
            const int64_t t_att = b.latest_attitude_ns();
            if (t_att > 0 && b.pose_at(t_att, p)) {
                std::printf("  ATTITUDE  %4.0f/s  tuoi %5.3f s | roll %6.2f pitch %6.2f yaw %7.2f do\n", hz(kMsgAttitude),
                            age_s(t_att, now), p.roll_rad * kDeg, p.pitch_rad * kDeg, p.yaw_rad * kDeg);
                if (p.has_position)
                    std::printf("  LOCAL_POS %4.0f/s | NED (%.3f, %.3f, %.3f) m | van toc (%.2f, %.2f, %.2f) m/s\n",
                                hz(kMsgLocalPositionNed), p.pos_ned_m[0], p.pos_ned_m[1], p.pos_ned_m[2],
                                s.vel_ned_m_s[0], s.vel_ned_m_s[1], s.vel_ned_m_s[2]);
                else
                    std::printf("  LOCAL_POS %4.0f/s | CHUA CO vi tri (EKF chua co nguon vi tri / chua gui SR2_POSITION)\n",
                                hz(kMsgLocalPositionNed));
                if (p.has_range) std::printf("  RANGEFNDR %4.0f/s | %.3f m\n", hz(kMsgDistanceSensor) + hz(kMsgRangefinder),
                                             p.range_m);
            } else {
                std::printf("  ATTITUDE  %4.0f/s | CHUA CO tu the\n", hz(kMsgAttitude));
            }
            int64_t ft = 0;
            double fd = kNaN;
            int q = 0;
            if (b.latest_flow(ft, fd, q)) {
                flow_min = std::min(flow_min, fd);
                flow_max = std::max(flow_max, fd);
                const double D = camera_floor_distance_m(fd, p.roll_rad, p.pitch_rad, distance);
                std::printf("  OPT_FLOW  %4.0f/s  tuoi %5.2f s | ground_distance %.3f m, quality %3d | -> D camera-san %.3f m\n",
                            hz(kMsgOpticalFlow), age_s(ft, now), fd, q, D);
            } else {
                std::printf("  OPT_FLOW  %4.0f/s | khong co ground_distance hop le\n", hz(kMsgOpticalFlow));
            }
            if (s.command_count != last_command_count) {
                commands_seen += s.command_count - last_command_count;
                last_command_count = s.command_count;
                (s.command_start ? saw_start : saw_stop) = true;
                std::printf("  >>> NHAN LENH tu may tinh: %s (lan %u)\n", s.command_start ? "BAT DAU" : "DUNG",
                            s.command_count);
            }
            const StartGate g = check_start(s, start, t0);
            std::printf("  start gate: lenh %s | hover %s%s%s\n", g.commanded ? "CO" : "chua",
                        g.hover() ? "DAT" : "chua", g.reason.empty() ? "" : " | ", g.reason.c_str());
            prev = counts;
        }

        const int64_t now = mono_ns();
        const FcStatus s = b.status(now, start.hover_window_s);
        std::printf("\n=== Tan so lon nhat moi loai tin (/s) ===\n");
        for (const auto& [id, n] : b.msg_counts())
            std::printf("  msgid %3u %-20s tong %6llu | max %5.0f/s\n", id, msg_name(id), static_cast<unsigned long long>(n),
                        max_hz[id]);
        std::printf("\n=== Ket luan ===\n");
        Verdict v;
        v.line(fc->bytes_received() > 0 ? "DAT" : "LOI", "Nhan du lieu MAVLink",
               std::to_string(fc->bytes_received()) + " byte, " + std::to_string(fc->crc_errors()) + " loi CRC" +
                   (fc->bytes_received() == 0 ? " -> day TX/RX, baud, SERIAL2_PROTOCOL=2, bridge" : ""));
        const double hb_age = age_s(s.heartbeat_ns, now);
        v.line(std::isfinite(hb_age) && hb_age < start.max_msg_age_s ? "DAT" : "LOI", "HEARTBEAT tu Pixhawk",
               std::isfinite(hb_age) ? "tuoi " + std::to_string(hb_age).substr(0, 4) + " s" : "chua nhan");
        const double att_hz = max_hz[kMsgAttitude];
        const double att_age = age_s(b.latest_attitude_ns(), now);
        v.line(att_hz >= kMinAttitudeHz && att_age < kMaxAttitudeAgeS ? "DAT" : att_hz > 0 ? "CANH BAO" : "LOI",
               "ATTITUDE >= 10/s (tu the moi khung)",
               std::to_string(static_cast<int>(att_hz)) + "/s" + (att_hz < kMinAttitudeHz ? " -> SR2_EXTRA1 = 20-50" : ""));
        v.line(b.n_position() > 0 ? "DAT" : "CANH BAO", "LOCAL_POSITION_NED (p_W, van toc)",
               b.n_position() > 0 ? std::to_string(b.n_position()) + " mau"
                                  : "khong co -> khong co p_W / van toc; SR2_POSITION, nguon vi tri EKF (optical flow)");
        const bool has_height = b.n_flow() > 0 || b.n_range() > 0;
        v.line(has_height ? "DAT" : (distance.source == "fc" ? "LOI" : "CANH BAO"), "Do cao (optical flow / rangefinder)",
               b.n_flow() > 0 ? "OPTICAL_FLOW " + std::to_string(b.n_flow()) + " mau, " + std::to_string(flow_min).substr(0, 5) +
                                    ".." + std::to_string(flow_max).substr(0, 5) + " m"
               : b.n_range() > 0 ? "rangefinder " + std::to_string(b.n_range()) + " mau"
                                 : "khong co -> D dung distance.fixed_m; SR2_EXTRA3 / FLOW_TYPE / RNGFND1_TYPE");
        v.line(count_of(b.msg_counts(), kMsgExtendedSysState) > 0 ? "DAT" : (start.require_hover ? "LOI" : "CANH BAO"),
               "EXTENDED_SYS_STATE (dang bay)", start.require_hover ? "can cho require_hover" : "chi de log");
        v.line(commands_seen > 0 ? "DAT" : "CANH BAO", "Lenh tu may tinh qua Pixhawk",
               commands_seen > 0 ? std::string("nhan ") + (saw_start ? "start " : "") + (saw_stop ? "stop" : "")
                                 : "chua nhan (thu gcs_command.py start trong luc chay; hoac dung SPACE tren viewer)");
        std::printf("\n%s: %d loi, %d canh bao\n", v.fails == 0 ? "PIXHAWK -> PI: DAT" : "PIXHAWK -> PI: CO LOI", v.fails,
                    v.warns);
        return v.fails == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "LOI: " << e.what() << '\n';
        return 1;
    }
}
