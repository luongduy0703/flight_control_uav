#include "uavp/pipeline/start_gate.hpp"

#include <algorithm>
#include <cmath>

namespace uavp {
namespace {

bool fresh(int64_t msg_ns, int64_t now_ns, double max_age_s) noexcept {
    return msg_ns > 0 && static_cast<double>(now_ns - msg_ns) / 1e9 <= max_age_s;
}

}  // namespace

bool stop_commanded(const FcStatus& s, int64_t since_ns) noexcept {
    return s.command_ns > since_ns && !s.command_start;
}

StartGate check_start(const FcStatus& s, const StartConfig& cfg, int64_t waiting_since_ns, bool viewer_command) {
    StartGate g;
    g.hover_required = cfg.require_hover;
    g.by_viewer = viewer_command && cfg.accept_viewer_key;
    g.commanded = (s.command_ns > waiting_since_ns && s.command_start) || g.by_viewer;
    const bool hb = fresh(s.heartbeat_ns, s.t_ns, cfg.max_msg_age_s);
    g.armed = hb && s.armed;
    g.mode_ok = hb && (cfg.modes.empty() || std::find(cfg.modes.begin(), cfg.modes.end(), s.custom_mode) != cfg.modes.end());
    g.in_air = fresh(s.landed_ns, s.t_ns, cfg.max_msg_age_s) && s.landed_state == kMavLandedStateInAir;
    const double v_h = std::hypot(s.vel_ned_m_s[0], s.vel_ned_m_s[1]);
    g.still = fresh(s.vel_ns, s.t_ns, cfg.max_msg_age_s) && v_h <= cfg.max_speed_m_s &&
              std::abs(s.vel_ned_m_s[2]) <= cfg.max_vspeed_m_s;
    const bool full_window = s.height_samples >= 2 &&
                             static_cast<double>(s.t_ns - s.height_oldest_ns) / 1e9 >= 0.9 * cfg.hover_window_s;
    g.height_stable = full_window && s.height_median_m >= cfg.min_height_m &&
                      s.height_max_m - s.height_min_m <= cfg.max_height_change_m;
    g.height_m = s.height_median_m;
    char b[160];
    if (!g.commanded) {
        g.reason = cfg.accept_viewer_key ? "cho lenh bat dau tu may tinh (gcs_command.py start hoac SPACE tren viewer)"
                                         : "cho lenh bat dau tu may tinh (tools/gcs_command.py start)";
        return g;
    }
    if (!cfg.require_hover) return g;  // chi can lenh: khong bao ly do hover
    if (!hb) g.reason = "khong co HEARTBEAT tu Pixhawk";
    else if (!g.armed) g.reason = "chua arm";
    else if (!g.mode_ok) g.reason = "che do bay " + std::to_string(s.custom_mode) + " khong nam trong start.modes";
    else if (!g.in_air) g.reason = "chua bay (EXTENDED_SYS_STATE landed_state " + std::to_string(s.landed_state) + ")";
    else if (!g.still) {
        std::snprintf(b, sizeof(b), "chua dung yen: ngang %.2f m/s, doc %.2f m/s", v_h, s.vel_ned_m_s[2]);
        g.reason = b;
    } else if (!g.height_stable) {
        std::snprintf(b, sizeof(b), "do cao chua on dinh: %d mau, %.3f..%.3f m (%s)", s.height_samples, s.height_min_m,
                      s.height_max_m, s.height_from_flow ? "optical flow" : "rangefinder");
        g.reason = b;
    }
    return g;
}

double camera_floor_distance_m(double sensor_height_m, double roll_rad, double pitch_rad,
                               const DistanceConfig& cfg) noexcept {
    return sensor_height_m * std::cos(roll_rad) * std::cos(pitch_rad) + cfg.camera_offset_m;
}

}  // namespace uavp
