// Dieu kien bat dau quet cua uavp_real_time (chay that, ket hop Pixhawk):
//   lenh tu may tinh     COMMAND_LONG (mavlink.command_id) param1 = 1 gui toi Pi qua Pixhawk, SAU luc bat dau cho
//   HEARTBEAT            da arm, custom_mode nam trong start.modes
//   EXTENDED_SYS_STATE   landed_state = IN_AIR
//   van toc              LOCAL_POSITION_NED (hoac VFR_HUD): ngang <= max_speed, doc <= max_vspeed
//   do cao               optical flow (hoac rangefinder): >= min_height, max - min <= max_height_change trong ca
//                        hover_window_s
// start.require_hover = false: chi can lenh (Pixhawk hoac SPACE tren viewer); cac dieu kien hover van duoc tinh de log.
// Ham thuan (test duoc). Khoang cach camera -> san cho s3 lay tu do cao luc bat dau (distance.source: fc).
#pragma once

#include <string>

#include "uavp/common/config.hpp"
#include "uavp/sources/mavlink.hpp"

namespace uavp {

struct StartGate {
    std::string reason;  // dieu kien dau tien chua dat (rong = dat het)
    double height_m{kNaN};
    bool commanded{false}, armed{false}, mode_ok{false}, in_air{false}, still{false}, height_stable{false};
    bool hover_required{true};
    bool by_viewer{false};  // lenh tu SPACE tren viewer (khong qua Pixhawk)
    [[nodiscard]] bool hover() const noexcept { return armed && mode_ok && in_air && still && height_stable; }
    [[nodiscard]] bool ok() const noexcept { return commanded && (!hover_required || hover()); }
};

// waiting_since_ns: luc bat dau cho (lenh bat dau phai den sau luc nay, tranh dung lai lenh cu).
// viewer_command: vua bam SPACE / s tren remote_viewer (chi tinh khi start.accept_viewer_key).
[[nodiscard]] StartGate check_start(const FcStatus& s, const StartConfig& cfg, int64_t waiting_since_ns,
                                   bool viewer_command = false);

// May tinh vua gui lenh dung (sau since_ns). Dung trong luc chay.
[[nodiscard]] bool stop_commanded(const FcStatus& s, int64_t since_ns) noexcept;

// Khoang cach camera -> san (m) tu do cao cam bien (doc theo truc than) va goc nghieng.
[[nodiscard]] double camera_floor_distance_m(double sensor_height_m, double roll_rad, double pitch_rad,
                                             const DistanceConfig& cfg) noexcept;

}  // namespace uavp
