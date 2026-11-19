// Doc tu the UAV tu Pixhawk (EKF ArduPilot / PX4) qua MAVLink (cong serial, chi doc). Tu parse goi v1 / v2, kiem
// CRC (X.25 + CRC_EXTRA), khong can thu vien MAVLink. Tin nhan dung:
//   ATTITUDE (30)            roll, pitch, yaw (rad, NED) -> R_WB
//   LOCAL_POSITION_NED (32)  x, y, z (m, NED, goc = diem EKF khoi tao) -> vi tri ngang (neu co)
//   DISTANCE_SENSOR (132)    rangefinder huong xuong (orientation 25), cm -> do cao so voi mat dat
//   RANGEFINDER (173)        ArduPilot: distance (m)
// Pixhawk phai gui cac tin nay toi cong cua Pi (ArduPilot: SERIALx_PROTOCOL = 2, SRx_EXTRA1 >= 20 Hz cho ATTITUDE,
// SRx_POSITION, SRx_EXTRA3; xem GUIDE.md).
// Dong ho: time_boot_ms cua FC -> mono_ns() cua Pi: offset = min(luc nhan - time_boot) tren cac tin gan day (tre
// truyen nho nhat) + latency_ms trong config.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/common.hpp"
#include "uavp/common/config.hpp"

namespace uavp {

// --- Goi MAVLink ------------------------------------------------------------------------------------------------
inline constexpr uint8_t kMavStxV1 = 0xFE;
inline constexpr uint8_t kMavStxV2 = 0xFD;
inline constexpr uint32_t kMsgHeartbeat = 0;
inline constexpr uint32_t kMsgAttitude = 30;
inline constexpr uint32_t kMsgLocalPositionNed = 32;
inline constexpr uint32_t kMsgDistanceSensor = 132;
inline constexpr uint32_t kMsgRangefinder = 173;
inline constexpr uint32_t kMsgCommandLong = 76;
inline constexpr uint32_t kMsgCommandAck = 77;
inline constexpr uint32_t kMsgVfrHud = 74;
inline constexpr uint32_t kMsgOpticalFlow = 100;
inline constexpr uint32_t kMsgExtendedSysState = 245;
inline constexpr uint8_t kMavSensorRotationPitch270 = 25;  // DISTANCE_SENSOR.orientation: huong xuong
inline constexpr uint8_t kMavModeFlagSafetyArmed = 128;    // HEARTBEAT.base_mode: da arm
inline constexpr uint8_t kMavTypeGcs = 6;                  // HEARTBEAT.type cua tram mat dat: bo qua
inline constexpr uint8_t kMavAutopilotInvalid = 8;         // HEARTBEAT.autopilot cua thiet bi khong phai FC: bo qua
inline constexpr uint8_t kMavLandedStateInAir = 2;         // EXTENDED_SYS_STATE.landed_state
inline constexpr uint8_t kMavTypeOnboardController = 18;   // HEARTBEAT.type cua Pi (may tinh dong hanh)
inline constexpr uint8_t kMavStateActive = 4;
inline constexpr uint8_t kMavResultAccepted = 0;

// CRC-16/MCRF4XX (X.25) cua MAVLink.
[[nodiscard]] uint16_t mavlink_crc(std::span<const uint8_t> data, uint16_t crc = 0xFFFF) noexcept;
// CRC_EXTRA cua tin nhan da biet; -1 neu khong dung tin nay.
[[nodiscard]] int mavlink_crc_extra(uint32_t msgid) noexcept;

struct MavMessage {
    std::array<uint8_t, 255> payload{};  // da them 0 o cuoi (MAVLink 2 cat byte 0 cuoi payload)
    uint32_t msgid{0};
    uint8_t len{0};
    uint8_t sysid{0}, compid{0};          // ben gui
};

// Tach goi tu dau buf (bo rac / goi hong / goi khong dung), xoa phan da dung. Them tin da biet, dung CRC vao out.
// Tra ve so goi loi CRC.
int split_mavlink(std::vector<uint8_t>& buf, std::vector<MavMessage>& out);

struct MavAttitude {
    uint32_t time_boot_ms{0};
    float roll_rad{0.f}, pitch_rad{0.f}, yaw_rad{0.f};
};
struct MavLocalPosition {
    uint32_t time_boot_ms{0};
    float x_m{0.f}, y_m{0.f}, z_m{0.f};     // NED
    float vx_m_s{0.f}, vy_m_s{0.f}, vz_m_s{0.f};
};
struct MavHeartbeat {
    uint32_t custom_mode{0};  // ArduCopter: 2 ALT_HOLD, 4 GUIDED, 5 LOITER, 16 POSHOLD...
    uint8_t type{0}, autopilot{0}, base_mode{0}, system_status{0};
    [[nodiscard]] bool from_autopilot() const noexcept { return type != kMavTypeGcs && autopilot != kMavAutopilotInvalid; }
    [[nodiscard]] bool armed() const noexcept { return (base_mode & kMavModeFlagSafetyArmed) != 0; }
};
// COMMAND_LONG tu may tinh (GCS) qua Pixhawk toi Pi: lenh bat dau / dung perception.
struct MavCommandLong {
    std::array<float, 7> param{};
    uint16_t command{0};
    uint8_t target_system{0}, target_component{0}, confirmation{0};
};
struct MavOpticalFlow {
    float ground_distance_m{-1.f};  // < 0: khong co do cao
    uint8_t quality{0};
};
struct MavVfrHud {
    float groundspeed_m_s{0.f}, climb_m_s{0.f};
};
struct MavDistance {
    uint32_t time_boot_ms{0};  // 0: tin khong co thoi gian (RANGEFINDER)
    float distance_m{0.f};
    bool downward{true};
};
[[nodiscard]] MavAttitude decode_attitude(const MavMessage& m) noexcept;
[[nodiscard]] MavLocalPosition decode_local_position(const MavMessage& m) noexcept;
[[nodiscard]] MavDistance decode_distance(const MavMessage& m) noexcept;  // DISTANCE_SENSOR hoac RANGEFINDER
[[nodiscard]] MavHeartbeat decode_heartbeat(const MavMessage& m) noexcept;
[[nodiscard]] MavCommandLong decode_command_long(const MavMessage& m) noexcept;
// Dong goi MAVLink 2 (khong ky) de Pi gui: HEARTBEAT (de Pixhawk biet duong chuyen lenh toi Pi), COMMAND_ACK.
[[nodiscard]] std::vector<uint8_t> pack_mavlink_v2(uint32_t msgid, std::span<const uint8_t> payload, uint8_t seq,
                                                   uint8_t sysid, uint8_t compid);
[[nodiscard]] std::vector<uint8_t> companion_heartbeat_payload();
[[nodiscard]] std::vector<uint8_t> command_ack_payload(uint16_t command, uint8_t result, uint8_t target_system,
                                                       uint8_t target_component);
[[nodiscard]] MavOpticalFlow decode_optical_flow(const MavMessage& m) noexcept;
[[nodiscard]] MavVfrHud decode_vfr_hud(const MavMessage& m) noexcept;
[[nodiscard]] inline uint8_t decode_landed_state(const MavMessage& m) noexcept { return m.payload[1]; }

// Trang thai FC de quyet dinh bat dau (che do bay that): moi muc kem thoi diem nhan (0 = chua co).
struct FcStatus {
    cv::Vec3d vel_ned_m_s{0.0, 0.0, 0.0};  // LOCAL_POSITION_NED; khong co thi VFR_HUD (ngang = groundspeed, doc = -climb)
    int64_t t_ns{0};                        // luc lay trang thai
    int64_t heartbeat_ns{0}, landed_ns{0}, vel_ns{0};
    int64_t command_ns{0};                  // lenh bat dau / dung gan nhat tu may tinh (0 = chua co)
    int64_t height_oldest_ns{0};            // mau do cao cu nhat trong cua so
    double height_min_m{kNaN}, height_max_m{kNaN}, height_median_m{kNaN};
    uint32_t custom_mode{0};
    int height_samples{0};
    uint32_t command_count{0};              // tang moi lenh (phan biet lenh moi / cu)
    uint8_t landed_state{0};
    bool command_start{false};              // lenh gan nhat: true = bat dau (param1 = 1), false = dung (param1 = 0)
    bool armed{false};
    bool height_from_flow{false};           // true: OPTICAL_FLOW.ground_distance; false: rangefinder
};

// --- Tu the FC theo thoi gian (cau hinh: MavlinkConfig trong uavp/common/config.hpp) ------------------------------
// Tu the UAV (than FRD trong NED) tai mot thoi diem (dong ho chung mono_ns).
struct FcPose {
    cv::Vec3d pos_ned_m{0.0, 0.0, 0.0};
    int64_t t_ns{0};
    double roll_rad{0.0}, pitch_rad{0.0}, yaw_rad{0.0};
    double range_m{0.0};
    bool has_position{false};  // LOCAL_POSITION_NED
    bool has_range{false};     // rangefinder huong xuong
};

// Bo dem tu the: luong doc MAVLink push_*; luong chinh pose_at (moi khung, khoa ngan).
class FcPoseBuffer {
public:
    explicit FcPoseBuffer(const MavlinkConfig& cfg);
    FcPoseBuffer(const FcPoseBuffer&) = delete;
    FcPoseBuffer& operator=(const FcPoseBuffer&) = delete;

    void push(const MavMessage& m, int64_t t_rx_ns);
    // Tu the luc t_ns: goc noi suy giua 2 mau ATTITUDE (yaw qua moc +-pi), vi tri / rangefinder: mau gan nhat trong
    // max_age_s. false neu khong co ATTITUDE quanh t_ns.
    [[nodiscard]] bool pose_at(int64_t t_ns, FcPose& out) const;
    [[nodiscard]] std::size_t n_attitude() const;
    [[nodiscard]] std::size_t n_position() const;
    [[nodiscard]] std::size_t n_range() const;
    [[nodiscard]] std::size_t n_flow() const;  // so mau OPTICAL_FLOW co ground_distance hop le
    // Mau OPTICAL_FLOW gan nhat (luc nhan, ground_distance m, quality 0-255). false neu chua co.
    [[nodiscard]] bool latest_flow(int64_t& t_ns, double& distance_m, int& quality) const;
    // So tin da nhan theo msgid (moi loai, ke ca loai khong giai ma) -> kiem tra Pixhawk gui gi, tan so bao nhieu.
    [[nodiscard]] std::vector<std::pair<uint32_t, uint64_t>> msg_counts() const;
    [[nodiscard]] int64_t latest_attitude_ns() const;  // 0 neu chua co
    // Trang thai FC luc now_ns; do cao: cac mau trong window_s cuoi (uu tien optical flow, khong co thi rangefinder).
    [[nodiscard]] FcStatus status(int64_t now_ns, double window_s) const;

private:
    struct Sample {
        int64_t t_ns{0};
        float a{0.f}, b{0.f}, c{0.f};
    };
    [[nodiscard]] int64_t to_mono_ns(uint32_t time_boot_ms, int64_t t_rx_ns);
    [[nodiscard]] static const Sample* nearest(const std::vector<Sample>& ring, std::size_t count, int64_t t_ns,
                                               int64_t max_age_ns) noexcept;

    MavlinkConfig cfg_;
    mutable std::mutex mutex_;
    std::vector<Sample> att_, pos_, rng_, flow_;  // ring, luy thua 2
    std::size_t n_att_{0}, n_pos_{0}, n_rng_{0}, n_flow_{0};
    std::vector<std::pair<uint32_t, uint64_t>> msg_counts_;  // (msgid, so tin)
    FcStatus st_;  // HEARTBEAT, RC, landed, van toc moi nhat (cac truong do cao tinh trong status())
    std::array<int64_t, 64> offsets_{};    // luc nhan - time_boot (ns) cua 64 tin gan nhat
    std::size_t n_offsets_{0};
    uint32_t last_boot_ms_{0};             // time_boot_ms lon nhat tu lan FC khoi dong gan nhat
    bool has_boot_ms_{false};
};

// Nguon tu the FC: doc thang cong serial (MavlinkReader) hoac doc RAM do uavp_fc_bridge ghi (MavlinkShmReader).
class IFcSource {
public:
    virtual ~IFcSource() = default;
    [[nodiscard]] virtual const FcPoseBuffer& poses() const noexcept = 0;
    [[nodiscard]] virtual std::size_t crc_errors() const noexcept = 0;
    [[nodiscard]] virtual std::size_t bytes_received() const noexcept = 0;
    [[nodiscard]] virtual std::string describe() const = 0;
};

// sources.mavlink.source: shm (mac dinh, can uavp_fc_bridge dang chay) | serial (mo cong truc tiep).
[[nodiscard]] std::unique_ptr<IFcSource> open_fc_source(const MavlinkConfig& cfg);

// Pixhawk qua cong serial: luong rieng doc theo khoi (poll co timeout).
class MavlinkReader final : public IFcSource {
public:
    explicit MavlinkReader(const MavlinkConfig& cfg);
    ~MavlinkReader() override;
    MavlinkReader(const MavlinkReader&) = delete;
    MavlinkReader& operator=(const MavlinkReader&) = delete;

    [[nodiscard]] const FcPoseBuffer& poses() const noexcept override { return poses_; }
    [[nodiscard]] std::size_t crc_errors() const noexcept override { return crc_errors_.load(); }
    [[nodiscard]] std::size_t bytes_received() const noexcept override { return bytes_.load(); }
    [[nodiscard]] std::string describe() const override { return "serial " + port_; }

private:
    void loop();

    FcPoseBuffer poses_;
    std::string port_;
    std::thread thread_;
    int fd_{-1};
    std::atomic<bool> running_{true};
    alignas(64) std::atomic<std::size_t> crc_errors_{0};
    alignas(64) std::atomic<std::size_t> bytes_{0};
};

class FcShmReader;

// Doc tin MAVLink tu RAM (/dev/shm, uavp_fc_bridge ghi) moi shm_poll_ms vao FcPoseBuffer rieng.
class MavlinkShmReader final : public IFcSource {
public:
    explicit MavlinkShmReader(const MavlinkConfig& cfg);
    ~MavlinkShmReader() override;
    MavlinkShmReader(const MavlinkShmReader&) = delete;
    MavlinkShmReader& operator=(const MavlinkShmReader&) = delete;

    [[nodiscard]] const FcPoseBuffer& poses() const noexcept override { return poses_; }
    [[nodiscard]] std::size_t crc_errors() const noexcept override;
    [[nodiscard]] std::size_t bytes_received() const noexcept override;
    [[nodiscard]] std::string describe() const override { return "shared memory " + name_; }

private:
    void loop();

    FcPoseBuffer poses_;
    std::string name_;
    std::unique_ptr<FcShmReader> shm_;
    std::thread thread_;
    std::chrono::milliseconds poll_;
    std::atomic<bool> running_{true};
};

}  // namespace uavp
