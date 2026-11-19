// uavp_fc_bridge: chay nen tren Pi (dich vu systemd uavp-fc-bridge) tu luc Pi khoi dong.
//   Doc MAVLink tu Pixhawk va ghi cac tin tu the (HEARTBEAT,
//   ATTITUDE, LOCAL_POSITION_NED, DISTANCE_SENSOR, RANGEFINDER) + thoi diem nhan vao RAM (/dev/shm/uavp_fc).
//   uavp_real_time / uavp_sensor_check doc RAM do (sources.mavlink.source: shm) -> world coordinate (NED cua EKF).
//   Pixhawk khoi dong sau / cap USB rut ra: bridge tu mo lai cong. EKF chua co goc toa do: chua co LOCAL_POSITION_NED.
//   Bridge GUI ve Pixhawk: HEARTBEAT (Pi = may tinh dong hanh sysid / compid trong config) moi heartbeat_period_s de
//   Pixhawk biet duong chuyen lenh cua may tinh (GCS) toi Pi; COMMAND_ACK khi nhan lenh bat dau / dung
//   (COMMAND_LONG mavlink.command_id, tools/gcs_command.py).
// Nguon (sources.mavlink.link):
//   udp    : qua mavlink-routerd (router giu UART TELEM2 <-> GPIO 14/15, chia cho tram mat dat 14550 va bridge). Bridge
//            lang nghe udp_address:udp_port (endpoint [UdpEndpoint uavp] Mode=Normal cua router), gui nguoc ve router.
//   serial : mo thang UART `port` (khi KHONG chay mavlink-routerd; 2 chuong trinh khong duoc mo chung 1 UART).
// Chay tay:
//   ./build/uavp_fc_bridge --cfg configs/run/real_time.yaml            # ghi (giong dich vu)
//   ./build/uavp_fc_bridge --cfg configs/run/real_time.yaml --print    # doc RAM, in tu the moi 1 s (kiem tra)
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "uavp/common/common.hpp"
#include "uavp/common/config.hpp"
#include "uavp/common/log.hpp"
#include "uavp/sources/fc_shm.hpp"
#include "uavp/sources/mavlink.hpp"
#include "uavp/sources/serial_port.hpp"
#include "uavp/support/cpu_affinity.hpp"

using namespace uavp;

namespace {

constexpr const char* kModule = "uavp.fc_bridge";
constexpr std::size_t kReadChunk = 8192;  // >= 1 goi UDP cua mavlink-routerd
constexpr int kPollTimeoutMs = 100;
constexpr double kPrintPeriodS = 1.0;
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

std::atomic<bool> g_stop{false};  // SIGINT / SIGTERM (systemctl stop)
void on_signal(int) { g_stop = true; }

struct BridgeConfig {
    std::vector<int> cores;
    std::size_t shm_capacity{4096};
    double status_period_s{5.0};
    double reopen_wait_s{2.0};
    double heartbeat_period_s{1.0};
};

// Gui ve Pixhawk (HEARTBEAT cua Pi, COMMAND_ACK). serial: ghi cong; udp: gui ve dia chi router (biet sau goi dau tien).
struct Sender {
    int fd;
    const MavlinkConfig& mav;
    bool udp{false};
    sockaddr_in peer{};
    bool has_peer{false};
    uint8_t seq{0};
    void send(uint32_t msgid, const std::vector<uint8_t>& payload) {
        if (udp && !has_peer) return;  // chua nhan goi nao tu router: chua biet gui ve dau
        const auto pkt = pack_mavlink_v2(msgid, payload, seq++, static_cast<uint8_t>(mav.companion_sysid),
                                         static_cast<uint8_t>(mav.companion_compid));
        const ssize_t n = udp ? ::sendto(fd, pkt.data(), pkt.size(), 0, reinterpret_cast<const sockaddr*>(&peer), sizeof(peer))
                              : ::write(fd, pkt.data(), pkt.size());
        if (n != static_cast<ssize_t>(pkt.size())) UAVP_LOG_WARN(kModule, "Gui loi / thieu byte: %s", std::strerror(errno));
    }
};

// Socket UDP lang nghe address:port (dau ra cua mavlink-routerd). -1 + log neu loi.
int open_udp(const MavlinkConfig& mav) {
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) throw std::runtime_error(std::string("socket UDP: ") + std::strerror(errno));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(mav.udp_port));
    if (::inet_pton(AF_INET, mav.udp_address.c_str(), &a.sin_addr) != 1) {
        ::close(fd);
        throw std::runtime_error("udp_address sai: " + mav.udp_address);
    }
    if (::bind(fd, reinterpret_cast<const sockaddr*>(&a), sizeof(a)) != 0) {
        const std::string err = std::strerror(errno);
        ::close(fd);
        throw std::runtime_error("Khong bind duoc UDP " + mav.udp_address + ":" + std::to_string(mav.udp_port) + " (" + err +
                                 "): cong dang bi chuong trinh khac giu? (ss -ulpn | grep " + std::to_string(mav.udp_port) + ")");
    }
    return fd;
}

// Lenh tu may tinh gui toi Pi: tra COMMAND_ACK, log.
void on_command(const MavMessage& m, const MavlinkConfig& mav, Sender& out) {
    const MavCommandLong c = decode_command_long(m);
    if (c.target_system != mav.companion_sysid || c.target_component != mav.companion_compid) return;
    if (c.command != mav.command_id) {
        UAVP_LOG_WARN(kModule, "Lenh %u khong ho tro (chi %d)", c.command, mav.command_id);
        return;
    }
    out.send(kMsgCommandAck, command_ack_payload(c.command, kMavResultAccepted, m.sysid, m.compid));
    UAVP_LOG_INFO(kModule, "Lenh tu may tinh (sysid %u compid %u): %s perception", m.sysid, m.compid,
                  c.param[0] >= 0.5f ? "BAT DAU" : "DUNG");
}

struct Args {
    std::string cfg;
    std::vector<std::string> overrides;
    bool print{false};
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) a.cfg = argv[++i];
        else if (std::strcmp(argv[i], "--print") == 0) a.print = true;
        else if (std::strcmp(argv[i], "--set") == 0)
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.overrides.emplace_back(argv[++i]);
    }
    if (a.cfg.empty()) throw std::invalid_argument("Dung: uavp_fc_bridge --cfg configs/run/real_time.yaml [--print] [--set k=v]");
    return a;
}

BridgeConfig bridge_config(const YAML::Node& r) {
    if (!r) throw std::runtime_error("Thieu muc fc_bridge trong config");
    return {r["cores"].as<std::vector<int>>(), r["shm_capacity"].as<std::size_t>(), r["status_period_s"].as<double>(),
            r["reopen_wait_s"].as<double>(), r["heartbeat_period_s"].as<double>()};
}

void sleep_s(double s) { std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int64_t>(s * 1000.0))); }

// Tin / s theo loai tu lan in truoc + EKF da co vi tri chua.
void log_status(const FcShmHeader& h, std::array<uint64_t, 5>& last, double period_s, const MavAttitude& att) {
    const std::array<uint64_t, 5> now{h.n_heartbeat.load(), h.n_attitude.load(), h.n_position.load(), h.n_range.load(),
                                      h.crc_errors.load()};
    std::array<double, 5> rate{};
    for (std::size_t k = 0; k < 5; ++k) rate[k] = static_cast<double>(now[k] - last[k]) / period_s;
    last = now;
    UAVP_LOG_INFO(kModule, "HEARTBEAT %.1f/s | ATTITUDE %.1f/s (roll %.1f pitch %.1f yaw %.1f do) | LOCAL_POSITION_NED "
                  "%.1f/s | rangefinder %.1f/s | loi CRC %.1f/s%s",
                  rate[0], rate[1], att.roll_rad * kRadToDeg, att.pitch_rad * kRadToDeg, att.yaw_rad * kRadToDeg, rate[2],
                  rate[3], rate[4],
                  rate[0] == 0.0 ? "  << CHUA THAY PIXHAWK"
                  : rate[2] == 0.0 ? "  << EKF chua co vi tri (chua co goc toa do / nguon vi tri)" : "");
}

enum class PumpEnd : uint8_t { kStopped, kLinkError, kShmGone };

// File /dev/shm/<ten> con khong. systemd-logind (RemoveIPC=yes, mac dinh Ubuntu) XOA shared memory cua user khi phien
// dang nhap cuoi ket thuc (dong ssh) du bridge van chay -> perception khong mo duoc -> phai tao lai.
bool shm_exists(const std::string& name) { return ::access(("/dev/shm" + name).c_str(), F_OK) == 0; }

// Doc cong / socket toi khi loi, bi dung, hoac file shm bi xoa.
PumpEnd pump(int fd, FcShmWriter& shm, const BridgeConfig& bc, const MavlinkConfig& mav) {
    Sender out{fd, mav, mav.link == "udp"};
    int64_t t_hb = 0;
    std::vector<uint8_t> buf;
    buf.reserve(4 * kReadChunk);
    std::vector<MavMessage> msgs;
    msgs.reserve(64);
    std::array<uint8_t, kReadChunk> chunk{};
    std::array<uint64_t, 5> last{};
    MavAttitude att;
    pollfd pfd{fd, POLLIN, 0};
    int64_t t_status = mono_ns();
    while (!g_stop) {
        if (::poll(&pfd, 1, kPollTimeoutMs) > 0) {
            sockaddr_in from{};
            socklen_t from_len = sizeof(from);
            const ssize_t n = out.udp ? ::recvfrom(fd, chunk.data(), chunk.size(), 0, reinterpret_cast<sockaddr*>(&from), &from_len)
                                      : ::read(fd, chunk.data(), chunk.size());
            const int64_t t_rx = mono_ns();
            if (out.udp && n > 0 && (!out.has_peer || from.sin_port != out.peer.sin_port ||
                                     from.sin_addr.s_addr != out.peer.sin_addr.s_addr)) {
                out.peer = from;  // dia chi cua mavlink-routerd: HEARTBEAT / ACK gui ve day
                out.has_peer = true;
                char ip[INET_ADDRSTRLEN] = "?";
                ::inet_ntop(AF_INET, &from.sin_addr, ip, sizeof(ip));
                UAVP_LOG_INFO(kModule, "Nhan MAVLink tu mavlink-routerd %s:%u", ip, ntohs(from.sin_port));
            }
            if (n < 0 && errno != EAGAIN) {
                UAVP_LOG_ERROR(kModule, "Loi doc cong: %s", std::strerror(errno));
                return PumpEnd::kLinkError;
            }
            if (n > 0) {
                buf.insert(buf.end(), chunk.begin(), chunk.begin() + n);
                const int crc = split_mavlink(buf, msgs);
                shm.add_io(static_cast<std::size_t>(n), static_cast<std::size_t>(crc));
                for (const auto& m : msgs) {
                    shm.push(m, t_rx);
                    if (m.msgid == kMsgAttitude) att = decode_attitude(m);
                    if (m.msgid == kMsgCommandLong) on_command(m, mav, out);
                }
                if (buf.size() > 8 * kReadChunk) buf.clear();
            }
        }
        if (static_cast<double>(mono_ns() - t_hb) / 1e9 >= bc.heartbeat_period_s) {
            out.send(kMsgHeartbeat, companion_heartbeat_payload());
            t_hb = mono_ns();
        }
        if (static_cast<double>(mono_ns() - t_status) / 1e9 >= bc.status_period_s) {
            log_status(shm.header(), last, static_cast<double>(mono_ns() - t_status) / 1e9, att);
            t_status = mono_ns();
            if (!shm_exists(mav.shm_name)) {
                UAVP_LOG_WARN(kModule, "/dev/shm%s bi xoa (systemd-logind RemoveIPC khi dong phien ssh?): tao lai",
                              mav.shm_name.c_str());
                return PumpEnd::kShmGone;
            }
        }
    }
    return PumpEnd::kStopped;
}

int run_writer(const MavlinkConfig& mav, const BridgeConfig& bc) {
    auto shm = std::make_unique<FcShmWriter>(mav.shm_name, bc.shm_capacity);
    const bool udp = mav.link == "udp";
    if (udp)
        UAVP_LOG_INFO(kModule, "Ghi RAM: /dev/shm%s (%zu tin) | nguon UDP %s:%d (mavlink-routerd)", mav.shm_name.c_str(),
                      bc.shm_capacity, mav.udp_address.c_str(), mav.udp_port);
    else
        UAVP_LOG_INFO(kModule, "Ghi RAM: /dev/shm%s (%zu tin) | cong %s @ %d", mav.shm_name.c_str(), bc.shm_capacity,
                      mav.port.c_str(), mav.baudrate);
    while (!g_stop) {
        int fd = -1;
        try {
            fd = udp ? open_udp(mav)
                     : open_serial_fd(mav.port, mav.baudrate,
                                      "Pixhawk TELEM2: dtoverlay=disable-bt, tat serial console, nhom dialout; neu "
                                      "mavlink-routerd dang giu cong: dung sources.mavlink.link: udp",
                                      true);  // ghi: HEARTBEAT + COMMAND_ACK
        } catch (const std::exception& e) {
            UAVP_LOG_WARN(kModule, "%s | thu lai sau %.1f s", e.what(), bc.reopen_wait_s);
            sleep_s(bc.reopen_wait_s);
            continue;
        }
        if (udp)
            UAVP_LOG_INFO(kModule, "Lang nghe UDP %s:%d, cho mavlink-routerd chuyen MAVLink toi", mav.udp_address.c_str(),
                          mav.udp_port);
        else
            UAVP_LOG_INFO(kModule, "Da mo %s, cho MAVLink tu Pixhawk", mav.port.c_str());
        const PumpEnd end = pump(fd, *shm, bc, mav);
        ::close(fd);
        if (end == PumpEnd::kLinkError) sleep_s(bc.reopen_wait_s);
        if (end == PumpEnd::kShmGone) {
            shm.reset();  // munmap vung cu (da bi unlink) truoc khi tao file moi
            shm = std::make_unique<FcShmWriter>(mav.shm_name, bc.shm_capacity);
        }
    }
    UAVP_LOG_INFO(kModule, "Dung bridge");
    return 0;
}

// --print: doc RAM nhu perception, in tu the + world position moi 1 s.
int run_print(MavlinkConfig mav) {
    mav.source = "shm";
    const auto fc = open_fc_source(mav);
    while (!g_stop) {
        sleep_s(kPrintPeriodS);
        FcPose p;
        const int64_t now = mono_ns();
        if (!fc->poses().pose_at(now, p)) {
            UAVP_LOG_WARN(kModule, "Chua co ATTITUDE moi trong %.2f s (Pixhawk / bridge?)", mav.max_age_s);
            continue;
        }
        const double age_ms = static_cast<double>(now - fc->poses().latest_attitude_ns()) / 1e6;
        char pos[96] = "chua co (EKF chua co vi tri)";
        if (p.has_position) std::snprintf(pos, sizeof(pos), "(%.3f, %.3f, %.3f) m", p.pos_ned_m[0], p.pos_ned_m[1], p.pos_ned_m[2]);
        char rng[48] = "chua co";
        if (p.has_range) std::snprintf(rng, sizeof(rng), "%.3f m", p.range_m);
        UAVP_LOG_INFO(kModule, "roll %6.2f pitch %6.2f yaw %7.2f do (tre %.0f ms) | NED %s | rangefinder %s",
                      p.roll_rad * kRadToDeg, p.pitch_rad * kRadToDeg, p.yaw_rad * kRadToDeg, age_ms, pos, rng);
    }
    return 0;
}

int run_main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const YAML::Node raw = load_raw(args.cfg, args.overrides);
    const MavlinkConfig mav = mavlink_config(raw["sources"]["mavlink"]);
    const BridgeConfig bc = bridge_config(raw["fc_bridge"]);
    const std::string cores = pin_process_to_cores(bc.cores);
    auto logger = std::make_unique<Logger>(Level::kInfo);
    set_logger(logger.get());
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    UAVP_LOG_INFO(kModule, "uavp_fc_bridge (%s), CPU %s", args.print ? "doc / in" : "ghi", cores.c_str());
    const int rc = args.print ? run_print(mav) : run_writer(mav, bc);
    set_logger(nullptr);
    return rc;
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
