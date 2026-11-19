// MAVLink (chi doc): tach goi, giai ma ATTITUDE / LOCAL_POSITION_NED / DISTANCE_SENSOR / RANGEFINDER, bo dem tu the.
#include "uavp/sources/mavlink.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>

#include <poll.h>
#include <unistd.h>

#include "uavp/common/common.hpp"
#include "uavp/common/log.hpp"
#include "uavp/sources/fc_shm.hpp"
#include "uavp/sources/serial_port.hpp"

namespace uavp {
namespace {

constexpr const char* kModule = "uavp.mavlink";
constexpr std::size_t kV1Header = 6, kV2Header = 10, kCrcBytes = 2, kSignatureBytes = 13;
constexpr uint8_t kIncompatSigned = 0x01;
constexpr std::size_t kReadChunk = 2048;
constexpr int kPollTimeoutMs = 100;
constexpr std::size_t kPosRing = 64, kRngRing = 64;
constexpr uint32_t kFcRebootJumpMs = 1000;  // time_boot_ms lui > 1 s = Pixhawk vua khoi dong lai

template <typename T>
T read_le(const MavMessage& m, std::size_t offset) noexcept {
    T v{};
    std::memcpy(&v, m.payload.data() + offset, sizeof(T));  // Pi / PC: little-endian nhu MAVLink
    return v;
}

constexpr bool is_pow2(std::size_t x) noexcept { return x != 0 && (x & (x - 1)) == 0; }

double unwrap_to(double angle, double ref) noexcept {  // goc gan ref nhat (cach 2 pi)
    while (angle - ref > CV_PI) angle -= 2.0 * CV_PI;
    while (angle - ref < -CV_PI) angle += 2.0 * CV_PI;
    return angle;
}

}  // namespace

uint16_t mavlink_crc(std::span<const uint8_t> data, uint16_t crc) noexcept {
    for (const uint8_t b : data) {
        auto tmp = static_cast<uint8_t>(b ^ static_cast<uint8_t>(crc & 0xFF));
        tmp = static_cast<uint8_t>(tmp ^ static_cast<uint8_t>(tmp << 4));
        crc = static_cast<uint16_t>((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4));
    }
    return crc;
}

int mavlink_crc_extra(uint32_t msgid) noexcept {
    switch (msgid) {  // tu dinh nghia XML cua MAVLink (common / ardupilotmega)
        case kMsgHeartbeat: return 50;
        case kMsgCommandLong: return 152;
        case kMsgCommandAck: return 143;
        case kMsgVfrHud: return 20;
        case kMsgOpticalFlow: return 175;
        case kMsgExtendedSysState: return 130;
        case kMsgAttitude: return 39;
        case kMsgLocalPositionNed: return 185;
        case kMsgDistanceSensor: return 85;
        case kMsgRangefinder: return 83;
        default: return -1;
    }
}

int split_mavlink(std::vector<uint8_t>& buf, std::vector<MavMessage>& out) {
    out.clear();
    int crc_errors = 0;
    std::size_t i = 0;
    while (i < buf.size()) {
        const uint8_t stx = buf[i];
        if (stx != kMavStxV1 && stx != kMavStxV2) {
            ++i;
            continue;
        }
        const bool v2 = stx == kMavStxV2;
        const std::size_t header = v2 ? kV2Header : kV1Header;
        if (buf.size() - i < header) break;  // cho them byte
        const uint8_t len = buf[i + 1];
        const bool signed_pkt = v2 && (buf[i + 2] & kIncompatSigned) != 0;
        const std::size_t total = header + len + kCrcBytes + (signed_pkt ? kSignatureBytes : 0);
        if (buf.size() - i < total) break;
        const uint32_t msgid = v2 ? static_cast<uint32_t>(buf[i + 7] | (buf[i + 8] << 8) | (buf[i + 9] << 16)) : buf[i + 5];
        const int extra = mavlink_crc_extra(msgid);
        if (extra < 0) {  // khong dung tin nay: bo qua ca goi (khong kiem duoc CRC)
            i += total;
            continue;
        }
        uint16_t crc = mavlink_crc(std::span(buf).subspan(i + 1, header - 1 + len));
        const auto extra_byte = static_cast<uint8_t>(extra);
        crc = mavlink_crc(std::span(&extra_byte, 1), crc);
        const std::size_t crc_at = i + header + len;
        if (crc != static_cast<uint16_t>(buf[crc_at] | (buf[crc_at + 1] << 8))) {
            ++crc_errors;
            ++i;  // co the la byte STX gia: tim tiep tu byte sau
            continue;
        }
        MavMessage& m = out.emplace_back();
        m.msgid = msgid;
        m.len = len;
        m.sysid = buf[i + (v2 ? 5 : 3)];
        m.compid = buf[i + (v2 ? 6 : 4)];
        std::copy_n(buf.begin() + static_cast<std::ptrdiff_t>(i + header), len, m.payload.begin());
        i += total;
    }
    buf.erase(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(i));
    return crc_errors;
}

MavAttitude decode_attitude(const MavMessage& m) noexcept {
    return {read_le<uint32_t>(m, 0), read_le<float>(m, 4), read_le<float>(m, 8), read_le<float>(m, 12)};
}

MavLocalPosition decode_local_position(const MavMessage& m) noexcept {
    return {read_le<uint32_t>(m, 0), read_le<float>(m, 4),  read_le<float>(m, 8), read_le<float>(m, 12),
            read_le<float>(m, 16),    read_le<float>(m, 20), read_le<float>(m, 24)};
}

MavHeartbeat decode_heartbeat(const MavMessage& m) noexcept {
    return {read_le<uint32_t>(m, 0), m.payload[4], m.payload[5], m.payload[6], m.payload[7]};
}

MavCommandLong decode_command_long(const MavMessage& m) noexcept {
    MavCommandLong c;  // param1..7 f32, command u16, target_system u8, target_component u8, confirmation u8
    for (std::size_t k = 0; k < c.param.size(); ++k) c.param[k] = read_le<float>(m, 4 * k);
    c.command = read_le<uint16_t>(m, 28);
    c.target_system = m.payload[30];
    c.target_component = m.payload[31];
    c.confirmation = m.payload[32];
    return c;
}

std::vector<uint8_t> pack_mavlink_v2(uint32_t msgid, std::span<const uint8_t> payload, uint8_t seq, uint8_t sysid,
                                     uint8_t compid) {
    constexpr std::size_t kHeader = 10;  // STX, len, incompat, compat, seq, sysid, compid, msgid (3 byte)
    const std::size_t n = payload.size();
    std::vector<uint8_t> p(kHeader + n + 2);  // cap phat mot lan: header + payload + CRC
    const std::array<uint8_t, kHeader> header{kMavStxV2, static_cast<uint8_t>(n), 0, 0, seq, sysid, compid,
                                              static_cast<uint8_t>(msgid & 0xFF), static_cast<uint8_t>((msgid >> 8) & 0xFF),
                                              static_cast<uint8_t>((msgid >> 16) & 0xFF)};
    std::copy(header.begin(), header.end(), p.begin());
    if (n > 0) std::memcpy(p.data() + kHeader, payload.data(), n);
    uint16_t crc = mavlink_crc(std::span<const uint8_t>(p.data() + 1, kHeader - 1 + n));
    const auto extra = static_cast<uint8_t>(mavlink_crc_extra(msgid));
    crc = mavlink_crc(std::span(&extra, 1), crc);
    p[kHeader + n] = static_cast<uint8_t>(crc & 0xFF);
    p[kHeader + n + 1] = static_cast<uint8_t>(crc >> 8);
    return p;
}

std::vector<uint8_t> companion_heartbeat_payload() {
    // custom_mode u32 = 0, type = ONBOARD_CONTROLLER, autopilot = INVALID, base_mode 0, ACTIVE, mavlink_version 3
    return {0, 0, 0, 0, kMavTypeOnboardController, kMavAutopilotInvalid, 0, kMavStateActive, 3};
}

std::vector<uint8_t> command_ack_payload(uint16_t command, uint8_t result, uint8_t target_system,
                                         uint8_t target_component) {
    // command u16, result u8, progress u8, result_param2 i32, target_system u8, target_component u8
    return {static_cast<uint8_t>(command & 0xFF), static_cast<uint8_t>(command >> 8), result, 0, 0, 0, 0, 0,
            target_system, target_component};
}

MavOpticalFlow decode_optical_flow(const MavMessage& m) noexcept {
    // time_usec u64, flow_comp_m_x / y f32, ground_distance f32, flow_x / y i16, sensor_id u8, quality u8
    return {read_le<float>(m, 16), m.payload[25]};
}

MavVfrHud decode_vfr_hud(const MavMessage& m) noexcept {
    return {read_le<float>(m, 4), read_le<float>(m, 12)};  // airspeed, groundspeed, alt, climb (f32)
}

MavDistance decode_distance(const MavMessage& m) noexcept {
    if (m.msgid == kMsgRangefinder) return {0, read_le<float>(m, 0), true};
    // DISTANCE_SENSOR: time_boot_ms u32, min / max / current u16 (cm), type, id, orientation, covariance u8
    const float cm = static_cast<float>(read_le<uint16_t>(m, 8));
    return {read_le<uint32_t>(m, 0), cm / 100.f, m.payload[12] == kMavSensorRotationPitch270};
}

// --- FcPoseBuffer -----------------------------------------------------------------------------------------------
FcPoseBuffer::FcPoseBuffer(const MavlinkConfig& cfg)
    : cfg_(cfg), att_(static_cast<std::size_t>(cfg.buffer_samples)), pos_(kPosRing), rng_(kRngRing), flow_(kRngRing) {
    if (!is_pow2(att_.size())) throw std::invalid_argument("mavlink.buffer_samples phai la luy thua 2");
}

int64_t FcPoseBuffer::to_mono_ns(uint32_t time_boot_ms, int64_t t_rx_ns) {
    // Pixhawk khoi dong lai (mat nguon cung Pi, reset, brownout): time_boot_ms ve gan 0 -> offset cu khong con dung.
    // Khong xoa thi min() giu offset cu them 64 tin va moi tu the moi bi lech ca thoi gian FC da chay truoc do.
    if (has_boot_ms_ && time_boot_ms + kFcRebootJumpMs < last_boot_ms_) n_offsets_ = 0;
    has_boot_ms_ = true;
    last_boot_ms_ = std::max(time_boot_ms, n_offsets_ == 0 ? time_boot_ms : last_boot_ms_);
    offsets_[n_offsets_++ % offsets_.size()] = t_rx_ns - static_cast<int64_t>(time_boot_ms) * kNsPerMs;
    const std::size_t n = std::min(n_offsets_, offsets_.size());
    const int64_t offset = *std::min_element(offsets_.begin(), offsets_.begin() + static_cast<std::ptrdiff_t>(n));
    return static_cast<int64_t>(time_boot_ms) * kNsPerMs + offset + static_cast<int64_t>(cfg_.latency_ms * 1e6);
}

void FcPoseBuffer::push(const MavMessage& m, int64_t t_rx_ns) {
    std::lock_guard lock(mutex_);
    const auto it = std::find_if(msg_counts_.begin(), msg_counts_.end(), [&](const auto& c) { return c.first == m.msgid; });
    if (it != msg_counts_.end()) ++it->second;
    else msg_counts_.emplace_back(m.msgid, 1);  // vai chuc loai tin: tim tuyen tinh re hon map
    switch (m.msgid) {
        case kMsgAttitude: {
            const MavAttitude a = decode_attitude(m);
            att_[n_att_++ & (att_.size() - 1)] = {to_mono_ns(a.time_boot_ms, t_rx_ns), a.roll_rad, a.pitch_rad, a.yaw_rad};
            break;
        }
        case kMsgLocalPositionNed: {
            const MavLocalPosition p = decode_local_position(m);
            pos_[n_pos_++ & (pos_.size() - 1)] = {to_mono_ns(p.time_boot_ms, t_rx_ns), p.x_m, p.y_m, p.z_m};
            st_.vel_ned_m_s = {p.vx_m_s, p.vy_m_s, p.vz_m_s};
            st_.vel_ns = t_rx_ns;
            break;
        }
        case kMsgHeartbeat: {
            const MavHeartbeat h = decode_heartbeat(m);
            if (!h.from_autopilot()) break;  // HEARTBEAT cua GCS / companion khac: bo
            st_.heartbeat_ns = t_rx_ns;
            st_.armed = h.armed();
            st_.custom_mode = h.custom_mode;
            break;
        }
        case kMsgCommandLong: {  // lenh tu may tinh gui toi Pi (sysid / compid / command trong config)
            const MavCommandLong c = decode_command_long(m);
            if (c.target_system != cfg_.companion_sysid || c.target_component != cfg_.companion_compid ||
                c.command != cfg_.command_id)
                break;
            st_.command_ns = t_rx_ns;
            st_.command_start = c.param[0] >= 0.5f;
            ++st_.command_count;
            break;
        }
        case kMsgExtendedSysState:
            st_.landed_state = decode_landed_state(m);
            st_.landed_ns = t_rx_ns;
            break;
        case kMsgVfrHud:
            if (st_.vel_ns == 0 || t_rx_ns - st_.vel_ns > static_cast<int64_t>(cfg_.max_age_s * 1e9)) {  // khong co NED
                const MavVfrHud v = decode_vfr_hud(m);
                st_.vel_ned_m_s = {v.groundspeed_m_s, 0.0, -v.climb_m_s};
                st_.vel_ns = t_rx_ns;
            }
            break;
        case kMsgOpticalFlow: {
            const MavOpticalFlow f = decode_optical_flow(m);
            if (f.ground_distance_m > 0.f)  // b = quality (0-255) cho uavp_fc_check
                flow_[n_flow_++ & (flow_.size() - 1)] = {t_rx_ns, f.ground_distance_m, static_cast<float>(f.quality), 0.f};
            break;
        }
        case kMsgDistanceSensor:
        case kMsgRangefinder: {
            const MavDistance d = decode_distance(m);
            if (!d.downward || d.distance_m <= 0.f) break;
            const int64_t t = d.time_boot_ms != 0 ? to_mono_ns(d.time_boot_ms, t_rx_ns)
                                                  : t_rx_ns - static_cast<int64_t>(cfg_.latency_ms * 1e6);
            rng_[n_rng_++ & (rng_.size() - 1)] = {t, d.distance_m, 0.f, 0.f};
            break;
        }
        default:
            break;
    }
}

const FcPoseBuffer::Sample* FcPoseBuffer::nearest(const std::vector<Sample>& ring, std::size_t count, int64_t t_ns,
                                                  int64_t max_age_ns) noexcept {
    const Sample* best = nullptr;
    const std::size_t n = std::min(count, ring.size());
    for (std::size_t k = 0; k < n; ++k) {
        const Sample& s = ring[(count - 1 - k) & (ring.size() - 1)];
        if (best == nullptr || std::abs(s.t_ns - t_ns) < std::abs(best->t_ns - t_ns)) best = &s;
    }
    return best != nullptr && std::abs(best->t_ns - t_ns) <= max_age_ns ? best : nullptr;
}

bool FcPoseBuffer::pose_at(int64_t t_ns, FcPose& out) const {
    const auto max_age = static_cast<int64_t>(cfg_.max_age_s * 1e9);
    std::lock_guard lock(mutex_);
    const std::size_t n = std::min(n_att_, att_.size());
    const Sample* before = nullptr;
    const Sample* after = nullptr;
    for (std::size_t k = 0; k < n; ++k) {  // moi -> cu: tim cap bao quanh t_ns
        const Sample& s = att_[(n_att_ - 1 - k) & (att_.size() - 1)];
        if (s.t_ns >= t_ns) after = &s;
        else {
            before = &s;
            break;
        }
    }
    if (before == nullptr && after == nullptr) return false;
    if (before != nullptr && t_ns - before->t_ns > max_age) before = nullptr;  // qua cu (vd truoc khi FC khoi dong lai)
    if (after != nullptr && after->t_ns - t_ns > max_age) after = nullptr;
    if (before == nullptr && after == nullptr) return false;
    if (before == nullptr) before = after;  // chi con 1 mau gan: dung mau do
    if (after == nullptr) after = before;
    const double span = static_cast<double>(after->t_ns - before->t_ns);
    const double w = span > 0.0 ? static_cast<double>(t_ns - before->t_ns) / span : 0.0;
    const auto lerp = [w](double a, double b) { return a + w * (b - a); };
    out = FcPose{};
    out.t_ns = t_ns;
    out.roll_rad = lerp(before->a, after->a);
    out.pitch_rad = lerp(before->b, after->b);
    out.yaw_rad = lerp(before->c, unwrap_to(after->c, before->c));
    if (const Sample* p = cfg_.use_local_position ? nearest(pos_, n_pos_, t_ns, max_age) : nullptr) {
        out.pos_ned_m = {p->a, p->b, p->c};
        out.has_position = true;
    }
    if (const Sample* r = cfg_.use_rangefinder ? nearest(rng_, n_rng_, t_ns, max_age) : nullptr) {
        out.range_m = r->a;
        out.has_range = true;
    }
    return true;
}

std::size_t FcPoseBuffer::n_attitude() const {
    std::lock_guard lock(mutex_);
    return n_att_;
}

std::size_t FcPoseBuffer::n_position() const {
    std::lock_guard lock(mutex_);
    return n_pos_;
}

std::size_t FcPoseBuffer::n_range() const {
    std::lock_guard lock(mutex_);
    return n_rng_;
}

std::size_t FcPoseBuffer::n_flow() const {
    std::lock_guard lock(mutex_);
    return n_flow_;
}

bool FcPoseBuffer::latest_flow(int64_t& t_ns, double& distance_m, int& quality) const {
    std::lock_guard lock(mutex_);
    if (n_flow_ == 0) return false;
    const Sample& f = flow_[(n_flow_ - 1) & (flow_.size() - 1)];
    t_ns = f.t_ns;
    distance_m = f.a;
    quality = static_cast<int>(f.b);
    return true;
}

std::vector<std::pair<uint32_t, uint64_t>> FcPoseBuffer::msg_counts() const {
    std::lock_guard lock(mutex_);
    return msg_counts_;
}

FcStatus FcPoseBuffer::status(int64_t now_ns, double window_s) const {
    std::lock_guard lock(mutex_);
    FcStatus s = st_;
    s.t_ns = now_ns;
    const auto window_ns = static_cast<int64_t>(window_s * 1e9);
    const bool use_flow = n_flow_ > 0 && now_ns - flow_[(n_flow_ - 1) & (flow_.size() - 1)].t_ns <= window_ns;
    const std::vector<Sample>& ring = use_flow ? flow_ : rng_;
    const std::size_t count = use_flow ? n_flow_ : n_rng_;
    std::vector<double> h;
    for (std::size_t k = 0; k < std::min(count, ring.size()); ++k) {
        const Sample& x = ring[(count - 1 - k) & (ring.size() - 1)];
        if (now_ns - x.t_ns > window_ns) break;
        h.push_back(x.a);
        s.height_oldest_ns = x.t_ns;
    }
    s.height_from_flow = use_flow;
    s.height_samples = static_cast<int>(h.size());
    if (!h.empty()) {
        const auto [lo, hi] = std::minmax_element(h.begin(), h.end());
        s.height_min_m = *lo;
        s.height_max_m = *hi;
        s.height_median_m = median_inplace(std::span<double>(h));
    }
    return s;
}

int64_t FcPoseBuffer::latest_attitude_ns() const {
    std::lock_guard lock(mutex_);
    return n_att_ == 0 ? 0 : att_[(n_att_ - 1) & (att_.size() - 1)].t_ns;
}

// --- MavlinkReader ----------------------------------------------------------------------------------------------
MavlinkReader::MavlinkReader(const MavlinkConfig& cfg)
    : poses_(cfg),
      port_(cfg.port),
      fd_(open_serial_fd(cfg.port, cfg.baudrate,
                         "Pixhawk: ls /dev/ttyAMA* /dev/ttyACM*, nhom dialout, tat serial console (raspi-config)")) {
    thread_ = std::thread([this] { loop(); });
    UAVP_LOG_INFO(kModule, "MAVLink: %s @ %d", cfg.port.c_str(), cfg.baudrate);
}

MavlinkReader::~MavlinkReader() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0) ::close(fd_);
}

void MavlinkReader::loop() {
    std::vector<uint8_t> buf;
    buf.reserve(4 * kReadChunk);
    std::vector<MavMessage> msgs;
    msgs.reserve(64);
    std::array<uint8_t, kReadChunk> chunk{};
    pollfd pfd{fd_, POLLIN, 0};
    while (running_) {
        if (::poll(&pfd, 1, kPollTimeoutMs) <= 0) continue;
        const ssize_t n = ::read(fd_, chunk.data(), chunk.size());
        const int64_t t_rx = mono_ns();
        if (n <= 0) {
            if (n < 0 && errno != EAGAIN) {
                UAVP_LOG_ERROR(kModule, "MAVLink loi doc: %s", std::strerror(errno));
                return;
            }
            continue;
        }
        bytes_ += static_cast<std::size_t>(n);
        buf.insert(buf.end(), chunk.begin(), chunk.begin() + n);
        crc_errors_ += static_cast<std::size_t>(split_mavlink(buf, msgs));
        for (const auto& m : msgs) poses_.push(m, t_rx);
        if (buf.size() > 8 * kReadChunk) buf.clear();  // rac khong co goi hop le: tranh phinh bo nho
    }
}

// --- MavlinkShmReader ---------------------------------------------------------------------------------------------
MavlinkShmReader::MavlinkShmReader(const MavlinkConfig& cfg)
    : poses_(cfg), name_(cfg.shm_name), shm_(std::make_unique<FcShmReader>(cfg.shm_name)),
      poll_(std::chrono::milliseconds(cfg.shm_poll_ms)) {
    thread_ = std::thread([this] { loop(); });
    UAVP_LOG_INFO(kModule, "MAVLink: doc tu shared memory %s (uavp_fc_bridge pid %d)", name_.c_str(),
                  shm_->header().writer_pid);
}

MavlinkShmReader::~MavlinkShmReader() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

std::size_t MavlinkShmReader::crc_errors() const noexcept {
    return static_cast<std::size_t>(shm_->header().crc_errors.load(std::memory_order_relaxed));
}

std::size_t MavlinkShmReader::bytes_received() const noexcept {
    return static_cast<std::size_t>(shm_->header().bytes.load(std::memory_order_relaxed));
}

void MavlinkShmReader::loop() {
    std::vector<MavMessage> msgs;
    std::vector<int64_t> t_rx;
    msgs.reserve(256);
    t_rx.reserve(256);
    std::size_t lost = 0;
    while (running_) {
        if (shm_->poll(msgs, t_rx, lost))
            for (std::size_t i = 0; i < msgs.size(); ++i) poses_.push(msgs[i], t_rx[i]);
        if (lost > 0) UAVP_LOG_WARN(kModule, "MAVLink shm: doc cham, mat %zu tin", lost);
        std::this_thread::sleep_for(poll_);
    }
}

std::unique_ptr<IFcSource> open_fc_source(const MavlinkConfig& cfg) {
    if (cfg.source == "serial") return std::make_unique<MavlinkReader>(cfg);
    return std::make_unique<MavlinkShmReader>(cfg);
}

}  // namespace uavp
