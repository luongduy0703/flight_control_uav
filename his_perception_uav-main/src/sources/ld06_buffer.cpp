// Bo dem vong quet LD06 + luong doc cong serial.
#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#include <poll.h>
#include <unistd.h>

#include "uavp/sources/ld06.hpp"
#include "uavp/sources/serial_port.hpp"
#include "uavp/common/log.hpp"

namespace uavp {
namespace {

constexpr const char* kModule = "uavp.sources";
constexpr std::size_t kReadChunk = 4096;
constexpr int kPollTimeoutMs = 100;
constexpr std::size_t kScanReservePoints = 600;  // LD06 ~450 diem / vong

std::chrono::nanoseconds to_ns(double seconds) {
    return std::chrono::nanoseconds(static_cast<int64_t>(seconds * 1e9));
}

void copy_scan(const Scan& from, Scan& to) {
    to.t_ns.assign(from.t_ns.begin(), from.t_ns.end());  // dung lai bo nho cua `to` neu du
    to.angle_deg.assign(from.angle_deg.begin(), from.angle_deg.end());
    to.range_mm.assign(from.range_mm.begin(), from.range_mm.end());
    to.intensity.assign(from.intensity.begin(), from.intensity.end());
}

}  // namespace

ScanBuffer::ScanBuffer(std::size_t capacity, int baudrate, double extra_latency_ms)
    : ring_(capacity), delay_ns_(packet_tx_ns(baudrate) + static_cast<int64_t>(extra_latency_ms * 1e6)) {
    for (auto& s : ring_) {
        s.t_ns.reserve(kScanReservePoints);
        s.angle_deg.reserve(kScanReservePoints);
        s.range_mm.reserve(kScanReservePoints);
        s.intensity.reserve(kScanReservePoints);
    }
}

void ScanBuffer::feed(std::span<const uint8_t> raw, int64_t t_rx_end_ns) {
    if (!parse_packet(raw, pkt_)) {
        ++crc_errors_;
        return;
    }
    point_times_ns(pkt_, t_rx_end_ns - delay_ns_, t_pts_);
    if (!assembler_.push(pkt_, t_pts_, completed_)) return;
    {
        std::lock_guard lock(mutex_);
        copy_scan(completed_, ring_[count_ % ring_.size()]);
        ++count_;
    }
    cv_.notify_all();
}

bool ScanBuffer::latest_scan(Scan& out) const {
    std::lock_guard lock(mutex_);
    if (count_ == 0) return false;
    copy_scan(ring_[(count_ - 1) % ring_.size()], out);
    return true;
}

bool ScanBuffer::wait_first_scan(double timeout_s) const {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, to_ns(timeout_s), [this] { return count_ > 0; });
}

void ScanBuffer::collected_since(int64_t t_ns, std::vector<Scan>& out, std::size_t& next_index, bool& truncated) const {
    std::lock_guard lock(mutex_);
    const std::size_t stored = std::min(count_, ring_.size());
    const std::size_t base = count_ - stored;
    out.clear();
    for (std::size_t i = base; i < count_; ++i) {
        const Scan& s = ring_[i % ring_.size()];
        if (!s.t_ns.empty() && s.t_ns.front() >= t_ns) copy_scan(s, out.emplace_back());
    }
    truncated = stored == ring_.size() && stored > 0 && ring_[base % ring_.size()].t_ns.front() > t_ns;
    next_index = count_;
}

bool ScanBuffer::wait_scan(std::size_t& index, double timeout_s, Scan& out) const {
    std::unique_lock lock(mutex_);
    if (!cv_.wait_for(lock, to_ns(timeout_s), [&] { return count_ > index; })) return false;
    const std::size_t base = count_ - std::min(count_, ring_.size());
    index = std::max(index, base);  // vong cu da bi day khoi bo dem
    copy_scan(ring_[index % ring_.size()], out);
    ++index;
    return true;
}

std::size_t ScanBuffer::count() const {
    std::lock_guard lock(mutex_);
    return count_;
}

bool ScanBuffer::wait_until(int64_t t_ns, double timeout_s) const {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, to_ns(timeout_s), [&] {
        if (count_ == 0) return false;
        const Scan& last = ring_[(count_ - 1) % ring_.size()];
        return !last.t_ns.empty() && last.t_ns.back() >= t_ns;
    });
}

Ld06Reader::Ld06Reader(const std::string& port, int baudrate, std::size_t buffer_scans, double extra_latency_ms)
    : buffer_(buffer_scans, baudrate, extra_latency_ms), fd_(open_serial_fd(port, baudrate, "LD06: ls -l /dev/serial/by-id/, user trong nhom dialout, chuong trinh khac dang giu cong?")), baudrate_(baudrate) {
    thread_ = std::thread([this] { loop(); });
    UAVP_LOG_INFO(kModule, "LD06: %s @ %d", port.c_str(), baudrate);
}

Ld06Reader::~Ld06Reader() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0) ::close(fd_);
}

void Ld06Reader::loop() {
    std::vector<uint8_t> buf;
    buf.reserve(4 * kReadChunk);
    std::vector<RawPacket> packets;
    packets.reserve(128);
    std::array<uint8_t, kReadChunk> chunk{};
    pollfd pfd{fd_, POLLIN, 0};
    while (running_) {
        if (::poll(&pfd, 1, kPollTimeoutMs) <= 0) continue;  // timeout: kiem tra running_
        const ssize_t n = ::read(fd_, chunk.data(), chunk.size());  // doc het byte dang cho
        const int64_t t_read = mono_ns();  // cung dong ho voi khung anh
        if (n <= 0) {
            if (n < 0 && errno != EAGAIN) {
                UAVP_LOG_ERROR(kModule, "LD06 loi doc: %s", std::strerror(errno));
                return;
            }
            continue;
        }
        buffer_.add_bytes(static_cast<std::size_t>(n));
        buf.insert(buf.end(), chunk.begin(), chunk.begin() + n);
        buffer_.add_crc_errors(static_cast<std::size_t>(split_packets(buf, packets)));
        for (const auto& p : packets) buffer_.feed(p.bytes, rx_end_ns(t_read, p.bytes_after, baudrate_));
    }
}

}  // namespace uavp
