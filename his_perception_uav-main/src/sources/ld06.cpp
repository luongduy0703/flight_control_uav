// Goi LD06: parse, thoi diem tung diem, tach goi, ghep vong quet (ham thuan, test duoc khong can cong serial).
#include "uavp/sources/ld06.hpp"

#include <cmath>
#include <cstring>

namespace uavp {
namespace {

uint16_t le16(std::span<const uint8_t> p, std::size_t off) noexcept {
    return static_cast<uint16_t>(p[off] | (p[off + 1] << 8));
}

}  // namespace

// --- Goi LD06 -------------------------------------------------------------------------------------------------
bool parse_packet(std::span<const uint8_t> pkt, Ld06Packet& out) noexcept {
    if (pkt.size() != kLd06PacketSize || pkt[0] != kLd06Header || pkt[1] != kLd06VerLen ||
        crc8(pkt.first(kLd06PacketSize - 1)) != pkt[kLd06PacketSize - 1])
        return false;
    const int start = le16(pkt, 4);
    const int end = le16(pkt, 42);
    const int span = ((end - start) % 36000 + 36000) % 36000;  // qua moc 360 do
    out.speed_dps = static_cast<float>(le16(pkt, 2));
    out.start_deg = static_cast<float>(start) / 100.f;
    out.span_deg = static_cast<float>(span) / 100.f;
    const double step = span / static_cast<double>(kLd06Points - 1);
    for (int i = 0; i < kLd06Points; ++i) {
        const auto k = static_cast<std::size_t>(i);
        const std::size_t off = 6 + 3 * k;
        out.range_mm[k] = le16(pkt, off);
        out.intensity[k] = pkt[off + 2];
        out.angle_deg[k] = static_cast<float>(py_mod(start + step * i, 36000.0) / 100.0);
    }
    return true;
}

int64_t packet_tx_ns(int baudrate) noexcept {
    return static_cast<int64_t>(std::llround(static_cast<double>(kLd06PacketSize) * kUartBitsPerByte * 1e9 / baudrate));
}

int64_t rx_end_ns(int64_t t_read_ns, std::size_t bytes_after, int baudrate) noexcept {
    return t_read_ns - static_cast<int64_t>(std::llround(static_cast<double>(bytes_after) * kUartBitsPerByte * 1e9 / baudrate));
}

void point_times_ns(const Ld06Packet& pkt, int64_t t_last_ns, std::array<int64_t, kLd06Points>& out) noexcept {
    if (pkt.speed_dps <= 0.f) {  // chua quay deu: khong noi suy
        out.fill(t_last_ns);
        return;
    }
    for (int i = 0; i < kLd06Points; ++i) {
        const double remaining_deg = pkt.span_deg * (1.0 - static_cast<double>(i) / (kLd06Points - 1));
        out[static_cast<std::size_t>(i)] = t_last_ns - static_cast<int64_t>(std::nearbyint(remaining_deg / pkt.speed_dps * 1e9));
    }
}

int split_packets(std::vector<uint8_t>& buf, std::vector<RawPacket>& out) {
    out.clear();
    int crc_errors = 0;
    std::size_t pos = 0;
    while (buf.size() - pos >= kLd06PacketSize) {
        if (buf[pos] != kLd06Header || buf[pos + 1] != kLd06VerLen) {
            std::size_t next = pos + 1;  // tim header tiep theo; khong thay -> giu byte cuoi
            while (next + 1 < buf.size() && !(buf[next] == kLd06Header && buf[next + 1] == kLd06VerLen)) ++next;
            pos = next + 1 < buf.size() ? next : buf.size() - 1;
            continue;
        }
        const std::span<const uint8_t> pkt(buf.data() + pos, kLd06PacketSize);
        if (crc8(pkt.first(kLd06PacketSize - 1)) != pkt[kLd06PacketSize - 1]) {
            ++crc_errors;
            ++pos;
            continue;
        }
        RawPacket raw;
        std::memcpy(raw.bytes.data(), pkt.data(), kLd06PacketSize);
        pos += kLd06PacketSize;
        raw.bytes_after = buf.size() - pos;
        out.push_back(raw);
    }
    buf.erase(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(pos));
    return crc_errors;
}

bool ScanAssembler::push(const Ld06Packet& pkt, const std::array<int64_t, kLd06Points>& t_ns, Scan& completed) {
    bool done = false;
    if (has_packet_ && pkt.start_deg + 180.f < last_start_deg_) {
        std::swap(completed, current_);  // khong chep: doi buffer
        current_.clear();
        done = true;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(kLd06Points); ++i) {
        current_.angle_deg.push_back(pkt.angle_deg[i]);
        current_.range_mm.push_back(pkt.range_mm[i]);
        current_.intensity.push_back(pkt.intensity[i]);
        current_.t_ns.push_back(t_ns[i]);
    }
    last_start_deg_ = pkt.start_deg;
    has_packet_ = true;
    return done;
}

}  // namespace uavp
