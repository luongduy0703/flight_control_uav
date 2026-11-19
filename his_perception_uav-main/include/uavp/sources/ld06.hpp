// LiDAR LD06 (giong phan LD06 cua src/uavp/sources.py). Goi 47 byte, little-endian:
//   [0] 0x54 header, [1] 0x2C (12 diem), [2:4] toc do (do/s), [4:6] goc dau (0.01 do),
//   [6:42] 12 x {khoang cach uint16 (mm), cuong do uint8}, [42:44] goc cuoi (0.01 do), [44:46] timestamp (ms),
//   [46] CRC8 (da thuc 0x4D) cua 46 byte dau.
// Thoi diem tung diem: t diem cuoi goi = luc nhan byte cuoi - thoi gian truyen goi - tre them;
//   t diem i = t diem cuoi - (goc cuoi - goc i) / toc do quay.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "uavp/common/common.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

inline constexpr uint8_t kLd06Header = 0x54;
inline constexpr uint8_t kLd06VerLen = 0x2C;
inline constexpr int kLd06Points = 12;
inline constexpr std::size_t kLd06PacketSize = 47;
inline constexpr int kUartBitsPerByte = 10;  // 8N1

constexpr std::array<uint8_t, 256> make_crc8_table(uint8_t poly) {
    std::array<uint8_t, 256> t{};
    for (int i = 0; i < 256; ++i) {
        auto c = static_cast<uint8_t>(i);
        for (int b = 0; b < 8; ++b) c = (c & 0x80) ? static_cast<uint8_t>((c << 1) ^ poly) : static_cast<uint8_t>(c << 1);
        t[static_cast<std::size_t>(i)] = c;
    }
    return t;
}
inline constexpr auto kCrc8 = make_crc8_table(0x4D);  // tinh luc bien dich

inline uint8_t crc8(std::span<const uint8_t> data) noexcept {
    uint8_t crc = 0;
    for (const uint8_t b : data) crc = kCrc8[static_cast<uint8_t>(crc ^ b)];
    return crc;
}

// Mot goi 12 diem (tho).
struct Ld06Packet {
    std::array<float, kLd06Points> angle_deg{};
    std::array<uint16_t, kLd06Points> range_mm{};
    std::array<uint8_t, kLd06Points> intensity{};
    float speed_dps{0.f};   // toc do quay (do/s)
    float start_deg{0.f};
    float span_deg{0.f};    // goc cuoi - goc dau (qua moc 360), >= 0
};

// 47 byte -> goi; sai header / do dai / CRC -> false.
[[nodiscard]] bool parse_packet(std::span<const uint8_t> pkt, Ld06Packet& out) noexcept;
[[nodiscard]] int64_t packet_tx_ns(int baudrate) noexcept;
[[nodiscard]] int64_t rx_end_ns(int64_t t_read_ns, std::size_t bytes_after, int baudrate) noexcept;
void point_times_ns(const Ld06Packet& pkt, int64_t t_last_ns, std::array<int64_t, kLd06Points>& out) noexcept;

// Tach goi 47 byte dung CRC tu dau buf (bo rac / goi hong), xoa phan da dung khoi buf.
struct RawPacket {
    std::array<uint8_t, kLd06PacketSize> bytes{};
    std::size_t bytes_after{0};  // so byte trong buf den SAU byte cuoi cua goi
};
int split_packets(std::vector<uint8_t>& buf, std::vector<RawPacket>& out);  // tra ve so loi CRC

// Ghep goi thanh vong quet: goc dau giam manh (> 180 do) so voi goi truoc = vua xong mot vong.
class ScanAssembler {
public:
    // Tra ve true va ghi `completed` neu goi nay bat dau vong moi (vong truoc xong).
    [[nodiscard]] bool push(const Ld06Packet& pkt, const std::array<int64_t, kLd06Points>& t_ns, Scan& completed);

private:
    Scan current_;
    float last_start_deg_{0.f};
    bool has_packet_{false};
};

// Bo dem `capacity` vong quet gan nhat (vong cu bi day ra). Luong doc goi feed(); luong khac doc vong quet.
class ScanBuffer {
public:
    ScanBuffer(std::size_t capacity, int baudrate, double extra_latency_ms);
    ScanBuffer(const ScanBuffer&) = delete;
    ScanBuffer& operator=(const ScanBuffer&) = delete;

    void feed(std::span<const uint8_t> raw, int64_t t_rx_end_ns);  // goi tho + luc nhan byte cuoi goi
    [[nodiscard]] bool latest_scan(Scan& out) const;
    [[nodiscard]] bool wait_first_scan(double timeout_s) const;
    // Vong quet DA THU bat dau tu t_ns (chep vao out); next_index: chi so vong moi tiep theo; truncated: bo dem da
    // day mat vong cu hon t_ns.
    void collected_since(int64_t t_ns, std::vector<Scan>& out, std::size_t& next_index, bool& truncated) const;
    // Vong quet thu `index` (cho toi da timeout_s); vong da bi day khoi bo dem -> nhay toi vong cu nhat con lai.
    [[nodiscard]] bool wait_scan(std::size_t& index, double timeout_s, Scan& out) const;
    // Tong so vong da xong (tang = co vong quet moi: dung de segment theo nhip LiDAR).
    [[nodiscard]] std::size_t count() const;
    // Cho toi khi vong moi nhat phu toi thoi diem t_ns (diem cuoi >= t_ns). false neu het timeout_s.
    [[nodiscard]] bool wait_until(int64_t t_ns, double timeout_s) const;

    [[nodiscard]] std::size_t crc_errors() const noexcept { return crc_errors_.load(); }
    [[nodiscard]] std::size_t bytes_received() const noexcept { return bytes_received_.load(); }
    void add_bytes(std::size_t n) noexcept { bytes_received_ += n; }
    void add_crc_errors(std::size_t n) noexcept { crc_errors_ += n; }

private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::vector<Scan> ring_;      // slot dung lai (cap phat lai khi vong lon hon)
    std::size_t count_{0};        // tong so vong da xong
    ScanAssembler assembler_;     // chi luong feed dung
    Scan completed_;
    Ld06Packet pkt_;
    std::array<int64_t, kLd06Points> t_pts_{};
    int64_t delay_ns_{0};
    alignas(64) std::atomic<std::size_t> crc_errors_{0};      // luong doc ghi, luong khac doc: tach dong cache
    alignas(64) std::atomic<std::size_t> bytes_received_{0};
};

// LD06 cam vao cong serial (vd /dev/ttyUSB0): doc theo khoi o luong rieng (poll co timeout), dong ho mono_ns().
class Ld06Reader {
public:
    Ld06Reader(const std::string& port, int baudrate, std::size_t buffer_scans, double extra_latency_ms);
    ~Ld06Reader();
    Ld06Reader(const Ld06Reader&) = delete;
    Ld06Reader& operator=(const Ld06Reader&) = delete;

    [[nodiscard]] const ScanBuffer& buffer() const noexcept { return buffer_; }

private:
    void loop();

    ScanBuffer buffer_;
    std::thread thread_;
    int fd_{-1};
    int baudrate_;
    std::atomic<bool> running_{true};
};

}  // namespace uavp
