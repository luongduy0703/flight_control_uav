// Goi LD06 tong hop (khong can cong serial), giong tests/test_sources.py phan LD06.
#include <gtest/gtest.h>

#include <cmath>

#include "uavp/sources/ld06.hpp"

using namespace uavp;

namespace {

constexpr int kBaud = 230400;
const std::array<double, 4> kStarts{0.0, 120.0, 240.0, 350.0};  // 4 goi / vong quet

std::vector<uint8_t> make_packet(double start_deg, double end_deg, const std::vector<int>& ranges, int intensity = 200,
                                 int speed_dps = 3600) {
    std::vector<uint8_t> b{0x54, 0x2C};
    const auto le16 = [&](int v) {
        b.push_back(static_cast<uint8_t>(v & 0xFF));
        b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    };
    le16(speed_dps);
    le16(static_cast<int>(std::lround(start_deg * 100)));
    for (const int r : ranges) {
        le16(r);
        b.push_back(static_cast<uint8_t>(intensity));
    }
    le16(static_cast<int>(std::lround(end_deg * 100)) % 36000);
    le16(1234);
    b.push_back(crc8(b));
    return b;
}

std::vector<int> flat(int r) { return std::vector<int>(12, r); }

}  // namespace

TEST(Ld06, ParsePacketAnglesRanges) {
    std::vector<int> ranges;
    for (int i = 0; i < 12; ++i) ranges.push_back(600 + i);
    const auto pkt = make_packet(10.0, 21.0, ranges);
    ASSERT_EQ(pkt.size(), kLd06PacketSize);
    Ld06Packet p;
    ASSERT_TRUE(parse_packet(pkt, p));
    EXPECT_FLOAT_EQ(p.start_deg, 10.f);
    EXPECT_FLOAT_EQ(p.speed_dps, 3600.f);
    EXPECT_NEAR(p.span_deg, 11.f, 1e-5);
    for (int i = 0; i < 12; ++i) {
        EXPECT_NEAR(p.angle_deg[static_cast<std::size_t>(i)], 10.0 + i, 1e-4);
        EXPECT_EQ(p.range_mm[static_cast<std::size_t>(i)], 600 + i);
    }
}

TEST(Ld06, ParsePacketWraps360) {
    Ld06Packet p;
    ASSERT_TRUE(parse_packet(make_packet(355.0, 6.0, flat(500)), p));
    EXPECT_NEAR(p.angle_deg[0], 355.f, 1e-4);
    EXPECT_NEAR(p.angle_deg[11], 6.f, 1e-4);
    EXPECT_NEAR(p.angle_deg[6], 1.f, 1e-4);
}

TEST(Ld06, BadCrcRejectedAndJunkSkipped) {
    const auto good = make_packet(10.0, 21.0, flat(600));
    auto bad = good;
    bad[10] ^= 0xFF;  // sai CRC
    std::vector<uint8_t> buf{0x00, 0x13, 0x54};
    buf.insert(buf.end(), bad.begin(), bad.end());
    buf.insert(buf.end(), good.begin(), good.end());
    buf.insert(buf.end(), good.begin(), good.begin() + 20);  // goi chua du
    std::vector<RawPacket> out;
    EXPECT_EQ(split_packets(buf, out), 1);
    ASSERT_EQ(out.size(), 1U);
    EXPECT_TRUE(std::equal(good.begin(), good.end(), out[0].bytes.begin()));
    EXPECT_EQ(out[0].bytes_after, 20U);
    EXPECT_EQ(buf, std::vector<uint8_t>(good.begin(), good.begin() + 20));  // phan chua du giu lai
}

TEST(Ld06, AssemblerCompletesScanWhenAngleWraps) {
    ScanAssembler asm_;
    Scan done;
    int completed = 0;
    for (int i = 0; i < 5; ++i) {
        const double s = std::array<double, 5>{0.0, 120.0, 240.0, 350.0, 5.0}[static_cast<std::size_t>(i)];
        Ld06Packet p;
        ASSERT_TRUE(parse_packet(make_packet(s, s + 11, flat(600)), p));
        std::array<int64_t, kLd06Points> t{};
        t.fill(i * 1000);
        if (asm_.push(p, t, done)) ++completed;
    }
    EXPECT_EQ(completed, 1);  // 5 < 350 - 180 -> xong vong 4 goi dau
    ASSERT_EQ(done.size(), 48U);
    EXPECT_EQ(done.t_ns[0], 0);
    EXPECT_EQ(done.t_ns[47], 3000);
}

TEST(Ld06, PointTimesInterpolatedByAngle) {
    Ld06Packet p;
    ASSERT_TRUE(parse_packet(make_packet(10.0, 21.0, flat(600)), p));  // 11 do / 3600 do/s = 3.056 ms
    std::array<int64_t, kLd06Points> t{};
    point_times_ns(p, 1'000'000'000, t);
    EXPECT_EQ(t[11], 1'000'000'000);
    EXPECT_EQ(t[0], 1'000'000'000 - std::llround(11.0 / 3600 * 1e9));
    for (std::size_t i = 1; i < t.size(); ++i) EXPECT_GT(t[i], t[i - 1]);
    Ld06Packet still;
    ASSERT_TRUE(parse_packet(make_packet(10.0, 21.0, flat(600), 200, 0), still));  // chua quay: khong noi suy
    point_times_ns(still, 5, t);
    for (const auto v : t) EXPECT_EQ(v, 5);
}

TEST(Ld06, UartDelays) {
    EXPECT_EQ(packet_tx_ns(kBaud), std::llround(47 * 10 / static_cast<double>(kBaud) * 1e9));  // ~2.04 ms
    EXPECT_EQ(rx_end_ns(1'000'000'000, 10, kBaud), 1'000'000'000 - std::llround(10 * 10 / static_cast<double>(kBaud) * 1e9));
}

TEST(Ld06, ScanTimesShiftedByUartDelayAndExtraLatency) {
    ScanBuffer buf(10, kBaud, 1.0);
    for (int i = 0; i < 5; ++i) {  // vong 1 = 4 goi dau
        const double s = kStarts[static_cast<std::size_t>(i % 4)];
        buf.feed(make_packet(s, s + 11, flat(600)), 1'000'000'000 + i * 1'000'000);
    }
    Scan scan;
    ASSERT_TRUE(buf.latest_scan(scan));
    EXPECT_EQ(scan.t_ns[11], 1'000'000'000 - packet_tx_ns(kBaud) - 1'000'000);  // tru truyen goi + tre them 1 ms
    EXPECT_LT(scan.t_ns[0], scan.t_ns[11]);
}

TEST(Ld06, CollectedSinceThenNewScans) {
    ScanBuffer buf(3, kBaud, 0.0);
    const auto feed = [&](int n, int64_t t0) {
        for (int i = 0; i < n; ++i) {
            const double s = kStarts[static_cast<std::size_t>(i % 4)];
            buf.feed(make_packet(s, s + 11, flat(600)), t0 + i * 25'000'000LL);
        }
    };
    feed(4 * 5 + 1, 1'000'000'000);  // 5 vong xong, bo dem giu 3 vong cuoi
    std::vector<Scan> collected;
    std::size_t next = 0;
    bool truncated = false;
    buf.collected_since(0, collected, next, truncated);
    EXPECT_EQ(collected.size(), 3U);
    EXPECT_TRUE(truncated);  // vong 1, 2 da bi day khoi bo dem
    const int64_t t1 = collected[1].t_ns.front();
    buf.collected_since(t1, collected, next, truncated);
    EXPECT_EQ(collected.size(), 2U);
    EXPECT_FALSE(truncated);
    feed(9, 2'000'000'000);  // goi 1 noi vao vong dang ghep, goi 5 va 9 ket thuc 2 vong
    Scan s;
    int got = 0;
    while (got < 5 && buf.wait_scan(next, 0.05, s)) ++got;
    EXPECT_EQ(got, 2);  // chi vong MOI sau luc goi collected_since; roi het 0.05 s -> dung
    EXPECT_EQ(buf.count(), 7U);
    EXPECT_TRUE(buf.wait_until(s.t_ns.back(), 0.01));  // vong moi nhat da phu toi diem cuoi cua no
    EXPECT_FALSE(buf.wait_until(s.t_ns.back() + 1'000'000'000, 0.01));
}
