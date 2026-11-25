// MAVLink: tach goi v1 / v2 (CRC + CRC_EXTRA), giai ma tu the, noi suy tu the theo thoi gian.
#include <gtest/gtest.h>

#include <cstring>

#include <sys/mman.h>

#include <unistd.h>

#include "uavp/sources/fc_shm.hpp"
#include "uavp/sources/mavlink.hpp"

using namespace uavp;

namespace {

// Dong goi MAVLink v2 (khong ky) giong thu vien chuan, de test.
std::vector<uint8_t> pack_v2(uint32_t msgid, const std::vector<uint8_t>& payload, uint8_t seq = 0) {
    std::vector<uint8_t> p{kMavStxV2, static_cast<uint8_t>(payload.size()), 0, 0, seq, 1, 1,
                           static_cast<uint8_t>(msgid & 0xFF), static_cast<uint8_t>((msgid >> 8) & 0xFF),
                           static_cast<uint8_t>((msgid >> 16) & 0xFF)};
    p.insert(p.end(), payload.begin(), payload.end());
    uint16_t crc = mavlink_crc(std::span(p).subspan(1));
    const auto extra = static_cast<uint8_t>(mavlink_crc_extra(msgid));
    crc = mavlink_crc(std::span(&extra, 1), crc);
    p.push_back(static_cast<uint8_t>(crc & 0xFF));
    p.push_back(static_cast<uint8_t>(crc >> 8));
    return p;
}

std::vector<uint8_t> pack_v1(uint32_t msgid, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> p{kMavStxV1, static_cast<uint8_t>(payload.size()), 0, 1, 1, static_cast<uint8_t>(msgid)};
    p.insert(p.end(), payload.begin(), payload.end());
    uint16_t crc = mavlink_crc(std::span(p).subspan(1));
    const auto extra = static_cast<uint8_t>(mavlink_crc_extra(msgid));
    crc = mavlink_crc(std::span(&extra, 1), crc);
    p.push_back(static_cast<uint8_t>(crc & 0xFF));
    p.push_back(static_cast<uint8_t>(crc >> 8));
    return p;
}

template <typename T>
void put(std::vector<uint8_t>& b, T v) {
    uint8_t raw[sizeof(T)];
    std::memcpy(raw, &v, sizeof(T));
    b.insert(b.end(), raw, raw + sizeof(T));
}

std::vector<uint8_t> attitude(uint32_t t_ms, float roll, float pitch, float yaw) {
    std::vector<uint8_t> b;
    put(b, t_ms);
    for (const float v : {roll, pitch, yaw, 0.f, 0.f, 0.f}) put(b, v);
    return b;
}

}  // namespace

TEST(Mavlink, CrcKnownValue) {
    const std::vector<uint8_t> data{'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    EXPECT_EQ(mavlink_crc(data), 0x6F91);  // CRC-16/MCRF4XX("123456789")
}

TEST(Mavlink, SplitV1V2SkipsGarbageUnknownAndBadCrc) {
    std::vector<uint8_t> buf{0x11, 0x22};  // rac truoc goi
    const auto a = pack_v2(kMsgAttitude, attitude(1000, 0.1f, -0.2f, 3.0f));
    buf.insert(buf.end(), a.begin(), a.end());
    auto unknown = pack_v2(24, std::vector<uint8_t>(30, 0));  // GPS_RAW_INT: khong dung -> bo qua ca goi
    buf.insert(buf.end(), unknown.begin(), unknown.end());
    auto bad = pack_v1(kMsgAttitude, attitude(2000, 0.f, 0.f, 0.f));
    bad[8] ^= 0xFF;  // hong payload
    buf.insert(buf.end(), bad.begin(), bad.end());
    std::vector<uint8_t> dist;
    put(dist, uint32_t{1500});
    for (const uint16_t v : {uint16_t{20}, uint16_t{1200}, uint16_t{153}}) put(dist, v);
    for (const uint8_t v : {uint8_t{0}, uint8_t{1}, kMavSensorRotationPitch270, uint8_t{0}}) dist.push_back(v);
    const auto d = pack_v2(kMsgDistanceSensor, dist);
    buf.insert(buf.end(), d.begin(), d.end());
    std::vector<MavMessage> out;
    const int crc_errors = split_mavlink(buf, out);
    EXPECT_GE(crc_errors, 1);
    ASSERT_EQ(out.size(), 2U);
    const MavAttitude att = decode_attitude(out[0]);
    EXPECT_EQ(att.time_boot_ms, 1000U);
    EXPECT_FLOAT_EQ(att.roll_rad, 0.1f);
    EXPECT_FLOAT_EQ(att.yaw_rad, 3.0f);
    const MavDistance dd = decode_distance(out[1]);
    EXPECT_FLOAT_EQ(dd.distance_m, 1.53f);
    EXPECT_TRUE(dd.downward);
    EXPECT_TRUE(buf.empty());
}

TEST(Mavlink, V2TrailingZerosTruncatedArePadded) {
    auto payload = attitude(5, 0.5f, 0.f, 0.f);
    payload.resize(8);  // MAVLink 2 cat cac byte 0 cuoi: pitch..yawspeed = 0
    auto pkt = pack_v2(kMsgAttitude, payload);
    std::vector<MavMessage> out;
    ASSERT_EQ(split_mavlink(pkt, out), 0);
    ASSERT_EQ(out.size(), 1U);
    const MavAttitude a = decode_attitude(out[0]);
    EXPECT_FLOAT_EQ(a.roll_rad, 0.5f);
    EXPECT_FLOAT_EQ(a.pitch_rad, 0.f);
}

TEST(Mavlink, PoseInterpolatesAttitudeAcrossYawWrapAndMapsClock) {
    MavlinkConfig cfg;
    cfg.max_age_s = 0.2;
    cfg.buffer_samples = 16;
    FcPoseBuffer buf(cfg);
    std::vector<MavMessage> out;
    auto p1 = pack_v2(kMsgAttitude, attitude(1000, 0.f, 0.f, 3.10f));
    auto p2 = pack_v2(kMsgAttitude, attitude(1020, 0.2f, 0.f, -3.10f));  // qua moc +-pi
    std::vector<uint8_t> all(p1);
    all.insert(all.end(), p2.begin(), p2.end());
    ASSERT_EQ(split_mavlink(all, out), 0);
    // nhan tre 5 ms va 8 ms -> offset = min = 5 ms: t_pi = time_boot + 5 ms
    buf.push(out[0], 1'005'000'000);
    buf.push(out[1], 1'028'000'000);
    FcPose pose;
    ASSERT_TRUE(buf.pose_at(1'015'000'000, pose));  // giua 2 mau
    EXPECT_NEAR(pose.roll_rad, 0.1, 1e-6);
    EXPECT_NEAR(std::abs(pose.yaw_rad), CV_PI, 1e-3);  // noi suy qua moc, khong ve 0
    EXPECT_FALSE(pose.has_position);
    EXPECT_FALSE(buf.pose_at(2'000'000'000, pose));  // qua max_age
}

TEST(FcShm, BridgeWritesPerceptionReadsAcrossRestart) {
    const std::string name = "/uavp_fc_test_" + std::to_string(::getpid());
    std::vector<MavMessage> msgs;
    auto a = pack_v2(kMsgAttitude, attitude(100, 0.1f, 0.2f, 0.3f));
    auto h = pack_v2(kMsgHeartbeat, std::vector<uint8_t>(9, 3));
    std::vector<uint8_t> all(a);
    all.insert(all.end(), h.begin(), h.end());
    ASSERT_EQ(split_mavlink(all, msgs), 0);
    ASSERT_EQ(msgs.size(), 2U);  // HEARTBEAT nay duoc dung (dem trong bridge)
    std::vector<MavMessage> out;
    std::vector<int64_t> t;
    std::size_t lost = 0;
    {
        FcShmWriter w(name, 8);
        FcShmReader r(name);
        w.push(msgs[0], 111);
        w.push(msgs[1], 222);
        ASSERT_TRUE(r.poll(out, t, lost));
        ASSERT_EQ(out.size(), 2U);
        EXPECT_EQ(t[1], 222);
        EXPECT_FLOAT_EQ(decode_attitude(out[0]).pitch_rad, 0.2f);
        EXPECT_EQ(w.header().n_attitude.load(), 1U);
        EXPECT_EQ(w.header().n_heartbeat.load(), 1U);
        for (int i = 0; i < 20; ++i) w.push(msgs[0], 1000 + i);  // nhieu hon 8 o: doc cham -> mat 12 tin
        ASSERT_TRUE(r.poll(out, t, lost));
        EXPECT_EQ(lost, 12U);
        ASSERT_EQ(out.size(), 8U);
        EXPECT_EQ(t.back(), 1019);
        ASSERT_TRUE(r.poll(out, t, lost));
        EXPECT_TRUE(out.empty());
        FcShmWriter restarted(name, 8);  // bridge khoi dong lai: nguoi doc dang mo tu doc tiep
        restarted.push(msgs[1], 5000);
        ASSERT_TRUE(r.poll(out, t, lost));
        ASSERT_EQ(out.size(), 1U);
        EXPECT_EQ(t[0], 5000);
    }
    ::shm_unlink(name.c_str());
}

TEST(Mavlink, FcRebootResetsClockOffset) {
    MavlinkConfig cfg;
    cfg.buffer_samples = 16;
    FcPoseBuffer buf(cfg);
    std::vector<MavMessage> out;
    auto before = pack_v2(kMsgAttitude, attitude(600'000, 0.f, 0.f, 0.f));   // FC da chay 600 s
    auto after = pack_v2(kMsgAttitude, attitude(2'000, 0.3f, 0.f, 0.f));    // FC khoi dong lai: 2 s
    std::vector<uint8_t> all(before);
    all.insert(all.end(), after.begin(), after.end());
    ASSERT_EQ(split_mavlink(all, out), 0);
    buf.push(out[0], 700'000'000'000);  // Pi: 700 s
    buf.push(out[1], 705'000'000'000);  // 5 s sau tren Pi
    FcPose p;
    ASSERT_TRUE(buf.pose_at(705'000'000'000, p));  // mau moi map dung ve luc nhan (khong lech 598 s)
    EXPECT_NEAR(p.roll_rad, 0.3, 1e-6);
}
