// uavp_real_time: lenh may tinh qua Pixhawk (COMMAND_LONG), trang thai FC, dieu kien hover; config _base_.
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>

#include "helpers.hpp"
#include "uavp/pipeline/start_gate.hpp"
#include "uavp/sources/mavlink.hpp"

using namespace uavp;

namespace {

constexpr int64_t kS = 1'000'000'000;

template <typename T>
void put(std::vector<uint8_t>& b, T v) {
    uint8_t raw[sizeof(T)];
    std::memcpy(raw, &v, sizeof(T));
    b.insert(b.end(), raw, raw + sizeof(T));
}

MavMessage msg(uint32_t id, const std::vector<uint8_t>& payload, uint8_t sysid = 1, uint8_t compid = 1) {
    auto pkt = pack_mavlink_v2(id, payload, 0, sysid, compid);
    std::vector<MavMessage> out;
    EXPECT_EQ(split_mavlink(pkt, out), 0);
    EXPECT_EQ(out.size(), 1U);
    return out.at(0);
}

std::vector<uint8_t> command_long(float p1, uint16_t cmd, uint8_t tsys, uint8_t tcomp) {
    std::vector<uint8_t> b;
    for (const float p : {p1, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}) put(b, p);
    put(b, cmd);
    b.insert(b.end(), {tsys, tcomp, 0});
    return b;
}

std::vector<uint8_t> heartbeat(uint32_t mode, uint8_t type, uint8_t autopilot, bool armed) {
    std::vector<uint8_t> b;
    put(b, mode);
    b.insert(b.end(), {type, autopilot, static_cast<uint8_t>(armed ? kMavModeFlagSafetyArmed : 0), 4, 3});
    return b;
}

std::vector<uint8_t> local_pos(float vx, float vy, float vz) {
    std::vector<uint8_t> b;
    put(b, uint32_t{1000});
    for (const float v : {0.f, 0.f, -1.f, vx, vy, vz}) put(b, v);
    return b;
}

std::vector<uint8_t> flow(float dist) {
    std::vector<uint8_t> b;
    put(b, uint64_t{0});
    for (const float v : {0.f, 0.f, dist}) put(b, v);
    put(b, int16_t{0});
    put(b, int16_t{0});
    b.insert(b.end(), {0, 200});
    return b;
}

// UAV hover tot: heartbeat arm LOITER, dang bay, van toc ~0, do cao flow 1.00 +- 1 cm trong 1.2 s; lenh bat dau.
void hover(FcPoseBuffer& buf, int64_t t0, bool with_start) {
    for (int k = 0; k <= 12; ++k) {
        const int64_t t = t0 + k * kS / 10;
        buf.push(msg(kMsgOpticalFlow, flow(1.0f + 0.01f * static_cast<float>(k % 2))), t);
        buf.push(msg(kMsgLocalPositionNed, local_pos(0.02f, -0.03f, 0.01f)), t);
        buf.push(msg(kMsgHeartbeat, heartbeat(5, 2, 3, true)), t);
        buf.push(msg(kMsgExtendedSysState, {0, kMavLandedStateInAir}), t);
    }
    if (with_start) buf.push(msg(kMsgCommandLong, command_long(1.f, 31010, 1, 191), 255, 190), t0 + 12 * kS / 10);
}

StartConfig start_cfg() {
    StartConfig c;
    c.modes = {5, 16};
    return c;
}

}  // namespace

TEST(Start, CommandPackParseAndAck) {
    const MavMessage m = msg(kMsgCommandLong, command_long(1.f, 31010, 1, 191), 255, 190);
    EXPECT_EQ(m.sysid, 255);
    EXPECT_EQ(m.compid, 190);
    const MavCommandLong c = decode_command_long(m);
    EXPECT_EQ(c.command, 31010);
    EXPECT_EQ(c.target_component, 191);
    EXPECT_FLOAT_EQ(c.param[0], 1.f);
    const MavMessage ack = msg(kMsgCommandAck, command_ack_payload(31010, kMavResultAccepted, 255, 190), 1, 191);
    EXPECT_EQ(ack.payload[0] | (ack.payload[1] << 8), 31010);
}

TEST(Start, HoverAndCommandGiveGoWithFlowHeight) {
    MavlinkConfig mc;
    FcPoseBuffer buf(mc);
    const int64_t t0 = 100 * kS;
    hover(buf, t0, true);
    const FcStatus s = buf.status(t0 + 12 * kS / 10, 1.0);
    EXPECT_TRUE(s.height_from_flow);
    const StartGate g = check_start(s, start_cfg(), t0);
    EXPECT_TRUE(g.ok()) << g.reason;
    EXPECT_NEAR(g.height_m, 1.0, 0.011);
    DistanceConfig d;
    d.camera_offset_m = -0.05;
    EXPECT_NEAR(camera_floor_distance_m(1.0, 0.1, 0.0, d), std::cos(0.1) - 0.05, 1e-12);
}

TEST(Start, MissingConditionsReported) {
    MavlinkConfig mc;
    FcPoseBuffer buf(mc);
    const int64_t t0 = 100 * kS;
    hover(buf, t0, false);
    const int64_t now = t0 + 12 * kS / 10;
    EXPECT_FALSE(check_start(buf.status(now, 1.0), start_cfg(), t0).commanded);  // chua co lenh
    buf.push(msg(kMsgCommandLong, command_long(1.f, 31010, 1, 191), 255, 190), t0 - kS);  // lenh CU truoc luc cho
    EXPECT_FALSE(check_start(buf.status(now, 1.0), start_cfg(), t0).commanded);
    buf.push(msg(kMsgCommandLong, command_long(1.f, 31010, 1, 50), 255, 190), now);  // gui nham compid khac
    EXPECT_FALSE(check_start(buf.status(now, 1.0), start_cfg(), t0).commanded);
    buf.push(msg(kMsgCommandLong, command_long(1.f, 31010, 1, 191), 255, 190), now);
    buf.push(msg(kMsgLocalPositionNed, local_pos(0.5f, 0.f, 0.f)), now);  // dang bay ngang 0.5 m/s
    const StartGate g = check_start(buf.status(now, 1.0), start_cfg(), t0);
    EXPECT_TRUE(g.commanded);
    EXPECT_FALSE(g.still);
    EXPECT_NE(g.reason.find("dung yen"), std::string::npos);
    buf.push(msg(kMsgHeartbeat, heartbeat(0, kMavTypeGcs, kMavAutopilotInvalid, false)), now);  // heartbeat GCS: bo qua
    EXPECT_TRUE(buf.status(now, 1.0).armed);
    buf.push(msg(kMsgCommandLong, command_long(0.f, 31010, 1, 191), 255, 190), now + 1);  // lenh dung
    EXPECT_TRUE(stop_commanded(buf.status(now + 2, 1.0), now));
}

TEST(Start, HeightMustBeStableOverWholeWindow) {
    MavlinkConfig mc;
    FcPoseBuffer buf(mc);
    const int64_t t0 = 100 * kS;
    hover(buf, t0, true);
    buf.push(msg(kMsgOpticalFlow, flow(1.2f)), t0 + 12 * kS / 10);  // nhay 20 cm
    const StartGate g = check_start(buf.status(t0 + 12 * kS / 10, 1.0), start_cfg(), t0);
    EXPECT_FALSE(g.height_stable);
    EXPECT_FALSE(g.ok());
}

// start.require_hover = false: chi can lenh (Pixhawk hoac SPACE tren viewer), UAV de tren ban (chua arm, duoi dat).
TEST(Start, CommandOnlyStartsWithoutHover) {
    MavlinkConfig mc;
    FcPoseBuffer buf(mc);
    const int64_t t0 = 100 * kS;
    buf.push(msg(kMsgHeartbeat, heartbeat(0, 2, 3, false)), t0);  // STABILIZE, chua arm
    StartConfig c = start_cfg();
    c.require_hover = false;
    EXPECT_FALSE(check_start(buf.status(t0, 1.0), c, t0).ok());  // chua co lenh
    const StartGate by_viewer_off = check_start(buf.status(t0, 1.0), c, t0, true);
    EXPECT_FALSE(by_viewer_off.ok());  // accept_viewer_key = false: SPACE khong tinh
    c.accept_viewer_key = true;
    const StartGate v = check_start(buf.status(t0, 1.0), c, t0, true);
    EXPECT_TRUE(v.ok());
    EXPECT_TRUE(v.by_viewer);
    EXPECT_FALSE(v.hover());
    buf.push(msg(kMsgCommandLong, command_long(1.f, 31010, 1, 191), 255, 190), t0 + kS);
    const StartGate g = check_start(buf.status(t0 + kS, 1.0), c, t0);
    EXPECT_TRUE(g.ok());
    EXPECT_FALSE(g.by_viewer);
    c.require_hover = true;  // bay that: lenh thoi chua du
    EXPECT_FALSE(check_start(buf.status(t0 + kS, 1.0), c, t0).ok());
}

TEST(Start, PerceptionConfigInheritsBase) {
    const auto old = std::filesystem::current_path();
    std::filesystem::current_path(test::kRoot);
    const YAML::Node raw = load_raw("configs/run/real_time_perception.yaml", {});
    EXPECT_EQ(distance_config(raw["distance"]).source, "fixed");
    EXPECT_DOUBLE_EQ(distance_config(raw["distance"]).fixed_m, 1.0);
    EXPECT_DOUBLE_EQ(raw["remote"]["max_fps"].as<double>(), 5.0);
    EXPECT_EQ(raw["remote"]["port"].as<int>(), 8100);  // khoa khong ghi de: lay tu real_time.yaml
    EXPECT_FALSE(raw["_base_"].IsDefined());
    const Config c = config_from_raw(raw);
    EXPECT_EQ(c.detector.input_w_px, 960);
    const YAML::Node full = load_raw("configs/run/real_time.yaml", {});
    EXPECT_EQ(distance_config(full["distance"]).source, "fc");
    EXPECT_EQ(mavlink_config(full["sources"]["mavlink"]).command_id, 31010);
    EXPECT_EQ(start_config(full["start"]).modes, (std::vector<uint32_t>{5, 16}));
    std::filesystem::current_path(old);
}
