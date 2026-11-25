// Cong chat luong khung truoc s1 (support/frame_gate.hpp): be rong quy dao gyro, rolling shutter, do net tuong doi.
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "uavp/support/frame_gate.hpp"

using namespace uavp;

namespace {

constexpr double kF = 823.3;  // fx cua K_1920x1080
constexpr int64_t kMs = 1'000'000;

// w(t) quanh truc y camera (dich anh theo x), lay mau deu `step_ns` tu 0 toi t_end_ns.
template <typename F>
std::vector<RateSample> rates_y(F w_of_t, int64_t t_end_ns, int64_t step_ns) {
    std::vector<RateSample> r;
    for (int64_t t = 0; t <= t_end_ns + step_ns; t += step_ns)
        r.push_back({t, {0.0, w_of_t(static_cast<double>(t) / 1e9), 0.0}});
    return r;
}

FrameGateConfig gate_cfg() {
    FrameGateConfig c;
    c.enabled = true;
    c.max_blur_px = 1.5;
    c.max_rs_px = 3.0;
    c.readout_ms = 18.0;
    c.roll_radius_px = 1101.0;
    c.max_roll_px = 1.5;
    c.history = 10;
    c.min_history = 5;
    c.min_ratio = 0.7;
    c.sharp_rows = 32;
    c.sharp_band = 0.5;
    return c;
}

cv::Mat checker(int cell) {
    cv::Mat img(1080, 1920, CV_8UC3);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            const uint8_t v = ((x / cell + y / cell) % 2) != 0 ? 220 : 30;
            img.at<cv::Vec3b>(y, x) = {v, v, v};
        }
    return img;
}

}  // namespace

// Rung 0.05 do, 150 Hz, cua so 10 ms (1.5 chu ky): dich chuyen rong = 0, chieu dai quy dao = 6 A f, nhung vet nhoe that
// chi la be rong dinh - dinh = 2 A f (~1.44 px). Ham phai tra ve be rong.
TEST(FrameGate, BlurIsTrajectoryWidthNotNetNorPathLength) {
    const double A = 0.05 * CV_PI / 180.0, fr = 150.0;
    const auto w = [&](double t) { return A * 2.0 * CV_PI * fr * std::cos(2.0 * CV_PI * fr * t); };
    const auto r = rates_y(w, 10 * kMs, 50'000);  // 20 kHz
    const auto b = blur_px(r, 0, 10 * kMs, kF);
    ASSERT_TRUE(b.has_value());
    EXPECT_NEAR(*b, 2.0 * A * kF, 0.02);  // 1.437 px
    double net = 0.0, length = 0.0;
    for (std::size_t i = 0; i + 1 < r.size() && r[i + 1].t_ns <= 10 * kMs; ++i) {
        net += r[i].w_C[1] * 50e-6;
        length += std::abs(r[i].w_C[1]) * 50e-6;
    }
    EXPECT_LT(kF * std::abs(net), 0.05);         // cach "dich chuyen rong": ~0 (sai)
    EXPECT_NEAR(kF * length, 6.0 * A * kF, 0.05);  // cach "chieu dai quy dao": ~4.3 px (thua)
}

TEST(FrameGate, ConstantRateBlurIsRateTimesExposure) {
    const auto r = rates_y([](double) { return 0.5; }, 20 * kMs, kMs);  // 0.5 rad/s
    const auto b = blur_px(r, 2 * kMs, 2 * kMs + 2'500'000, kF);        // phoi sang 2.5 ms
    ASSERT_TRUE(b.has_value());
    EXPECT_NEAR(*b, kF * 0.5 * 2.5e-3, 1e-6);  // 1.03 px
    const auto m = max_perp_rate(r, 0, 18 * kMs);
    ASSERT_TRUE(m.has_value());
    EXPECT_NEAR(*m, 0.5, 1e-12);
}

TEST(FrameGate, WindowNotCoveredGivesNothing) {
    const auto r = rates_y([](double) { return 0.1; }, 10 * kMs, kMs);
    EXPECT_FALSE(blur_px(r, 5 * kMs, 30 * kMs, kF).has_value());
    EXPECT_FALSE(blur_px({}, 0, kMs, kF).has_value());
}

TEST(FrameGate, RatesFromPosesInCameraFrame) {
    std::vector<CamPoseSample> poses(3);
    const double step = 0.002;  // rad moi 20 ms quanh truc y camera -> 0.1 rad/s
    for (int i = 0; i < 3; ++i) {
        cv::Matx33d R;
        cv::Rodrigues(cv::Vec3d(0.0, step * i, 0.0), R);
        poses[static_cast<std::size_t>(i)] = {CamPose{R, {0, 0, 0}, false}, i * 20 * kMs};
    }
    const auto r = rates_from_poses(poses);
    ASSERT_EQ(r.size(), 3U);
    EXPECT_NEAR(r[0].w_C[1], 0.1, 1e-9);
    EXPECT_NEAR(r[0].w_C[0], 0.0, 1e-12);
    EXPECT_NEAR(r[1].w_C[1], 0.1, 1e-9);
}

TEST(FrameGate, StripSharpnessDropsWhenBlurred) {
    const cv::Mat sharp = checker(8);
    cv::Mat blurred;
    cv::GaussianBlur(sharp, blurred, cv::Size(0, 0), 3.0);
    const double s = strip_sharpness(sharp, 32, 0.5), b = strip_sharpness(blurred, 32, 0.5);
    EXPECT_GT(s, 0.0);
    EXPECT_LT(b, 0.3 * s);
    EXPECT_DOUBLE_EQ(strip_sharpness(cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(90, 90, 90)), 32, 0.5), 0.0);
}

// Trung vi truot: cho du lich su, khung nhoe (so voi chinh canh do) bi loai; canh tron (diem thap) KHONG bi loai
// neu cac khung truoc cung la canh do.
TEST(FrameGate, SharpnessRelativeToOwnRollingMedian) {
    FrameGate gate(gate_cfg());
    const cv::Mat sharp = checker(8);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(gate.evaluate(sharp, 0, 2500, kF, {}).reason, "warmup");
    EXPECT_TRUE(gate.evaluate(sharp, 0, 2500, kF, {}).pass);
    cv::Mat blurred;
    cv::GaussianBlur(sharp, blurred, cv::Size(0, 0), 3.0);
    const GateResult r = gate.evaluate(blurred, 0, 2500, kF, {});
    EXPECT_FALSE(r.pass);
    EXPECT_EQ(r.reason, "sharp");
    EXPECT_FALSE(r.gyro);

    FrameGate plain(gate_cfg());  // canh it chi tiet (o 40 px): diem tuyet doi thap hon nhung on dinh -> qua
    const cv::Mat coarse = checker(40);
    for (int i = 0; i < 5; ++i) (void)plain.evaluate(coarse, 0, 2500, kF, {});
    EXPECT_TRUE(plain.evaluate(coarse, 0, 2500, kF, {}).pass);
}

TEST(FrameGate, GyroRejectsBeforeReadingPixels) {
    FrameGate gate(gate_cfg());
    const cv::Mat img = checker(8);
    for (int i = 0; i < 6; ++i) (void)gate.evaluate(img, 0, 2500, kF, {});
    // 1 rad/s quanh y: nhoe f w t_exp = 2.06 px > 1.5
    const auto fast = rates_y([](double) { return 1.0; }, 40 * kMs, kMs);
    const GateResult r = gate.evaluate(img, kMs, 2500, kF, fast);
    EXPECT_TRUE(r.gyro);
    EXPECT_EQ(r.reason, "blur");
    // 0.15 rad/s: nhoe 0.31 px dat, rolling shutter f w T = 823 x 0.15 x 0.018 = 2.2 px dat
    const auto slow = rates_y([](double) { return 0.15; }, 40 * kMs, kMs);
    EXPECT_TRUE(gate.evaluate(img, kMs, 2500, kF, slow).pass);
    // 0.25 rad/s, phoi sang ngan 0.5 ms: nhoe 0.1 px dat nhung rolling shutter 3.7 px > 3
    const auto mid = rates_y([](double) { return 0.25; }, 40 * kMs, kMs);
    EXPECT_EQ(gate.evaluate(img, kMs, 500, kF, mid).reason, "rolling_shutter");
}

TEST(FrameGate, RollAboutOpticalAxisAtImageCorner) {
    FrameGate gate(gate_cfg());
    const cv::Mat img = checker(8);
    for (int i = 0; i < 6; ++i) (void)gate.evaluate(img, 0, 2500, kF, {});
    std::vector<RateSample> r;
    for (int64_t t = 0; t <= 40 * kMs; t += kMs) r.push_back({t, {0.0, 0.0, 1.0}});  // w_z 1 rad/s
    const GateResult g = gate.evaluate(img, kMs, 2500, kF, r);
    EXPECT_NEAR(g.blur_px, 0.0, 1e-9);        // khong dich tam anh
    EXPECT_NEAR(g.roll_px, 1101.0 * 2.5e-3, 1e-6);  // 2.75 px o goc anh
    EXPECT_EQ(g.reason, "roll");
}
