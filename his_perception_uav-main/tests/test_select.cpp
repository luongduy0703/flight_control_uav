// s3: decide, rank, tracker, measure (giong tests/test_decide.py, test_rank.py, test_measure.py).
#include <gtest/gtest.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "helpers.hpp"
#include "uavp/stages/s3_select.hpp"

using namespace uavp;

namespace {

const DecideConfig kCfg{5, 80.0, 0.7, 5.0, 10};
constexpr double kD_m = 2.0;

Observation obs(double W_mm, float conf, double area = 100.0) {
    Observation o;
    o.W_mm = W_mm;
    o.conf = conf;
    o.area_px = area;
    return o;
}

// 4 dinh hinh chu nhat tren san (met), canh dai l theo huong angle, chieu len anh co meo.
std::vector<cv::Point2d> floor_rect_px(cv::Point2d c, double w_m, double l_m, double angle_deg, const CameraModel& cam) {
    const double a = angle_deg * CV_PI / 180.0;
    const cv::Point2d u(std::cos(a) * l_m / 2, std::sin(a) * l_m / 2), v(-std::sin(a) * w_m / 2, std::cos(a) * w_m / 2);
    std::vector<cv::Point3d> pts;
    for (const auto& p : {c - u - v, c + u - v, c + u + v, c - u + v}) pts.emplace_back(p.x, p.y, kD_m);
    std::vector<cv::Point2d> px;
    cv::projectPoints(pts, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), cam.K_px, cam.dist_coeffs, px);
    return px;
}

Detection det_at(double cx, double cy) {
    return {static_cast<float>(cx - 10), static_cast<float>(cy - 10), static_cast<float>(cx + 10),
            static_cast<float>(cy + 10), 0.9f, 0, {}};
}

}  // namespace

TEST(Decide, DeferUntilNMinThenOk) {
    Track t;
    for (int i = 0; i < 4; ++i) update_state(t, obs(60, 0.9f), kCfg);
    EXPECT_EQ(t.state, State::kDefer);
    update_state(t, obs(60, 0.9f), kCfg);
    EXPECT_EQ(t.state, State::kOk);
}

TEST(Decide, TooBigUsesMedianAndCheckedBeforeLowConf) {
    Track t;
    for (const double w : {90.0, 90.0, 90.0, 10.0, 10.0}) update_state(t, obs(w, 0.5f), kCfg);
    EXPECT_EQ(t.state, State::kAbort);
    EXPECT_EQ(t.reason, Reason::kTooBig);
}

TEST(Decide, LowConfUsesMean) {
    Track t;
    for (const float c : {0.9f, 0.9f, 0.5f, 0.5f, 0.5f}) update_state(t, obs(60, c), kCfg);
    EXPECT_EQ(t.reason, Reason::kLowConf);
}

TEST(Decide, SmallMaskStreakAbortsAndGoodMaskResets) {
    Track t;
    for (int i = 0; i < 9; ++i) update_state(t, obs(kNaN, 0.9f, 0.0), kCfg);
    EXPECT_EQ(t.state, State::kDefer);
    update_state(t, obs(60, 0.9f), kCfg);  // lan do tot: dat lai chuoi
    EXPECT_EQ(t.mask_small_streak, 0);
    for (int i = 0; i < 10; ++i) update_state(t, obs(kNaN, 0.9f, 0.0), kCfg);
    EXPECT_EQ(t.reason, Reason::kMaskSmall);
    update_state(t, obs(60, 0.9f), kCfg);  // da quyet dinh: giu nguyen
    EXPECT_EQ(t.state, State::kAbort);
}

TEST(Tracker, FirstGroupNearestCenterThenSkipChecked) {
    std::vector<Track> tracks;
    const std::vector<Detection> dets{det_at(900, 500), det_at(100, 100), det_at(960, 540), det_at(1500, 900)};
    const auto g = select_group(tracks, dets, {960, 540}, 2, 40, 0);
    ASSERT_EQ(g, (std::vector<int>{0, 1}));
    EXPECT_NEAR(tracks[0].center_px.x, 960, 1e-9);  // gan tam nhat truoc
    EXPECT_NEAR(tracks[1].center_px.x, 900, 1e-9);
    const auto g2 = select_group(tracks, dets, {960, 540}, 2, 40, 1);  // bo vat da xet
    ASSERT_EQ(g2.size(), 2U);
    EXPECT_NEAR(tracks[2].center_px.x, 1500, 1e-9);
}

TEST(Tracker, MatchRadiusAndOneDetectionPerTrack) {
    std::vector<Track> tracks(2);
    tracks[0].detection = det_at(100, 100);  // ghep theo tam bbox du doan (van toc 0: = bbox hien tai)
    tracks[1].detection = det_at(110, 100);
    tracks[0].center_px = {100, 100};
    tracks[1].center_px = {110, 100};
    const std::vector<Detection> dets{det_at(105, 100), det_at(300, 300)};
    match(tracks, dets, 40, 3);
    EXPECT_EQ(tracks[0].last_seen_frame + tracks[1].last_seen_frame, 3U);  // chi mot track lay detection gan
    EXPECT_NE(tracks[0].center_px, tracks[1].center_px);
}

TEST(Rank, ByDistanceToCenterOkOnly) {
    std::vector<Track> tracks(3);
    for (int i = 0; i < 3; ++i) {
        SegObject s;
        s.center_px = {960.0 + 100.0 * (2 - i), 540.0};
        Observation o = obs(60, 0.9f);
        o.seg = s;
        tracks[static_cast<std::size_t>(i)].observations.push_back(o);
        tracks[static_cast<std::size_t>(i)].state = i == 2 ? State::kAbort : State::kOk;
    }
    EXPECT_EQ(rank_ok(tracks, {960, 540}), (std::vector<int>{1, 0}));
}

TEST(Measure, Rotated50x100mmRectangleGivesW50mm) {
    const CameraModel cam = test::camera();
    for (const cv::Point2d c : {cv::Point2d(0, 0), cv::Point2d(0.6, -0.3)}) {  // giua anh va gan goc (meo lon)
        const auto px = floor_rect_px(c, 0.050, 0.100, 30.0, cam);
        std::vector<cv::Point2d> norm;
        cv::undistortPoints(px, norm, cam.K_px, cam.dist_coeffs);
        std::vector<cv::Point2f> m;
        for (const auto& p : norm) m.emplace_back(static_cast<float>(p.x * kD_m), static_cast<float>(p.y * kD_m));
        const SizeMm s = rect_size_mm(m);
        EXPECT_NEAR(s.W_mm, 50.0, 0.1);
        EXPECT_NEAR(s.L_mm, 100.0, 0.1);
        EXPECT_NEAR(s.yaw_deg, 30.0, 0.1);
    }
}

TEST(Measure, FromRasterizedMask) {
    const CameraModel cam = test::camera();
    const auto px = floor_rect_px({0.1, 0.05}, 0.050, 0.100, 30.0, cam);
    int x0 = 1 << 20, y0 = 1 << 20;
    for (const auto& p : px) x0 = std::min(x0, static_cast<int>(p.x) - 20), y0 = std::min(y0, static_cast<int>(p.y) - 20);
    cv::Mat mask = cv::Mat::zeros(120, 120, CV_8U);
    std::vector<cv::Point> poly;
    for (const auto& p : px) poly.emplace_back(round_half_even(p.x - x0), round_half_even(p.y - y0));
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, 1);
    SegObject obj;
    ASSERT_TRUE(seg_object({mask, x0, y0, 0.9f, 0}, obj));
    EXPECT_NEAR(obj.yaw_deg, 30.0, 2.0);
    EXPECT_DOUBLE_EQ(obj.area_px, cv::countNonZero(mask));
    const Observation o = measure(obj, 0.8f, 7, cam, kD_m);
    EXPECT_NEAR(o.W_mm, 50.0, 4.0);
    EXPECT_NEAR(o.L_mm, 100.0, 4.0);
    EXPECT_EQ(o.frame_idx, 7U);
    ASSERT_TRUE(o.seg.has_value());
    const Observation none = measure(std::nullopt, 0.9f, 0, cam, kD_m);
    EXPECT_EQ(none.area_px, 0.0);
    EXPECT_TRUE(std::isnan(none.W_mm));
}

TEST(Tracker, PredictedBoxFollowsVelocityAndMaskCorrects) {
    std::vector<Track> tracks;
    const std::vector<Detection> dets{det_at(500, 400)};
    (void)select_group(tracks, dets, {960, 540}, 1, 40, 0);
    Track& t = tracks[0];
    update_from_mask(t, {502, 401}, 0, 0.5);  // khung 0: s1 + mask
    EXPECT_NEAR(t.center_px.x, 500, 1e-9);    // bbox tu s1 o khung nay: giu nguyen
    update_from_mask(t, {522, 401}, 1, 0.5);  // khung 1 (chi s2): mask dich 20 px
    EXPECT_NEAR(t.vel_px.x, 10.0, 1e-9);      // 0.5 x 20 + 0.5 x 0
    EXPECT_NEAR(t.center_px.x, 520, 1e-9);    // bbox dich theo mask thuc
    EXPECT_NEAR(predicted_center(t, 3).x, 540, 1e-9);  // 2 khung x 10 px
    // khung 3: s1 thay vat o 541 -> ghep nho du doan (cach tam cu 21 px nhung cach du doan 1 px)
    match(tracks, std::vector<Detection>{det_at(541, 400)}, 5, 3);
    EXPECT_EQ(t.last_seen_frame, 3U);
    EXPECT_NEAR(t.center_px.x, 541, 1e-9);
}

TEST(Tracker, SelectGroupSkipsTrackAtPredictedPosition) {
    std::vector<Track> tracks;
    (void)select_group(tracks, std::vector<Detection>{det_at(500, 400)}, {960, 540}, 1, 40, 0);
    tracks[0].vel_px = {30, 0};  // 30 px / khung
    const auto g = select_group(tracks, std::vector<Detection>{det_at(590, 400)}, {960, 540}, 1, 40, 3);
    EXPECT_TRUE(g.empty());  // 590 = du doan khung 3 -> vat da co track
}
