// s3: decide, rank, tracker, measure (giong tests/test_decide.py, test_rank.py, test_measure.py).
#include <gtest/gtest.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "helpers.hpp"
#include "uavp/pipeline/session.hpp"
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

// To giay ArUco diem cat canh: bbox chua tam marker hoac tam bbox gan tam marker -> khong chon.
TEST(Exclude, BoxContainingOrNearMarkerIsExcluded) {
    const ExcludeZone z{{800.0, 900.0}, 100.0};
    Detection paper;
    paper.x1 = 700.f; paper.y1 = 840.f; paper.x2 = 910.f; paper.y2 = 1040.f;  // chua (800, 900)
    Detection near_box;
    near_box.x1 = 850.f; near_box.y1 = 950.f; near_box.x2 = 900.f; near_box.y2 = 1000.f;  // tam (875, 975): cach 106 px
    Detection object;
    object.x1 = 250.f; object.y1 = 520.f; object.x2 = 350.f; object.y2 = 610.f;  // hop carton ben trai
    EXPECT_TRUE(in_zone(paper, z));
    EXPECT_FALSE(in_zone(near_box, z));
    EXPECT_TRUE(in_zone(near_box, ExcludeZone{{800.0, 900.0}, 120.0}));
    EXPECT_FALSE(excluded(object, {z}));
    EXPECT_FALSE(excluded(paper, {}));
    EXPECT_EQ(name(Reason::kLaunchMarker), "launch_marker");
}

// Buoc detection: canh ngan bbox doi ra mm bang D (do cao EKF) -> loai som vat qua to.
TEST(DetectSize, BboxShortSideInMmAtFloorDistance) {
    const CameraModel cam = test::camera();
    const double f = cam.K_px(0, 0);
    Detection d;
    d.x1 = 100.f; d.y1 = 100.f; d.x2 = 100.f + static_cast<float>(f * 0.25); d.y2 = 100.f + static_cast<float>(f * 0.10);
    EXPECT_NEAR(bbox_short_mm(d, 1.0, cam), 100.0, 0.5);  // canh ngan f x 0.1 px tai 1 m = 100 mm
    EXPECT_NEAR(bbox_short_mm(d, 0.8, cam), 80.0, 0.5);
    EXPECT_TRUE(std::isnan(bbox_short_mm(d, kNaN, cam)));
    EXPECT_TRUE(std::isnan(bbox_short_mm(d, 0.0, cam)));
}

// planA: khoa abort do kich thuoc -> vat 92 mm (> w_max_mm) van OK; W van duoc do.
TEST(Decide, SizeAbortDisabledKeepsBigObject) {
    DecideConfig cfg = kCfg;
    cfg.size_abort = false;
    Track t;
    for (int i = 0; i < 5; ++i) update_state(t, obs(92, 0.86f), cfg);
    EXPECT_EQ(t.state, State::kOk);
    EXPECT_NEAR(median_W_mm(t), 92.0, 1e-9);
}

// Bay 2026-11-22 20:07: UAV hover, van toc tam mask (rung) bi ngoai suy qua 10 khung mat mask -> ROI / diem ghep troi
// ~90 px khoi vat, s1 thay vat nhung khong ghep duoc -> mask_small. Mat mask -> van toc = 0, giu vi tri biet cuoi.
TEST(Tracker, LostMaskStopsVelocityExtrapolation) {
    std::vector<Track> tracks;
    (void)select_group(tracks, std::vector<Detection>{det_at(500, 400)}, {960, 540}, 1, 40, 0);
    Track& t = tracks[0];
    update_from_mask(t, {500, 400}, 0, 0.5);
    update_from_mask(t, {520, 400}, 1, 0.5);  // van toc 10 px / khung
    ASSERT_NEAR(t.vel_px.x, 10.0, 1e-9);
    SegmenterConfig cfg;
    cfg.classes = {"box_top", "bottle_top"};
    cfg.input_px = 32;
    cfg.max_det = 10;
    const std::vector<std::vector<float>> none{{16}, {16}, {16}, {8}, {0.01f}, {0.01f}, {1.f}};  // diem < nguong
    Segmenter seg(cfg, std::make_unique<test::FakeInfer>(
                           std::vector<Tensor>{test::tensor2(none), test::tensor3(1, 8, 8, 1.f)}));
    SegObject obj;
    for (uint32_t k = 2; k < 12; ++k) EXPECT_FALSE(track_segment(t, k, cv::Mat::zeros(1080, 1920, CV_8UC3), {0.5}, seg, obj));
    EXPECT_NEAR(t.vel_px.x, 0.0, 1e-9);
    EXPECT_NEAR(predicted_center(t, 12).x, 520, 1e-9);  // truoc day: 520 + 11 x 10 = 630
    match(tracks, std::vector<Detection>{det_at(530, 405)}, 40, 12);  // s1 thay lai vat -> ghep duoc
    EXPECT_EQ(t.last_seen_frame, 12U);
    EXPECT_NEAR(t.center_px.x, 530, 1e-9);
}

// --- Che do the gioi (2026-11-22): vat = diem tren san, du doan bang tu the camera luc chup ----------------------
namespace {

CameraModel test_camera() {
    CameraModel c;
    c.K_px = cv::Matx33d(823.3, 0, 960, 0, 823.3, 540, 0, 0, 1);
    c.dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    return c;
}

// Camera nhin thang xuong (z_C = xuong), nghieng `tilt_deg` quanh truc Bac, dat o (n, e, z) mm.
FloorView view_at(const CameraModel& cam, double tilt_deg, cv::Vec3d t_mm, double floor_z_mm = 1000.0) {
    const double a = tilt_deg * CV_PI / 180.0;
    const cv::Matx33d Rx(1, 0, 0, 0, std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a));
    const cv::Matx33d down(0, -1, 0, 1, 0, 0, 0, 0, 1);  // x_C -> Dong, y_C -> Nam, z_C -> xuong
    FloorView v;
    v.pose.R_WC = Rx * down;
    v.pose.t_WC_mm = t_mm;
    v.pose.has_position = true;
    v.floor_z_mm = floor_z_mm;
    v.camera = &cam;
    v.radius_mm = 80.0;
    v.same_mm = 100.0;
    return v;
}

Detection det_px(cv::Point2d c) { return det_at(c.x, c.y); }

}  // namespace

TEST(WorldTrack, FloorPointRoundTrip) {
    const CameraModel cam = test_camera();
    const FloorView v = view_at(cam, 8.0, {30, -20, 0});
    const cv::Point3d p{250, -120, 1000};
    const auto uv = floor_pixel(v, p);
    ASSERT_TRUE(uv);
    const auto q = floor_point(v, *uv);
    ASSERT_TRUE(q);
    EXPECT_NEAR(q->x, p.x, 1e-6);
    EXPECT_NEAR(q->y, p.y, 1e-6);
    EXPECT_NEAR(q->z, 1000.0, 1e-6);
}

// UAV nghieng 10 do + troi 5 cm giua 2 khung: vat dich > 140 px tren anh. Bam bang px (40 px) mat vat; bam bang diem
// tren san du doan dung vi tri va ghep lai detection.
TEST(WorldTrack, PredictsThroughTiltAndMatchesInMm) {
    const CameraModel cam = test_camera();
    const cv::Point3d obj{200, 100, 1000};
    const FloorView v0 = view_at(cam, 0.0, {0, 0, 0});
    std::vector<Track> tracks;
    ASSERT_EQ(select_group(tracks, std::vector<Detection>{det_px(*floor_pixel(v0, obj))}, {960, 540}, 1, 40, 0, &v0)
                  .size(), 1U);
    Track& t = tracks[0];
    ASSERT_TRUE(t.has_world);
    EXPECT_NEAR(t.p_W_mm.x, obj.x, 0.01);  // bbox luu float
    const FloorView v1 = view_at(cam, 10.0, {50, 0, 0});
    const cv::Point2d u1 = *floor_pixel(v1, obj);
    EXPECT_GT(cv::norm(u1 - t.center_px), 140.0);                    // vat dich manh tren anh
    EXPECT_LT(cv::norm(predicted_center(t, 1, &v1) - u1), 0.01);      // du doan bang tu the: dung cho
    EXPECT_GT(cv::norm(predicted_center(t, 1) - u1), 140.0);          // khong tu the (px cu): lech
    match(tracks, std::vector<Detection>{det_px(u1 + cv::Point2d(20, -15))}, 40, 1, &v1);  // ~3 cm tren san
    EXPECT_EQ(t.last_seen_frame, 1U);
    // mask o khung 2 (nghieng nguoc): p_W cap nhat theo mask, van toc anh khong dung
    const FloorView v2 = view_at(cam, -10.0, {40, 10, 0});
    update_from_mask(t, *floor_pixel(v2, obj), 2, 0.5, &v2);
    EXPECT_EQ(t.vel_px, cv::Point2d(0, 0));
    EXPECT_LT(std::hypot(t.p_W_mm.x - obj.x, t.p_W_mm.y - obj.y), 40.0);
}

// Bay 2026-11-22 20:07: mot hop sinh ra 3 track (#0, #1 ABORT, #2). Tam tren san gan nhau -> CUNG MOT VAT.
TEST(WorldTrack, CloseCentersAreOneObject) {
    const CameraModel cam = test_camera();
    const cv::Point3d obj{200, 100, 1000};
    const FloorView v0 = view_at(cam, 0.0, {0, 0, 0});
    std::vector<Track> tracks;
    (void)select_group(tracks, std::vector<Detection>{det_px(*floor_pixel(v0, obj))}, {960, 540}, 1, 40, 0, &v0);
    tracks[0].state = State::kAbort;  // vd mask_small
    // Khung sau: UAV nghieng 12 do, troi; cung hop (lech 60 mm do sai so) + mot vat that khac cach 300 mm.
    const FloorView v1 = view_at(cam, 12.0, {-60, 40, 0});
    const cv::Point3d same{obj.x + 60, obj.y, 1000}, other{obj.x, obj.y - 300, 1000};
    const auto g = select_group(tracks,
                                std::vector<Detection>{det_px(*floor_pixel(v1, same)), det_px(*floor_pixel(v1, other))},
                                {960, 540}, 1, 40, 1, &v1);
    ASSERT_EQ(g.size(), 1U);  // khong tao lai track cho hop da ABORT
    EXPECT_NEAR(tracks[static_cast<std::size_t>(g[0])].p_W_mm.y, other.y, 0.01);
    // 2 detection cung khung cach 50 mm tren san -> mot vat (giu cai gan tam anh)
    std::vector<Track> t2;
    const cv::Point3d a{0, 0, 1000}, b{0, 50, 1000};
    const auto g2 = select_group(t2, std::vector<Detection>{det_px(*floor_pixel(v0, b)), det_px(*floor_pixel(v0, a))},
                                 {960, 540}, 2, 40, 0, &v0);
    ASSERT_EQ(g2.size(), 1U);
    EXPECT_NEAR(t2[0].p_W_mm.y, a.y, 0.01);
}
