// Bam vat sau khi khoa: he the gioi (tu the FC), Kalman, doi mask, optical flow / NCC, LockedTracker.
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "helpers.hpp"
#include "uavp/stages/s2_segment/mask_shape.hpp"
#include "uavp/tracking/locked_tracker.hpp"

using namespace uavp;

namespace {

LockConfig lock_cfg() {
    LockConfig c;
    c.search_scale = 2.0;
    return c;
}

// Anh van ngau nhien (nhieu goc) lon, cat cua so 1920 x 1080 dich (dx, dy): vat (moi thu) dich (dx, dy) tren anh.
cv::Mat textured(int dx_px, int dy_px) {
    static const cv::Mat base = [] {
        cv::Mat b(1280, 2200, CV_8UC3);
        cv::RNG rng(7);
        rng.fill(b, cv::RNG::UNIFORM, 0, 255);
        cv::GaussianBlur(b, b, cv::Size(7, 7), 2.0);
        return b;
    }();
    return base(cv::Rect(100 - dx_px, 100 - dy_px, 1920, 1080)).clone();
}

SegObject square_object(cv::Point2d center_px, int half_px) {
    const int side = 4 * half_px;
    cv::Mat mask = cv::Mat::zeros(side, side, CV_8U);
    mask(cv::Rect(side / 2 - half_px, side / 2 - half_px, 2 * half_px, 2 * half_px)).setTo(1);
    SegObject obj;
    EXPECT_TRUE(seg_object({mask, round_half_even(center_px.x) - side / 2, round_half_even(center_px.y) - side / 2, 0.9f, 0}, obj));
    return obj;
}

Detection box_at(cv::Point2d c, double half) {
    return {static_cast<float>(c.x - half), static_cast<float>(c.y - half), static_cast<float>(c.x + half),
            static_cast<float>(c.y + half), 0.9f, 0, {}};
}

}  // namespace

TEST(WorldPose, NedFromBodyAndRoundTrip) {
    const cv::Matx33d R = rotation_ned_from_body(0.0, 0.0, CV_PI / 2);  // mui UAV huong Dong
    const cv::Vec3d fwd = R * cv::Vec3d(1, 0, 0);
    EXPECT_NEAR(fwd[1], 1.0, 1e-12);
    const cv::Matx33d R2 = rotation_ned_from_body(0.1, -0.2, 0.3);
    EXPECT_NEAR(cv::norm(R2 * R2.t() - cv::Matx33d::eye()), 0.0, 1e-12);
    FcPose fc;
    fc.roll_rad = 0.05;
    fc.pitch_rad = -0.03;
    fc.yaw_rad = 1.0;
    fc.pos_ned_m = {2.0, -1.0, -5.0};
    fc.has_position = true;
    fc.range_m = 1.5;
    fc.has_range = true;
    BodyCamera bc;
    bc.t_BC_mm = {50, 0, 30};
    const CamPose cp = camera_pose(fc, bc);
    // x, y tu LOCAL_POSITION_NED; z = -range cos(roll) cos(pitch) (rangefinder thay z cua EKF); + offset camera
    const cv::Matx33d R_WB = rotation_ned_from_body(0.05, -0.03, 1.0);
    const cv::Vec3d expect = R_WB * bc.t_BC_mm + cv::Vec3d(2000.0, -1000.0, -1500.0 * std::cos(0.05) * std::cos(-0.03));
    EXPECT_NEAR(cv::norm(cp.t_WC_mm - expect), 0.0, 1e-9);
    EXPECT_TRUE(cp.has_position);
    const cv::Point3d p_C(12, -34, 900);
    const cv::Point3d back = to_camera(cp, to_world(cp, p_C));
    EXPECT_NEAR(cv::norm(back - p_C), 0.0, 1e-9);
}

TEST(WorldPose, RotatePixelMatchesHomographyNearCenter) {
    const CameraModel cam = test::camera();
    const cv::Matx33d R = rotation_ned_from_body(0.02, -0.01, 0.0);  // quay nho
    const cv::Point2d c = cam.center_px();
    const auto uv = rotate_pixel(c, R, cam);
    ASSERT_TRUE(uv.has_value());
    const cv::Vec3d h = rotation_homography(R, cam) * cv::Vec3d(c.x, c.y, 1);
    EXPECT_NEAR(uv->x, h[0] / h[2], 1.0);  // gan tam anh meo nho
    EXPECT_NEAR(uv->y, h[1] / h[2], 1.0);
    const auto same = rotate_pixel({300, 200}, cv::Matx33d::eye(), cam);
    EXPECT_NEAR(same->x, 300, 1e-6);
}

TEST(RoiKalman, FollowsConstantVelocityAndUsesControl) {
    RoiKalman kf(300.0, 50.0);
    kf.init({100, 100}, 0);
    for (int k = 1; k <= 30; ++k) {
        kf.predict(k * 33'000'000LL, {0, 0});
        kf.update({100 + 3.0 * k, 100}, 2.0);  // 3 px / khung = ~91 px/s
    }
    EXPECT_NEAR(kf.vel_px_s().x, 3.0 / 0.033, 5.0);
    const cv::Point2d before = kf.center_px();
    kf.predict(31 * 33'000'000LL, {0, 20});  // dau vao dieu khien tu tu the: +20 px theo y
    EXPECT_NEAR(kf.center_px().y - before.y, 20.0, 1e-6);
    EXPECT_NEAR(kf.center_px().x - before.x, 3.0, 0.5);
}

TEST(WarpMask, TranslationMovesOriginKeepsPixels) {
    SegObject o = square_object({500, 400}, 10);
    SegMask m = o.mask;
    const int area = cv::countNonZero(m.mask);
    cv::Mat buf;
    ASSERT_TRUE(warp_mask(m, cv::Matx33d(1, 0, 15, 0, 1, -8, 0, 0, 1), buf));
    EXPECT_EQ(m.x0_px, o.mask.x0_px + 15);
    EXPECT_EQ(m.y0_px, o.mask.y0_px - 8);
    EXPECT_EQ(cv::countNonZero(m.mask), area);
    EXPECT_EQ(cv::countNonZero(o.mask.mask), area);  // mask cu (co the dang o MaskFrame) khong bi ghi de
}

TEST(RoiFlow, OpticalFlowMeasuresShiftOnTexture) {
    RoiFlow flow(lock_cfg());
    const SegObject obj = square_object({960, 540}, 40);
    flow.reset(textured(0, 0), obj.mask, obj.center_px, 200);
    const FlowMeasure m = flow.measure(textured(12, -7), {960, 540});  // du doan sai 12 px: flow sua
    ASSERT_EQ(m.method, FlowMethod::kFlow);
    EXPECT_GE(m.n_good, 6);
    EXPECT_NEAR(m.center_px.x, 972, 1.0);
    EXPECT_NEAR(m.center_px.y, 533, 1.0);
    EXPECT_NEAR(m.M_prev_to_cur(0, 2), 12, 1.5);
}

TEST(RoiFlow, FewCornersFallsBackToNcc) {
    LockConfig cfg = lock_cfg();
    cfg.flow_min_points = 50;  // ep that bai flow
    RoiFlow flow(cfg);
    const SegObject obj = square_object({960, 540}, 40);
    flow.reset(textured(0, 0), obj.mask, obj.center_px, 240);
    const FlowMeasure m = flow.measure(textured(-10, 6), {955, 545});
    ASSERT_EQ(m.method, FlowMethod::kNcc);
    EXPECT_GT(m.ncc, 0.8);  // anh thu 1/2: lech nua pixel giua 2 cua so
    EXPECT_NEAR(m.center_px.x, 950, 2.0);
    EXPECT_NEAR(m.center_px.y, 546, 2.0);
}

TEST(LockedTracker, FollowsImageMotionWithoutPose) {
    const CameraModel cam = test::camera();
    LockedTracker tr(lock_cfg(), cam, BodyCamera{});
    const SegObject obj = square_object({960, 540}, 40);
    tr.lock(textured(0, 0), obj, box_at({960, 540}, 50), {0, 0, 800}, 0, std::nullopt);
    for (int k = 1; k <= 10; ++k) {
        const LockStep st = tr.step(textured(4 * k, 2 * k), k * 33'000'000LL, std::nullopt);
        EXPECT_EQ(st.mode, PoseMode::kNone);
        EXPECT_EQ(st.meas.method, FlowMethod::kFlow);
    }
    EXPECT_NEAR(tr.roi_box().center_px().x, 1000, 2.0);
    EXPECT_NEAR(tr.object().center_px.x, 1000, 2.0);  // mask da doi theo
    EXPECT_NEAR(tr.object().center_px.y, 560, 2.0);
    EXPECT_FALSE(tr.has_world());
}

TEST(LockedTracker, WorldModePredictsFromFcPose) {
    const CameraModel cam = test::camera();
    LockedTracker tr(lock_cfg(), cam, BodyCamera{});
    const cv::Point3d p_C0(0, 0, 800);
    const auto c0 = project_point_px(p_C0, cam);
    const SegObject obj = square_object(*c0, 40);
    CamPose pose0;
    pose0.has_position = true;
    tr.lock(textured(0, 0), obj, box_at(*c0, 50), p_C0, 0, pose0);
    ASSERT_TRUE(tr.has_world());
    EXPECT_NEAR(tr.p_W_mm().z, 800, 1e-9);
    CamPose pose1 = pose0;
    pose1.t_WC_mm = {-20, 0, 0};  // camera dich -20 mm theo x -> vat dich +20 mm tren anh he camera
    const LockStep st = tr.step(textured(0, 0), 33'000'000, pose1);  // anh khong doi: flow do 0, tu the du doan dich
    EXPECT_EQ(st.mode, PoseMode::kWorld);
    EXPECT_NEAR(st.p_C_pose_mm.x, 20.0, 1e-9);
    const auto c1 = project_point_px({20, 0, 800}, cam);
    EXPECT_NEAR(st.pose_pred_px.x, c1->x, 1e-6);
    EXPECT_GT(st.center_px.x, c0->x);  // Kalman: giua du doan tu the va do (flow = 0)
    EXPECT_LT(st.center_px.x, c1->x + 0.5);
}
