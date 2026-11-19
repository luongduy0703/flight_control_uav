// s4 fusion voi du lieu tong hop: camera nhin xuong, mat top hop 80 x 50 mm nghieng (a, b) quanh z = 600 mm, san
// 700 mm. LiDAR gia: R_CL dua mat quet (z_L = 0) ve mat y_C = 0 (u = x_C, w = -y_C), t = 0. UAV troi theo y (vuong
// goc mat quet) va troi do cao -> moi vong quet cat mat top o vi tri khac -> do duoc nghieng ca 2 huong.
#include <gtest/gtest.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "helpers.hpp"
#include "uavp/stages/s4_fusion.hpp"

using namespace uavp;

namespace {

constexpr double kTop = 600.0, kFloor = 700.0, kHalfL = 40.0, kHalfW = 25.0;
constexpr int64_t kFrameNs = 100'000'000;  // 10 Hz: moi khung mot vong quet

Extrinsics ext() {
    Extrinsics e;
    e.R_CL = cv::Matx33d(0, -1, 0, 0, 0, -1, 1, 0, 0);
    e.t_CL_mm = {0, 0, 0};
    e.angle_clockwise = false;
    return e;
}

FusionConfig cfg() {
    FusionConfig c;
    c.extrinsics = ext();
    c.min_pts = 30;
    c.lidar_window_s = 0.05;
    return c;
}

// Mat top o khung: z = kTop + dz + a x + b (y - dy) (he camera), vat dich dy theo y.
struct Motion {
    double dy_mm{0.0}, dz_mm{0.0};
};

SegObject top_object(const CameraModel& cam, double a, double b, Motion m) {
    std::vector<cv::Point3d> corners;
    for (const auto& [x, y] : {std::pair{-kHalfL, -kHalfW}, {kHalfL, -kHalfW}, {kHalfL, kHalfW}, {-kHalfL, kHalfW}})
        corners.emplace_back(x, y + m.dy_mm, kTop + m.dz_mm + a * x + b * y);
    std::vector<cv::Point2d> uv;
    cv::projectPoints(corners, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), cam.K_px, cam.dist_coeffs, uv);
    double xmin = 1e9, ymin = 1e9;
    for (const auto& p : uv) xmin = std::min(xmin, p.x), ymin = std::min(ymin, p.y);
    const int x0 = static_cast<int>(xmin) - 10, y0 = static_cast<int>(ymin) - 10;
    cv::Mat mask = cv::Mat::zeros(130, 160, CV_8U);
    std::vector<cv::Point> poly;
    for (const auto& p : uv) poly.emplace_back(round_half_even(p.x - x0), round_half_even(p.y - y0));
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, 1);
    SegObject obj;
    EXPECT_TRUE(seg_object({mask, x0, y0, 0.9f, 0}, obj));
    return obj;
}

// Tia theta (he L) -> huong (-sin, 0, cos) he C (mat quet y_C = 0). Cat mat top neu |x| <= L/2 va |y_vat| <= W/2.
Scan scan(double a, double b, Motion m, int64_t t0_ns, double step_deg = 0.5) {
    Scan s;
    int i = 0;
    const double y_obj = -m.dy_mm;  // duong quet y_C = 0 -> y tren vat
    for (double th = -30.0; th < 30.0 - 1e-9; th += step_deg, ++i) {
        const double t = th * CV_PI / 180.0;
        const double top = kTop + m.dz_mm + b * y_obj;
        const double s_top = top / (std::cos(t) + a * std::sin(t));
        const double x = -s_top * std::sin(t);
        const bool on_top = std::abs(x) <= kHalfL && std::abs(y_obj) <= kHalfW;
        const double r = on_top ? s_top : (kFloor + m.dz_mm) / std::cos(t);
        s.angle_deg.push_back(static_cast<float>(th));
        s.range_mm.push_back(static_cast<uint16_t>(std::lround(r)));
        s.intensity.push_back(200);
        s.t_ns.push_back(t0_ns + i * 1000);
    }
    return s;
}

struct Sequence {
    std::vector<MaskFrame> frames;
    std::vector<Scan> scans;
};

// n khung: vat troi dy = amp sin(...) theo y, troi do cao dz = drift_mm_s * t. with_pose: tu the camera that.
Sequence sequence(const CameraModel& cam, double a, double b, double amp_mm, double drift_mm_s, bool with_pose, int n = 12) {
    Sequence q;
    for (int k = 0; k < n; ++k) {
        const int64_t t = k * kFrameNs;
        const Motion m{amp_mm * std::sin(2.0 * CV_PI * k / n), drift_mm_s * static_cast<double>(t) / 1e9};
        std::optional<CamPose> pose;
        if (with_pose) {  // he W = he camera khung 0; vat dung yen -> camera dich (0, -dy, -dz)
            CamPose p;
            p.t_WC_mm = {0.0, -m.dy_mm, -m.dz_mm};
            p.has_position = true;
            pose = p;
        }
        q.frames.push_back({top_object(cam, a, b, m), pose, t, true});
        q.scans.push_back(scan(a, b, m, t));
    }
    return q;
}

double angle_deg(const cv::Vec3d& u, const cv::Vec3d& v) {
    return std::acos(std::clamp(u.dot(v) / (cv::norm(u) * cv::norm(v)), -1.0, 1.0)) * 180.0 / CV_PI;
}

}  // namespace

TEST(Fusion, LidarAngleClockwiseAndWrap) {
    EXPECT_NEAR(lidar_angle_rad(90.0, true, 0.0), -CV_PI / 2, 1e-12);
    EXPECT_NEAR(lidar_angle_rad(270.0, true, 0.0), CV_PI / 2, 1e-12);
    EXPECT_NEAR(lidar_angle_rad(350.0, false, 0.0), -10.0 * CV_PI / 180.0, 1e-12);
}

TEST(Fusion, ScanFiltersRangeIntensityAndAngle) {
    Scan s;
    for (const auto& [a, r, inten] : std::vector<std::tuple<float, uint16_t, uint8_t>>{
             {0.f, 600, 200}, {0.f, 0, 200}, {0.f, 10, 200}, {0.f, 13000, 200}, {180.f, 600, 200}, {0.f, 600, 50}}) {
        s.angle_deg.push_back(a);
        s.range_mm.push_back(r);
        s.intensity.push_back(inten);
        s.t_ns.push_back(static_cast<int64_t>(s.t_ns.size()));
    }
    std::vector<cv::Point3f> pts;
    std::vector<int64_t> t;
    scan_to_camera_mm(s, ext(), cfg(), pts, t);
    ASSERT_EQ(pts.size(), 1U);  // chi diem 0 con lai
    EXPECT_NEAR(pts[0].z, 600.f, 1e-4);
    EXPECT_EQ(t[0], 0);
}

TEST(Plane, RansacRecoversTiltAndDriftWithOutliers) {
    PlaneData d;
    std::mt19937_64 gen(1);
    std::uniform_real_distribution<double> U(-40, 40), W(-25, 25), T(-1, 0), N(-2, 2);
    for (int i = 0; i < 300; ++i) {
        const double u = U(gen), w = W(gen), dt = T(gen);
        const bool outlier = i % 5 == 0;  // 20 % diem san (cach 100 mm)
        d.u.push_back(u);
        d.w.push_back(w);
        d.dt.push_back(dt);
        d.z.push_back((outlier ? 700.0 : 600.0) + 0.1 * u - 0.2 * w + 15.0 * dt + N(gen));
        d.scan.push_back(static_cast<uint16_t>(i % 10));
    }
    std::mt19937_64 rng(0);
    PlaneFit fit;
    ransac_plane(d, {true, true}, 200, 6.0, std::tan(30.0 * CV_PI / 180.0), rng, fit);
    EXPECT_EQ(fit.n_inliers, 240);
    EXPECT_NEAR(fit.m.a, 0.1, 0.01);
    EXPECT_NEAR(fit.m.b, -0.2, 0.01);
    EXPECT_NEAR(fit.m.e, 15.0, 1.0);
    EXPECT_NEAR(median_offset_by_scan(d, fit), 600.0, 1.0);
    EXPECT_NEAR(fit.c2, 700.0, 2.0);  // san: mat song song thu 2
    EXPECT_EQ(fit.n2, 60);
}

TEST(Plane, MedianAcrossScansSmoothsRangeSteps) {
    // Moi vong: z bi lam tron theo bac 15.6 mm nhung lech pha khac nhau (UAV troi) -> median giua cac vong ~ that.
    PlaneData d;
    constexpr double kStep = 15.6, kTrue = 607.0;
    for (int s = 0; s < 11; ++s) {
        const double shift = kStep * (s - 5) / 11.0;  // vi tri tuong doi cua bac
        const double z_q = std::round((kTrue + shift) / kStep) * kStep - shift;
        for (int i = 0; i < 8; ++i) {
            d.u.push_back(i);
            d.w.push_back(0);
            d.dt.push_back(0);
            d.z.push_back(z_q);
            d.scan.push_back(static_cast<uint16_t>(s));
        }
    }
    PlaneFit fit;
    fit.inliers.assign(d.size(), 1);
    fit.m = {0, 0, 0, 0};
    EXPECT_NEAR(median_offset_by_scan(d, fit), kTrue, 3.0);  // 1 vong: sai toi 7.8 mm
}

TEST(Fusion, TiltedTopWithLateralDriftImageCompensation) {
    const CameraModel cam = test::camera();
    const double a = std::tan(6.0 * CV_PI / 180.0), b = std::tan(-8.0 * CV_PI / 180.0);
    const Sequence q = sequence(cam, a, b, 18.0, 20.0, false);
    Fusion fusion(cfg(), cam);
    const FusionResult r = fusion(q.frames, q.scans);
    ASSERT_EQ(r.status, FusionStatus::kOk);
    EXPECT_FALSE(r.pose_compensated);
    EXPECT_TRUE(r.normal_observable);
    EXPECT_GT(r.spread_w_mm, 20.0);
    const cv::Vec3d n_true = cv::normalize(cv::Vec3d(-a, -b, 1.0));
    EXPECT_LT(angle_deg(r.axis_z, n_true), 2.0);
    EXPECT_NEAR(r.tilt_deg, angle_deg(n_true, {0, 0, 1}), 2.0);
    EXPECT_NEAR(r.drift_mm_s, 20.0, 5.0);
    // khung tham chieu = khung cuoi: tam mat top o (0, dy, z)
    const double dy = 18.0 * std::sin(2.0 * CV_PI * 11 / 12), dz = 20.0 * 1.1;
    EXPECT_NEAR(r.p_obj_mm.y, dy, 3.0);
    EXPECT_NEAR(r.d_mm, kTop + dz + b * 0.0, 3.0);
    EXPECT_NEAR(std::abs(r.axis_x.dot(cv::Vec3d(1, 0, 0))), 1.0, 0.01);  // canh dai theo x
    EXPECT_NEAR(r.axis_y.dot(r.axis_z.cross(r.axis_x)), 1.0, 1e-9);     // he phai
    EXPECT_NEAR(r.W_mm, 2 * kHalfW, 4.0);
    EXPECT_NEAR(r.L_mm, 2 * kHalfL, 4.0);
}

TEST(Fusion, PoseCompensationFromFc) {
    const CameraModel cam = test::camera();
    const double a = std::tan(-5.0 * CV_PI / 180.0), b = std::tan(10.0 * CV_PI / 180.0);
    const Sequence q = sequence(cam, a, b, 18.0, 30.0, true);
    Fusion fusion(cfg(), cam);
    const FusionResult r = fusion(q.frames, q.scans);
    ASSERT_EQ(r.status, FusionStatus::kOk);
    EXPECT_TRUE(r.pose_compensated);
    EXPECT_TRUE(r.normal_observable);
    EXPECT_TRUE(std::isnan(r.drift_mm_s));  // tu the da bu do cao
    EXPECT_LT(angle_deg(r.axis_z, cv::normalize(cv::Vec3d(-a, -b, 1.0))), 1.5);
}

TEST(Fusion, NoLateralMotionNormalNotObservable) {
    const CameraModel cam = test::camera();
    const double a = std::tan(6.0 * CV_PI / 180.0);
    const Sequence q = sequence(cam, a, 0.0, 0.0, 0.0, false);
    Fusion fusion(cfg(), cam);
    const FusionResult r = fusion(q.frames, q.scans);
    ASSERT_EQ(r.status, FusionStatus::kOk);
    EXPECT_FALSE(r.normal_observable);  // chi do duoc nghieng doc mat quet
    EXPECT_NEAR(r.axis_z[1], 0.0, 1e-9);
    EXPECT_LT(angle_deg(r.axis_z, cv::normalize(cv::Vec3d(-a, 0.0, 1.0))), 1.5);
    EXPECT_NEAR(r.d_mm, kTop, 2.0);
}

TEST(Fusion, OnlyLidarNearMaskFramesIsUsed) {
    const CameraModel cam = test::camera();
    Sequence q = sequence(cam, 0.0, 0.0, 10.0, 0.0, false);
    for (auto& s : q.scans)
        for (auto& t : s.t_ns) t += 3 * kFrameNs / 10;  // quet sau khung 30 ms: trong cua so 50 ms -> dung
    Fusion fusion(cfg(), cam);
    EXPECT_GT(fusion(q.frames, q.scans).n_points, 0);
    for (auto& s : q.scans)
        for (auto& t : s.t_ns) t += 3 * kFrameNs / 10;  // 60 ms sau khung k = 40 ms truoc khung k + 1
    FusionConfig narrow = cfg();
    narrow.lidar_window_s = 0.02;  // cua so 20 ms: moi diem cach khung co mask gan nhat >= 40 ms -> bo
    Fusion fusion2(narrow, cam);
    const FusionResult r = fusion2(q.frames, q.scans);
    EXPECT_EQ(r.n_points, 0);
    EXPECT_EQ(r.status, FusionStatus::kInsufficient);
    EXPECT_TRUE(std::isnan(r.d_mm));
}

TEST(Fusion, OffFovPointFoldedByDistortionIsRejected) {
    // X/Z ~ 1.94 (ngoai anh): projectPoints gap ve gan tam anh do da thuc meo (K v1) -> khong tinh la trong mask.
    CameraModel cam;
    cam.K_px = cv::Matx33d(736.0713, 0.0, 979.9493, 0.0, 732.7414, 536.8896, 0.0, 0.0, 1.0);
    cam.dist_coeffs = (cv::Mat_<double>(1, 5) << -0.0810998, 0.0757834, 0.0010750, 0.0008024, -0.0317086);
    const std::vector<cv::Point3f> far{{-1165.8f, 65.9f, 600.f}};
    std::vector<cv::Point2f> uv;
    project_px(far, cam, uv);
    ASSERT_TRUE(uv[0].x > 0 && uv[0].x < 1920 && uv[0].y > 0 && uv[0].y < 1080);  // bi gap vao trong anh
    const SegMask seg{cv::Mat::ones(100, 100, CV_8U), static_cast<int>(uv[0].x) - 50, static_cast<int>(uv[0].y) - 50, 0.9f, 0};
    std::vector<uint8_t> keep;
    in_mask(far, seg.mask, seg, cam, keep);
    EXPECT_EQ(keep[0], 0);
}

TEST(Fusion, VisiblePxDropsFoldedAndBehindPoints) {
    const CameraModel cam = test::camera();
    const std::vector<cv::Point3f> pts{{0, 0, 600}, {-1165.8f, 65.9f, 600}, {0, 0, -600}, {100, 50, 700}};
    std::vector<cv::Point2f> uv;
    visible_px(pts, cam, 1920, 1080, uv);
    ASSERT_EQ(uv.size(), 2U);
    EXPECT_NEAR(uv[0].x, cam.center_px().x, 1e-3);
}
