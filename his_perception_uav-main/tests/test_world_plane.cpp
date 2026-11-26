// s4 method world: slerp tu the, RANSAC 3 diem + PCA, dieu kien PLANE, giao tia, hinh chu nhat tren mat phang, va
// Fusion tren canh tong hop trong he the gioi: mat top hop 80 x 50 mm nghieng (a, b), z_W = 600 mm duoi camera luc dau,
// san 700 mm. LiDAR gia nhu test_fusion (R_CL: mat quet z_L = 0 -> y_C = 0, t = 0). UAV dich ngang vuong goc duong
// quet va lac roll trong khi quet -> diem phai deskew theo tu the luc t_i moi nam tren mot mat.
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "helpers.hpp"
#include "uavp/stages/s4_fusion.hpp"

using namespace uavp;

namespace {

constexpr double kDeg = CV_PI / 180.0;
constexpr double kTop = 600.0, kFloor = 700.0, kHalfL = 40.0, kHalfW = 25.0;
constexpr double kA = 0.10, kB = -0.15;  // z_W = kTop + kA x + kB y tren mat top

cv::Matx33d rot_x(double rad) {
    const double c = std::cos(rad), s = std::sin(rad);
    return {1, 0, 0, 0, c, -s, 0, s, c};
}
cv::Matx33d rot_z(double rad) {
    const double c = std::cos(rad), s = std::sin(rad);
    return {c, -s, 0, s, c, 0, 0, 0, 1};
}

double angle_deg(const cv::Vec3d& a, const cv::Vec3d& b) {
    return std::acos(std::clamp(std::abs(a.dot(b)) / (cv::norm(a) * cv::norm(b)), 0.0, 1.0)) / kDeg;
}

// --- Canh tong hop ------------------------------------------------------------------------------------------------
struct Motion {
    double sweep_y_mm{40.0};  // dich ngang vuong goc duong quet trong 2 s
    double roll_deg{3.0};     // lac roll (quanh truc doc duong quet), 2 Hz
    double y0_mm{kNaN};       // vi tri y luc t = 0 (nan: -sweep_y_mm / 2)
};

CamPose pose_at_t(double t_s, Motion m) {
    CamPose p;
    p.R_WC = rot_x(m.roll_deg * kDeg * std::sin(2.0 * CV_PI * 2.0 * t_s));
    const double y0 = std::isfinite(m.y0_mm) ? m.y0_mm : -m.sweep_y_mm / 2.0;
    p.t_WC_mm = {0.0, y0 + m.sweep_y_mm * t_s / 2.0, 0.0};
    p.has_position = true;
    return p;
}

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
    c.method = FusionMethod::kWorld;
    c.min_pts = 30;
    c.lidar_window_s = 0.05;
    c.world.window_s = 2.0;
    c.world.min_inliers = 40;
    return c;
}

// Do dai tia tu goc o, huong d (he W) toi mat top (neu trung trong hop) hoac san.
double cast(const cv::Vec3d& o, const cv::Vec3d& d) {
    const double s_top = (kTop + kA * o[0] + kB * o[1] - o[2]) / (d[2] - kA * d[0] - kB * d[1]);
    const cv::Vec3d hit = o + s_top * d;
    if (s_top > 0 && std::abs(hit[0]) <= kHalfL && std::abs(hit[1]) <= kHalfW) return s_top;
    return (kFloor - o[2]) / d[2];
}

// Mot vong quet bat dau t0: -30..30 do buoc 0.5 do, 100 ms / 360 do (thoi gian rieng tung diem).
Scan scan_at(double t0_s, Motion m) {
    Scan s;
    for (double th = -30.0; th < 30.0 - 1e-9; th += 0.5) {
        const double t = t0_s + (th + 30.0) / 360.0 * 0.1;
        const CamPose p = pose_at_t(t, m);
        const cv::Vec3d d_C(-std::sin(th * kDeg), 0.0, std::cos(th * kDeg));
        const double r = cast(p.t_WC_mm, p.R_WC * d_C);
        s.angle_deg.push_back(static_cast<float>(th));
        s.range_mm.push_back(static_cast<uint16_t>(std::lround(r)));
        s.intensity.push_back(200);
        s.t_ns.push_back(static_cast<int64_t>(t * 1e9));
    }
    return s;
}

// Mask mat top nhin tu camera luc t (chieu 4 goc).
SegObject top_mask(const CameraModel& cam, const CamPose& p) {
    std::vector<cv::Point3d> corners_C;
    for (const auto& [x, y] : {std::pair{-kHalfL, -kHalfW}, {kHalfL, -kHalfW}, {kHalfL, kHalfW}, {-kHalfL, kHalfW}}) {
        const cv::Vec3d w(x, y, kTop + kA * x + kB * y);
        const cv::Vec3d c = p.R_WC.t() * (w - p.t_WC_mm);
        corners_C.emplace_back(c[0], c[1], c[2]);
    }
    std::vector<cv::Point2d> uv;
    cv::projectPoints(corners_C, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), cam.K_px, cam.dist_coeffs, uv);
    double xmin = 1e9, ymin = 1e9;
    for (const auto& q : uv) xmin = std::min(xmin, q.x), ymin = std::min(ymin, q.y);
    const int x0 = static_cast<int>(xmin) - 10, y0 = static_cast<int>(ymin) - 10;
    cv::Mat mask = cv::Mat::zeros(140, 170, CV_8U);
    std::vector<cv::Point> poly;
    for (const auto& q : uv) poly.emplace_back(round_half_even(q.x - x0), round_half_even(q.y - y0));
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, 1);
    SegObject obj;
    EXPECT_TRUE(seg_object({mask, x0, y0, 0.9f, 0}, obj));
    return obj;
}

struct Scene {
    std::vector<MaskFrame> frames;
    std::vector<Scan> scans;
    std::vector<CamPoseSample> poses;
};

Scene make_scene(const CameraModel& cam, Motion m, bool with_position = true, int n_scans = 20) {
    Scene s;
    for (int k = 0; k < n_scans; ++k) {  // mac dinh 2 s: 20 vong quet, mot khung co mask moi vong
        const double t = 0.1 * k;
        s.scans.push_back(scan_at(t, m));
        const double tf = t + 0.05;
        s.frames.push_back({top_mask(cam, pose_at_t(tf, m)), pose_at_t(tf, m), static_cast<int64_t>(tf * 1e9), true});
    }
    for (int i = 0; i <= 10 * n_scans + 20; ++i) {  // ATTITUDE 100 Hz
        CamPose p = pose_at_t(0.01 * i, m);
        p.has_position = with_position;
        s.poses.push_back({p, static_cast<int64_t>(i) * 10'000'000});
    }
    return s;
}

}  // namespace

// --- Ham thuan ----------------------------------------------------------------------------------------------------
TEST(WorldPlane, QuaternionRoundTripAndSlerpHalfway) {
    const cv::Matx33d R = rot_z(90 * kDeg) * rot_x(20 * kDeg);
    const cv::Matx33d back = matrix_from_quat(quat_from_matrix(R));
    EXPECT_LT(cv::norm(R - back), 1e-9);
    const cv::Vec4d q0 = quat_from_matrix(cv::Matx33d::eye()), q1 = quat_from_matrix(rot_z(90 * kDeg));
    EXPECT_LT(cv::norm(matrix_from_quat(slerp(q0, q1, 0.5)) - rot_z(45 * kDeg)), 1e-9);
    EXPECT_LT(cv::norm(matrix_from_quat(slerp(q0, -q1, 0.5)) - rot_z(45 * kDeg)), 1e-9);  // q, -q: duong ngan
}

TEST(WorldPlane, InterpolatePoseSlerpAndRejectsGaps) {
    CamPose a, b;
    b.R_WC = rot_z(30 * kDeg);
    b.t_WC_mm = {100, 0, -20};
    a.has_position = b.has_position = true;
    const std::vector<CamPoseSample> s{{a, 0}, {b, 10'000'000}};
    CamPose out;
    ASSERT_TRUE(interpolate_pose(s, 5'000'000, 10'000'000, out));
    EXPECT_LT(cv::norm(out.R_WC - rot_z(15 * kDeg)), 1e-9);
    EXPECT_NEAR(out.t_WC_mm[0], 50.0, 1e-9);
    EXPECT_NEAR(out.t_WC_mm[2], -10.0, 1e-9);
    EXPECT_FALSE(interpolate_pose(s, 5'000'000, 4'000'000, out));    // 2 mau cach 10 ms > 2 x 4 ms
    EXPECT_FALSE(interpolate_pose(s, 40'000'000, 10'000'000, out));  // qua xa sau mau cuoi
    EXPECT_TRUE(interpolate_pose(s, 15'000'000, 10'000'000, out));   // ngay sau mau cuoi: dung mau cuoi
}

TEST(WorldPlane, RansacPcaRecoversNormalWithOutliers) {
    std::mt19937_64 gen(1);
    std::normal_distribution<double> noise(0.0, 2.0);
    std::uniform_real_distribution<double> uni(-40.0, 40.0);
    std::vector<cv::Vec3d> pts;
    for (int i = 0; i < 300; ++i) {
        const double x = uni(gen), y = uni(gen);
        pts.emplace_back(x, y, kTop + kA * x + kB * y + noise(gen));
    }
    for (int i = 0; i < 80; ++i) pts.emplace_back(uni(gen), uni(gen), kFloor + noise(gen));  // san lan vao
    std::mt19937_64 rng(0);
    PlanePca p;
    ransac_pca_plane(pts, 200, 6.0, {0, 0, 1}, 30.0, rng, p);
    EXPECT_GE(p.n_inliers, 290);
    EXPECT_LE(p.n_inliers, 300);
    EXPECT_LT(angle_deg(p.n, {-kA, -kB, 1.0}), 1.0);
    EXPECT_GE(p.lambda_mm2[0], p.lambda_mm2[1]);
    EXPECT_GE(p.lambda_mm2[1], p.lambda_mm2[2]);
    EXPECT_NEAR(std::sqrt(p.lambda_mm2[2]), 2.0 * std::cos(std::atan(std::hypot(kA, kB))), 0.4);
}

TEST(WorldPlane, SingleScanLineIsNotAPlane) {
    // Mot duong quet: diem thang hang -> lambda2 ~ 0 -> phap tuyen quay tu do quanh duong -> khong chap nhan PLANE.
    std::vector<cv::Vec3d> pts;
    std::mt19937_64 gen(2);
    std::normal_distribution<double> noise(0.0, 1.0);
    for (int i = 0; i < 200; ++i) pts.emplace_back(-40.0 + 0.4 * i, noise(gen), kTop + noise(gen));
    std::mt19937_64 rng(0);
    PlanePca p;
    ransac_pca_plane(pts, 200, 10.0, {0, 0, 1}, 30.0, rng, p);
    WorldPlaneConfig c;
    const PlaneQuality q = accept_plane(p, c);
    EXPECT_FALSE(q.accepted);
    if (p.n_inliers > 0) EXPECT_LT(q.sqrt_l2_mm, c.min_sqrt_l2_mm);
}

TEST(WorldPlane, AcceptanceFollowsSigmaThetaFormula) {
    PlanePca p;
    p.n_inliers = 100;
    p.lambda_mm2 = {400.0, 100.0, 4.0};  // sqrt(l2) = 10 mm, sqrt(l3) = 2 mm < sigma LiDAR 5 mm
    WorldPlaneConfig c;
    c.lidar_sigma_mm = 5.0;
    c.max_normal_sigma_deg = 3.0;
    c.min_sqrt_l2_mm = 10.0;
    const PlaneQuality q = accept_plane(p, c);
    EXPECT_NEAR(q.sigma_perp_mm, 5.0, 1e-12);
    EXPECT_NEAR(q.sigma_theta_deg, 5.0 / std::sqrt(100.0 * 100.0) / kDeg, 1e-9);  // 0.5 rad / 10 -> 2.86 do
    const double k = 5.0 / (3.0 * kDeg * 10.0);
    EXPECT_NEAR(q.n_required, k * k, 1e-9);  // ~ 91.2
    EXPECT_TRUE(q.accepted);
    p.n_inliers = 80;  // < N can
    EXPECT_FALSE(accept_plane(p, c).accepted);
    p.n_inliers = 1000;
    p.lambda_mm2[1] = 64.0;  // sqrt(l2) = 8 < 10 mm: du N nhung trai qua hep
    EXPECT_FALSE(accept_plane(p, c).accepted);
}

// Nhieu tu the khi tich luy nhieu vong: sigma_eff^2 = sigma_lidar^2 + (rho sigma_att)^2. rho 0.8 m:
// 0.5 do -> 7.0 mm, 0.2 do -> 2.8 mm, 0.1 do -> 1.4 mm; LiDAR 5 mm + 0.5 do -> 8.6 mm (tu the chi phoi).
TEST(WorldPlane, AttitudeNoiseInflatesNormalUncertainty) {
    WorldPlaneConfig c;
    c.lidar_sigma_mm = 5.0;
    c.attitude_sigma_deg = 0.0;
    EXPECT_NEAR(effective_sigma_mm(c, 800.0), 5.0, 1e-12);  // mo hinh cu
    c.attitude_sigma_deg = 0.5;
    EXPECT_NEAR(effective_sigma_mm(c, 800.0), std::hypot(5.0, 800.0 * 0.5 * kDeg), 1e-9);
    EXPECT_NEAR(effective_sigma_mm(c, 800.0), 8.6, 0.05);
    c.attitude_sigma_deg = 0.2;
    EXPECT_NEAR(800.0 * c.attitude_sigma_deg * kDeg, 2.8, 0.05);
    c.attitude_sigma_deg = 0.1;
    EXPECT_NEAR(800.0 * c.attitude_sigma_deg * kDeg, 1.4, 0.05);

    // Cung mat phang: dat khi chi tinh LiDAR, KHONG dat khi tinh ca tu the o 0.8 m (sigma_theta lac quan truoc day).
    PlanePca p;
    p.n_inliers = 100;
    p.lambda_mm2 = {400.0, 100.0, 4.0};  // sqrt(l2) 10 mm, sqrt(l3) 2 mm
    c.max_normal_sigma_deg = 3.0;
    c.min_sqrt_l2_mm = 2.0;
    c.attitude_sigma_deg = 0.5;
    const PlaneQuality lidar_only = accept_plane(p, c, 0.0);
    const PlaneQuality with_att = accept_plane(p, c, 800.0);
    EXPECT_TRUE(lidar_only.accepted);
    EXPECT_FALSE(with_att.accepted);
    EXPECT_NEAR(with_att.sigma_perp_mm, effective_sigma_mm(c, 800.0), 1e-12);
    EXPECT_NEAR(with_att.sigma_theta_deg / lidar_only.sigma_theta_deg, effective_sigma_mm(c, 800.0) / 5.0, 1e-9);
    EXPECT_GT(with_att.n_required, 2.9 * lidar_only.n_required);  // (8.6 / 5)^2 = 2.96
    p.n_inliers = 300;  // du N
    EXPECT_TRUE(accept_plane(p, c, 800.0).accepted);
    // sqrt(lambda2) >= 2 mm: trai 3 mm du dieu kien trai, chi con sigma_theta quyet dinh
    p.lambda_mm2 = {400.0, 9.0, 4.0};
    p.n_inliers = 100000;
    EXPECT_TRUE(accept_plane(p, c, 800.0).accepted);
    p.lambda_mm2[1] = 3.0;  // sqrt 1.7 mm < 2 mm
    EXPECT_FALSE(accept_plane(p, c, 800.0).accepted);
}

TEST(WorldPlane, Height1dIgnoresFloorPoints) {
    std::vector<cv::Vec3d> pts;
    for (int i = 0; i < 60; ++i) pts.emplace_back(i, 0, kTop + (i % 3) - 1);
    for (int i = 0; i < 25; ++i) pts.emplace_back(i, 5, kFloor);
    std::mt19937_64 rng(0);
    std::vector<uint8_t> in;
    EXPECT_NEAR(height_1d(pts, {0, 0, 1}, 100, 5.0, rng, in), kTop, 1e-9);
    EXPECT_EQ(std::count(in.begin(), in.end(), uint8_t{1}), 60);
}

TEST(WorldPlane, RayPlaneMatchesClosedForm) {
    const cv::Vec3d n = cv::normalize(cv::Vec3d(0.1, -0.2, 1.0)), c(10, -5, 600);
    const cv::Vec3d r(0.05, 0.02, 1.0);
    cv::Vec3d p;
    ASSERT_TRUE(ray_plane(r, n, c, p));
    EXPECT_NEAR(n.dot(p - c), 0.0, 1e-9);                       // p tren mat phang
    EXPECT_NEAR(cv::norm(p.cross(r)), 0.0, 1e-9);               // p tren tia
    EXPECT_NEAR(p[2], n.dot(c) / n.dot(r), 1e-9);               // p = (n.c / n.r) r
    EXPECT_FALSE(ray_plane({1, 0, 0}, {0, 0, 1}, c, p));        // tia song song mat
    EXPECT_FALSE(ray_plane({0, 0, 1}, {0, 0, 1}, {0, 0, -5}, p));  // mat sau camera
}

TEST(WorldPlane, TopRectOnFrontoParallelPlaneMatchesSize) {
    const CameraModel cam = test::camera();
    CamPose p;  // camera nhin thang xuong, mat top khong nghieng
    std::vector<cv::Point3d> corners;
    const SegObject obj = [&] {
        std::vector<cv::Point3d> cs{{-kHalfL, -kHalfW, kTop}, {kHalfL, -kHalfW, kTop}, {kHalfL, kHalfW, kTop},
                                    {-kHalfL, kHalfW, kTop}};
        std::vector<cv::Point2d> uv;
        cv::projectPoints(cs, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), cam.K_px, cam.dist_coeffs, uv);
        const int x0 = static_cast<int>(uv[0].x) - 10, y0 = static_cast<int>(uv[0].y) - 10;
        cv::Mat mask = cv::Mat::zeros(140, 170, CV_8U);
        std::vector<cv::Point> poly;
        for (const auto& q : uv) poly.emplace_back(round_half_even(q.x - x0), round_half_even(q.y - y0));
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, 1);
        SegObject o;
        EXPECT_TRUE(seg_object({mask, x0, y0, 0.9f, 0}, o));
        return o;
    }();
    TopRect r;
    ASSERT_TRUE(top_rect_on_plane(obj.mask, cam, {0, 0, 1}, {0, 0, kTop}, r));
    EXPECT_NEAR(r.L_mm, 2 * kHalfL, 2.0);  // pixel hoa mask ~ 1 px ~ 0.7 mm
    EXPECT_NEAR(r.W_mm, 2 * kHalfW, 2.0);
    EXPECT_NEAR(std::abs(r.axis_x_C[0]), 1.0, 1e-3);  // canh dai doc truc x camera
}

// --- Fusion method world ------------------------------------------------------------------------------------------
TEST(WorldFusion, DeskewedSweepGivesPlaneWithNormalAndCenter) {
    const CameraModel cam = test::camera();
    const Scene s = make_scene(cam, Motion{});
    Fusion fusion(cfg(), cam);
    const FusionResult r = fusion(s.frames, s.scans, s.poses);
    EXPECT_EQ(r.mode, PlaneMode::kPlane);
    EXPECT_EQ(r.status, FusionStatus::kOk);
    EXPECT_TRUE(r.normal_observable);
    EXPECT_TRUE(r.pose_compensated);
    EXPECT_EQ(r.n_pose_missing, 0);
    EXPECT_LT(angle_deg(r.normal_W, {-kA, -kB, 1.0}), 1.5) << "sigma_theta " << r.sigma_theta_deg;
    EXPECT_GT(r.normal_W[2], 0.0);  // huong xuong, ra xa camera
    // Tam mat top (he the gioi) gan tam hop; tam mask ~ tam hinh chieu cua tu giac
    EXPECT_NEAR(r.p_obj_W_mm.x, 0.0, 3.0);
    EXPECT_NEAR(r.p_obj_W_mm.y, 0.0, 3.0);
    EXPECT_NEAR(r.p_obj_W_mm.z, kTop + kA * r.p_obj_W_mm.x + kB * r.p_obj_W_mm.y, 2.0);
    EXPECT_NEAR(r.L_mm, 2 * kHalfL, 4.0);
    EXPECT_NEAR(r.W_mm, 2 * kHalfW, 4.0);
    EXPECT_GE(r.sqrt_l2_mm, 10.0);  // 40 mm quet vuong goc ~ 40 / sqrt(12) = 11.5 mm (cat bot o mep hop)
    EXPECT_LE(r.sigma_theta_deg, 3.0);
}

// Khoang cach camera -> mat top do bang LiDAR (chi diem trong mask).
TEST(WorldFusion, PlaneDistanceFromMaskPoints) {
    const CameraModel cam = test::camera();
    const Scene s = make_scene(cam, Motion{});
    Fusion fusion(cfg(), cam);
    const FusionResult r = fusion(s.frames, s.scans, s.poses);
    ASSERT_EQ(r.mode, PlaneMode::kPlane);
    EXPECT_GT(r.plane_dist_mm, 560.0);  // mat top ~600 mm duoi camera, nghieng nhe
    EXPECT_LT(r.plane_dist_mm, 620.0);
    EXPECT_GT(r.n_window_points, 40);
    EXPECT_DOUBLE_EQ(r.window_used_s, 2.0);  // du trai ngay cua so dau
}

// Troi cham (5 mm / s, khong lac): 2 s chi trai ~10 mm (sqrt(l2) ~ 3 mm) -> mo rong cua so toi khi du trai
// (sqrt(l2) >= 10 mm <=> trai >= 35 mm <=> ~7 s). Khong cho mo rong -> HEIGHT_ONLY.
TEST(WorldFusion, AdaptiveWindowGrowsUntilNormalIsObservable) {
    const CameraModel cam = test::camera();
    const Scene s = make_scene(cam, Motion{10.0, 0.0, -22.0}, true, 90);  // 9 s, y -22 -> +23 mm (trong hop)
    FusionConfig c = cfg();
    Fusion fixed(c, cam);
    const FusionResult r0 = fixed(s.frames, s.scans, s.poses);
    EXPECT_EQ(r0.mode, PlaneMode::kHeightOnly);
    EXPECT_DOUBLE_EQ(r0.window_used_s, 2.0);
    c.world.max_window_s = 10.0;
    c.world.window_step_s = 1.0;
    Fusion adaptive(c, cam);
    const FusionResult r = adaptive(s.frames, s.scans, s.poses);
    EXPECT_EQ(r.mode, PlaneMode::kPlane) << "sqrt_l2 " << r.sqrt_l2_mm;
    EXPECT_GE(r.window_used_s, 6.0);
    EXPECT_LE(r.window_used_s, 9.0);
    EXPECT_LT(r.n_window_points, r.n_points + 1);
    EXPECT_LT(angle_deg(r.normal_W, {-kA, -kB, 1.0}), 3.0);
}

TEST(WorldFusion, NoMotionFallsBackToHeightOnly) {
    const CameraModel cam = test::camera();
    const Scene s = make_scene(cam, Motion{0.0, 0.0});
    Fusion fusion(cfg(), cam);
    const FusionResult r = fusion(s.frames, s.scans, s.poses);
    EXPECT_EQ(r.mode, PlaneMode::kHeightOnly);  // moi vong cung mot duong: lambda2 ~ 0
    EXPECT_FALSE(r.normal_observable);
    EXPECT_LT(angle_deg(r.normal_W, {0, 0, 1}), 1e-9);
    // Duong quet di qua y = 0 -> do cao tren duong quet = kTop + kA x, median quanh kTop
    EXPECT_NEAR(r.p_obj_W_mm.z, kTop, 6.0);
}

TEST(WorldFusion, WithoutDeskewTheSameSweepIsWorse) {
    // Cung canh, nhung tu the chi lay 1 mau / vong (giua vong): bo deskew -> diem lech theo roll trong vong quet.
    const CameraModel cam = test::camera();
    Motion m;
    m.roll_deg = 6.0;
    Scene s = make_scene(cam, m);
    // Chi so sanh do phang (rms): noi nguong sqrt(l2) / sigma_theta de ket qua co deskew la PLANE.
    FusionConfig c = cfg();
    c.world.max_normal_sigma_deg = 10.0;
    c.world.min_sqrt_l2_mm = 5.0;
    Fusion good(c, cam);
    const FusionResult r_good = good(s.frames, s.scans, s.poses);
    std::vector<CamPoseSample> coarse;
    for (const auto& p : s.poses)
        if ((p.t_ns / 10'000'000) % 10 == 5) coarse.push_back(p);  // 10 Hz
    c.world.pose_max_gap_s = 0.06;
    Fusion bad(c, cam);
    const FusionResult r_bad = bad(s.frames, s.scans, coarse);
    ASSERT_EQ(r_good.mode, PlaneMode::kPlane);
    ASSERT_GT(r_bad.n_points, 0);
    EXPECT_LT(r_good.rms_mm, r_bad.rms_mm);
}

TEST(WorldFusion, PoseWithoutPositionUsesImageMethod) {
    const CameraModel cam = test::camera();
    const Scene s = make_scene(cam, Motion{}, false);
    Fusion fusion(cfg(), cam);
    const FusionResult r = fusion(s.frames, s.scans, s.poses);
    EXPECT_EQ(r.mode, PlaneMode::kImage);
}

TEST(WorldPlane, FillWorldGivesNorthReferencedYaw) {
    FusionResult r;
    r.p_obj_mm = {10.0, 20.0, 600.0};
    r.axis_x = {1.0, 0.0, 0.0};
    r.axis_z = {0.0, 0.0, 1.0};
    CamPose pose;  // R_WC = quay 90 do quanh z_W: x_C -> Dong, y_C -> Nam
    pose.R_WC = rot_z(90.0 * kDeg);
    pose.t_WC_mm = {1000.0, -500.0, -800.0};
    pose.has_position = true;
    fill_world(r, pose);
    EXPECT_NEAR(r.p_obj_W_mm.x, 1000.0 - 20.0, 1e-9);
    EXPECT_NEAR(r.p_obj_W_mm.y, -500.0 + 10.0, 1e-9);
    EXPECT_NEAR(r.p_obj_W_mm.z, -800.0 + 600.0, 1e-9);
    EXPECT_NEAR(r.normal_W[2], 1.0, 1e-9);
    EXPECT_NEAR(r.yaw_W_deg, 90.0, 1e-9);  // canh dai theo Dong
    r.axis_x = {-1.0, 0.0, 0.0};             // nguoc chieu: cung canh
    fill_world(r, pose);
    EXPECT_NEAR(r.yaw_W_deg, 90.0, 1e-9);
    FusionResult none;
    none.p_obj_mm = r.p_obj_mm;
    fill_world(none, CamPose{});  // khong vi tri -> bo qua
    EXPECT_FALSE(std::isfinite(none.yaw_W_deg));
}
