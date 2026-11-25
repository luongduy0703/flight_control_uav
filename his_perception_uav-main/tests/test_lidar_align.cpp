// Dieu kien LiDAR quet qua vat: e_px (tam -> duong LiDAR tren anh), e_mm = e_px d / f_x, vector chinh delta_C.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <opencv2/calib3d.hpp>

#include "helpers.hpp"
#include "uavp/stages/s4_fusion/lidar_align.hpp"

using namespace uavp;

namespace {

// Duong quet thang: y_C = y0, z_C = d, x_C tu -400 den 400 mm, buoc 5 mm.
std::vector<cv::Point3f> line(double y0_mm, double d_mm, double gap_from = 1e9, double gap_to = -1e9) {
    std::vector<cv::Point3f> pts;
    for (double x = -400.0; x <= 400.0; x += 5.0)
        if (x < gap_from || x > gap_to) pts.emplace_back(static_cast<float>(x), static_cast<float>(y0_mm), static_cast<float>(d_mm));
    return pts;
}

cv::Point2d project(const CameraModel& cam, cv::Point3d p) {
    std::vector<cv::Point2d> uv;
    cv::projectPoints(std::vector<cv::Point3d>{p}, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), cam.K_px, cam.dist_coeffs, uv);
    return uv[0];
}

}  // namespace

TEST(LidarAlign, LineThroughCenterIsAligned) {
    const CameraModel cam = test::camera();
    const cv::Point2d c = project(cam, {12.0, 0.0, 600.0});
    const LidarAlign a = lidar_alignment(line(0.0, 600.0), c, cam, 1920, 1080, 40.0, 10.0);
    ASSERT_TRUE(a.valid);
    EXPECT_TRUE(a.ok);
    EXPECT_LT(a.e_px, 1.0);
    EXPECT_NEAR(a.d_mm, 600.0, 1e-3);
    EXPECT_LT(cv::norm(a.delta_C_mm), 2.0);
}

TEST(LidarAlign, OffsetLineGivesMetricErrorAndCorrection) {
    const CameraModel cam = test::camera();
    const cv::Point2d c = project(cam, {0.0, 30.0, 600.0});  // tam vat 30 mm canh duong quet (truc y camera)
    const LidarAlign a = lidar_alignment(line(0.0, 600.0), c, cam, 1920, 1080, 40.0, 10.0);
    ASSERT_TRUE(a.valid);
    EXPECT_FALSE(a.ok);
    EXPECT_NEAR(a.e_mm, 30.0, 1.5);  // e_px d / f ~ khoang cach that o do sau d
    EXPECT_NEAR(a.e_px, 30.0 * cam.K_px(1, 1) / 600.0, 3.0);
    EXPECT_NEAR(a.delta_C_mm[1], 30.0, 1.0);  // dich duong quet +30 mm theo y_C thi trung tam
    EXPECT_NEAR(a.delta_C_mm[0], 0.0, 1.0);
    EXPECT_NEAR(a.delta_C_mm[2], 0.0, 1e-6);
    EXPECT_TRUE(lidar_alignment(line(0.0, 600.0), c, cam, 1920, 1080, 40.0, 35.0).ok);  // nguong rong hon
}

TEST(LidarAlign, GapIsNotBridged) {
    const CameraModel cam = test::camera();
    // Lo hong x in (-100, 100): tam (0, 0) khong duoc noi qua lo -> khoang cach toi dau mut gan nhat (~100 mm).
    const cv::Point2d c = project(cam, {0.0, 0.0, 600.0});
    const LidarAlign a = lidar_alignment(line(0.0, 600.0, -100.0, 100.0), c, cam, 1920, 1080, 40.0, 10.0);
    ASSERT_TRUE(a.valid);
    EXPECT_NEAR(a.e_mm, 100.0, 6.0);
    EXPECT_FALSE(a.ok);
}

TEST(LidarAlign, NoVisiblePointsIsInvalid) {
    const CameraModel cam = test::camera();
    const std::vector<cv::Point3f> behind{{0.0F, 0.0F, -500.0F}, {10.0F, 0.0F, -500.0F}};
    const LidarAlign a = lidar_alignment(behind, cam.center_px(), cam, 1920, 1080, 40.0, 10.0);
    EXPECT_FALSE(a.valid);
    EXPECT_FALSE(a.ok);
    EXPECT_EQ(a.n_visible, 0);
    EXPECT_FALSE(lidar_alignment({}, cam.center_px(), cam, 1920, 1080, 40.0, 10.0).valid);
}
