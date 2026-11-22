// Chan doan video: do net giam khi mo (va theo huong mo), dich anh giua 2 khung, lech tren / duoi (rolling shutter),
// toc do quay + dich anh du doan tu goc quay.
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <opencv2/imgproc.hpp>

#include "helpers.hpp"
#include "uavp/support/video_diag.hpp"

using namespace uavp;

namespace {

cv::Mat texture(int w, int h, unsigned seed = 3) {
    cv::Mat g(h / 4, w / 4, CV_8UC1);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> u(0, 255);
    for (int y = 0; y < g.rows; ++y)
        for (int x = 0; x < g.cols; ++x) g.at<uint8_t>(y, x) = static_cast<uint8_t>(u(rng));
    cv::Mat big, rgb;
    cv::resize(g, big, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(big, rgb, cv::COLOR_GRAY2RGB);
    return rgb;
}

cv::Mat shifted(const cv::Mat& img, double dx, double dy) {
    const cv::Matx23d M(1, 0, dx, 0, 1, dy);
    cv::Mat out;
    cv::warpAffine(img, out, M, img.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return out;
}

cv::Matx33d rot_x(double rad) {
    const double c = std::cos(rad), s = std::sin(rad);
    return {1, 0, 0, 0, c, -s, 0, s, c};
}

}  // namespace

TEST(VideoDiag, SharpnessDropsWithBlurAndShowsDirection) {
    const cv::Mat img = texture(1280, 720);
    const SharpStats sharp = sharp_stats(img, 256);
    cv::Mat blur_x;
    cv::blur(img, blur_x, cv::Size(15, 1));  // mo theo x (rung ngang)
    const SharpStats bx = sharp_stats(blur_x, 256);
    EXPECT_LT(bx.sharp, 0.5 * sharp.sharp);
    EXPECT_LT(bx.grad_x, 0.5 * bx.grad_y);  // gradient theo huong mo giam manh hon
    EXPECT_NEAR(bx.luma, sharp.luma, 3.0);
}

TEST(VideoDiag, ShiftMeterMeasuresContentMotionAndSkew) {
    const cv::Mat a = texture(1280, 720);
    ShiftMeter m(320);
    ImageShift s;
    EXPECT_FALSE(m.update(a, s));  // khung dau
    ASSERT_TRUE(m.update(shifted(a, 12.0, -8.0), s));
    EXPECT_NEAR(s.dx_px, 12.0, 1.5);  // noi dung dich sang phai 12 px, len 8 px
    EXPECT_NEAR(s.dy_px, -8.0, 1.5);
    EXPECT_NEAR(s.skew_x_px, 0.0, 1.5);
    EXPECT_GT(s.response, 0.2);
    // Rolling shutter: dai tren dich 16 px, dai duoi khong dich (dich theo hang)
    const cv::Mat b = shifted(a, 12.0, -8.0);
    cv::Mat c = b.clone();
    cv::Mat top = shifted(b, 16.0, 0.0);
    top(cv::Rect(0, 0, b.cols, b.rows / 3)).copyTo(c(cv::Rect(0, 0, b.cols, b.rows / 3)));
    ASSERT_TRUE(m.update(c, s));
    EXPECT_NEAR(s.skew_x_px, 16.0, 2.5);
}

TEST(VideoDiag, RatesAndRotationPrediction) {
    const CameraModel cam = test::camera();
    const cv::Matx33d R0 = cv::Matx33d::eye(), R1 = rot_x(0.01);  // quay 0.01 rad quanh x camera
    const cv::Vec3d w = angular_rate(R0, R1, 0.02);
    EXPECT_NEAR(w[0], 0.5, 1e-9);
    EXPECT_NEAR(w[1], 0.0, 1e-9);
    const cv::Point2d p = rotation_shift_px(R0, R1, cam);
    // Quay +x: truc quang ngoc len (-y anh) -> diem o xa dang o tam anh truot xuong (dy > 0)
    EXPECT_NEAR(p.y, cam.K_px(1, 1) * std::tan(0.01), 1e-6);
    EXPECT_NEAR(p.x, 0.0, 1e-9);
    EXPECT_NEAR(blur_pred_px(w, 1000.0, 800.0), 0.4, 1e-9);  // 0.5 rad/s * 800 px * 1 ms
    EXPECT_TRUE(std::isnan(blur_pred_px(w, -1.0, 800.0)));
}
