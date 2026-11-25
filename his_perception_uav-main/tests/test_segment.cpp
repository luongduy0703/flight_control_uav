// s2 segment voi model gia (giong tests/test_models.py phan Segmenter, tests/test_measure.py phan mask).
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "helpers.hpp"
#include "uavp/stages/s2_segment.hpp"

using namespace uavp;
using test::FakeInfer;
using test::tensor2;
using test::tensor3;

namespace {

// out0 (4 + nc + nm, A) tu danh sach instance.
Tensor raw_seg(const std::vector<std::array<float, 4>>& boxes, const std::vector<std::array<float, 2>>& scores,
               const std::vector<float>& coeffs) {
    std::vector<std::vector<float>> rows(7);
    for (std::size_t a = 0; a < boxes.size(); ++a) {
        for (std::size_t k = 0; k < 4; ++k) rows[k].push_back(boxes[a][k]);
        rows[4].push_back(scores[a][0]);
        rows[5].push_back(scores[a][1]);
        rows[6].push_back(coeffs[a]);
    }
    return tensor2(rows);
}

Detection det_box(float x1, float y1, float x2, float y2) { return {x1, y1, x2, y2, 0.9f, 0, {}}; }

}  // namespace

TEST(Segment, RoiWindowSquareScaledAndClamped) {
    auto r = roi_window(det_box(100, 200, 140, 220), 1.5, 1920, 1080);
    EXPECT_EQ(std::tie(r.x0_px, r.y0_px, r.side_px), std::make_tuple(90, 180, 60));
    r = roi_window(det_box(0, 0, 40, 20), 1.5, 1920, 1080);
    EXPECT_EQ(std::tie(r.x0_px, r.y0_px, r.side_px), std::make_tuple(0, 0, 60));
    r = roi_window(det_box(1900, 1070, 1920, 1080), 1.5, 1920, 1080);
    EXPECT_EQ(std::tie(r.x0_px, r.y0_px, r.side_px), std::make_tuple(1890, 1050, 30));
    EXPECT_EQ(roi_window(det_box(0, 0, 1000, 900), 1.5, 1920, 1080).side_px, 1080);
}

TEST(Segment, SelectInstanceUsesIoaNotIou) {
    SegInstances inst;
    inst.xyxy_crop_px = {{40, 0, 60, 20}, {90, 0, 190, 100}};  // nap chai nho; vat ben canh
    inst.conf = {0.5, 0.9};
    EXPECT_EQ(select_instance(inst, {0, 0, 100, 300}, 0.5), 0);
    inst.xyxy_crop_px.erase(inst.xyxy_crop_px.begin());
    inst.conf.erase(inst.conf.begin());
    EXPECT_EQ(select_instance(inst, {0, 0, 100, 300}, 0.5), -1);
}

TEST(Segment, DecodeSegFiltersNmsAndClips) {
    const Tensor raw = raw_seg({{50, 50, 40, 40}, {52, 50, 40, 40}, {300, 300, 60, 60}, {10, 10, 4, 4}},
                               {{0.9f, 0.f}, {0.8f, 0.f}, {0.f, 0.7f}, {0.1f, 0.f}}, {1, 2, 3, 4});
    SegInstances inst;
    decode_seg(raw, 2, 0.25f, 0.7, 10, 320, inst);
    ASSERT_EQ(inst.conf.size(), 2U);  // box 2 bi NMS, box 4 duoi nguong
    EXPECT_NEAR(inst.conf[0], 0.9, 1e-6);
    EXPECT_NEAR(inst.conf[1], 0.7, 1e-6);
    EXPECT_EQ(inst.class_id[1], 1);
    EXPECT_DOUBLE_EQ(inst.xyxy_crop_px[1].x1, 270.0);
    EXPECT_DOUBLE_EQ(inst.xyxy_crop_px[1].x2, 320.0);  // cat vao crop
}

TEST(Segment, InstanceMaskIsCroppedToBox) {
    const Tensor protos = tensor3(1, 8, 8, 1.f);
    const Tensor raw = tensor2({{0}, {0}, {0}, {0}, {0}, {0}, {1.f}});  // he so mask = 1 o hang 6
    cv::Mat small, logits(32, 32, CV_32F), mask;
    instance_mask(protos, raw, 0, 6, {4, 8, 20, 16}, 0.f, small, logits, mask);
    EXPECT_EQ(cv::countNonZero(mask), 16 * 8);
    EXPECT_EQ(mask.at<uint8_t>(8, 4), 1);
    EXPECT_EQ(mask.at<uint8_t>(16, 4), 0);
    EXPECT_EQ(mask.at<uint8_t>(8, 20), 0);
}

// ROI tren anh letterbox 960 x 544 lay tu anh goc 1920 x 1080 = letterbox ca anh roi cat (cach tao data/images_960×544).
TEST(Segment, LetterboxRoiMatchesResizeThenCrop) {
    cv::Mat image(1080, 1920, CV_8UC3);
    cv::randu(image, cv::Scalar::all(0), cv::Scalar::all(256));
    const LetterboxParams lb = letterbox_params(1920, 1080, 960, 544);
    ASSERT_EQ(lb.pad_y_px, 2);
    cv::Mat small(544, 960, CV_8UC3, cv::Scalar::all(114));
    cv::resize(image, small(cv::Rect(0, 2, 960, 540)), {960, 540}, 0, 0, cv::INTER_LINEAR);
    for (const RoiWindow roi : {RoiWindow{301, 117, 90}, RoiWindow{0, 0, 75}, RoiWindow{900, 480, 60}}) {
        cv::Mat out;
        letterbox_roi(image, lb, roi, 114, out);
        const cv::Mat ref = small(cv::Rect(roi.x0_px, roi.y0_px, roi.side_px, roi.side_px));
        EXPECT_EQ(cv::norm(out, ref, cv::NORM_INF), 0.0) << roi.x0_px << ", " << roi.y0_px;
    }
}

// Che do 960 x 544: input model = ROI cua anh letterbox; mask / tam tra ve toa do anh goc, cat phan vung dem.
TEST(Segment, SegmenterLetterboxModeReturnsMaskInOriginalImage) {
    SegmenterConfig cfg;
    cfg.classes = {"box_top", "bottle_top"};
    cfg.input_px = 32;
    cfg.max_det = 10;
    cfg.roi_image_w_px = 960;
    cfg.roi_image_h_px = 544;
    // bbox anh goc (400, 300)-(480, 340) -> anh letterbox (200, 152)-(240, 172): ROI 1.5 x 40 = 60 tai (190, 132).
    // Instance = vung giua crop 16 x 8 (= 30 x 15 px anh letterbox = 60 x 30 px anh goc).
    auto fake = std::make_unique<FakeInfer>(
        std::vector<Tensor>{raw_seg({{16, 16, 16, 8}}, {{0.9f, 0.1f}}, {1.f}), tensor3(1, 8, 8, 1.f)});
    Segmenter seg(cfg, std::move(fake));
    SegObject obj;
    ASSERT_TRUE(seg(cv::Mat::zeros(1080, 1920, CV_8UC3), det_box(400, 300, 480, 340), obj));
    EXPECT_EQ(obj.mask.x0_px, 380);  // (190 - 0) / 0.5
    EXPECT_EQ(obj.mask.y0_px, 260);  // (132 - 2) / 0.5
    EXPECT_EQ(obj.mask.mask.cols, 120);
    const cv::Rect bb = cv::boundingRect(obj.mask.mask);
    EXPECT_NEAR(bb.width, 60, 4);
    EXPECT_NEAR(bb.height, 30, 4);
    EXPECT_NEAR(obj.center_px.x, 440, 2.0);  // tam bbox anh goc
    EXPECT_NEAR(obj.center_px.y, 320, 2.0);
    // Vat sat mep tren: ROI kep o y = 0 cua anh letterbox (2 hang dau la vung dem) -> vat nam o nua tren crop
    // (y crop ~1-12) -> mask cat bot vung dem, van nam trong anh goc.
    auto fake2 = std::make_unique<FakeInfer>(
        std::vector<Tensor>{raw_seg({{16, 6, 16, 8}}, {{0.9f, 0.1f}}, {1.f}), tensor3(1, 8, 8, 1.f)});
    Segmenter seg2(cfg, std::move(fake2));
    ASSERT_TRUE(seg2(cv::Mat::zeros(1080, 1920, CV_8UC3), det_box(400, 0, 480, 40), obj));
    EXPECT_EQ(obj.mask.y0_px, 0);
    EXPECT_LE(obj.mask.y0_px + obj.mask.mask.rows, 1080);
}

TEST(Segment, SegmenterReturnsMaskInImageRoi) {
    // bbox 40x20 tai (60, 50), ROI 1.5x -> canh 60 px, resize 32. Instance = vung giua crop 16x8 px (= 30x15 px anh).
    SegmenterConfig cfg;
    cfg.classes = {"box_top", "bottle_top"};
    cfg.input_px = 32;
    cfg.max_det = 10;
    auto fake = std::make_unique<FakeInfer>(
        std::vector<Tensor>{raw_seg({{16, 16, 16, 8}}, {{0.9f, 0.1f}}, {1.f}), tensor3(1, 8, 8, 1.f)});
    FakeInfer* f = fake.get();
    Segmenter seg(cfg, std::move(fake));
    SegObject obj;
    ASSERT_TRUE(seg(cv::Mat::zeros(100, 200, CV_8UC3), det_box(40, 40, 80, 60), obj));
    ASSERT_EQ(f->shapes.size(), 1U);
    EXPECT_EQ(f->shapes[0], (std::array<int, 3>{3, 32, 32}));
    EXPECT_EQ(obj.mask.class_id, 0);
    EXPECT_EQ(obj.mask.x0_px, 30);
    EXPECT_EQ(obj.mask.y0_px, 20);
    EXPECT_EQ(obj.mask.mask.cols, 60);
    const cv::Rect bb = cv::boundingRect(obj.mask.mask);
    EXPECT_NEAR(bb.width, 30, 2);
    EXPECT_NEAR(bb.height, 15, 2);
    EXPECT_NEAR(obj.center_px.x, 59.5, 1.0);
    EXPECT_NEAR(obj.center_px.y, 49.5, 1.0);
    EXPECT_NEAR(std::min(obj.yaw_deg, 180.0 - obj.yaw_deg), 0.0, 1.0);  // canh dai nam ngang
}

TEST(Segment, SegObjectEmptyMask) {
    SegObject obj;
    EXPECT_FALSE(seg_object({cv::Mat::zeros(10, 10, CV_8U), 0, 0, 0.9f, 0}, obj));
}

TEST(Segment, MinAreaRectYaw) {
    const std::vector<cv::Point2f> pts{{0, 0}, {10, 10}, {9, 11}, {-1, 1}};  // canh dai huong 45 do
    const RectSize r = min_area_rect(pts);
    EXPECT_NEAR(r.yaw_deg, 45.0, 1e-3);
    EXPECT_NEAR(r.long_side, std::sqrt(200.0), 1e-3);
}
