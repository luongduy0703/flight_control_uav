// s1 detect voi model gia (giong tests/test_models.py phan Detector).
#include <gtest/gtest.h>

#include "helpers.hpp"
#include "uavp/stages/s1_detect.hpp"

using namespace uavp;
using test::FakeInfer;
using test::tensor2;

TEST(Detect, TilesCoverImageAndHaveFullSize) {
    const auto w = windows_for_mode(DetectMode::kTile, 1920, 1080, 640, 0.2);
    ASSERT_EQ(w.size(), 8U);  // 4 x 2 tile
    int xmin = 10000, xmax = 0, ymax = 0;
    for (const auto& t : w) {
        EXPECT_EQ(t.x2 - t.x1, 640);
        EXPECT_EQ(t.y2 - t.y1, 640);
        xmin = std::min(xmin, t.x1);
        xmax = std::max(xmax, t.x2);
        ymax = std::max(ymax, t.y2);
    }
    EXPECT_EQ(xmin, 0);
    EXPECT_EQ(xmax, 1920);
    EXPECT_EQ(ymax, 1080);
}

TEST(Detect, Letterbox1920x1080To640) {
    const LetterboxParams p = letterbox_params(1920, 1080, 640, 640);
    EXPECT_EQ(p.resized_w_px, 640);
    EXPECT_EQ(p.resized_h_px, 360);
    EXPECT_EQ(p.pad_x_px, 0);
    EXPECT_EQ(p.pad_y_px, 140);
}

TEST(Detect, LetterboxFullImageTo960x544) {
    const LetterboxParams p = letterbox_params(1920, 1080, 960, 544);
    EXPECT_DOUBLE_EQ(p.scale, 0.5);
    EXPECT_EQ(p.resized_w_px, 960);
    EXPECT_EQ(p.resized_h_px, 540);
    EXPECT_EQ(p.pad_x_px, 0);
    EXPECT_EQ(p.pad_y_px, 2);  // 4 px dem chia 2 ben
}

TEST(Detect, FullModeRectInputRunsModelOnce) {
    DetectorConfig cfg;
    cfg.classes = {"box", "bottle"};
    cfg.mode = DetectMode::kFull;
    cfg.input_w_px = 960;
    cfg.input_h_px = 544;
    cfg.conf_thr = 0.5f;
    // 1 box o (480, 272) input -> anh goc (960, 540) sau khi bo dem 2 px va chia ti le 0.5
    auto infer = std::make_unique<FakeInfer>(std::vector<Tensor>{tensor2({{480}, {272}, {40}, {20}, {0.9f}, {0.1f}})});
    FakeInfer& fake = *infer;
    Detector det(cfg, std::move(infer));
    const cv::Mat image(1080, 1920, CV_8UC3, cv::Scalar(10, 20, 30));
    std::vector<Detection> out;
    det(image, out);
    ASSERT_EQ(fake.shapes.size(), 1U);  // 1 lan chay model / khung
    EXPECT_EQ(fake.shapes[0], (std::array<int, 3>{3, 544, 960}));
    ASSERT_EQ(out.size(), 1U);
    EXPECT_NEAR(out[0].x1, 920.f, 1e-3);
    EXPECT_NEAR(out[0].y1, 520.f, 1e-3);
    EXPECT_NEAR(out[0].x2, 1000.f, 1e-3);
    EXPECT_NEAR(out[0].y2, 560.f, 1e-3);
}

TEST(Detect, RoundHalfEvenLikePython) {
    EXPECT_EQ(round_half_even(2.5), 2);
    EXPECT_EQ(round_half_even(3.5), 4);
    EXPECT_EQ(round_half_even(-0.5), 0);
}

TEST(Detect, DecodeOne2Many) {
    const Tensor raw = tensor2({{100, 10}, {50, 10}, {20, 4}, {10, 4}, {0.9f, 0.1f}, {0.2f, 0.3f}});
    std::vector<Detection> out;
    decode_one2many(raw, 2, 0.5f, out);
    ASSERT_EQ(out.size(), 1U);
    EXPECT_FLOAT_EQ(out[0].x1, 90.f);
    EXPECT_FLOAT_EQ(out[0].y1, 45.f);
    EXPECT_FLOAT_EQ(out[0].x2, 110.f);
    EXPECT_FLOAT_EQ(out[0].y2, 55.f);
    EXPECT_FLOAT_EQ(out[0].conf, 0.9f);
    EXPECT_EQ(out[0].class_id, 0);
}

TEST(Detect, MergeRemovesBoxCutByTileEdge) {
    const std::vector<Detection> dets{{0, 0, 100, 100, 0.9f, 0, {}}, {60, 0, 100, 100, 0.8f, 0, {}}};  // 2 nam trong 1
    const std::vector<uint8_t> windows{0, 1};
    MergeScratch s;
    std::vector<Detection> kept;
    merge_detections(dets, windows, 0.7, 0.8, 10, s, kept);
    ASSERT_EQ(kept.size(), 1U);
    EXPECT_FLOAT_EQ(kept[0].conf, 0.9f);
}

TEST(Detect, MapsBackToImageFullMode) {
    // Anh 1280x720 letterbox vao 640: scale 0.5, pad y 140. Box o input (320, 320, w 40, h 20) -> anh goc.
    DetectorConfig cfg;
    cfg.classes = {"box", "bottle"};
    cfg.mode = DetectMode::kFull;
    cfg.input_w_px = cfg.input_h_px = 640;  // input vuong
    cfg.conf_thr = 0.5f;
    const Tensor raw = tensor2({{320, 0}, {320, 0}, {40, 0}, {20, 0}, {0.1f, 0}, {0.95f, 0}});
    Detector det(cfg, std::make_unique<FakeInfer>(std::vector<Tensor>{raw}));
    std::vector<Detection> out;
    det(cv::Mat::zeros(720, 1280, CV_8UC3), out);
    ASSERT_EQ(out.size(), 1U);
    EXPECT_EQ(out[0].class_id, 1);
    EXPECT_NEAR(out[0].conf, 0.95, 1e-6);
    EXPECT_NEAR(out[0].x1, 600, 1e-3);
    EXPECT_NEAR(out[0].y1, 340, 1e-3);
    EXPECT_NEAR(out[0].x2, 680, 1e-3);
    EXPECT_NEAR(out[0].y2, 380, 1e-3);
}

TEST(Detect, FillInputPadsAndScales) {
    const cv::Mat img(4, 8, CV_8UC3, cv::Scalar(255, 0, 0));  // RGB: do
    const LetterboxParams p = letterbox_params(8, 4, 8, 8);
    InputCHW in;
    in.reset(3, 8, 8);
    fill_input_chw(img, p, in, nullptr, 114, 1);
    EXPECT_FLOAT_EQ(in.plane(0)[0], kU8ToUnit[114]);  // hang 0: dem
    EXPECT_FLOAT_EQ(in.plane(0)[2 * 8], 1.f);          // hang 2: anh, kenh R
    EXPECT_FLOAT_EQ(in.plane(1)[2 * 8], 0.f);
}
