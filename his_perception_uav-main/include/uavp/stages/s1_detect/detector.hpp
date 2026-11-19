// s1 detect / detector: anh RGB -> (ca anh | tile) -> letterbox ve input chu nhat (vd 960 x 544) -> YOLO26n (NCNN,
// head one-to-many) -> giai ma -> ghep box.
#pragma once

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/common.hpp"
#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"
#include "uavp/stages/s1_detect/letterbox.hpp"
#include "uavp/stages/s1_detect/tiling.hpp"

namespace uavp {

// Output (4 + nc, A): cx, cy, w, h (px input model) + diem class (da sigmoid) -> them detection (px input) vao out.
void decode_one2many(const Tensor& raw, int num_classes, float conf_thr, std::vector<Detection>& out);

// s1: anh RGB -> detection (anh goc, conf giam dan). Buffer cap phat theo kich thuoc anh lan dau.
class Detector {
public:
    Detector(DetectorConfig cfg, std::unique_ptr<IInfer> infer);
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    void operator()(const cv::Mat& image_rgb, std::vector<Detection>& out);
    [[nodiscard]] const DetectorConfig& config() const noexcept { return cfg_; }
    [[nodiscard]] std::size_t num_windows() const noexcept { return windows_.size(); }

private:
    void prepare(int width_px, int height_px);
    void detect_window(const cv::Mat& image_rgb, std::size_t index);

    DetectorConfig cfg_;
    std::unique_ptr<IInfer> infer_;
    cv::Size size_{0, 0};
    std::vector<WindowPx> windows_;
    std::vector<LetterboxParams> params_;
    std::vector<cv::Mat> resize_buffers_;  // rong neu tile khong can resize
    InputCHW input_;
    std::vector<Tensor> outputs_;
    std::vector<Detection> candidates_;    // moi tile, anh goc
    std::vector<uint8_t> window_ids_;
    MergeScratch scratch_;
};

}  // namespace uavp
