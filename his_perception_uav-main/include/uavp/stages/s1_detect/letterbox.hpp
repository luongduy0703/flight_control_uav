// s1 detect / letterbox: anh (tile) -> input model float32 CHW [0, 1], can giua giong ultralytics LetterBox.
// Kich thuoc input (input_px [w, h], co the chu nhat) va cach resize nam trong configs/models/s1_detect_ncnn.yaml.
#pragma once

#include <opencv2/core.hpp>

#include "uavp/common/common.hpp"

namespace uavp {

// Anh nguon (w, h) -> input: resize theo `scale`, dem pad_*_px o tren / trai.
struct LetterboxParams {
    double scale{1.0};
    int pad_x_px{0};
    int pad_y_px{0};
    int resized_w_px{0};
    int resized_h_px{0};
};

[[nodiscard]] LetterboxParams letterbox_params(int src_w_px, int src_h_px, int input_w_px, int input_h_px) noexcept;
[[nodiscard]] inline bool needs_resize(const LetterboxParams& p, int src_w_px, int src_h_px) noexcept {
    return p.resized_w_px != src_w_px || p.resized_h_px != src_h_px;
}

// Letterbox `src_rgb` (uint8 HWC RGB, view) vao `out` (3, H, W) float [0, 1] co san. `resized`: buffer resize
// (nullptr neu khong can resize). Khong cap phat khi buffer da dung kich thuoc.
void fill_input_chw(const cv::Mat& src_rgb, const LetterboxParams& p, InputCHW& out, cv::Mat* resized,
                    int pad_value, int interpolation);

// Letterbox `src_rgb` vao input (input.c, input.h, input.w) roi chay model. Model ho tro anh (NcnnModel, chi khi
// interpolation = linear): resize / dem / chuan hoa trong ncnn, khong dung `input` / `resized`. Khong thi
// fill_input_chw + run (model gia trong test, interpolation area).
void run_letterboxed(IInfer& infer, const cv::Mat& src_rgb, const LetterboxParams& p, InputCHW& input,
                     cv::Mat* resized, int pad_value, int interpolation, std::vector<Tensor>& outputs);

}  // namespace uavp
