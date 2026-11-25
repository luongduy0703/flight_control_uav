#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "uavp/stages/s1_detect/letterbox.hpp"

namespace uavp {

LetterboxParams letterbox_params(int src_w_px, int src_h_px, int input_w_px, int input_h_px) noexcept {
    const double scale = std::min(static_cast<double>(input_w_px) / src_w_px, static_cast<double>(input_h_px) / src_h_px);
    LetterboxParams p;
    p.scale = scale;
    p.resized_w_px = round_half_even(src_w_px * scale);
    p.resized_h_px = round_half_even(src_h_px * scale);
    p.pad_x_px = round_half_even((input_w_px - p.resized_w_px) / 2.0 - 0.1);
    p.pad_y_px = round_half_even((input_h_px - p.resized_h_px) / 2.0 - 0.1);
    return p;
}

void fill_input_chw(const cv::Mat& src_rgb, const LetterboxParams& p, InputCHW& out, cv::Mat* resized,
                    int pad_value, int interpolation) {
    const cv::Mat* region = &src_rgb;
    if (resized != nullptr) {
        resized->create(p.resized_h_px, p.resized_w_px, CV_8UC3);  // da dung kich thuoc: khong cap phat
        cv::resize(src_rgb, *resized, resized->size(), 0, 0, interpolation);
        region = resized;
    }
    const bool covers_all = p.pad_x_px == 0 && p.pad_y_px == 0 && region->cols == out.w && region->rows == out.h;
    if (!covers_all) std::fill(out.data.begin(), out.data.end(), kU8ToUnit[static_cast<std::size_t>(pad_value)]);
    float* __restrict r_plane = out.plane(0);
    float* __restrict g_plane = out.plane(1);
    float* __restrict b_plane = out.plane(2);
    const auto w = static_cast<std::size_t>(out.w);
    for (int r = 0; r < region->rows; ++r) {
        const uint8_t* __restrict px = region->ptr<uint8_t>(r);
        const std::size_t base = static_cast<std::size_t>(p.pad_y_px + r) * w + static_cast<std::size_t>(p.pad_x_px);
        for (int c = 0; c < region->cols; ++c) {
            const auto k = static_cast<std::size_t>(c);
            r_plane[base + k] = kU8ToUnit[px[3 * k]];  // anh RGB: kenh 0 = R
            g_plane[base + k] = kU8ToUnit[px[3 * k + 1]];
            b_plane[base + k] = kU8ToUnit[px[3 * k + 2]];
        }
    }
}

void run_letterboxed(IInfer& infer, const cv::Mat& src_rgb, const LetterboxParams& p, InputCHW& input,
                     cv::Mat* resized, int pad_value, int interpolation, std::vector<Tensor>& outputs) {
    if (interpolation == cv::INTER_LINEAR) {
        const ImagePrep prep{p.resized_w_px, p.resized_h_px, p.pad_x_px, p.pad_y_px, input.w, input.h, pad_value};
        if (infer.run_rgb(src_rgb, prep, outputs)) return;
    }
    fill_input_chw(src_rgb, p, input, resized, pad_value, interpolation);
    infer.run(input, outputs);
}

}  // namespace uavp
