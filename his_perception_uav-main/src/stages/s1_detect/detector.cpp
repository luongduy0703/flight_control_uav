#include <stdexcept>

#include "uavp/stages/s1_detect/detector.hpp"

namespace uavp {

void decode_one2many(const Tensor& raw, int num_classes, float conf_thr, std::vector<Detection>& out) {
    constexpr int kBoxRows = 4;
    if (raw.rows() != kBoxRows + num_classes) throw std::runtime_error("Output detector khong khop (4 + nc, A)");
    const int anchors = raw.cols();
    const float* cx = raw.row(0);
    const float* cy = raw.row(1);
    const float* bw = raw.row(2);
    const float* bh = raw.row(3);
    for (int a = 0; a < anchors; ++a) {
        int best = 0;  // argmax: lay class dau tien neu bang nhau (giong numpy)
        float conf = raw.row(kBoxRows)[a];
        for (int k = 1; k < num_classes; ++k) {
            const float s = raw.row(kBoxRows + k)[a];
            if (s > conf) {
                conf = s;
                best = k;
            }
        }
        if (conf < conf_thr) continue;  // da sigmoid trong model: so truc tiep voi nguong
        Detection d;
        d.x1 = cx[a] - bw[a] / 2.f;
        d.y1 = cy[a] - bh[a] / 2.f;
        d.x2 = cx[a] + bw[a] / 2.f;
        d.y2 = cy[a] + bh[a] / 2.f;
        d.conf = conf;
        d.class_id = static_cast<uint8_t>(best);
        out.push_back(d);
    }
}

Detector::Detector(DetectorConfig cfg, std::unique_ptr<IInfer> infer) : cfg_(std::move(cfg)), infer_(std::move(infer)) {
    if (!infer_) throw std::invalid_argument("Detector: infer = nullptr");
    input_.reset(3, cfg_.input_h_px, cfg_.input_w_px);  // (c, h, w): input chu nhat, vd 960 x 544
    candidates_.reserve(1024);
    window_ids_.reserve(1024);
}

void Detector::prepare(int width_px, int height_px) {
    windows_ = windows_for_mode(cfg_.mode, width_px, height_px, cfg_.tile_px, cfg_.overlap);
    params_.clear();
    resize_buffers_.clear();
    for (const auto& w : windows_) {
        const LetterboxParams p = letterbox_params(w.x2 - w.x1, w.y2 - w.y1, cfg_.input_w_px, cfg_.input_h_px);
        params_.push_back(p);
        resize_buffers_.push_back(needs_resize(p, w.x2 - w.x1, w.y2 - w.y1)
                                      ? cv::Mat(p.resized_h_px, p.resized_w_px, CV_8UC3)
                                      : cv::Mat());
    }
    size_ = {width_px, height_px};
}

void Detector::detect_window(const cv::Mat& image_rgb, std::size_t index) {
    const cv::Mat view = image_rgb(windows_[index].rect());  // view, khong copy
    cv::Mat* resized = resize_buffers_[index].empty() ? nullptr : &resize_buffers_[index];
    run_letterboxed(*infer_, view, params_[index], input_, resized, cfg_.pad_value, cfg_.interpolation, outputs_);
    const std::size_t first = candidates_.size();
    decode_one2many(outputs_.at(0), static_cast<int>(cfg_.classes.size()), cfg_.conf_thr, candidates_);
    input_to_image(std::span(candidates_).subspan(first), params_[index], windows_[index]);
    window_ids_.resize(candidates_.size(), static_cast<uint8_t>(index));
}

void Detector::operator()(const cv::Mat& image_rgb, std::vector<Detection>& out) {
    if (image_rgb.type() != CV_8UC3) throw std::invalid_argument("Detector: anh phai la uint8 RGB 3 kenh");
    if (size_ != image_rgb.size()) prepare(image_rgb.cols, image_rgb.rows);
    candidates_.clear();
    window_ids_.clear();
    for (std::size_t i = 0; i < windows_.size(); ++i) detect_window(image_rgb, i);
    merge_detections(candidates_, window_ids_, cfg_.nms_iou, cfg_.cross_window_ios, cfg_.max_det, scratch_, out);
}

}  // namespace uavp
