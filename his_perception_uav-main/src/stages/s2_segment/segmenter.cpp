// Segmenter YOLO26n-seg (giong segmenter.py). Output: out0 (4 + nc + nm, A), out1 (nm, S/4, S/4) proto.
#include <algorithm>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "uavp/stages/s1_detect/letterbox.hpp"  // fill_input_chw dung chung
#include "uavp/stages/s2_segment/segmenter.hpp"

namespace uavp {

RoiWindow roi_window(const Detection& det, double roi_scale, int width_px, int height_px) noexcept {
    const double x1 = det.x1, y1 = det.y1, x2 = det.x2, y2 = det.y2;
    int side = std::min({round_half_even(roi_scale * std::max(x2 - x1, y2 - y1)), width_px, height_px});
    side = std::max(side, 1);
    const int x0 = std::min(std::max(round_half_even((x1 + x2) / 2.0 - side / 2.0), 0), width_px - side);
    const int y0 = std::min(std::max(round_half_even((y1 + y2) / 2.0 - side / 2.0), 0), height_px - side);
    return {x0, y0, side};
}

void decode_seg(const Tensor& raw, int num_classes, float conf_thr, double nms_iou, int max_det, int size_px,
                SegInstances& out) {
    constexpr int kBoxRows = 4;
    if (raw.dims != 2 || raw.rows() <= kBoxRows + num_classes) throw std::runtime_error("out0 segment khong khop");
    out.all_boxes.clear();
    out.all_conf.clear();
    out.all_class.clear();
    out.all_anchor.clear();
    for (int a = 0; a < raw.cols(); ++a) {
        int best = 0;
        float conf = raw.row(kBoxRows)[a];
        for (int k = 1; k < num_classes; ++k) {
            if (raw.row(kBoxRows + k)[a] > conf) {
                conf = raw.row(kBoxRows + k)[a];
                best = k;
            }
        }
        if (conf < conf_thr) continue;
        const double cx = raw.row(0)[a], cy = raw.row(1)[a], w = raw.row(2)[a], h = raw.row(3)[a];
        out.all_boxes.push_back({cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0});
        out.all_conf.push_back(conf);
        out.all_class.push_back(static_cast<uint8_t>(best));
        out.all_anchor.push_back(a);
    }
    nms_per_class(out.all_boxes, out.all_conf, out.all_class, nms_iou, Overlap::kIoU, out.keep, out.order, out.removed);
    const std::size_t n = std::min(out.keep.size(), static_cast<std::size_t>(max_det));
    const auto lim = static_cast<double>(size_px);
    out.xyxy_crop_px.clear();
    out.conf.clear();
    out.class_id.clear();
    out.anchor.clear();
    for (std::size_t i = 0; i < n; ++i) {
        const auto k = static_cast<std::size_t>(out.keep[i]);
        const BoxD& b = out.all_boxes[k];
        out.xyxy_crop_px.push_back({std::clamp(b.x1, 0.0, lim), std::clamp(b.y1, 0.0, lim), std::clamp(b.x2, 0.0, lim),
                                    std::clamp(b.y2, 0.0, lim)});
        out.conf.push_back(out.all_conf[k]);
        out.class_id.push_back(out.all_class[k]);
        out.anchor.push_back(out.all_anchor[k]);
    }
}

int select_instance(const SegInstances& inst, const BoxD& target_crop_px, double min_ioa) noexcept {
    int best = -1;  // argmax conf trong cac ung vien (lay ung vien dau tien neu bang nhau)
    for (std::size_t i = 0; i < inst.conf.size(); ++i) {
        if (ioa(inst.xyxy_crop_px[i], target_crop_px) < min_ioa) continue;
        if (best < 0 || inst.conf[i] > inst.conf[static_cast<std::size_t>(best)]) best = static_cast<int>(i);
    }
    return best;
}

void instance_mask(const Tensor& protos, const Tensor& raw, int anchor, int coeff_row0, const BoxD& box_crop_px,
                   float logit_thr, cv::Mat& small, cv::Mat& logits, cv::Mat& mask_out) {
    const int nm = protos.shape[0], ph = protos.shape[1], pw = protos.shape[2];
    small.create(ph, pw, CV_32F);
    small.setTo(0.f);
    float* __restrict acc = small.ptr<float>(0);
    const std::size_t plane = static_cast<std::size_t>(ph) * static_cast<std::size_t>(pw);
    for (int k = 0; k < nm; ++k) {  // logits = he so @ proto (float32)
        const float coeff = raw.row(coeff_row0 + k)[anchor];
        const float* __restrict p = protos.data.data() + static_cast<std::size_t>(k) * plane;
        for (std::size_t i = 0; i < plane; ++i) acc[i] += coeff * p[i];
    }
    // ultralytics: F.interpolate bilinear, align_corners=False == cv::INTER_LINEAR khi phong to
    cv::resize(small, logits, logits.size(), 0, 0, cv::INTER_LINEAR);
    const int size = logits.rows;
    mask_out.create(size, size, CV_8U);
    for (int r = 0; r < size; ++r) {
        const float* __restrict lr = logits.ptr<float>(r);
        uint8_t* __restrict mr = mask_out.ptr<uint8_t>(r);
        const bool row_in = r >= box_crop_px.y1 && r < box_crop_px.y2;  // cat theo box, giong crop_mask
        for (int c = 0; c < size; ++c)
            mr[c] = static_cast<uint8_t>(row_in && c >= box_crop_px.x1 && c < box_crop_px.x2 && lr[c] > logit_thr);
    }
}

Segmenter::Segmenter(SegmenterConfig cfg, std::unique_ptr<IInfer> infer) : cfg_(std::move(cfg)), infer_(std::move(infer)) {
    if (!infer_) throw std::invalid_argument("Segmenter: infer = nullptr");
    input_.reset(3, cfg_.input_px, cfg_.input_px);
    crop_rgb_.create(cfg_.input_px, cfg_.input_px, CV_8UC3);
    logits_.create(cfg_.input_px, cfg_.input_px, CV_32F);
}

void letterbox_roi(const cv::Mat& image_rgb, const LetterboxParams& lb, const RoiWindow& roi, int pad_value,
                   cv::Mat& out) {
    out.create(roi.side_px, roi.side_px, CV_8UC3);
    out.setTo(cv::Scalar::all(pad_value));
    // Vung ROI (anh letterbox) tren anh goc. Thu 0.5 lan: goc ROI chan, INTER_LINEAR 2x = trung binh 2x2 -> pixel
    // trung khop voi resize ca anh roi cat (cach tao anh 960 x 544 luc train).
    const double inv = 1.0 / lb.scale;
    const int fx0 = round_half_even((roi.x0_px - lb.pad_x_px) * inv);
    const int fy0 = round_half_even((roi.y0_px - lb.pad_y_px) * inv);
    const int fside = round_half_even(roi.side_px * inv);
    const cv::Rect full = cv::Rect(fx0, fy0, fside, fside) & cv::Rect(0, 0, image_rgb.cols, image_rgb.rows);
    if (full.empty()) return;
    const int sx0 = round_half_even((full.x - fx0) * lb.scale), sy0 = round_half_even((full.y - fy0) * lb.scale);
    const int sw = std::min(round_half_even(full.width * lb.scale), roi.side_px - sx0);
    const int sh = std::min(round_half_even(full.height * lb.scale), roi.side_px - sy0);
    if (sw <= 0 || sh <= 0) return;
    cv::Mat dst = out(cv::Rect(sx0, sy0, sw, sh));
    cv::resize(image_rgb(full), dst, dst.size(), 0, 0, cv::INTER_LINEAR);
}

bool Segmenter::operator()(const cv::Mat& image_rgb, const Detection& det, SegObject& out) {
    const int s = cfg_.input_px;
    // Anh tinh ROI: letterbox roi_image (giong s1 va luc train) hoac anh goc (scale 1, khong dem).
    const bool letterbox = cfg_.roi_image_w_px > 0 &&
                           (cfg_.roi_image_w_px != image_rgb.cols || cfg_.roi_image_h_px != image_rgb.rows);
    const LetterboxParams lb = letterbox ? letterbox_params(image_rgb.cols, image_rgb.rows, cfg_.roi_image_w_px,
                                                            cfg_.roi_image_h_px)
                                         : LetterboxParams{1.0, 0, 0, image_rgb.cols, image_rgb.rows};
    const double sc = lb.scale, px = lb.pad_x_px, py = lb.pad_y_px;
    Detection det_l = det;  // bbox tren anh letterbox
    det_l.x1 = static_cast<float>(det.x1 * sc + px);
    det_l.y1 = static_cast<float>(det.y1 * sc + py);
    det_l.x2 = static_cast<float>(det.x2 * sc + px);
    det_l.y2 = static_cast<float>(det.y2 * sc + py);
    const int img_w = letterbox ? cfg_.roi_image_w_px : image_rgb.cols;
    const int img_h = letterbox ? cfg_.roi_image_h_px : image_rgb.rows;
    const RoiWindow roi = roi_window(det_l, cfg_.roi_scale, img_w, img_h);
    if (letterbox) letterbox_roi(image_rgb, lb, roi, cfg_.roi_pad_value, roi_rgb_);
    const cv::Mat view = letterbox ? roi_rgb_ : image_rgb(cv::Rect(roi.x0_px, roi.y0_px, roi.side_px, roi.side_px));
    const LetterboxParams p{static_cast<double>(s) / roi.side_px, 0, 0, s, s};  // resize thang ve S x S
    run_letterboxed(*infer_, view, p, input_, &crop_rgb_, 0, cfg_.interpolation, outputs_);
    const bool first_is_raw = outputs_.at(0).dims < outputs_.at(1).dims;
    const Tensor& raw = first_is_raw ? outputs_[0] : outputs_[1];
    const Tensor& protos = first_is_raw ? outputs_[1] : outputs_[0];
    const auto nc = static_cast<int>(cfg_.classes.size());
    decode_seg(raw, nc, cfg_.conf_thr, cfg_.nms_iou, cfg_.max_det, s, inst_);
    const double k = static_cast<double>(s) / roi.side_px;
    const double x0 = roi.x0_px, y0 = roi.y0_px;  // double giong ban Python (xyxy float64)
    const BoxD target{(det_l.x1 - x0) * k, (det_l.y1 - y0) * k, (det_l.x2 - x0) * k, (det_l.y2 - y0) * k};
    const int index = select_instance(inst_, target, cfg_.min_select_ioa);
    if (index < 0) return false;
    const auto i = static_cast<std::size_t>(index);
    instance_mask(protos, raw, inst_.anchor[i], 4 + nc, inst_.xyxy_crop_px[i], cfg_.mask_logit_thr, small_, logits_,
                  mask_s_);
    SegMask seg;
    if (!letterbox) {
        cv::resize(mask_s_, seg.mask, cv::Size(roi.side_px, roi.side_px), 0, 0, cv::INTER_NEAREST);  // mask moi
        seg.x0_px = roi.x0_px;
        seg.y0_px = roi.y0_px;
    } else {  // ROI anh letterbox -> anh goc: phong mask ve canh ROI goc, bo phan nam ngoai anh (vung dem)
        const int fx0 = round_half_even((roi.x0_px - px) / sc), fy0 = round_half_even((roi.y0_px - py) / sc);
        const int fside = round_half_even(roi.side_px / sc);
        cv::resize(mask_s_, mask_full_, cv::Size(fside, fside), 0, 0, cv::INTER_NEAREST);
        const cv::Rect in_image = cv::Rect(fx0, fy0, fside, fside) & cv::Rect(0, 0, image_rgb.cols, image_rgb.rows);
        if (in_image.empty()) return false;
        seg.mask = mask_full_(in_image - cv::Point(fx0, fy0)).clone();  // mask moi: luu vao track
        seg.x0_px = in_image.x;
        seg.y0_px = in_image.y;
    }
    seg.conf = static_cast<float>(inst_.conf[i]);
    seg.class_id = inst_.class_id[i];
    return seg_object(std::move(seg), out);
}

}  // namespace uavp
