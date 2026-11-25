// s2 segment / segmenter: ROI quanh bbox -> YOLO26n-seg (NCNN) -> instance cua vat -> mask -> tam, yaw.
// ROI: configs/stages/s2_segment.yaml; input_px (320) va model: configs/models/s2_segment_ncnn.yaml.
#pragma once

#include <memory>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/boxes.hpp"
#include "uavp/common/common.hpp"
#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"
#include "uavp/stages/s1_detect/letterbox.hpp"
#include "uavp/stages/s2_segment/mask_shape.hpp"

namespace uavp {

// --- Segmenter ------------------------------------------------------------------------------------------------
struct RoiWindow {
    int x0_px{0};
    int y0_px{0};
    int side_px{1};
};

// ROI vuong: canh = roi_scale * canh dai bbox, tam = tam bbox, kep trong anh.
[[nodiscard]] RoiWindow roi_window(const Detection& det, double roi_scale, int width_px, int height_px) noexcept;

// Instance sau loc conf + NMS (toa do crop, px input model), conf giam dan. `anchor`: cot trong out0.
struct SegInstances {
    std::vector<BoxD> xyxy_crop_px;
    std::vector<double> conf;
    std::vector<uint8_t> class_id;
    std::vector<int> anchor;
    // buffer tam
    std::vector<BoxD> all_boxes;
    std::vector<double> all_conf;
    std::vector<uint8_t> all_class;
    std::vector<int> all_anchor, keep, order;
    std::vector<uint8_t> removed;
};

// out0 (4 + nc + nm, A) -> instance da loc conf, NMS theo class, cat box vao [0, size_px].
void decode_seg(const Tensor& raw, int num_classes, float conf_thr, double nms_iou, int max_det, int size_px,
                SegInstances& out);

// Instance cua vat: box nam >= min_ioa trong bbox vat, conf cao nhat; -1 neu khong co.
[[nodiscard]] int select_instance(const SegInstances& inst, const BoxD& target_crop_px, double min_ioa) noexcept;

// Mask (S, S) uint8 0 / 1 cua mot instance (giong ultralytics retina_masks): he so @ proto -> resize tuyen tinh ->
// > logit_thr -> cat theo box. `small`, `logits`: buffer dung lai.
void instance_mask(const Tensor& protos, const Tensor& raw, int anchor, int coeff_row0, const BoxD& box_crop_px,
                   float logit_thr, cv::Mat& small, cv::Mat& logits, cv::Mat& mask_out);

// ROI tren anh letterbox (roi_image_px, vd 960 x 544) cua anh goc: lay pixel ROI dung nhu khi letterbox ca anh roi cat
// (thu nho chi vung ROI cua anh goc; phan ngoai anh = pad_value), ghi vao `out` (side, side) uint8 RGB.
void letterbox_roi(const cv::Mat& image_rgb, const LetterboxParams& lb, const RoiWindow& roi, int pad_value,
                   cv::Mat& out);

// s2: (anh goc RGB, bbox vat o anh goc) -> SegObject (mask, tam o anh goc). ROI tinh tren anh letterbox
// (roi_image_px, giong luc train) hoac tren anh goc (roi_image_px = [0, 0]). false neu khong co instance cua vat.
class Segmenter {
public:
    Segmenter(SegmenterConfig cfg, std::unique_ptr<IInfer> infer);
    Segmenter(const Segmenter&) = delete;
    Segmenter& operator=(const Segmenter&) = delete;

    [[nodiscard]] bool operator()(const cv::Mat& image_rgb, const Detection& det, SegObject& out);
    [[nodiscard]] const SegmenterConfig& config() const noexcept { return cfg_; }

private:
    SegmenterConfig cfg_;
    std::unique_ptr<IInfer> infer_;
    cv::Mat crop_rgb_;  // (S, S) uint8
    cv::Mat roi_rgb_;   // ROI tren anh letterbox (side, side) uint8
    cv::Mat mask_full_; // mask (S, S) phong ve anh goc
    InputCHW input_;
    std::vector<Tensor> outputs_;
    SegInstances inst_;
    cv::Mat small_, logits_, mask_s_;  // proto (P, P) f32, (S, S) f32, (S, S) u8
};

}  // namespace uavp
