// s1 detect / tiling: cat anh thanh tile co overlap (tile_px, overlap trong configs/stages/s1_detect.yaml), doi box
// ve anh goc, ghep box giua cac tile (NMS + bo box bi cat o mep tile).
#pragma once

#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/boxes.hpp"
#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"
#include "uavp/stages/s1_detect/letterbox.hpp"

namespace uavp {

struct WindowPx {
    int x1{0}, y1{0}, x2{0}, y2{0};
    [[nodiscard]] cv::Rect rect() const noexcept { return {x1, y1, x2 - x1, y2 - y1}; }
};

// Vi tri bat dau tile tren mot truc; tile cuoi sat mep anh.
[[nodiscard]] std::vector<int> tile_starts(int length_px, int tile_px, double overlap);
[[nodiscard]] std::vector<WindowPx> windows_for_mode(DetectMode mode, int width_px, int height_px, int tile_px,
                                                     double overlap);

// Doi xyxy tu input model ve anh goc (sua tai cho, float32 giong ban Python).
void input_to_image(std::span<Detection> dets, const LetterboxParams& p, const WindowPx& window) noexcept;

// Buffer tam cho ghep box (cap phat mot lan, dung lai moi khung).
struct MergeScratch {
    std::vector<BoxD> boxes;
    std::vector<double> scores;
    std::vector<uint8_t> class_ids;
    std::vector<int> keep, order;
    std::vector<uint8_t> removed;
};

// NMS IoU theo class, roi bo box bi cat o mep tile (khac tile, IoS > cross_window_ios), toi da max_det.
void merge_detections(std::span<const Detection> dets, std::span<const uint8_t> window_ids, double nms_iou,
                      double cross_window_ios, int max_det, MergeScratch& scratch, std::vector<Detection>& out);

}  // namespace uavp
