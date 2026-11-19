// Ham box dung chung (s1, s2), giong src/uavp/boxes.py. Box xyxy_px, double (giong ban Python, de khop so).
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace uavp {

struct BoxD {
    double x1{0.0}, y1{0.0}, x2{0.0}, y2{0.0};
};

enum class Overlap : uint8_t { kIoU, kIoS, kCount };

inline double box_area(const BoxD& b) noexcept { return std::max(b.x2 - b.x1, 0.0) * std::max(b.y2 - b.y1, 0.0); }

inline double intersection(const BoxD& a, const BoxD& b) noexcept {
    const double w = std::max(std::min(a.x2, b.x2) - std::max(a.x1, b.x1), 0.0);
    const double h = std::max(std::min(a.y2, b.y2) - std::max(a.y1, b.y1), 0.0);
    return w * h;
}

// IoU, hoac IoS (giao / dien tich box nho hon).
inline double overlap(const BoxD& a, const BoxD& b, Overlap metric) noexcept {
    const double inter = intersection(a, b);
    const double area_a = box_area(a);
    const double area_b = box_area(b);
    const double denom = metric == Overlap::kIoU ? area_a + area_b - inter : std::min(area_a, area_b);
    return inter / std::max(denom, 1e-9);
}

// Ti le dien tich `pred` nam trong `target`.
inline double ioa(const BoxD& pred, const BoxD& target) noexcept {
    const double area = box_area(pred);
    return area > 0.0 ? intersection(pred, target) / std::max(area, 1e-9) : 0.0;
}

// NMS tham lam rieng tung class (giong batched_nms_indices): sap score giam dan, on dinh theo chi so; box sau bi
// bo neu cung class va overlap > thr voi mot box da giu. `keep` (chi so, score giam dan), `order` / `removed`:
// buffer cua nguoi goi (khong cap phat moi khi da du dung luong).
void nms_per_class(std::span<const BoxD> boxes, std::span<const double> scores, std::span<const uint8_t> class_ids,
                   double thr, Overlap metric, std::vector<int>& keep, std::vector<int>& order,
                   std::vector<uint8_t>& removed);

}  // namespace uavp
