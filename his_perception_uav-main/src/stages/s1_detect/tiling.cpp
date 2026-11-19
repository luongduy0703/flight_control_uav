// Tile, giai ma output detector, ghep box giua cac tile (giong tiling.py, detector.py).
#include <algorithm>
#include <stdexcept>

#include "uavp/stages/s1_detect/tiling.hpp"

namespace uavp {

std::vector<int> tile_starts(int length_px, int tile_px, double overlap) {
    if (length_px <= tile_px) return {0};
    const int stride_px = std::max(1, round_half_even(tile_px * (1.0 - overlap)));
    std::vector<int> starts;
    for (int s = 0; s < length_px - tile_px; s += stride_px) starts.push_back(s);
    starts.push_back(length_px - tile_px);
    std::sort(starts.begin(), starts.end());
    starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
    return starts;
}

std::vector<WindowPx> windows_for_mode(DetectMode mode, int width_px, int height_px, int tile_px, double overlap) {
    switch (mode) {
        case DetectMode::kFull:
            return {{0, 0, width_px, height_px}};
        case DetectMode::kTile: {
            std::vector<WindowPx> windows;
            for (const int y0 : tile_starts(height_px, tile_px, overlap))
                for (const int x0 : tile_starts(width_px, tile_px, overlap))
                    windows.push_back({x0, y0, std::min(x0 + tile_px, width_px), std::min(y0 + tile_px, height_px)});
            return windows;
        }
        case DetectMode::kCount:
            break;
    }
    throw std::invalid_argument("DetectMode khong hop le");
}

void input_to_image(std::span<Detection> dets, const LetterboxParams& p, const WindowPx& window) noexcept {
    const auto scale = static_cast<float>(p.scale);
    const auto pad_x = static_cast<float>(p.pad_x_px);
    const auto pad_y = static_cast<float>(p.pad_y_px);
    const auto wx = static_cast<float>(window.x1);
    const auto wy = static_cast<float>(window.y1);
    for (auto& d : dets) {
        d.x1 = (d.x1 - pad_x) / scale + wx;
        d.x2 = (d.x2 - pad_x) / scale + wx;
        d.y1 = (d.y1 - pad_y) / scale + wy;
        d.y2 = (d.y2 - pad_y) / scale + wy;
    }
}

namespace {

// Bo box bi tile cat: hai box cung class, khac tile, IoS > thr -> giu box conf cao hon (da sap conf giam dan).
void suppress_cross_window(std::vector<Detection>& kept, std::vector<uint8_t>& kept_windows, double ios_thr,
                           std::vector<uint8_t>& removed) {
    const std::size_t n = kept.size();
    removed.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (removed[i] != 0) continue;
        const BoxD a{kept[i].x1, kept[i].y1, kept[i].x2, kept[i].y2};
        for (std::size_t j = i + 1; j < n; ++j) {
            if (kept[j].class_id != kept[i].class_id || kept_windows[j] == kept_windows[i]) continue;
            if (overlap(a, {kept[j].x1, kept[j].y1, kept[j].x2, kept[j].y2}, Overlap::kIoS) > ios_thr) removed[j] = 1;
        }
    }
    std::size_t w = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (removed[i] == 0) kept[w++] = kept[i];
    kept.resize(w);
}

}  // namespace

void merge_detections(std::span<const Detection> dets, std::span<const uint8_t> window_ids, double nms_iou,
                      double cross_window_ios, int max_det, MergeScratch& s, std::vector<Detection>& out) {
    out.clear();
    if (dets.empty()) return;
    s.boxes.clear();
    s.scores.clear();
    s.class_ids.clear();
    for (const auto& d : dets) {
        s.boxes.push_back({d.x1, d.y1, d.x2, d.y2});
        s.scores.push_back(d.conf);
        s.class_ids.push_back(d.class_id);
    }
    nms_per_class(s.boxes, s.scores, s.class_ids, nms_iou, Overlap::kIoU, s.keep, s.order, s.removed);
    std::vector<uint8_t>& kept_windows = s.class_ids;  // dung lai buffer: class_ids khong con can
    kept_windows.clear();
    bool many_windows = false;
    for (const int k : s.keep) {
        const auto idx = static_cast<std::size_t>(k);
        out.push_back(dets[idx]);
        kept_windows.push_back(window_ids[idx]);
        many_windows = many_windows || window_ids[idx] != kept_windows.front();
    }
    if (many_windows) suppress_cross_window(out, kept_windows, cross_window_ios, s.removed);
    if (out.size() > static_cast<std::size_t>(max_det)) out.resize(static_cast<std::size_t>(max_det));
}

}  // namespace uavp
