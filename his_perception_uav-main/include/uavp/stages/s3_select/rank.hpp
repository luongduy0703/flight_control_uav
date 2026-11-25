// s3 select / rank: xep track OK theo khoang cach tam toi (cx, cy); dau tien = selected. Giong rank.py.
#pragma once

#include <optional>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

// --- rank -----------------------------------------------------------------------------------------------------
[[nodiscard]] const SegObject* last_seg(const Track& track) noexcept;
[[nodiscard]] double distance_to_center_px(const Track& track, cv::Point2d center_px) noexcept;
// Chi so track OK, gan tam truoc. Phan tu dau la selected.
[[nodiscard]] std::vector<int> rank_ok(const std::vector<Track>& tracks, cv::Point2d center_px);
// Track selected (nullptr neu khong co vat OK).
[[nodiscard]] const Track* selected_track(const std::vector<Track>& tracks, cv::Point2d center_px);

}  // namespace uavp
