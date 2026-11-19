// s3 select / tracker: chon nhom n vat gan tam anh, ghep detection cac khung sau theo tam DU DOAN gan nhat; giua cac
// lan s1, bbox cua track dich theo van toc tam mask (s2) de s2 cat ROI dung cho du UAV troi.
#pragma once

#include <optional>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

// --- tracker --------------------------------------------------------------------------------------------------
// Bbox du doan o khung `frame`: bbox o box_frame dich vel_px * (frame - box_frame).
[[nodiscard]] Detection predicted_box(const Track& t, uint32_t frame) noexcept;
[[nodiscard]] inline cv::Point2d predicted_center(const Track& t, uint32_t frame) noexcept {
    return predicted_box(t, frame).center_px();
}

// Nhom moi: n_targets detection gan (cx, cy) nhat, bo detection cach tam du doan cua mot track da co <= radius_px.
// Track moi duoc them vao cuoi `tracks`; tra ve chi so cua chung (nhom).
std::vector<int> select_group(std::vector<Track>& tracks, std::span<const Detection> detections, cv::Point2d center_px,
                              int n_targets, double radius_px, uint32_t frame_idx);

// Moi track lay detection gan tam du doan nhat trong ban kinh; xet cap gan nhat truoc; moi detection mot track.
void match(std::vector<Track>& tracks, std::span<const Detection> detections, double radius_px, uint32_t frame_idx);

// s2 tim thay mask (tam mask_center_px) o khung `frame`: cap nhat van toc (lam muot alpha) va, neu bbox chua do s1
// dat o khung nay, dich bbox theo vi tri mask thuc (sua du doan).
void update_from_mask(Track& t, cv::Point2d mask_center_px, uint32_t frame, double alpha) noexcept;

}  // namespace uavp
