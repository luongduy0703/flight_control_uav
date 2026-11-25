// s3 select / tracker: chon nhom n vat gan tam anh, ghep detection cac khung sau theo tam DU DOAN gan nhat; giua cac
// lan s1, bbox cua track dich theo van toc tam mask (s2) de s2 cat ROI dung cho du UAV troi.
#pragma once

#include <cmath>
#include <optional>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

// --- che do the gioi -------------------------------------------------------------------------------------------
// Tu the camera LUC CHUP khung hien tai + mat san z = floor_z_mm (NED W, huong xuong). Diem anh <-> diem tren san.
struct FloorView {
    CamPose pose;
    double floor_z_mm{0.0};
    const CameraModel* camera{nullptr};
    double radius_mm{80.0};   // ghep detection voi track dang bam (dung sai du doan)
    double same_mm{100.0};    // hai tam tren san cach <= same_mm -> CUNG MOT VAT (ca track da ABORT, ca 2 detection)
};
// Hai diem tren san (mm) la cung mot vat?
[[nodiscard]] inline bool same_object(const cv::Point3d& a, const cv::Point3d& b, double same_mm) noexcept {
    return std::hypot(a.x - b.x, a.y - b.y) <= same_mm;
}
// Tia qua diem anh (khu meo) cat mat san. nullopt neu tia khong huong xuong san.
[[nodiscard]] std::optional<cv::Point3d> floor_point(const FloorView& v, cv::Point2d uv_px);
// Chieu diem W len anh (co meo). nullopt neu sau camera.
[[nodiscard]] std::optional<cv::Point2d> floor_pixel(const FloorView& v, const cv::Point3d& p_W_mm);

// --- tracker --------------------------------------------------------------------------------------------------
// Bbox du doan o khung `frame`. Co `view` + track co p_W: bbox (kich thuoc luc do) dat tai hinh chieu p_W bang tu the
// khung nay (bu UAV nghieng / troi). Khong: bbox o box_frame dich vel_px * (frame - box_frame).
[[nodiscard]] Detection predicted_box(const Track& t, uint32_t frame, const FloorView* view = nullptr);
[[nodiscard]] inline cv::Point2d predicted_center(const Track& t, uint32_t frame, const FloorView* view = nullptr) {
    return predicted_box(t, frame, view).center_px();
}

// Nhom moi: n_targets detection gan (cx, cy) nhat, bo detection da thuoc MOT track bat ky (ca track da ABORT): co
// `view` -> cach p_W cua track <= view->radius_mm tren san; khong -> cach tam du doan <= radius_px.
// Track moi duoc them vao cuoi `tracks`; tra ve chi so cua chung (nhom).
std::vector<int> select_group(std::vector<Track>& tracks, std::span<const Detection> detections, cv::Point2d center_px,
                              int n_targets, double radius_px, uint32_t frame_idx, const FloorView* view = nullptr);

// Moi track lay detection gan nhat trong ban kinh (mm tren san neu co `view`, khong thi px tu tam du doan); xet cap
// gan nhat truoc; moi detection mot track. Detection ghep duoc dat lai bbox + p_W cua track.
void match(std::vector<Track>& tracks, std::span<const Detection> detections, double radius_px, uint32_t frame_idx,
           const FloorView* view = nullptr);

// s2 tim thay mask (tam mask_center_px) o khung `frame`: neu bbox chua do s1 dat o khung nay, dich bbox theo vi tri
// mask thuc. Co `view`: cap nhat p_W tu bbox moi (khong dung van toc). Khong: cap nhat van toc (lam muot alpha).
void update_from_mask(Track& t, cv::Point2d mask_center_px, uint32_t frame, double alpha,
                      const FloorView* view = nullptr);

}  // namespace uavp
