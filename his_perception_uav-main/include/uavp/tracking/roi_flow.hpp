// Hieu chinh tam ROI bang anh (1-3 ms): chi xu ly cua so quanh ROI, anh xam thu 1/2.
//   1. Optical flow: 30-50 diem goc trong mask (goodFeaturesToTrack), calcOpticalFlowPyrLK khung truoc -> khung nay,
//      kiem tra forward-backward, median do dich = do dich tam; >= flow_min_points diem tot: them affine (doi mask).
//   2. It goc (mat vat tron, hop mot mau): template matching NCC voi anh cat quanh mask luc segment gan nhat.
// Cua so co canh co dinh tu luc khoa (search_scale x canh ROI) de anh xam 2 khung cung kich thuoc.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

enum class FlowMethod : uint8_t { kNone, kFlow, kNcc, kCount };
inline constexpr std::array<std::string_view, static_cast<std::size_t>(FlowMethod::kCount)> kFlowMethodNames{
    "none", "flow", "ncc"};
constexpr std::string_view name(FlowMethod m) noexcept { return kFlowMethodNames[static_cast<std::size_t>(m)]; }

struct FlowMeasure {
    cv::Matx33d M_prev_to_cur{cv::Matx33d::eye()};  // anh goc: khung truoc -> khung nay (affine / tinh tien)
    cv::Point2d center_px{0.0, 0.0};                // tam vat do duoc (anh goc)
    double ncc{0.0};
    int n_good{0};
    FlowMethod method{FlowMethod::kNone};
};

class RoiFlow {
public:
    explicit RoiFlow(const LockConfig& cfg) : cfg_(cfg) {}
    RoiFlow(const RoiFlow&) = delete;
    RoiFlow& operator=(const RoiFlow&) = delete;

    // Moc moi (luc khoa / sau moi lan segment): anh xam cua so, diem goc trong mask, template NCC.
    // side_px: canh cua so (anh goc); 0 = giu canh cu.
    void reset(const cv::Mat& image_rgb, const SegMask& mask, cv::Point2d center_px, int side_px);
    // Khung moi: cua so quanh tam du doan -> do. method = kNone neu ca flow va NCC deu that bai.
    [[nodiscard]] FlowMeasure measure(const cv::Mat& image_rgb, cv::Point2d predicted_center_px);
    // Sau khi mask da doi sang khung nay: tim lai diem goc neu con it (< 1/2 flow_max_corners).
    void refresh_points(const SegMask& mask);
    [[nodiscard]] int side_px() const noexcept { return side_px_; }

private:
    [[nodiscard]] cv::Point window_origin(cv::Point2d center_px, cv::Size image) const noexcept;
    void gray_half(const cv::Mat& image_rgb, cv::Point origin, cv::Mat& out);
    void mask_half(const SegMask& mask, cv::Point origin, cv::Mat& out);
    [[nodiscard]] bool track_lk(cv::Point2d predicted_center_px, FlowMeasure& m);
    [[nodiscard]] bool match_ncc(FlowMeasure& m);

    LockConfig cfg_;
    int side_px_{0};
    cv::Point prev_origin_, cur_origin_;
    cv::Point2d prev_center_px_;
    cv::Mat crop_gray_, prev_gray_, cur_gray_, mask_half_, mask_full_, ncc_out_;
    cv::Mat templ_;
    cv::Point2d templ_center_half_;  // tam vat trong template (px anh 1/2)
    std::vector<cv::Point2f> prev_pts_, next_pts_, back_pts_, good_prev_, good_next_;
    std::vector<uint8_t> status_, status_back_;
    std::vector<float> err_;
    std::vector<double> dx_, dy_;
};

}  // namespace uavp
