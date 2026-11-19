#include "uavp/tracking/roi_flow.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "uavp/common/common.hpp"

namespace uavp {
namespace {

cv::Matx33d translation(cv::Point2d d) noexcept { return {1, 0, d.x, 0, 1, d.y, 0, 0, 1}; }

}  // namespace

cv::Point RoiFlow::window_origin(cv::Point2d center_px, cv::Size image) const noexcept {
    const int x0 = std::clamp(round_half_even(center_px.x - side_px_ / 2.0), 0, image.width - side_px_);
    const int y0 = std::clamp(round_half_even(center_px.y - side_px_ / 2.0), 0, image.height - side_px_);
    return {x0, y0};
}

void RoiFlow::gray_half(const cv::Mat& image_rgb, cv::Point origin, cv::Mat& out) {
    cv::cvtColor(image_rgb(cv::Rect(origin.x, origin.y, side_px_, side_px_)), crop_gray_, cv::COLOR_RGB2GRAY);
    cv::resize(crop_gray_, out, cv::Size(side_px_ / 2, side_px_ / 2), 0, 0, cv::INTER_AREA);
}

// Mask (anh goc, tai x0, y0) -> mask 1/2 trong toa do cua so tai origin (255 = vat).
void RoiFlow::mask_half(const SegMask& mask, cv::Point origin, cv::Mat& out) {
    cv::Mat& full = mask_full_;  // buffer tam dung lai
    full.create(side_px_, side_px_, CV_8U);
    full.setTo(0);
    const cv::Rect win(origin.x, origin.y, side_px_, side_px_);
    const cv::Rect m(mask.x0_px, mask.y0_px, mask.mask.cols, mask.mask.rows);
    const cv::Rect inter = win & m;
    if (inter.area() > 0) {
        const cv::Mat src = mask.mask(inter - m.tl()) > 0;
        src.copyTo(full(inter - win.tl()));
    }
    cv::resize(full, out, cv::Size(side_px_ / 2, side_px_ / 2), 0, 0, cv::INTER_NEAREST);
}

void RoiFlow::reset(const cv::Mat& image_rgb, const SegMask& mask, cv::Point2d center_px, int side_px) {
    if (side_px > 0) {
        side_px_ = std::min({side_px + (side_px & 1), image_rgb.cols & ~1, image_rgb.rows & ~1});  // chan
    }
    prev_origin_ = window_origin(center_px, image_rgb.size());
    prev_center_px_ = center_px;
    gray_half(image_rgb, prev_origin_, prev_gray_);
    mask_half(mask, prev_origin_, mask_half_);
    prev_pts_.clear();
    cv::goodFeaturesToTrack(prev_gray_, prev_pts_, cfg_.flow_max_corners, cfg_.flow_quality, cfg_.flow_min_distance_px,
                            mask_half_);
    const cv::Rect box = cv::boundingRect(mask_half_);  // template NCC: anh quanh mask
    templ_.release();
    if (box.area() > 0) {
        prev_gray_(box).copyTo(templ_);
        templ_center_half_ = (center_px - cv::Point2d(prev_origin_)) * 0.5 - cv::Point2d(box.tl());
    }
}

bool RoiFlow::track_lk(cv::Point2d predicted_center_px, FlowMeasure& m) {
    if (static_cast<int>(prev_pts_.size()) < cfg_.flow_min_points) return false;
    const cv::Point2d guess = (cv::Point2d(prev_origin_ - cur_origin_) + (predicted_center_px - prev_center_px_)) * 0.5;
    next_pts_.resize(prev_pts_.size());
    for (std::size_t i = 0; i < prev_pts_.size(); ++i)
        next_pts_[i] = prev_pts_[i] + cv::Point2f(static_cast<float>(guess.x), static_cast<float>(guess.y));
    const cv::Size win(cfg_.flow_win_px, cfg_.flow_win_px);
    const cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::calcOpticalFlowPyrLK(prev_gray_, cur_gray_, prev_pts_, next_pts_, status_, err_, win, cfg_.flow_levels, crit,
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    back_pts_ = prev_pts_;
    cv::calcOpticalFlowPyrLK(cur_gray_, prev_gray_, next_pts_, back_pts_, status_back_, err_, win, cfg_.flow_levels,
                             crit, cv::OPTFLOW_USE_INITIAL_FLOW);
    good_prev_.clear();
    good_next_.clear();
    dx_.clear();
    dy_.clear();
    const auto fb_max2 = static_cast<float>(cfg_.flow_fb_max_px * cfg_.flow_fb_max_px);
    for (std::size_t i = 0; i < prev_pts_.size(); ++i) {
        const cv::Point2f e = back_pts_[i] - prev_pts_[i];
        if (status_[i] == 0 || status_back_[i] == 0 || e.dot(e) > fb_max2) continue;
        const cv::Point2f p = prev_pts_[i] * 2.f + cv::Point2f(prev_origin_);  // anh goc
        const cv::Point2f q = next_pts_[i] * 2.f + cv::Point2f(cur_origin_);
        good_prev_.push_back(p);
        good_next_.push_back(q);
        dx_.push_back(static_cast<double>(q.x - p.x));
        dy_.push_back(static_cast<double>(q.y - p.y));
    }
    m.n_good = static_cast<int>(good_prev_.size());
    if (m.n_good < cfg_.flow_min_points) return false;
    const cv::Point2d shift(median_inplace(std::span<double>(dx_)), median_inplace(std::span<double>(dy_)));
    m.center_px = prev_center_px_ + shift;
    m.M_prev_to_cur = translation(shift);
    const cv::Mat A = cv::estimateAffinePartial2D(good_prev_, good_next_, cv::noArray(), cv::RANSAC, cfg_.affine_ransac_px);
    if (!A.empty())
        m.M_prev_to_cur = {A.at<double>(0, 0), A.at<double>(0, 1), A.at<double>(0, 2),
                           A.at<double>(1, 0), A.at<double>(1, 1), A.at<double>(1, 2), 0, 0, 1};
    m.method = FlowMethod::kFlow;
    return true;
}

bool RoiFlow::match_ncc(FlowMeasure& m) {
    if (templ_.empty() || templ_.cols >= cur_gray_.cols || templ_.rows >= cur_gray_.rows) return false;
    cv::matchTemplate(cur_gray_, templ_, ncc_out_, cv::TM_CCOEFF_NORMED);
    double best = 0.0;
    cv::Point loc;
    cv::minMaxLoc(ncc_out_, nullptr, &best, nullptr, &loc);
    m.ncc = best;
    if (best < cfg_.ncc_min) return false;
    m.center_px = cv::Point2d(cur_origin_) + 2.0 * (cv::Point2d(loc) + templ_center_half_);
    m.M_prev_to_cur = translation(m.center_px - prev_center_px_);
    m.method = FlowMethod::kNcc;
    return true;
}

FlowMeasure RoiFlow::measure(const cv::Mat& image_rgb, cv::Point2d predicted_center_px) {
    FlowMeasure m;
    if (side_px_ <= 0 || prev_gray_.empty()) return m;
    cur_origin_ = window_origin(predicted_center_px, image_rgb.size());
    gray_half(image_rgb, cur_origin_, cur_gray_);
    if (!track_lk(predicted_center_px, m) && !match_ncc(m)) m.method = FlowMethod::kNone;
    // khung nay thanh khung truoc cua lan sau
    std::swap(prev_gray_, cur_gray_);
    prev_origin_ = cur_origin_;
    prev_center_px_ = m.method == FlowMethod::kNone ? predicted_center_px : m.center_px;
    if (m.method == FlowMethod::kFlow) {
        prev_pts_.clear();
        for (const auto& q : good_next_) prev_pts_.push_back((q - cv::Point2f(cur_origin_)) * 0.5f);
    } else {
        prev_pts_.clear();  // tim lai trong refresh_points (mask da doi)
    }
    return m;
}

void RoiFlow::refresh_points(const SegMask& mask) {
    if (side_px_ <= 0 || prev_gray_.empty() || static_cast<int>(prev_pts_.size()) * 2 >= cfg_.flow_max_corners) return;
    mask_half(mask, prev_origin_, mask_half_);
    cv::goodFeaturesToTrack(prev_gray_, prev_pts_, cfg_.flow_max_corners, cfg_.flow_quality, cfg_.flow_min_distance_px,
                            mask_half_);
}

}  // namespace uavp
