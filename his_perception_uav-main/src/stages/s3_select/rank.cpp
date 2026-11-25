#include "uavp/stages/s3_select/rank.hpp"

#include <algorithm>
#include <cmath>

namespace uavp {

namespace {

double dist(cv::Point2d a, cv::Point2d b) noexcept { return std::hypot(a.x - b.x, a.y - b.y); }

}  // namespace

// --- rank -----------------------------------------------------------------------------------------------------
const SegObject* last_seg(const Track& track) noexcept {
    if (track.observations.empty() || !track.observations.back().seg) return nullptr;
    return &*track.observations.back().seg;
}

double distance_to_center_px(const Track& track, cv::Point2d center_px) noexcept {
    const SegObject* seg = last_seg(track);
    return seg == nullptr ? kNaN : dist(seg->center_px, center_px);
}

std::vector<int> rank_ok(const std::vector<Track>& tracks, cv::Point2d center_px) {
    std::vector<int> ok;
    for (std::size_t i = 0; i < tracks.size(); ++i)
        if (tracks[i].state == State::kOk) ok.push_back(static_cast<int>(i));
    std::stable_sort(ok.begin(), ok.end(), [&](int a, int b) {
        return distance_to_center_px(tracks[static_cast<std::size_t>(a)], center_px) <
               distance_to_center_px(tracks[static_cast<std::size_t>(b)], center_px);
    });
    return ok;
}

const Track* selected_track(const std::vector<Track>& tracks, cv::Point2d center_px) {
    const std::vector<int> ranked = rank_ok(tracks, center_px);
    if (ranked.empty()) return nullptr;
    const Track& t = tracks[static_cast<std::size_t>(ranked.front())];
    return last_seg(t) == nullptr ? nullptr : &t;
}

}  // namespace uavp
