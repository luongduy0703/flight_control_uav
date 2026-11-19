#include "uavp/stages/s3_select/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace uavp {

namespace {

double dist(cv::Point2d a, cv::Point2d b) noexcept { return std::hypot(a.x - b.x, a.y - b.y); }

Detection shifted(Detection d, cv::Point2d shift_px) noexcept {
    const auto dx = static_cast<float>(shift_px.x), dy = static_cast<float>(shift_px.y);
    d.x1 += dx;
    d.x2 += dx;
    d.y1 += dy;
    d.y2 += dy;
    return d;
}

void set_box(Track& t, const Detection& d, uint32_t frame) noexcept {
    t.detection = d;
    t.center_px = d.center_px();
    t.box_frame = frame;
}

}  // namespace

// --- tracker --------------------------------------------------------------------------------------------------
Detection predicted_box(const Track& t, uint32_t frame) noexcept {
    const double frames = static_cast<double>(frame) - static_cast<double>(t.box_frame);
    return shifted(t.detection, t.vel_px * frames);
}

std::vector<int> select_group(std::vector<Track>& tracks, std::span<const Detection> detections, cv::Point2d center_px,
                              int n_targets, double radius_px, uint32_t frame_idx) {
    std::vector<const Detection*> fresh;
    for (const auto& d : detections) {
        const bool known = std::any_of(tracks.begin(), tracks.end(), [&](const Track& t) {
            return dist(d.center_px(), predicted_center(t, frame_idx)) <= radius_px;
        });
        if (!known) fresh.push_back(&d);
    }
    std::stable_sort(fresh.begin(), fresh.end(), [&](const Detection* a, const Detection* b) {
        return dist(a->center_px(), center_px) < dist(b->center_px(), center_px);
    });
    if (fresh.size() > static_cast<std::size_t>(n_targets)) fresh.resize(static_cast<std::size_t>(n_targets));
    std::vector<int> group;
    const auto first_id = static_cast<int>(tracks.size());
    for (std::size_t i = 0; i < fresh.size(); ++i) {
        Track t;
        t.track_id = first_id + static_cast<int>(i);
        t.class_id = fresh[i]->class_id;
        set_box(t, *fresh[i], frame_idx);
        t.last_seen_frame = frame_idx;
        tracks.push_back(std::move(t));
        group.push_back(first_id + static_cast<int>(i));
    }
    return group;
}

void match(std::vector<Track>& tracks, std::span<const Detection> detections, double radius_px, uint32_t frame_idx) {
    std::vector<std::tuple<double, std::size_t, std::size_t>> pairs;  // (khoang cach, track, detection)
    for (std::size_t ti = 0; ti < tracks.size(); ++ti) {
        const cv::Point2d pred = predicted_center(tracks[ti], frame_idx);
        for (std::size_t di = 0; di < detections.size(); ++di)
            pairs.emplace_back(dist(detections[di].center_px(), pred), ti, di);
    }
    std::sort(pairs.begin(), pairs.end());  // giong sorted() cua Python tren tuple
    std::vector<uint8_t> used_t(tracks.size(), 0), used_d(detections.size(), 0);
    for (const auto& [d, ti, di] : pairs) {
        if (d > radius_px) break;
        if (used_t[ti] != 0 || used_d[di] != 0) continue;
        Track& t = tracks[ti];
        set_box(t, detections[di], frame_idx);
        t.last_seen_frame = frame_idx;
        used_t[ti] = 1;
        used_d[di] = 1;
    }
}

void update_from_mask(Track& t, cv::Point2d mask_center_px, uint32_t frame, double alpha) noexcept {
    if (t.box_frame != frame) {  // bbox du doan: dat lai theo vi tri mask thuc
        const Detection pred = predicted_box(t, frame);
        const cv::Point2d pred_mask = t.has_mask ? t.last_mask_center_px +
                                                       t.vel_px * (static_cast<double>(frame) - t.last_mask_frame)
                                                 : mask_center_px;
        set_box(t, shifted(pred, mask_center_px - pred_mask), frame);
    }
    if (t.has_mask && frame > t.last_mask_frame) {
        const cv::Point2d v_meas = (mask_center_px - t.last_mask_center_px) / static_cast<double>(frame - t.last_mask_frame);
        t.vel_px = alpha * v_meas + (1.0 - alpha) * t.vel_px;
    }
    t.last_mask_center_px = mask_center_px;
    t.last_mask_frame = frame;
    t.has_mask = true;
}

}  // namespace uavp
