#include "uavp/stages/s3_select/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

#include <opencv2/calib3d.hpp>

#include "uavp/tracking/world_pose.hpp"

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

// Bbox moi o khung nay -> p_W (tam bbox tren san). Che do the gioi khong dung van toc anh.
void set_world(Track& t, const FloorView* view) {
    if (view == nullptr) return;
    if (const auto p = floor_point(*view, t.center_px)) {
        t.p_W_mm = *p;
        t.has_world = true;
        t.vel_px = {0.0, 0.0};
    }
}

// Khoang cach chuan hoa (<= 1: trong ban kinh) giua detection va track: mm tren san neu ca hai co diem san, khong thi px.
double gate(const Track& t, const Detection& d, uint32_t frame, double radius_px, const FloorView* view) {
    if (view != nullptr && t.has_world) {
        if (const auto p = floor_point(*view, d.center_px()))
            return std::hypot(p->x - t.p_W_mm.x, p->y - t.p_W_mm.y) / view->radius_mm;
    }
    return dist(d.center_px(), predicted_center(t, frame, view)) / radius_px;
}

}  // namespace

// --- tracker --------------------------------------------------------------------------------------------------
std::optional<cv::Point3d> floor_point(const FloorView& v, cv::Point2d uv_px) {
    if (v.camera == nullptr) return std::nullopt;
    const std::vector<cv::Point2d> in{uv_px};
    std::vector<cv::Point2d> norm;
    cv::undistortPoints(in, norm, v.camera->K_px, v.camera->dist_coeffs);
    const cv::Vec3d ray = v.pose.R_WC * cv::Vec3d(norm[0].x, norm[0].y, 1.0);
    if (!(ray[2] > 1e-3)) return std::nullopt;  // tia khong huong xuong
    const double s = (v.floor_z_mm - v.pose.t_WC_mm[2]) / ray[2];
    if (!(s > 0.0) || !std::isfinite(s)) return std::nullopt;
    const cv::Vec3d p = v.pose.t_WC_mm + s * ray;
    return cv::Point3d(p[0], p[1], p[2]);
}

std::optional<cv::Point2d> floor_pixel(const FloorView& v, const cv::Point3d& p_W_mm) {
    if (v.camera == nullptr) return std::nullopt;
    return project_point_px(to_camera(v.pose, p_W_mm), *v.camera);
}

Detection predicted_box(const Track& t, uint32_t frame, const FloorView* view) {
    if (view != nullptr && t.has_world) {
        if (const auto uv = floor_pixel(*view, t.p_W_mm)) return shifted(t.detection, *uv - t.center_px);
    }
    const double frames = static_cast<double>(frame) - static_cast<double>(t.box_frame);
    return shifted(t.detection, t.vel_px * frames);
}

std::vector<int> select_group(std::vector<Track>& tracks, std::span<const Detection> detections, cv::Point2d center_px,
                              int n_targets, double radius_px, uint32_t frame_idx, const FloorView* view) {
    std::vector<const Detection*> fresh;
    std::vector<std::optional<cv::Point3d>> fresh_W;  // diem san cua detection (che do the gioi)
    for (const auto& d : detections) {
        const std::optional<cv::Point3d> pW = view != nullptr ? floor_point(*view, d.center_px()) : std::nullopt;
        const bool known = std::any_of(tracks.begin(), tracks.end(), [&](const Track& t) {
            if (pW && t.has_world) return same_object(*pW, t.p_W_mm, view->same_mm);  // ca track da ABORT
            return gate(t, d, frame_idx, radius_px, view) <= 1.0;
        });
        if (known) continue;
        fresh.push_back(&d);
        fresh_W.push_back(pW);
    }
    std::vector<std::size_t> order(fresh.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return dist(fresh[a]->center_px(), center_px) < dist(fresh[b]->center_px(), center_px);
    });
    {  // 2 detection cung khung co tam tren san qua gan -> mot vat: giu cai gan tam anh hon
        std::vector<const Detection*> kept;
        std::vector<std::optional<cv::Point3d>> kept_W;
        for (const std::size_t i : order) {
            const bool dup = fresh_W[i] && std::any_of(kept_W.begin(), kept_W.end(), [&](const auto& k) {
                                 return k && same_object(*fresh_W[i], *k, view->same_mm);
                             });
            if (dup) continue;
            kept.push_back(fresh[i]);
            kept_W.push_back(fresh_W[i]);
        }
        fresh = std::move(kept);
    }
    if (fresh.size() > static_cast<std::size_t>(n_targets)) fresh.resize(static_cast<std::size_t>(n_targets));
    std::vector<int> group;
    const auto first_id = static_cast<int>(tracks.size());
    for (std::size_t i = 0; i < fresh.size(); ++i) {
        Track t;
        t.track_id = first_id + static_cast<int>(i);
        t.class_id = fresh[i]->class_id;
        set_box(t, *fresh[i], frame_idx);
        set_world(t, view);
        t.last_seen_frame = frame_idx;
        tracks.push_back(std::move(t));
        group.push_back(first_id + static_cast<int>(i));
    }
    return group;
}

void match(std::vector<Track>& tracks, std::span<const Detection> detections, double radius_px, uint32_t frame_idx,
           const FloorView* view) {
    std::vector<std::tuple<double, std::size_t, std::size_t>> pairs;  // (khoang cach / ban kinh, track, detection)
    for (std::size_t ti = 0; ti < tracks.size(); ++ti)
        for (std::size_t di = 0; di < detections.size(); ++di)
            pairs.emplace_back(gate(tracks[ti], detections[di], frame_idx, radius_px, view), ti, di);
    std::sort(pairs.begin(), pairs.end());  // giong sorted() cua Python tren tuple
    std::vector<uint8_t> used_t(tracks.size(), 0), used_d(detections.size(), 0);
    for (const auto& [d, ti, di] : pairs) {
        if (d > 1.0) break;
        if (used_t[ti] != 0 || used_d[di] != 0) continue;
        Track& t = tracks[ti];
        set_box(t, detections[di], frame_idx);
        set_world(t, view);
        t.last_seen_frame = frame_idx;
        used_t[ti] = 1;
        used_d[di] = 1;
    }
}

void update_from_mask(Track& t, cv::Point2d mask_center_px, uint32_t frame, double alpha, const FloorView* view) {
    if (view != nullptr && t.has_world) {  // the gioi: bbox dat quanh tam mask (giu lech bbox - mask), p_W theo bbox
        if (t.box_frame != frame) {
            const Detection pred = predicted_box(t, frame, view);
            const cv::Point2d off = t.has_mask ? t.center_px - t.last_mask_center_px : cv::Point2d(0.0, 0.0);
            set_box(t, shifted(pred, mask_center_px + off - pred.center_px()), frame);
            set_world(t, view);
        }
        t.last_mask_center_px = mask_center_px;
        t.last_mask_frame = frame;
        t.has_mask = true;
        return;
    }
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
