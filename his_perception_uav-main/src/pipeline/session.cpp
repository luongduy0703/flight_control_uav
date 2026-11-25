#include "uavp/pipeline/session.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "uavp/common/log.hpp"
#include "uavp/stages/s3_select.hpp"

namespace uavp {
namespace {

constexpr const char* kModule = "uavp.run_session";

bool any_state(const Session& s, State st) {
    return std::any_of(s.group.begin(), s.group.end(),
                       [&](int i) { return s.tracks[static_cast<std::size_t>(i)].state == st; });
}

// Track DEFER cua nhom khong co mask o khung truoc (s2 tren ROI du doan that bai) -> can s1 de ghep lai.
bool any_lost(const Session& s, uint32_t frame_idx) {
    return std::any_of(s.group.begin(), s.group.end(), [&](int i) {
        const Track& t = s.tracks[static_cast<std::size_t>(i)];
        return t.state == State::kDefer && frame_idx > 0 && (!t.has_mask || t.last_mask_frame + 1 < frame_idx);
    });
}

// s1 chi chay khi: dau phien / chua co vat / vat truoc bi loai (need_detect, nhom rong) hoac mat vat (s2 khong ra mask o
// khung truoc -> can s1 ghep lai bbox). Con lai s2 bam ROI du doan.
bool should_detect(const Session& s, uint32_t frame_idx) {
    return s.need_detect || s.group.empty() || any_lost(s, frame_idx);
}

// s2 + do W moi track DEFER cua nhom (ROI quanh bbox du doan), roi cap nhat trang thai. Tra ve (ms s2, ms quyet dinh).
std::pair<double, double> segment_and_decide(const cv::Mat& image, Session& s, uint32_t frame_idx, int64_t t_ns,
                                             Segmenter& segmenter, const Config& cfg, const FloorView* view) {
    const int64_t start = mono_ns();
    std::vector<std::pair<int, Observation>> measured;
    for (const int i : s.group) {
        Track& t = s.tracks[static_cast<std::size_t>(i)];
        if (t.state != State::kDefer) continue;
        SegObject obj;
        std::optional<SegObject> seg;
        if (track_segment(t, frame_idx, image, cfg.track, segmenter, obj, view)) seg = std::move(obj);
        Observation o = measure(std::move(seg), t.detection.conf, frame_idx, cfg.camera, cfg.D_m);
        o.t_ns = t_ns;
        measured.emplace_back(i, std::move(o));
    }
    const double segment_ms = ms_since(start);
    const int64_t start2 = mono_ns();
    for (auto& [i, obs] : measured) update_state(s.tracks[static_cast<std::size_t>(i)], std::move(obs), cfg.decide);
    return {segment_ms, ms_since(start2)};
}

// Detection duoc phep chon nhom moi: bo detection trong vung loai (vd to giay ArUco diem cat canh) va detection qua to
// (canh ngan bbox doi ra mm bang D tu do cao EKF > bbox_max_mm) -> khong ton thoi gian segment vat khong gap duoc.
std::vector<Detection> pickable(Session& s, const Config& cfg) {
    const bool size_gate = cfg.bbox_max_mm > 0.0 && cfg.decide.size_abort;
    if (s.exclude.empty() && !size_gate) return s.detections;
    std::vector<Detection> out;
    for (const auto& d : s.detections) {
        if (excluded(d, s.exclude)) {
            ++s.excluded_detections;
            continue;
        }
        if (size_gate) {
            const double w = bbox_short_mm(d, cfg.D_m, cfg.camera);
            if (std::isfinite(w) && w > cfg.bbox_max_mm) {
                ++s.too_big_detections;
                UAVP_LOG_DEBUG(kModule, "Bo detection qua to: bbox canh ngan %.0f mm > %.0f mm (D %.2f m)", w,
                               cfg.bbox_max_mm, cfg.D_m);
                continue;
            }
        }
        out.push_back(d);
    }
    return out;
}

// Track DEFER cua nhom nam trong vung loai (vd da chon to giay marker truoc khi thay marker) -> ABORT launch_marker.
void abort_excluded(Session& s, uint32_t frame_idx, const FloorView* view) {
    if (s.exclude.empty()) return;
    for (const int i : s.group) {
        Track& t = s.tracks[static_cast<std::size_t>(i)];
        if (t.state != State::kDefer || !excluded(predicted_box(t, frame_idx, view), s.exclude)) continue;
        t.state = State::kAbort;
        t.reason = Reason::kLaunchMarker;
        UAVP_LOG_INFO(kModule, "  #%d nam tren ArUco diem cat canh -> ABORT launch_marker (khong gap)", t.track_id);
    }
}

// s3_select.single_shot: s1 MOT lan -> loc conf -> xet tu vat gan tam: s2 mask >= mask_min_px -> chon. Vat khac ->
// next_targets (p_W tren san neu co tu the FC). Phien ket thuc sau khung nay (co hoac khong co vat duoc chon).
StepTimes single_shot_step(Session& s, uint32_t frame_idx, int64_t t_ns, const cv::Mat& image_rgb, const Config& cfg,
                           Detector& detector, Segmenter& segmenter) {
    StepTimes t;
    const cv::Point2d center = cfg.camera.center_px();
    const std::optional<FloorView> fv = floor_view(s, cfg);
    const FloorView* view = fv ? &*fv : nullptr;
    s.detections.clear();
    const int64_t start = mono_ns();
    detector(image_rgb, s.detections);
    t.detect_ms = ms_since(start);
    t.detected = true;
    s.need_detect = false;
    const int64_t start_select = mono_ns();
    std::vector<Detection> candidates;
    int low_conf = 0;
    for (const auto& d : pickable(s, cfg)) {  // vung loai (ArUco diem cat canh) + qua to (khi size_abort)
        if (d.conf < cfg.decide.conf_min) {
            ++low_conf;
            continue;
        }
        candidates.push_back(d);
    }
    // Moi vat -> track (p_W tren san neu co tu the), gan tam truoc; 2 detection cung diem san -> 1 vat.
    const std::vector<int> order = select_group(s.tracks, candidates, center, static_cast<int>(candidates.size()),
                                                cfg.match_radius_px, frame_idx, view);
    double segment_ms = 0.0;
    int chosen = -1;
    for (const int i : order) {
        Track& tr = s.tracks[static_cast<std::size_t>(i)];
        if (chosen >= 0) {  // da co vat duoc chon: con lai chi nho vi tri, khong ton s2
            s.next_targets.push_back(i);
            continue;
        }
        const int64_t start_seg = mono_ns();
        SegObject obj;
        std::optional<SegObject> seg;
        if (track_segment(tr, frame_idx, image_rgb, cfg.track, segmenter, obj, view)) seg = std::move(obj);
        Observation o = measure(std::move(seg), tr.detection.conf, frame_idx, cfg.camera, cfg.D_m);
        o.t_ns = t_ns;
        segment_ms += ms_since(start_seg);
        if (o.area_px < cfg.decide.mask_min_px) {
            tr.state = State::kAbort;
            tr.reason = Reason::kMaskSmall;
            s.next_targets.push_back(i);  // van la vat detect duoc: nho vi tri (khung sau co the ra mask)
            continue;
        }
        tr.observations.push_back(std::move(o));
        if (cfg.decide.size_abort && median_W_mm(tr) > cfg.decide.w_max_mm) {  // tat bang size_abort: false
            tr.state = State::kAbort;
            tr.reason = Reason::kTooBig;
            continue;  // qua to cho tay gap: khong phai dich tiep theo
        }
        tr.state = State::kOk;
        chosen = i;
    }
    t.segment_ms = segment_ms;
    t.select_ms = std::max(0.0, ms_since(start_select) - segment_ms);
    s.group.clear();
    if (chosen >= 0) s.group.push_back(chosen);
    s.done = true;
    log_frame(frame_idx, s.detections.size(), s, t, cfg);
    UAVP_LOG_INFO(kModule, "single_shot: %zu detection, %d bo conf < %.2f, %zu ung vien -> %s, %zu vat nho lam dich tiep",
                  s.detections.size(), low_conf, cfg.decide.conf_min, candidates.size(),
                  chosen >= 0 ? ("chon #" + std::to_string(s.tracks[static_cast<std::size_t>(chosen)].track_id)).c_str()
                              : "KHONG chon duoc vat nao",
                  s.next_targets.size());
    return t;
}

}  // namespace

double bbox_short_mm(const Detection& d, double D_m, const CameraModel& camera) noexcept {
    const double f = camera.K_px(0, 0);
    if (!(D_m > 0.0) || !std::isfinite(D_m) || !(f > 0.0)) return kNaN;
    const double px = std::min(static_cast<double>(d.x2 - d.x1), static_cast<double>(d.y2 - d.y1));
    return px * D_m * 1000.0 / f;
}

bool in_zone(const Detection& d, const ExcludeZone& z) noexcept {
    const bool contains = z.center_px.x >= d.x1 && z.center_px.x <= d.x2 && z.center_px.y >= d.y1 && z.center_px.y <= d.y2;
    const cv::Point2d c = d.center_px();
    return contains || std::hypot(c.x - z.center_px.x, c.y - z.center_px.y) < z.radius_px;
}

bool excluded(const Detection& d, const std::vector<ExcludeZone>& zones) noexcept {
    return std::any_of(zones.begin(), zones.end(), [&](const ExcludeZone& z) { return in_zone(d, z); });
}

bool track_segment(Track& t, uint32_t frame_idx, const cv::Mat& image_rgb, const TrackConfig& cfg,
                   Segmenter& segmenter, SegObject& out, const FloorView* view) {
    t.last_seen_frame = frame_idx;
    if (!segmenter(image_rgb, predicted_box(t, frame_idx, view), out)) {
        // Mat mask: KHONG ngoai suy van toc tiep (UAV hover -> van toc chi la rung; ngoai suy nhieu khung day ROI
        // va diem ghep detection ra khoi vat -> s1 thay vat nhung khong ghep duoc -> mask_small). Giu o vi tri biet cuoi.
        t.vel_px = {0.0, 0.0};
        return false;
    }
    update_from_mask(t, out.center_px, frame_idx, cfg.velocity_alpha, view);
    return true;
}

std::optional<FloorView> floor_view(Session& s, const Config& cfg) {
    if (!cfg.world_track || !s.pose) return std::nullopt;
    const CamPose& p = *s.pose;
    if (!std::isfinite(s.floor_z_W_mm)) {  // chot mat san (D tu luc bat dau: EKF / ToF)
        if (!(cfg.D_m > 0.0) || !(p.R_WC(2, 2) > 0.5)) return std::nullopt;
        s.world_position = p.has_position;
        s.floor_z_W_mm = (p.has_position ? p.t_WC_mm[2] : 0.0) + 1000.0 * cfg.D_m * p.R_WC(2, 2);
        UAVP_LOG_INFO(kModule, "s3 bam vat theo the gioi: mat san z = %.0f mm (%s), ghep <= %.0f mm, cung vat <= %.0f mm",
                      s.floor_z_W_mm, p.has_position ? "vi tri EKF" : "chi goc quay", cfg.match_radius_mm,
                      cfg.same_object_mm);
    }
    if (p.has_position != s.world_position) return std::nullopt;
    FloorView v;
    v.pose = p;
    if (!p.has_position) v.pose.t_WC_mm = {0.0, 0.0, 0.0};
    v.floor_z_mm = s.floor_z_W_mm;
    v.camera = &cfg.camera;
    v.radius_mm = cfg.match_radius_mm;
    v.same_mm = cfg.same_object_mm;
    return v;
}

bool will_detect(const Session& s, const Config& cfg, uint32_t frame_idx) {
    if (s.done) return false;
    return cfg.single_shot || should_detect(s, frame_idx);
}

StepTimes session_step(Session& s, uint32_t frame_idx, int64_t t_ns, const cv::Mat& image_rgb, const Config& cfg,
                       Detector& detector, Segmenter& segmenter) {
    if (cfg.single_shot) return single_shot_step(s, frame_idx, t_ns, image_rgb, cfg, detector, segmenter);
    StepTimes t;
    const cv::Point2d center = cfg.camera.center_px();
    const std::optional<FloorView> fv = floor_view(s, cfg);
    const FloorView* view = fv ? &*fv : nullptr;
    s.detections.clear();
    if (should_detect(s, frame_idx)) {
        const int64_t start = mono_ns();
        detector(image_rgb, s.detections);
        t.detect_ms = ms_since(start);
        t.detected = true;
        s.need_detect = false;
        match(s.tracks, s.detections, cfg.match_radius_px, frame_idx, view);  // ca track nhom cu: giu detection vat da xet
        if (s.group.empty()) {  // dau phien / chua co vat / nhom truoc ABORT het: chon nhom tu khung nay
            s.group = select_group(s.tracks, pickable(s, cfg), center, kTargets, cfg.match_radius_px, frame_idx, view);
            if (s.group.empty() && s.reselect) {  // nhom truoc ABORT het, khung nay khong con vat moi -> ket thuc
                UAVP_LOG_INFO(kModule, "Ca nhom ABORT -> chon nhom khac: khong con vat nao");
                log_frame(frame_idx, s.detections.size(), s, t, cfg);
                s.done = true;
                return t;
            }
            s.reselect = false;
        }
    }
    abort_excluded(s, frame_idx, view);
    std::tie(t.segment_ms, t.select_ms) = segment_and_decide(image_rgb, s, frame_idx, t_ns, segmenter, cfg, view);
    log_frame(frame_idx, s.detections.size(), s, t, cfg);
    if (s.group.empty() || any_state(s, State::kDefer)) return t;
    if (any_state(s, State::kOk)) {
        s.done = true;
        return t;
    }
    if (!t.detected) {  // ca nhom ABORT o khung khong chay s1: chon nhom moi o khung sau (can detection moi)
        s.need_detect = true;
        s.reselect = true;
        s.group.clear();
        return t;
    }
    s.group = select_group(s.tracks, pickable(s, cfg), center, kTargets, cfg.match_radius_px, frame_idx, view);
    std::string ids;
    for (const int i : s.group) ids += std::to_string(s.tracks[static_cast<std::size_t>(i)].track_id) + " ";
    UAVP_LOG_INFO(kModule, "Ca nhom ABORT -> chon nhom khac: %s", ids.empty() ? "khong con vat nao" : ids.c_str());
    s.done = s.group.empty();
    return t;
}

void log_frame(uint32_t frame_idx, std::size_t n_detections, const Session& s, const StepTimes& t,
               const Config& cfg) {
    if (t.detected)
        UAVP_LOG_INFO(kModule, "khung %u | s1 detect %zu vat %.1f ms | s2 segment %.1f ms | s3 select %.2f ms", frame_idx,
                      n_detections, t.detect_ms, t.segment_ms, t.select_ms);
    else
        UAVP_LOG_INFO(kModule, "khung %u | s1 -- (bam ROI du doan) | s2 segment %.1f ms | s3 select %.2f ms", frame_idx,
                      t.segment_ms, t.select_ms);
    for (const int i : s.group) {
        const Track& tr = s.tracks[static_cast<std::size_t>(i)];
        const SegObject* seg = last_seg(tr);
        char pose[160] = "-";
        if (seg != nullptr && tr.has_world)
            std::snprintf(pose, sizeof(pose), "tam (%.0f, %.0f) yaw %5.1f san W (%.0f, %.0f) mm", seg->center_px.x,
                          seg->center_px.y, seg->yaw_deg, tr.p_W_mm.x, tr.p_W_mm.y);
        else if (seg != nullptr)
            std::snprintf(pose, sizeof(pose), "tam (%.0f, %.0f) yaw %5.1f v (%+.1f, %+.1f) px/khung", seg->center_px.x,
                          seg->center_px.y, seg->yaw_deg, tr.vel_px.x, tr.vel_px.y);
        const std::string_view reason = name(tr.reason);
        UAVP_LOG_INFO(kModule, "  #%d %-6s %-5s n_obs %zu  W %6.1f mm  conf %.3f  %s  %s%s", tr.track_id,
                      cfg.detector.classes.at(tr.class_id).c_str(), std::string(name(tr.state)).c_str(),
                      tr.observations.size(), median_W_mm(tr), mean_conf(tr), pose,
                      reason.empty() ? "-" : std::string(reason).c_str(),
                      tr.state != State::kDefer || tr.last_mask_frame == frame_idx ? "" : "  (khong co mask o khung nay)");
    }
}

}  // namespace uavp
