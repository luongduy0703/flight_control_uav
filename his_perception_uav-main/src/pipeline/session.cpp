#include "uavp/pipeline/session.hpp"

#include <algorithm>
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
                                             Segmenter& segmenter, const Config& cfg) {
    const int64_t start = mono_ns();
    std::vector<std::pair<int, Observation>> measured;
    for (const int i : s.group) {
        Track& t = s.tracks[static_cast<std::size_t>(i)];
        if (t.state != State::kDefer) continue;
        SegObject obj;
        std::optional<SegObject> seg;
        if (track_segment(t, frame_idx, image, cfg.track, segmenter, obj)) seg = std::move(obj);
        Observation o = measure(std::move(seg), t.detection.conf, frame_idx, cfg.camera, cfg.D_m);
        o.t_ns = t_ns;
        measured.emplace_back(i, std::move(o));
    }
    const double segment_ms = ms_since(start);
    const int64_t start2 = mono_ns();
    for (auto& [i, obs] : measured) update_state(s.tracks[static_cast<std::size_t>(i)], std::move(obs), cfg.decide);
    return {segment_ms, ms_since(start2)};
}

}  // namespace

bool track_segment(Track& t, uint32_t frame_idx, const cv::Mat& image_rgb, const TrackConfig& cfg,
                   Segmenter& segmenter, SegObject& out) {
    t.last_seen_frame = frame_idx;
    if (!segmenter(image_rgb, predicted_box(t, frame_idx), out)) return false;
    update_from_mask(t, out.center_px, frame_idx, cfg.velocity_alpha);
    return true;
}

StepTimes session_step(Session& s, uint32_t frame_idx, int64_t t_ns, const cv::Mat& image_rgb, const Config& cfg,
                       Detector& detector, Segmenter& segmenter) {
    StepTimes t;
    const cv::Point2d center = cfg.camera.center_px();
    s.detections.clear();
    if (should_detect(s, frame_idx)) {
        const int64_t start = mono_ns();
        detector(image_rgb, s.detections);
        t.detect_ms = ms_since(start);
        t.detected = true;
        s.need_detect = false;
        match(s.tracks, s.detections, cfg.match_radius_px, frame_idx);  // ca track nhom cu: giu detection vat da xet
        if (s.group.empty()) {  // dau phien / chua co vat / nhom truoc ABORT het: chon nhom tu khung nay
            s.group = select_group(s.tracks, s.detections, center, kTargets, cfg.match_radius_px, frame_idx);
            if (s.group.empty() && s.reselect) {  // nhom truoc ABORT het, khung nay khong con vat moi -> ket thuc
                UAVP_LOG_INFO(kModule, "Ca nhom ABORT -> chon nhom khac: khong con vat nao");
                log_frame(frame_idx, s.detections.size(), s, t, cfg);
                s.done = true;
                return t;
            }
            s.reselect = false;
        }
    }
    std::tie(t.segment_ms, t.select_ms) = segment_and_decide(image_rgb, s, frame_idx, t_ns, segmenter, cfg);
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
    s.group = select_group(s.tracks, s.detections, center, kTargets, cfg.match_radius_px, frame_idx);
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
        char pose[128] = "-";
        if (seg != nullptr)
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
