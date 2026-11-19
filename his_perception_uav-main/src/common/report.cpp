// Thoi gian tung buoc + cac muc result.json (cung khoa voi ban Python de compare_runs doc duoc ca hai).
#include <algorithm>
#include <cmath>

#include "uavp/common/log.hpp"
#include "uavp/common/run_files.hpp"
#include "uavp/stages/s3_select.hpp"

namespace uavp {
namespace {

constexpr double kP95 = 95.0;

double round1(double v) { return std::round(v * 10.0) / 10.0; }

void stats(const std::vector<double>& v, double& median, double& p95) {
    std::vector<double> a(v);
    median = median_inplace(std::span<double>(a));
    p95 = percentile_inplace(std::span<double>(a), kP95);
}

}  // namespace

// --- Timings ----------------------------------------------------------------------------------------------------
void Timings::add(const std::string& name, double ms) {
    const auto it = std::find_if(items_.begin(), items_.end(), [&](const auto& kv) { return kv.first == name; });
    if (it == items_.end()) items_.emplace_back(name, std::vector<double>{ms});
    else it->second.push_back(ms);
}

const std::vector<double>* Timings::get(const std::string& name) const {
    const auto it = std::find_if(items_.begin(), items_.end(), [&](const auto& kv) { return kv.first == name; });
    return it == items_.end() ? nullptr : &it->second;
}

void Timings::write_json(JsonWriter& w) const {
    w.begin_object();
    for (const auto& [name, v] : items_) {
        if (v.empty()) continue;
        double med = 0.0, p95 = 0.0;
        stats(v, med, p95);
        w.key(name).begin_object().key("median_ms").num(med).key("p95_ms").num(p95).key("n").value(v.size()).end_object();
    }
    w.end_object();
}

void Timings::log_summary() const {
    for (const auto& [name, v] : items_) {
        if (v.empty()) continue;
        double med = 0.0, p95 = 0.0;
        stats(v, med, p95);
        UAVP_LOG_INFO("uavp.real_time", "Thoi gian %-10s median %7.2f ms, p95 %7.2f ms (%zu lan)", name.c_str(), med, p95,
                      v.size());
    }
}

// --- result.json ------------------------------------------------------------------------------------------------
void write_track_summary(JsonWriter& w, const Track& t, const Config& cfg) {
    const SegObject* seg = last_seg(t);
    const std::string_view reason = name(t.reason);
    w.begin_object().key("track_id").value(t.track_id).key("class").value(cfg.detector.classes.at(t.class_id));
    w.key("state").value(name(t.state)).key("reason");
    if (reason.empty()) w.null();
    else w.value(reason);
    w.key("n_obs").value(t.observations.size()).key("W_mm_median").num(median_W_mm(t)).key("L_mm_median");
    if (seg != nullptr) w.num(median_L_mm(t));
    else w.null();
    w.key("center_px");
    if (seg != nullptr) w.begin_array().value(round1(seg->center_px.x)).value(round1(seg->center_px.y)).end_array();
    else w.null();
    w.key("yaw_deg");
    if (seg != nullptr) w.num(seg->yaw_deg);
    else w.null();
    w.key("conf_mean").num(mean_conf(t)).key("dist_to_center_px").num(distance_to_center_px(t, cfg.camera.center_px()));
    w.end_object();
}

namespace {

void vec3(JsonWriter& w, const cv::Vec3d& v) { w.begin_array().num(v[0]).num(v[1]).num(v[2]).end_array(); }

}  // namespace

void write_fusion_summary(JsonWriter& w, const FusionResult& r) {
    w.begin_object().key("status").value(name(r.status)).key("p_obj_mm").begin_array();
    w.num(r.p_obj_mm.x).num(r.p_obj_mm.y).num(r.p_obj_mm.z).end_array();
    w.key("d_mm").num(r.d_mm).key("dist_mm").num(r.dist_mm).key("W_mm").num(r.W_mm).key("L_mm").num(r.L_mm);
    w.key("yaw_deg").num(r.yaw_deg);
    w.key("axes").begin_object().key("x");  // truc vat trong he camera: z = phap tuyen mat top
    vec3(w, r.axis_x);
    w.key("y");
    vec3(w, r.axis_y);
    w.key("z");
    vec3(w, r.axis_z);
    w.end_object().key("tilt_deg").num(r.tilt_deg).key("normal_observable").value(r.normal_observable);
    w.key("spread_w_mm").num(r.spread_w_mm).key("drift_mm_s").num(r.drift_mm_s);
    w.key("pose_compensated").value(r.pose_compensated).key("rms_mm").num(r.rms_mm);
    w.key("n_frames").value(r.n_frames).key("n_scans").value(r.n_scans).key("n_points").value(r.n_points);
    w.key("n_inliers").value(r.n_inliers).key("inlier_ratio").num(r.inlier_ratio).key("t_image_ns").value(r.t_image_ns);
    w.key("t_lidar_ns");
    if (r.points_t_ns.empty()) w.null();
    else w.begin_array().value(r.t_lidar_first_ns).value(r.t_lidar_last_ns).end_array();
    w.key("plane2").begin_object().key("d_mm").num(r.d2_mm).key("n_points").value(r.n2_inliers).end_object();
    w.key("ambiguous").value(r.ambiguous);
    w.key("points_file").value("lidar_points.csv").end_object();
}

void write_aruco_summary(JsonWriter& w, const ArucoPose& pose, cv::Point3d p, const FusionResult& r) {
    const double d_a = p.z, dist_a = std::sqrt(p.dot(p));
    const cv::Point3d diff = r.p_obj_mm - p;
    w.begin_object().key("found").value(true).key("marker_id").value(pose.marker_id).key("marker_center_mm");
    w.begin_array().num(pose.t_CM_mm[0]).num(pose.t_CM_mm[1]).num(pose.t_CM_mm[2]).end_array();
    w.key("tilt_deg").num(pose.tilt_deg).key("reproj_px").num(pose.reproj_px).key("p_obj_mm");
    w.begin_array().num(p.x).num(p.y).num(p.z).end_array();
    w.key("d_mm").num(d_a).key("dist_mm").num(dist_a).key("error_s4_minus_aruco").begin_object();
    w.key("d_mm").num(r.d_mm - d_a).key("d_pct").num(100.0 * (r.d_mm - d_a) / d_a);
    w.key("dist_mm").num(r.dist_mm - dist_a).key("p_obj_mm").num(std::sqrt(diff.dot(diff))).end_object();
    w.end_object();
}

void write_session_result(JsonWriter& w, const std::vector<Track>& tracks, const Config& cfg, int frames) {
    const std::vector<int> ranked = rank_ok(tracks, cfg.camera.center_px());
    const cv::Point2d c = cfg.camera.center_px();
    w.begin_object().key("frames").value(frames).key("D_m").value(cfg.D_m).key("center_px");
    w.begin_array().value(round1(c.x)).value(round1(c.y)).end_array().key("selected");
    if (ranked.empty()) w.null();
    else write_track_summary(w, tracks[static_cast<std::size_t>(ranked[0])], cfg);
    w.key("waiting").begin_array();
    for (std::size_t i = 1; i < ranked.size(); ++i) write_track_summary(w, tracks[static_cast<std::size_t>(ranked[i])], cfg);
    w.end_array();
    for (const State st : {State::kDefer, State::kAbort}) {
        w.key(name(st)).begin_array();
        for (const auto& t : tracks)
            if (t.state == st) write_track_summary(w, t, cfg);
        w.end_array();
    }
    w.key("mask_file").value(ranked.empty() ? "" : kSelectedMaskFile).end_object();
}

}  // namespace uavp
