// Doc config (cho lanh, chay mot lan luc khoi dong: uu tien de doc, CPP_RULES.md muc 0).
#include "uavp/common/config.hpp"

#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace uavp {
namespace {

// Khoa bat buoc: thieu -> exception kem ten khoa.
YAML::Node req(const YAML::Node& node, const std::string& key) {
    YAML::Node v = node[key];
    if (!v.IsDefined() || v.IsNull()) throw std::runtime_error("Thieu khoa config: " + key);
    return v;
}

template <typename T>
T get(const YAML::Node& node, const std::string& key) {
    return req(node, key).as<T>();
}

// Khoa true / false (khong nhan chuoi "yes", so 1...), giong as_bool cua ban Python.
bool get_bool(const YAML::Node& node, const std::string& key) {
    const YAML::Node v = req(node, key);
    const std::string text = v.Scalar();
    if (text != "true" && text != "false") throw std::runtime_error(key + " = " + text + ", phai la true / false");
    return text == "true";
}

std::vector<std::string> get_strings(const YAML::Node& node, const std::string& key) {
    return req(node, key).as<std::vector<std::string>>();
}

void apply_override(YAML::Node& raw, const std::string& item) {
    const auto eq = item.find('=');
    if (eq == std::string::npos) throw std::runtime_error("--set " + item + ": thieu '='");
    const std::string key = item.substr(0, eq);
    std::vector<std::string> parts;
    for (std::size_t start = 0;;) {
        const auto dot = key.find('.', start);
        parts.push_back(key.substr(start, dot - start));
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    YAML::Node node = raw;  // yaml-cpp: Node la tham chieu
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) node.reset(node[parts[i]]);
    if (!node[parts.back()].IsDefined()) throw std::runtime_error("Khong co khoa " + key + " trong config");
    node[parts.back()] = YAML::Load(item.substr(eq + 1));
}

int interpolation_of(const YAML::Node& m) {
    const auto interp = get<std::string>(m, "interpolation");
    if (interp != "linear" && interp != "area") throw std::runtime_error("interpolation = " + interp + ": linear | area");
    return interp == "linear" ? cv::INTER_LINEAR : cv::INTER_AREA;
}

// input_px: [w, h] hoac mot so (vuong). Phai la boi so cua stride lon nhat cua model (32).
void input_size(const YAML::Node& m, int& w_px, int& h_px) {
    constexpr int kStride = 32;
    const YAML::Node v = req(m, "input_px");
    if (v.IsSequence()) {
        if (v.size() != 2) throw std::runtime_error("input_px phai la [w, h] hoac mot so");
        w_px = v[0].as<int>();
        h_px = v[1].as<int>();
    } else {
        w_px = h_px = v.as<int>();
    }
    if (w_px <= 0 || h_px <= 0 || w_px % kStride != 0 || h_px % kStride != 0)
        throw std::runtime_error("input_px = [" + std::to_string(w_px) + ", " + std::to_string(h_px) +
                                 "]: phai > 0 va chia het cho 32");
}

DetectorConfig detector_config(const YAML::Node& d) {
    DetectorConfig c;
    const YAML::Node m = req(d, "model");  // file model (model_cfg)
    c.model = model_config(m);
    c.classes = get_strings(m, "classes");
    input_size(m, c.input_w_px, c.input_h_px);
    c.pad_value = get<int>(m, "pad_value");
    c.interpolation = interpolation_of(m);
    const auto mode = get<std::string>(d, "mode");
    if (mode != "tile" && mode != "full") throw std::runtime_error("s1_detect.mode = " + mode + ": tile | full");
    c.mode = mode == "tile" ? DetectMode::kTile : DetectMode::kFull;
    c.tile_px = get<int>(d, "tile_px");
    c.overlap = get<double>(d, "overlap");
    c.conf_thr = get<float>(d, "conf_thr");
    c.nms_iou = get<double>(d, "nms_iou");
    c.cross_window_ios = get<double>(d, "cross_window_ios");
    c.max_det = get<int>(d, "max_det");
    return c;
}

SegmenterConfig segmenter_config(const YAML::Node& s) {
    SegmenterConfig c;
    const YAML::Node m = req(s, "model");  // file model (model_cfg)
    c.model = model_config(m);
    c.classes = get_strings(m, "classes");
    c.input_px = get<int>(m, "input_px");
    c.interpolation = interpolation_of(m);
    const auto roi_image = get<std::vector<int>>(s, "roi_image_px");  // [w, h]; [0, 0] = anh goc
    if (roi_image.size() != 2 || roi_image[0] < 0 || roi_image[1] < 0)
        throw std::runtime_error("s2_segment.roi_image_px phai la [w, h] >= 0");
    c.roi_image_w_px = roi_image[0];
    c.roi_image_h_px = roi_image[1];
    c.roi_pad_value = get<int>(s, "roi_pad_value");
    c.roi_scale = get<double>(s, "roi_scale");
    c.conf_thr = get<float>(s, "conf_thr");
    c.nms_iou = get<double>(s, "nms_iou");
    c.max_det = get<int>(s, "max_det");
    c.mask_logit_thr = get<float>(s, "mask_logit_thr");
    c.min_select_ioa = get<double>(s, "min_select_ioa");
    return c;
}

FusionConfig fusion_config(const YAML::Node& f) {
    FusionConfig c;
    c.extrinsics = load_extrinsics(get<std::string>(f, "calib_T_CL"));
    c.min_pts = get<int>(f, "min_pts");
    c.lidar_window_s = get<double>(f, "lidar_window_s");
    c.plane_iters = get<int>(f, "plane_iters");
    c.ransac_tau_mm = get<double>(f, "ransac_tau_mm");
    c.ransac_seed = get<uint64_t>(f, "ransac_seed");
    c.max_tilt_deg = get<double>(f, "max_tilt_deg");
    c.min_spread_mm = get<double>(f, "min_spread_mm");
    c.drift_min_span_s = get<double>(f, "drift_min_span_s");
    c.max_motion_time_corr = get<double>(f, "max_motion_time_corr");
    c.min_inlier_ratio = get<double>(f, "min_inlier_ratio");
    c.ambiguity_ratio = get<double>(f, "ambiguity_ratio");
    c.erode_px = get<int>(f, "erode_px");
    c.intensity_min = get<double>(f, "intensity_min");
    c.r_min_mm = get<double>(f, "r_min_mm");
    c.r_max_mm = get<double>(f, "r_max_mm");
    c.scan_angle_min_deg = get<double>(f, "scan_angle_min_deg");
    c.scan_angle_max_deg = get<double>(f, "scan_angle_max_deg");
    const auto method = get<std::string>(f, "method");
    if (method == "world") c.method = FusionMethod::kWorld;
    else if (method == "image") c.method = FusionMethod::kImage;
    else throw std::runtime_error("s4_fusion.method phai la world hoac image, nhan: " + method);
    const YAML::Node w = req(f, "world");
    c.world.window_s = get<double>(w, "window_s");
    c.world.pose_max_gap_s = get<double>(w, "pose_max_gap_s");
    c.world.lidar_sigma_mm = get<double>(w, "lidar_sigma_mm");
    c.world.min_sqrt_l2_mm = get<double>(w, "min_sqrt_l2_mm");
    c.world.max_normal_sigma_deg = get<double>(w, "max_normal_sigma_deg");
    c.world.min_inliers = get<int>(w, "min_inliers");
    if (w["max_window_s"]) c.world.max_window_s = w["max_window_s"].as<double>();
    if (w["window_step_s"]) c.world.window_step_s = w["window_step_s"].as<double>();
    if (!(c.world.window_s > 0.0 && c.world.window_s <= 10.0 && c.world.pose_max_gap_s > 0.0 &&
          c.world.lidar_sigma_mm > 0.0 && c.world.min_sqrt_l2_mm >= 0.0 && c.world.max_normal_sigma_deg > 0.0 &&
          c.world.min_inliers >= 3 && c.world.max_window_s >= 0.0 && c.world.max_window_s <= 15.0 &&
          c.world.window_step_s > 0.0))
        throw std::runtime_error("s4_fusion.world: gia tri ngoai pham vi");
    return c;
}

cv::Matx33d matx33(const YAML::Node& rows) {
    cv::Matx33d m;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m(r, c) = rows[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)].as<double>();
    return m;
}

}  // namespace

// Gop `over` vao `base` (map long nhau gop tung khoa, con lai thay the). Tra ve node moi.
static YAML::Node merge_nodes(const YAML::Node& base, const YAML::Node& over) {
    if (!base.IsMap() || !over.IsMap()) return YAML::Clone(over);
    YAML::Node out = YAML::Clone(base);
    for (const auto& kv : over) {
        const std::string key = kv.first.as<std::string>();
        out[key] = out[key].IsDefined() ? merge_nodes(out[key], kv.second) : YAML::Clone(kv.second);
    }
    return out;
}

static YAML::Node load_with_base(const std::filesystem::path& path) {
    YAML::Node raw = YAML::LoadFile(path.string());
    if (!raw["_base_"].IsDefined()) return raw;
    const std::filesystem::path base = path.parent_path() / raw["_base_"].as<std::string>();
    raw.remove("_base_");
    return merge_nodes(load_with_base(base), raw);
}

YAML::Node load_raw(const std::filesystem::path& path, const std::vector<std::string>& overrides) {
    YAML::Node raw = load_with_base(path);
    for (const auto& kv : req(raw, "stages")) {
        YAML::Node stage = YAML::LoadFile(kv.second.as<std::string>());
        if (stage["model_cfg"].IsDefined()) stage["model"] = YAML::LoadFile(stage["model_cfg"].as<std::string>());
        raw[kv.first.as<std::string>()] = stage;
    }
    for (const auto& item : overrides) apply_override(raw, item);
    return raw;
}

ModelConfig model_config(const YAML::Node& r) {
    ModelConfig m;
    m.dir = get<std::string>(r, "dir");
    m.param_file = get<std::string>(r, "param_file");
    m.bin_file = get<std::string>(r, "bin_file");
    m.num_threads = get<int>(r, "num_threads");
    m.use_vulkan = get_bool(r, "use_vulkan");
    m.use_fp16_storage = get_bool(r, "use_fp16_storage");
    return m;
}

CameraModel load_camera(const std::filesystem::path& path) {
    const YAML::Node raw = YAML::LoadFile(path.string());
    CameraModel cam;
    cam.K_px = matx33(req(raw, "camera_matrix"));
    const auto dist = get<std::vector<double>>(raw, "dist_coeffs");
    cam.dist_coeffs = cv::Mat(dist, true).reshape(1, 1);
    return cam;
}

Extrinsics load_extrinsics(const std::filesystem::path& path) {
    const YAML::Node raw = YAML::LoadFile(path.string());
    Extrinsics e;
    e.R_CL = matx33(req(raw, "R_CL"));
    const auto t_m = get<std::vector<double>>(raw, "t_CL_m");
    e.t_CL_mm = cv::Vec3d(1000.0 * t_m.at(0), 1000.0 * t_m.at(1), 1000.0 * t_m.at(2));
    const YAML::Node angle = req(raw, "lidar_angle");
    const YAML::Node rng = req(raw, "range_correction");
    e.angle_clockwise = get_bool(angle, "clockwise");
    e.angle_offset_deg = get<double>(angle, "offset_deg");
    e.range_scale = get<double>(rng, "range_scale");
    e.range_offset_mm = 1000.0 * get<double>(rng, "range_offset_m");
    return e;
}

CaptureConfig load_capture(const std::filesystem::path& calib_K_path) {
    const YAML::Node s = req(YAML::LoadFile(calib_K_path.string()), "sensor_mode");
    CaptureConfig c;
    const auto size = get<std::vector<int>>(s, "image_size_px");
    const auto sensor = get<std::vector<int>>(s, "sensor_mode_px");
    c.width_px = size.at(0);
    c.height_px = size.at(1);
    c.sensor_width_px = sensor.at(0);
    c.sensor_height_px = sensor.at(1);
    c.lens_position = get<double>(s, "lens_position");
    c.rotate_deg = get<int>(s, "rotate_deg");
    c.fps = get<double>(s, "fps");
    return c;
}

CaptureConfig load_capture(const std::filesystem::path& calib_K_path, const SourcesConfig& sources) {
    CaptureConfig c = load_capture(calib_K_path);
    const double frame_us = 1e6 / c.fps;
    if (!(sources.camera_exposure_us >= 0.0) || sources.camera_exposure_us > frame_us)
        throw std::invalid_argument("sources.camera_exposure_us phai trong [0, " + std::to_string(frame_us) +
                                    "] us (0 = tu dong; khong dai hon 1 khung)");
    if (!(sources.camera_analogue_gain == 0.0 ||
          (sources.camera_analogue_gain >= 1.0 && sources.camera_analogue_gain <= 16.0)))
        throw std::invalid_argument("sources.camera_analogue_gain phai = 0 (tu dong) hoac trong [1, 16]");
    c.exposure_us = sources.camera_exposure_us;
    c.analogue_gain = sources.camera_analogue_gain;
    return c;
}

Config config_from_raw(const YAML::Node& raw) {
    const YAML::Node s3 = req(raw, "s3_select");
    Config c;
    c.D_m = get<double>(s3, "D_m");
    c.match_radius_px = get<double>(s3, "match_radius_px");
    if (s3["bbox_max_mm"]) c.bbox_max_mm = s3["bbox_max_mm"].as<double>();
    if (s3["match_radius_mm"]) c.match_radius_mm = s3["match_radius_mm"].as<double>();
    if (s3["world_track"]) c.world_track = s3["world_track"].as<bool>();
    if (s3["same_object_mm"]) c.same_object_mm = s3["same_object_mm"].as<double>();
    if (!(c.same_object_mm > 0.0)) throw std::runtime_error("s3_select.same_object_mm phai > 0");
    if (!(c.match_radius_mm > 0.0)) throw std::runtime_error("s3_select.match_radius_mm phai > 0");
    if (c.bbox_max_mm < 0.0) throw std::runtime_error("s3_select.bbox_max_mm phai >= 0");
    c.decide = {get<int>(s3, "n_min"), get<double>(s3, "w_max_mm"), get<double>(s3, "conf_min"),
                get<double>(s3, "mask_min_px"), get<int>(s3, "mask_small_max_frames")};
    // SAU khi gan c.decide = {...} (gan lai ca struct se dua size_abort ve mac dinh true)
    if (s3["size_abort"]) c.decide.size_abort = s3["size_abort"].as<bool>();
    c.track = {get<double>(s3, "velocity_alpha")};
    c.cpu.cores = req(req(raw, "cpu"), "cores").as<std::vector<int>>();
    c.max_frames = get<int>(raw, "max_frames");
    c.runs_root = get<std::string>(raw, "runs_root");
    c.camera = load_camera(get<std::string>(raw, "calib_K"));
    c.detector = detector_config(req(raw, "s1_detect"));
    c.segmenter = segmenter_config(req(raw, "s2_segment"));
    c.fusion = fusion_config(req(raw, "s4_fusion"));
    return c;
}

SourcesConfig sources_config(const YAML::Node& r) {
    SourcesConfig s;
    s.camera_buffer_count = get<int>(r, "camera_buffer_count");
    s.sensor_bit_depth = get<int>(r, "sensor_bit_depth");
    s.connect_timeout_s = get<double>(r, "connect_timeout_s");
    s.frame_timeout_s = get<double>(r, "frame_timeout_s");
    s.lens_position_tol = get<double>(r, "lens_position_tol");
    s.clock_skew_max_s = get<double>(r, "clock_skew_max_s");
    s.lidar_port = get<std::string>(r, "lidar_port");
    s.lidar_baudrate = get<int>(r, "lidar_baudrate");
    s.lidar_buffer_scans = get<int>(r, "lidar_buffer_scans");
    s.lidar_extra_latency_ms = get<double>(r, "lidar_extra_latency_ms");
    s.scan_timeout_s = get<double>(r, "scan_timeout_s");
    // tuy chon (file cau hinh cu khong co): 0 = tu dong
    if (r["camera_exposure_us"]) s.camera_exposure_us = get<double>(r, "camera_exposure_us");
    if (r["camera_analogue_gain"]) s.camera_analogue_gain = get<double>(r, "camera_analogue_gain");
    return s;
}

ArucoConfig aruco_config(const YAML::Node& r) {
    return {get<std::string>(r, "dictionary"), get<int>(r, "marker_id"), get<double>(r, "side_mm"),
            get_bool(r, "check_aruco")};
}

LaunchMarkerConfig launch_marker_config(const YAML::Node& r) {
    LaunchMarkerConfig m{get_bool(r, "enabled"), get<std::string>(r, "dictionary"), get<int>(r, "marker_id"),
                         get<double>(r, "side_mm"), get<double>(r, "period_s"), get<double>(r, "max_reproj_px"),
                         get<double>(r, "max_tilt_deg"), get_bool(r, "track_object")};
    if (r["track_during_scan"]) m.track_during_scan = get_bool(r, "track_during_scan");
    if (r["exclude_radius_mm"]) m.exclude_radius_mm = get<double>(r, "exclude_radius_mm");
    if (!(m.exclude_radius_mm >= 0.0)) throw std::runtime_error("launch_marker.exclude_radius_mm phai >= 0");
    if (m.side_mm <= 0.0 || m.period_s <= 0.0 || m.max_reproj_px <= 0.0 || m.max_tilt_deg <= 0.0)
        throw std::runtime_error("launch_marker: side_mm, period_s, max_reproj_px, max_tilt_deg phai > 0");
    return m;
}

VideoDiagConfig video_diag_config(const YAML::Node& r) {
    VideoDiagConfig c;
    c.enabled = get_bool(r, "enabled");
    c.every_n = get<int>(r, "every_n");
    c.crop_px = get<int>(r, "crop_px");
    c.motion_width_px = get<int>(r, "motion_width_px");
    c.gyro_half_window_s = get<double>(r, "gyro_half_window_s");
    c.flush_period_s = get<double>(r, "flush_period_s");
    c.log_period_s = get<double>(r, "log_period_s");
    if (c.every_n < 1 || c.crop_px < 32 || c.motion_width_px < 64 || c.gyro_half_window_s <= 0.0 ||
        c.flush_period_s <= 0.0 || c.log_period_s <= 0.0)
        throw std::runtime_error("video_diag: every_n >= 1, crop_px >= 32, motion_width_px >= 64, cac chu ky > 0");
    return c;
}

DisplayConfig display_config(const YAML::Node& r) {
    return {get<int>(r, "window_width_px"), get<int>(r, "window_height_px"), get_bool(r, "show")};
}

LockConfig lock_config(const YAML::Node& r) {
    LockConfig c;
    c.calib_T_BC = get<std::string>(r, "calib_T_BC");
    c.search_scale = get<double>(r, "search_scale");
    c.flow_max_corners = get<int>(r, "flow_max_corners");
    c.flow_quality = get<double>(r, "flow_quality");
    c.flow_min_distance_px = get<double>(r, "flow_min_distance_px");
    c.flow_win_px = get<int>(r, "flow_win_px");
    c.flow_levels = get<int>(r, "flow_levels");
    c.flow_fb_max_px = get<double>(r, "flow_fb_max_px");
    c.flow_min_points = get<int>(r, "flow_min_points");
    c.affine_ransac_px = get<double>(r, "affine_ransac_px");
    c.ncc_min = get<double>(r, "ncc_min");
    c.kf_accel_px_s2 = get<double>(r, "kf_accel_px_s2");
    c.kf_init_vel_px_s = get<double>(r, "kf_init_vel_px_s");
    c.kf_meas_px = get<double>(r, "kf_meas_px");
    c.max_duration_s = get<double>(r, "max_duration_s");
    c.min_segments = get<int>(r, "min_segments");
    c.stop_when_converged = get_bool(r, "stop_when_converged");
    c.warped_masks_for_lidar = get_bool(r, "warped_masks_for_lidar");
    c.align_gate = get_bool(r, "align_gate");
    if (r["wait_scan_command"]) c.wait_scan_command = get_bool(r, "wait_scan_command");
    if (r["scan_max_duration_s"]) c.scan_max_duration_s = r["scan_max_duration_s"].as<double>();
    if (c.scan_max_duration_s < 0.0) throw std::invalid_argument("lock_track.scan_max_duration_s: phai >= 0");
    c.align_e_max_mm = get<double>(r, "align_e_max_mm");
    c.align_max_gap_px = get<double>(r, "align_max_gap_px");
    c.align_log_period_s = get<double>(r, "align_log_period_s");
    if (!(c.align_e_max_mm > 0.0) || !(c.align_max_gap_px > 0.0) || !(c.align_log_period_s > 0.0))
        throw std::invalid_argument("lock_track.align_*: phai > 0");
    return c;
}

StartConfig start_config(const YAML::Node& r) {
    StartConfig c;
    c.require_hover = get_bool(r, "require_hover");
    c.accept_viewer_key = get_bool(r, "accept_viewer_key");
    c.modes = req(r, "modes").as<std::vector<uint32_t>>();
    c.max_speed_m_s = get<double>(r, "max_speed_m_s");
    c.max_vspeed_m_s = get<double>(r, "max_vspeed_m_s");
    c.max_height_change_m = get<double>(r, "max_height_change_m");
    c.hover_window_s = get<double>(r, "hover_window_s");
    c.min_height_m = get<double>(r, "min_height_m");
    c.max_msg_age_s = get<double>(r, "max_msg_age_s");
    c.status_period_s = get<double>(r, "status_period_s");
    return c;
}

DistanceConfig distance_config(const YAML::Node& r) {
    DistanceConfig c;
    c.source = get<std::string>(r, "source");
    if (c.source != "fixed" && c.source != "fc") throw std::runtime_error("distance.source = " + c.source + ": fixed | fc");
    c.fixed_m = get<double>(r, "fixed_m");
    c.camera_offset_m = get<double>(r, "camera_offset_m");
    return c;
}

MavlinkConfig mavlink_config(const YAML::Node& r) {
    MavlinkConfig c;
    c.enabled = get_bool(r, "enabled");
    c.source = get<std::string>(r, "source");
    if (c.source != "shm" && c.source != "serial") throw std::runtime_error("mavlink.source = " + c.source + ": shm | serial");
    c.shm_name = get<std::string>(r, "shm_name");
    c.shm_poll_ms = get<int>(r, "shm_poll_ms");
    c.companion_sysid = get<int>(r, "companion_sysid");
    c.companion_compid = get<int>(r, "companion_compid");
    c.command_id = get<int>(r, "command_id");
    c.link = get<std::string>(r, "link");
    if (c.link != "serial" && c.link != "udp") throw std::runtime_error("mavlink.link = " + c.link + ": serial | udp");
    c.udp_address = get<std::string>(r, "udp_address");
    c.udp_port = get<int>(r, "udp_port");
    c.port = get<std::string>(r, "port");
    c.baudrate = get<int>(r, "baudrate");
    c.latency_ms = get<double>(r, "latency_ms");
    c.max_age_s = get<double>(r, "max_age_s");
    c.buffer_samples = get<int>(r, "buffer_samples");
    c.use_local_position = get_bool(r, "use_local_position");
    c.use_rangefinder = get_bool(r, "use_rangefinder");
    return c;
}

MonitorConfig monitor_config(const YAML::Node& r) { return {get<double>(r, "period_s"), get_bool(r, "enabled")}; }

}  // namespace uavp
