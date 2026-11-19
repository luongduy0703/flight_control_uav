// Doc config. Cau truc configs/ (duong dan tinh tu thu muc goc uav_perception_pi/):
//   run/<real_time|offline>.yaml  config chay: max_frames, calib, runs_root, `stages` (+ sources, display, ...)
//   stages/s1..s4_*.yaml          tham so tung stage; stage co model tro toi file model bang `model_cfg`
//   models/s*_ncnn.yaml           model NCNN: file, classes, pixel input (input_px), tien xu ly, so luong
//   calib/                        K (camera), T_CL (LiDAR -> camera)
// File model duoc nap vao muc `model` cua stage (ghi de: --set "s1_detect.model.input_px=[960,544]").
// Thieu khoa -> exception luc khoi dong (khong co gia tri mac dinh trong code).
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

namespace uavp {

// Model NCNN (configs/models/*.yaml): file, so luong, fp16 storage. Phep tinh fp16 luon tat (Pi 4 / A72 khong co).
struct ModelConfig {
    std::filesystem::path dir;
    std::string param_file;
    std::string bin_file;
    int num_threads{4};
    bool use_vulkan{false};
    bool use_fp16_storage{false};
};

enum class DetectMode : uint8_t { kTile, kFull, kCount };

// s1: model (configs/models/s1_detect_ncnn.yaml: classes, input_px, pad_value, interpolation) + stage (full / tile,
// nguong). input_px trong YAML: [w, h] (chu nhat, vd [960, 544]) hoac mot so (vuong).
struct DetectorConfig {
    ModelConfig model;
    std::vector<std::string> classes;  // class id = vi tri (file model)
    DetectMode mode{DetectMode::kFull};
    int input_w_px{960};               // pixel input model (file model), boi so cua stride 32
    int input_h_px{544};
    int tile_px{640};
    double overlap{0.2};
    float conf_thr{0.6f};
    double nms_iou{0.7};
    double cross_window_ios{0.8};
    int max_det{100};
    int pad_value{114};    // file model
    int interpolation{1};  // cv::INTER_LINEAR / cv::INTER_AREA (file model)
};

// s2: model (configs/models/s2_segment_ncnn.yaml: classes, input_px, interpolation) + stage (ROI, nguong).
struct SegmenterConfig {
    ModelConfig model;
    std::vector<std::string> classes;
    int input_px{320};     // pixel input model, vuong: ROI resize ve input_px x input_px
    int interpolation{1};
    // ROI cat tren anh letterbox roi_image_w x h (vd 960 x 544, giong s1 va luc train); 0 = cat thang tren anh goc.
    int roi_image_w_px{0};
    int roi_image_h_px{0};
    int roi_pad_value{114};  // mau dem cua anh letterbox (= pad_value cua s1)
    double roi_scale{1.5};
    float conf_thr{0.25f};
    double nms_iou{0.7};
    int max_det{300};
    float mask_logit_thr{0.f};
    double min_select_ioa{0.5};
};

struct DecideConfig {
    int n_min{5};
    double w_max_mm{80.0};
    double conf_min{0.7};
    double mask_min_px{5.0};
    int mask_small_max_frames{10};
};

// s3 tracking (configs/stages/s3_select.yaml): s1 chi chay khi chua co vat / mat vat; cac khung khac s2 chay tren ROI
// du doan (van toc tam mask, px / khung, lam muot theo velocity_alpha).
struct TrackConfig {
    double velocity_alpha{0.5};
};

// K + he so meo cua anh nhan duoc (OpenCV pinhole + k1 k2 p1 p2 k3).
struct CameraModel {
    cv::Matx33d K_px;
    cv::Mat dist_coeffs;  // 1x5 CV_64F
    [[nodiscard]] cv::Point2d center_px() const noexcept { return {K_px(0, 2), K_px(1, 2)}; }
};

// LiDAR -> camera: p_C_mm = R_CL p_L_mm + t_CL_mm; quy uoc goc / tam LD06 luc hieu chuan.
struct Extrinsics {
    cv::Matx33d R_CL;
    cv::Vec3d t_CL_mm;
    double angle_offset_deg{0.0};
    double range_scale{1.0};
    double range_offset_mm{0.0};
    bool angle_clockwise{true};
};

// s4 (configs/stages/s4_fusion.yaml): LiDAR quanh tung khung co mask -> RANSAC mat phang top (bu chuyen dong UAV).
struct FusionConfig {
    Extrinsics extrinsics;
    int min_pts{40};
    double lidar_window_s{0.1};         // diem LiDAR cach khung co mask gan nhat <= 0.1 s moi dung
    int plane_iters{200};
    double ransac_tau_mm{10.0};
    uint64_t ransac_seed{0};
    double max_tilt_deg{30.0};          // gia thuyet mat phang nghieng hon -> bo (mat ben, san lan)
    double min_spread_mm{15.0};         // trai diem vuong goc mat quet < 15 mm -> khong do duoc nghieng theo huong do
    double drift_min_span_s{0.3};       // diem trai < 0.3 s -> khong uoc luong troi do cao
    double max_motion_time_corr{0.9};   // |corr(w, t)| > 0.9: nghieng va troi lan nhau -> bo nghieng theo w
    double min_inlier_ratio{0.6};
    double ambiguity_ratio{0.7};
    int erode_px{2};
    double intensity_min{100.0};
    double r_min_mm{20.0};
    double r_max_mm{12000.0};
    double scan_angle_min_deg{-90.0};
    double scan_angle_max_deg{90.0};
};

// Muc sensor_mode cua file K: camera phai mo dung nhu luc hieu chuan.
struct CaptureConfig {
    int width_px{1920};
    int height_px{1080};
    int sensor_width_px{2304};
    int sensor_height_px{1296};
    double lens_position{1.25};
    int rotate_deg{180};
    double fps{30.0};
};

struct SourcesConfig {
    std::string lidar_port;
    int camera_buffer_count{4};
    int sensor_bit_depth{10};  // che do cam bien (IMX708: 10 bit)
    double connect_timeout_s{5.0};
    double frame_timeout_s{5.0};
    double lens_position_tol{0.02};
    double clock_skew_max_s{0.5};
    int lidar_baudrate{230400};
    int lidar_buffer_scans{600};
    double lidar_extra_latency_ms{0.0};
    double scan_timeout_s{1.0};
};

struct ArucoConfig {
    std::string dictionary;
    int marker_id{0};
    double side_mm{36.4};
    bool check_aruco{false};
};

struct DisplayConfig {
    int window_width_px{1600};
    int window_height_px{900};
    bool show{false};
};

struct MonitorConfig {
    double period_s{0.5};
    bool enabled{true};
};

// Khoa + bam vat sau s3 (configs/stages/lock_track.yaml): du doan ROI tu tu the FC, sua bang optical flow / NCC,
// Kalman tam ROI; segment theo nhip LiDAR.
struct LockConfig {
    std::filesystem::path calib_T_BC;
    double search_scale{2.0};        // cua so tim (flow, NCC) = 2 x canh ROI, quanh tam du doan
    int flow_max_corners{40};
    double flow_quality{0.01};
    double flow_min_distance_px{4.0};  // anh xam thu 1/2
    int flow_win_px{15};
    int flow_levels{2};
    double flow_fb_max_px{1.0};      // loi forward-backward > 1 px (anh 1/2) -> bo diem
    int flow_min_points{6};          // >= 6 diem tot: do + affine doi mask; it hon -> NCC
    double affine_ransac_px{2.0};
    double ncc_min{0.6};             // diem NCC toi thieu de nhan ket qua template matching
    double kf_accel_px_s2{300.0};
    double kf_init_vel_px_s{50.0};
    double kf_meas_px{3.0};
    double max_duration_s{3.0};      // bam toi da 3 s sau khi khoa
    int min_segments{5};             // it nhat 5 lan segment (theo LiDAR) truoc khi dung som
    bool stop_when_converged{true};  // s4 da ok + do duoc nghieng -> dung
    bool warped_masks_for_lidar{false};  // loc LiDAR ca bang mask doi (giua 2 lan segment)
};

// Pixhawk qua MAVLink (run/real_time.yaml: sources.mavlink).
struct MavlinkConfig {
    // uavp_fc_bridge nhan MAVLink tu dau: serial = mo thang `port` (UART TELEM2) | udp = qua mavlink-routerd (router
    // giu UART, chuyen tiep toi udp_address:udp_port; bridge gui HEARTBEAT / COMMAND_ACK nguoc lai router).
    std::string link{"serial"};
    std::string udp_address{"127.0.0.1"};
    int udp_port{14551};
    std::string port;
    int baudrate{921600};
    double latency_ms{0.0};      // tre them (xu ly FC + USB / UART) ngoai tre nho nhat da tru
    double max_age_s{0.2};       // mau gan nhat cach thoi diem can > 0.2 s -> coi nhu khong co
    int buffer_samples{1024};    // so mau ATTITUDE giu lai (~20 s o 50 Hz), luy thua 2
    std::string source{"shm"};      // shm: doc RAM do uavp_fc_bridge ghi | serial: mo cong truc tiep
    std::string shm_name{"/uavp_fc"};
    int shm_poll_ms{5};
    int companion_sysid{1};     // Pi = may tinh dong hanh: cung sysid voi Pixhawk
    int companion_compid{191};  // MAV_COMP_ID_ONBOARD_COMPUTER
    int command_id{31010};      // MAV_CMD_USER_1: may tinh gui COMMAND_LONG param1 = 1 bat dau, 0 dung
    bool enabled{false};
    bool use_local_position{true};
    bool use_rangefinder{true};
};

// Dieu kien bat dau cua uavp_real_time (run/real_time.yaml, muc start): may tinh gui lenh bat dau qua Pixhawk
// (COMMAND_LONG, mavlink.command_id) VA Pi tu nhan UAV dang hover (da arm, dung che do bay, dang bay, van toc ~0,
// do cao on dinh trong hover_window_s). Lenh dung (param1 = 0) giua chung -> dung.
struct StartConfig {
    bool require_hover{true};         // false: CHI can lenh may tinh (gcs_command.py start / SPACE viewer), bo dieu kien hover
    bool accept_viewer_key{false};    // SPACE / s tren remote_viewer cung la lenh bat dau (ngoai lenh qua Pixhawk)
    std::vector<uint32_t> modes;      // custom_mode duoc phep (ArduCopter: 5 LOITER, 16 POSHOLD, 4 GUIDED); [] = moi che do
    double max_speed_m_s{0.15};       // van toc ngang
    double max_vspeed_m_s{0.10};      // van toc doc
    double max_height_change_m{0.05}; // do cao max - min trong cua so
    double hover_window_s{1.0};
    double min_height_m{0.3};
    double max_msg_age_s{1.5};        // tin FC cu hon -> coi nhu khong co
    double status_period_s{2.0};
};

// Khoang cach camera -> san cho s3 (doi kich thuoc mask px -> mm, loai vat qua to cho tay gap).
//   fixed: fixed_m (uavp_real_time_perception: 1 m) | fc: do cao optical flow (rangefinder) luc bat dau x cos(roll) cos(pitch)
//   - camera_offset_m. Khoang cach THAT toi mat top van do o s4 (LiDAR).
struct DistanceConfig {
    std::string source{"fixed"};
    double fixed_m{1.0};
    double camera_offset_m{0.0};  // camera cao hon (+) / thap hon (-) cam bien do cao, theo truc thang dung
};

// CPU (run/*.yaml, muc cpu): ca chuong trinh chi chay tren cac nhan nay (vd [2, 3]); [] = khong gioi han.
struct CpuConfig {
    std::vector<int> cores;
};

// Moi tham so cua mot phien.
struct Config {
    DetectorConfig detector;
    SegmenterConfig segmenter;
    FusionConfig fusion;
    DecideConfig decide;
    TrackConfig track;
    CpuConfig cpu;
    CameraModel camera;
    std::filesystem::path runs_root;
    double D_m{0.66};
    double match_radius_px{40.0};
    int max_frames{100};
};

// Config phien (YAML) + noi dung file cua tung stage (raw[<ten stage>]) roi ghi de `--set khoa.con=gia_tri`.
// File co `_base_: <file>` (duong dan tinh tu thu muc cua file): nap file goc roi ghi de tung khoa (gop map long nhau).
[[nodiscard]] YAML::Node load_raw(const std::filesystem::path& path, const std::vector<std::string>& overrides);
[[nodiscard]] Config config_from_raw(const YAML::Node& raw);
[[nodiscard]] CameraModel load_camera(const std::filesystem::path& path);
[[nodiscard]] Extrinsics load_extrinsics(const std::filesystem::path& path);
[[nodiscard]] CaptureConfig load_capture(const std::filesystem::path& calib_K_path);
[[nodiscard]] SourcesConfig sources_config(const YAML::Node& r);
[[nodiscard]] ArucoConfig aruco_config(const YAML::Node& r);
[[nodiscard]] DisplayConfig display_config(const YAML::Node& r);
[[nodiscard]] MonitorConfig monitor_config(const YAML::Node& r);
[[nodiscard]] ModelConfig model_config(const YAML::Node& r);
[[nodiscard]] LockConfig lock_config(const YAML::Node& r);
[[nodiscard]] MavlinkConfig mavlink_config(const YAML::Node& r);
[[nodiscard]] StartConfig start_config(const YAML::Node& r);
[[nodiscard]] DistanceConfig distance_config(const YAML::Node& r);

}  // namespace uavp
