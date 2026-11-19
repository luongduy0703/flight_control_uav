// Kieu du lieu cua pipeline (giong src/uavp/types.py). Toa do anh goc theo quy uoc OpenCV, anh uint8 HWC RGB.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/common.hpp"

namespace uavp {

enum class State : uint8_t { kDefer, kOk, kAbort, kCount };
enum class Reason : uint8_t { kNone, kTooBig, kLowConf, kMaskSmall, kCount };
enum class FusionStatus : uint8_t { kOk, kLowConfidence, kInsufficient, kCount };

// Ten trong log / result.json, giong ban Python (bang tra theo chi so enum, CPP_RULES.md muc 7).
inline constexpr std::array<std::string_view, static_cast<std::size_t>(State::kCount)> kStateNames{"defer", "ok",
                                                                                                  "abort"};
inline constexpr std::array<std::string_view, static_cast<std::size_t>(Reason::kCount)> kReasonNames{
    "", "too_big", "low_conf", "mask_small"};
inline constexpr std::array<std::string_view, static_cast<std::size_t>(FusionStatus::kCount)> kFusionNames{
    "ok", "low_confidence", "insufficient"};

constexpr std::string_view name(State s) noexcept { return kStateNames[static_cast<std::size_t>(s)]; }
constexpr std::string_view name(Reason r) noexcept { return kReasonNames[static_cast<std::size_t>(r)]; }
constexpr std::string_view name(FusionStatus f) noexcept { return kFusionNames[static_cast<std::size_t>(f)]; }

// Mot vat detector tim thay (anh goc, px). 24 byte: 2–3 detection mot dong cache.
struct Detection {
    float x1{0.f}, y1{0.f}, x2{0.f}, y2{0.f};
    float conf{0.f};
    uint8_t class_id{0};
    std::array<uint8_t, 3> pad_{};

    [[nodiscard]] cv::Point2d center_px() const noexcept {
        return {(static_cast<double>(x1) + x2) / 2.0, (static_cast<double>(y1) + y2) / 2.0};
    }
};
static_assert(sizeof(Detection) == 24, "Detection layout changed");

// Mask mat tren o do phan giai anh goc, trong ROI: pixel (r, c) cua mask = pixel (y0 + r, x0 + c) cua anh.
struct SegMask {
    cv::Mat mask;  // CV_8U, 0 / 1, (side, side)
    int x0_px{0};
    int y0_px{0};
    float conf{0.f};  // conf instance segment
    uint8_t class_id{0};
};

// Dau ra s2 cho mot vat (va dau ra s3 cho vat selected): tam, yaw, mask (giu nguyen pixel).
struct SegObject {
    cv::Point2d center_px;  // trong tam mask, anh goc
    double yaw_deg{0.0};    // huong canh dai minAreaRect cua mask tren anh (px), [0, 180)
    double area_px{0.0};    // so pixel mask
    SegMask mask;
};

// Mot lan do cua mot track.
struct Observation {
    std::optional<SegObject> seg;  // dau ra s2 o khung nay; khong co mask -> nullopt
    double W_mm{kNaN};             // canh ngan minAreaRect tren mat san
    double L_mm{kNaN};
    double area_px{0.0};           // 0 neu khong co mask
    int64_t t_ns{0};               // thoi diem khung (dong ho chung, = SensorTimestamp): s4 gop LiDAR quanh day
    float conf{0.f};               // conf detection (lan s1 gan nhat cua track)
    uint32_t frame_idx{0};
};

// Mot vat dang theo doi trong phien. Bbox `detection` dung o khung `box_frame`; o khung k bbox du doan = bbox dich
// vel_px * (k - box_frame) (UAV hover van troi -> vat dich deu tren anh trong vai khung).
struct Track {
    std::vector<Observation> observations;  // chi lan do hop le
    Detection detection;                    // bbox (anh goc) o khung box_frame: tu s1, hoac dich theo tam mask
    cv::Point2d center_px;                  // tam `detection`
    cv::Point2d vel_px{0.0, 0.0};           // van toc tam mask, px / khung (lam muot)
    cv::Point2d last_mask_center_px;        // tam mask lan s2 thanh cong gan nhat
    uint32_t box_frame{0};
    uint32_t last_mask_frame{0};
    uint32_t last_seen_frame{0};            // khung gan nhat track duoc xet (s1 ghep hoac s2 tren ROI du doan)
    int track_id{0};
    int mask_small_streak{0};               // so lan do lien tiep khong co mask / mask qua nho
    uint8_t class_id{0};
    State state{State::kDefer};
    Reason reason{Reason::kNone};
    bool has_mask{false};                   // da co it nhat mot mask (co last_mask_center_px)
};

// Mot vong quet LD06 (THO, chua doi goc / hieu chinh tam). Mang song song (SoA), cap phat lai khi dung lai slot.
struct Scan {
    std::vector<int64_t> t_ns;       // thoi diem do tung diem (dong ho chung)
    std::vector<float> angle_deg;    // goc tho LD06
    std::vector<uint16_t> range_mm;  // tam tho (0 = khong phan xa)
    std::vector<uint8_t> intensity;

    [[nodiscard]] std::size_t size() const noexcept { return range_mm.size(); }
    void clear() noexcept {
        t_ns.clear();
        angle_deg.clear();
        range_mm.clear();
        intensity.clear();
    }
};

// Tu the camera trong he the gioi W (NED cua EKF Pixhawk, mm): p_W = R_WC p_C + t_WC_mm.
struct CamPose {
    cv::Matx33d R_WC{cv::Matx33d::eye()};
    cv::Vec3d t_WC_mm{0.0, 0.0, 0.0};
    bool has_position{false};  // false: chi co goc quay (t_WC = 0, khong biet UAV dich chuyen)
};

// Mot khung co mask cua vat selected: s4 gop LiDAR quanh t_ns. cv::Mat cua mask chia du lieu (khong chep pixel).
struct MaskFrame {
    SegObject obj;
    std::optional<CamPose> pose;  // tu the camera tu FC luc t_ns (nullopt: khong co MAVLink)
    int64_t t_ns{0};
    bool segmented{true};         // false: mask cu da doi bang flow / homography (khong chay s2 o khung nay)
};

// Dau ra s4, he camera (OpenCV: x phai, y xuong, z theo truc quang) cua KHUNG THAM CHIEU (khung co mask moi nhat), mm.
// Toa do vat: goc = tam mat top p_obj_mm; truc z = phap tuyen mat top (huong ra xa camera, z_C > 0), truc x = canh
// dai (yaw mask) nam tren mat top, truc y = z x x.
struct FusionResult {
    std::vector<cv::Point3f> points_C_mm;    // diem trong mask (he camera luc do diem)
    std::vector<cv::Point3f> points_ref_mm;  // da bu dich chuyen ngang ve khung tham chieu
    std::vector<int64_t> points_t_ns;
    std::vector<uint16_t> points_frame;      // chi so khung co mask cua diem (trong danh sach frames dua vao)
    std::vector<uint16_t> points_scan;       // chi so vong quet
    std::vector<uint8_t> inliers;            // 0 / 1, inlier RANSAC mat phang
    cv::Point3d p_obj_mm{kNaN, kNaN, kNaN};  // tam mat top (Z = d)
    cv::Vec3d axis_x{kNaN, kNaN, kNaN};
    cv::Vec3d axis_y{kNaN, kNaN, kNaN};
    cv::Vec3d axis_z{kNaN, kNaN, kNaN};      // phap tuyen mat top
    double d_mm{kNaN};
    double dist_mm{kNaN};
    double W_mm{kNaN};
    double L_mm{kNaN};
    double yaw_deg{kNaN};                    // tren mat top (W / L), giong truoc
    double tilt_deg{kNaN};                   // goc phap tuyen - truc quang
    double inlier_ratio{kNaN};
    double rms_mm{kNaN};                     // sai so inlier so voi mat phang
    double d2_mm{kNaN};                      // mat phang song song thu 2 (tai tam), nan neu khong co
    double spread_w_mm{kNaN};                // trai diem vuong goc mat quet (p90 - p10) sau bu chuyen dong
    double drift_mm_s{kNaN};                 // troi do cao UAV uoc luong (mm/s), nan neu khong dung
    int64_t t_image_ns{0};                   // thoi diem khung tham chieu
    int64_t t_lidar_first_ns{0};             // khong co diem -> 0
    int64_t t_lidar_last_ns{0};
    int n_frames{0};                         // so khung co mask dua vao
    int n_scans{0};                          // so vong quet co diem trong mask
    int n_points{0};
    int n_inliers{0};
    int n2_inliers{0};
    FusionStatus status{FusionStatus::kInsufficient};
    bool ambiguous{false};
    bool normal_observable{false};           // do duoc nghieng ca 2 huong (UAV troi du vuong goc mat quet)
    bool pose_compensated{false};            // bu chuyen dong bang tu the FC (khong: bang dich chuyen tam vat tren anh)
};

}  // namespace uavp
