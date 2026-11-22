// s4 fusion / fusion: LiDAR quanh cac khung co mask -> bu chuyen dong ve khung tham chieu -> RANSAC mat phang top
// -> tam mat top, 3 truc, d, dist, W, L, yaw (he camera khung tham chieu, mm).
//   1. Diem LiDAR (thoi diem tung diem) gan khung co mask nao nhat; cach > lidar_window_s -> bo. Kiem diem trong mask
//      cua khung do, mask dich theo tam vat noi suy giua 2 khung (UAV troi giua luc chup va luc quet).
//   2. Bu chuyen dong ve khung tham chieu (khung co mask moi nhat):
//      - moi khung co tu the FC: q = T_CW(t_ref) T_WC(t_f) p   (ca quay lan dich chuyen)
//      - khong: chi dich ngang theo tam vat tren anh: q_xy = p_xy + d0 (tia_ref - tia(t)), troi do cao -> hang e.
//   3. RANSAC mat phang (plane.hpp), d = median theo vong roi median giua cac vong.
// method world (cfg.method, mac dinh): deskew tung diem theo tu the FC + tich luy trong he the gioi + RANSAC 3 diem / PCA
// (world_plane.hpp). Can `poses` co vi tri phu khung co mask moi nhat; khong co -> dung cach image o tren.
#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"
#include "uavp/stages/s4_fusion/lidar.hpp"
#include "uavp/stages/s4_fusion/plane.hpp"
#include "uavp/stages/s4_fusion/world_plane.hpp"

namespace uavp {

// --- Vi tri, kich thuoc -----------------------------------------------------------------------------------------
// p_obj = d (x_n, y_n, 1) voi (x_n, y_n) = undistort(tam).
[[nodiscard]] cv::Point3d object_position_mm(cv::Point2d center_px, double d_mm, const CameraModel& camera);
struct TopSize {
    double W_mm{kNaN}, L_mm{kNaN}, yaw_deg{kNaN};
};
[[nodiscard]] TopSize top_size(const SegMask& seg, double d_mm, const CameraModel& camera);

class Fusion {
public:
    Fusion(FusionConfig cfg, CameraModel camera);
    // frames: cac khung co mask cua vat (thu tu bat ky); scans: vong quet LD06 phu cac khung do.
    // poses: mau tu the camera tang dan theo thoi gian (method world), rong -> cach image.
    [[nodiscard]] FusionResult operator()(std::span<const MaskFrame> frames, std::span<const Scan> scans,
                                          std::span<const CamPoseSample> poses = {});

private:
    struct FrameInfo {
        cv::Mat mask;           // da co erode_px
        cv::Point2d center_px;  // tam mask
        cv::Point2d ray;        // tia chuan hoa (khu meo) qua tam
        double roi_r{0.0};      // ban kinh chuan hoa lon nhat cua ROI (bo diem bi meo gap)
        int64_t t_ns{0};
        std::size_t index{0};   // vi tri trong `frames`
    };
    void prepare(std::span<const MaskFrame> frames);
    [[nodiscard]] std::size_t nearest_frame(int64_t t_ns) const noexcept;
    [[nodiscard]] cv::Point2d interp(int64_t t_ns, bool ray) const noexcept;
    void collect(std::span<const MaskFrame> frames, std::span<const Scan> scans, FusionResult& r);
    void compensate(std::span<const MaskFrame> frames, FusionResult& r);
    void fit(std::span<const MaskFrame> frames, FusionResult& r);
    void set_pose(const SegObject& ref, const PlaneFit& fit, double c_mm, FusionResult& r) const;
    // method world. false: thieu tu the co vi tri cho khung tham chieu -> goi lai cach image.
    [[nodiscard]] bool fuse_world(std::span<const MaskFrame> frames, std::span<const Scan> scans,
                                  std::span<const CamPoseSample> poses, FusionResult& r);
    void collect_world(std::span<const MaskFrame> frames, std::span<const Scan> scans,
                       std::span<const CamPoseSample> poses, FusionResult& r);
    void solve_world(std::span<const MaskFrame> frames, FusionResult& r);

    FusionConfig cfg_;
    CameraModel camera_;
    std::mt19937_64 rng_;
    cv::Point2d u_dir_, w_dir_;  // truc u (doc mat quet), w (vuong goc) trong mat anh x-y
    // buffer dung lai
    std::vector<FrameInfo> infos_;
    std::vector<cv::Point3f> scan_points_;
    std::vector<int64_t> scan_t_;
    std::vector<cv::Point2f> uv_;
    PlaneData data_;
    PlaneFit fit_;
    // method world
    std::vector<CamPose> frame_pose_;   // tu the camera cua infos_[k]
    std::vector<uint8_t> frame_ok_;     // infos_[k] co tu the co vi tri
    std::vector<cv::Vec3d> points_W_;   // diem da deskew, he the gioi (mm), song song r.points_C_mm
    PlanePca pca_;
};

}  // namespace uavp
