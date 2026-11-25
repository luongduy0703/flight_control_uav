// Bam vat da khoa (sau s3 + s4 lan dau), chay moi khung camera (30 Hz):
//   1. Du doan tu tu the FC: co vi tri (LOCAL_POSITION_NED / rangefinder) -> p_C(t) = T_CW(t) p_W chieu len anh;
//      chi co goc quay -> homography H = K R(t <- t-1) K^-1 doi tam ROI va mask cu. Dich chuyen du doan = dau vao
//      dieu khien cua Kalman.
//   2. Do trong cua so quanh ROI: optical flow (affine doi mask) hoac NCC (RoiFlow).
//   3. Kalman (u, v, vu, vv) gop du doan + do -> tam ROI; bbox ROI (kich thuoc luc khoa) cho s2.
// Segment (s2) KHONG chay moi khung: app goi on_segment() khi co vong quet LiDAR moi (10 Hz); giua 2 lan segment,
// mask cu duoc doi bang flow / homography.
#pragma once

#include <array>
#include <optional>
#include <string_view>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"
#include "uavp/tracking/roi_flow.hpp"
#include "uavp/tracking/roi_kalman.hpp"
#include "uavp/tracking/world_pose.hpp"

namespace uavp {

enum class PoseMode : uint8_t { kNone, kRotation, kWorld, kCount };
inline constexpr std::array<std::string_view, static_cast<std::size_t>(PoseMode::kCount)> kPoseModeNames{
    "none", "rotation", "world"};
constexpr std::string_view name(PoseMode m) noexcept { return kPoseModeNames[static_cast<std::size_t>(m)]; }

// Doi mask (anh goc) theo M (anh goc khung truoc -> khung nay). Mask giu kich thuoc, goc (x0, y0) dich theo tam.
// `buffer`: Mat dung lai. false neu mask doi ra rong (ra ngoai anh).
[[nodiscard]] bool warp_mask(SegMask& mask, const cv::Matx33d& M, cv::Mat& buffer);

struct LockStep {
    cv::Point2d center_px;                    // tam ROI sau Kalman
    cv::Point2d pose_pred_px{kNaN, kNaN};     // tam du doan tu tu the FC (world: chieu p_W; rotation: H)
    cv::Point3d p_C_pose_mm{kNaN, kNaN, kNaN};  // p_C(t) = T_CW(t) p_W (che do world)
    FlowMeasure meas;
    double kf_sigma_px{0.0};
    PoseMode mode{PoseMode::kNone};
};

class LockedTracker {
public:
    LockedTracker(const LockConfig& cfg, const CameraModel& camera, const BodyCamera& body_camera);
    LockedTracker(const LockedTracker&) = delete;
    LockedTracker& operator=(const LockedTracker&) = delete;

    // Khoa vat: mask + tam (khung t0), bbox (kich thuoc ROI cho s2), tam 3D he camera (s4), tu the camera t0.
    void lock(const cv::Mat& image_rgb, const SegObject& obj, const Detection& box, const cv::Point3d& p_C_mm,
              int64_t t_ns, const std::optional<CamPose>& pose);
    // Mot khung camera.
    LockStep step(const cv::Mat& image_rgb, int64_t t_ns, const std::optional<CamPose>& pose);
    // s2 vua chay o khung hien tai (co LiDAR moi): mask moi -> dat lai flow + cap nhat Kalman.
    void on_segment(const cv::Mat& image_rgb, const SegObject& obj);

    [[nodiscard]] Detection roi_box() const noexcept;
    [[nodiscard]] const SegObject& object() const noexcept { return obj_; }  // mask moi hoac da doi
    [[nodiscard]] bool has_world() const noexcept { return has_world_; }
    [[nodiscard]] cv::Point3d p_W_mm() const noexcept { return p_W_mm_; }

private:
    [[nodiscard]] PoseMode pose_control(const std::optional<CamPose>& pose, cv::Point2d& control_px,
                                        cv::Matx33d& M_pose, LockStep& out) const;
    void move_mask(const cv::Matx33d& M);

    LockConfig cfg_;
    CameraModel camera_;
    BodyCamera body_camera_;
    RoiFlow flow_;
    RoiKalman kf_;
    SegObject obj_;
    Detection box_;              // kich thuoc ROI luc khoa
    cv::Point3d p_W_mm_;
    std::optional<CamPose> last_pose_;
    cv::Point2d last_pose_px_{kNaN, kNaN};
    cv::Mat warp_buf_;
    bool has_world_{false};
};

}  // namespace uavp
