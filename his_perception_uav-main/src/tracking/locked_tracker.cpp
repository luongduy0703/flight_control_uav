#include "uavp/tracking/locked_tracker.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "uavp/stages/s2_segment/mask_shape.hpp"

namespace uavp {
namespace {

cv::Point2d apply(const cv::Matx33d& M, cv::Point2d p) noexcept {
    const cv::Vec3d q = M * cv::Vec3d(p.x, p.y, 1.0);
    return {q[0] / q[2], q[1] / q[2]};
}

cv::Matx33d translation(cv::Point2d d) noexcept { return {1, 0, d.x, 0, 1, d.y, 0, 0, 1}; }

bool finite(cv::Point2d p) noexcept { return std::isfinite(p.x) && std::isfinite(p.y); }

}  // namespace

bool warp_mask(SegMask& mask, const cv::Matx33d& M, cv::Mat& buffer) {
    const cv::Point2d half(mask.mask.cols / 2.0, mask.mask.rows / 2.0);
    const cv::Point2d c_new = apply(M, cv::Point2d(mask.x0_px, mask.y0_px) + half);
    const int x0 = round_half_even(c_new.x - half.x), y0 = round_half_even(c_new.y - half.y);
    // toa do mask cu -> anh -> M -> anh moi -> toa do mask moi
    const cv::Matx33d M_local = translation({-static_cast<double>(x0), -static_cast<double>(y0)}) * M *
                                translation({static_cast<double>(mask.x0_px), static_cast<double>(mask.y0_px)});
    buffer.release();  // bo nho moi: mask cu co the dang duoc MaskFrame cua s4 giu (cv::Mat chia du lieu)
    cv::warpPerspective(mask.mask, buffer, M_local, mask.mask.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
    if (cv::countNonZero(buffer) == 0) return false;
    std::swap(mask.mask, buffer);
    buffer.release();
    mask.x0_px = x0;
    mask.y0_px = y0;
    return true;
}

LockedTracker::LockedTracker(const LockConfig& cfg, const CameraModel& camera, const BodyCamera& body_camera)
    : cfg_(cfg), camera_(camera), body_camera_(body_camera), flow_(cfg), kf_(cfg.kf_accel_px_s2, cfg.kf_init_vel_px_s) {}

void LockedTracker::lock(const cv::Mat& image_rgb, const SegObject& obj, const Detection& box, const cv::Point3d& p_C_mm,
                         int64_t t_ns, const std::optional<CamPose>& pose) {
    obj_ = obj;
    box_ = box;
    const double roi_side = std::max(box.x2 - box.x1, box.y2 - box.y1);
    flow_.reset(image_rgb, obj.mask, obj.center_px, round_half_even(cfg_.search_scale * roi_side));
    kf_.init(obj.center_px, t_ns);
    last_pose_ = pose;
    has_world_ = pose.has_value() && pose->has_position;
    p_W_mm_ = pose ? to_world(*pose, p_C_mm) : cv::Point3d(kNaN, kNaN, kNaN);
    const auto uv = project_point_px(p_C_mm, camera_);  // cung cach tinh voi cac khung sau (khong lech he thong)
    last_pose_px_ = uv ? *uv : obj.center_px;
}

PoseMode LockedTracker::pose_control(const std::optional<CamPose>& pose, cv::Point2d& control_px, cv::Matx33d& M_pose,
                                     LockStep& out) const {
    control_px = {0.0, 0.0};
    M_pose = cv::Matx33d::eye();
    if (!pose || !last_pose_) return PoseMode::kNone;
    if (has_world_ && pose->has_position) {  // p_C(t) = T_CW(t) p_W -> anh
        out.p_C_pose_mm = to_camera(*pose, p_W_mm_);
        const auto uv = project_point_px(out.p_C_pose_mm, camera_);
        if (uv && finite(last_pose_px_)) {
            out.pose_pred_px = *uv;
            control_px = *uv - last_pose_px_;
            M_pose = translation(control_px);
            return PoseMode::kWorld;
        }
    }
    const cv::Matx33d R_new_old = pose->R_WC.t() * last_pose_->R_WC;  // chi goc quay
    const cv::Point2d c = kf_.center_px();
    if (const auto uv = rotate_pixel(c, R_new_old, camera_)) {
        out.pose_pred_px = *uv;
        control_px = *uv - c;
        M_pose = rotation_homography(R_new_old, camera_);
        return PoseMode::kRotation;
    }
    return PoseMode::kNone;
}

void LockedTracker::move_mask(const cv::Matx33d& M) {
    SegMask m = obj_.mask;  // header chia du lieu; warp_mask ghi ra bo nho moi
    if (!warp_mask(m, M, warp_buf_)) return;  // ra ngoai anh: giu mask cu
    SegObject moved;
    if (seg_object(std::move(m), moved)) obj_ = std::move(moved);
}

LockStep LockedTracker::step(const cv::Mat& image_rgb, int64_t t_ns, const std::optional<CamPose>& pose) {
    LockStep out;
    cv::Point2d control;
    cv::Matx33d M_pose;
    out.mode = pose_control(pose, control, M_pose, out);
    kf_.predict(t_ns, control);
    out.meas = flow_.measure(image_rgb, kf_.center_px());
    if (out.meas.method != FlowMethod::kNone) kf_.update(out.meas.center_px, cfg_.kf_meas_px);
    // mask cu -> khung nay: affine tu flow > tinh tien NCC > homography / tinh tien tu the > theo Kalman
    switch (out.meas.method) {
        case FlowMethod::kFlow:
        case FlowMethod::kNcc:
            move_mask(out.meas.M_prev_to_cur);
            break;
        case FlowMethod::kNone:
            move_mask(out.mode == PoseMode::kNone ? translation(kf_.center_px() - obj_.center_px) : M_pose);
            break;
        case FlowMethod::kCount:
            break;
    }
    flow_.refresh_points(obj_.mask);
    out.center_px = kf_.center_px();
    out.kf_sigma_px = kf_.sigma_px();
    last_pose_ = pose;
    if (out.mode == PoseMode::kWorld) last_pose_px_ = out.pose_pred_px;
    return out;
}

void LockedTracker::on_segment(const cv::Mat& image_rgb, const SegObject& obj) {
    obj_ = obj;
    kf_.update(obj.center_px, cfg_.kf_meas_px);
    flow_.reset(image_rgb, obj.mask, obj.center_px, 0);
}

Detection LockedTracker::roi_box() const noexcept {
    const cv::Point2d c = kf_.center_px();
    const double hw = (box_.x2 - box_.x1) / 2.0, hh = (box_.y2 - box_.y1) / 2.0;
    Detection d = box_;
    d.x1 = static_cast<float>(c.x - hw);
    d.x2 = static_cast<float>(c.x + hw);
    d.y1 = static_cast<float>(c.y - hh);
    d.y2 = static_cast<float>(c.y + hh);
    return d;
}

}  // namespace uavp
