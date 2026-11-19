// s4 fusion: LiDAR quanh khung co mask -> bu chuyen dong -> RANSAC mat phang top -> tam + 3 truc.
// Chay vai lan moi giay tren vai tram diem (cho am): uu tien de doc, buffer dung lai.
#include "uavp/stages/s4_fusion/fusion.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>

#include "uavp/stages/s3_select/measure.hpp"  // W, L tren mat z = d

namespace uavp {
namespace {

cv::Point2d undistort_px(cv::Point2d px, const CameraModel& camera) {
    const std::vector<cv::Point2d> in{px};
    std::vector<cv::Point2d> out;
    cv::undistortPoints(in, out, camera.K_px, camera.dist_coeffs);
    return out[0];
}

double pearson(std::span<const double> a, std::span<const double> b) {
    const auto n = static_cast<double>(a.size());
    if (a.size() < 2) return 0.0;
    double ma = 0, mb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) ma += a[i], mb += b[i];
    ma /= n;
    mb /= n;
    double sab = 0, saa = 0, sbb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    return saa > 0 && sbb > 0 ? sab / std::sqrt(saa * sbb) : 0.0;
}

}  // namespace

cv::Point3d object_position_mm(cv::Point2d center_px, double d_mm, const CameraModel& camera) {
    const cv::Point2d n = undistort_px(center_px, camera);
    return {d_mm * n.x, d_mm * n.y, d_mm};
}

TopSize top_size(const SegMask& seg, double d_mm, const CameraModel& camera) {
    std::vector<cv::Point> contour;
    if (!largest_contour_px(seg, contour) || contour.size() < 3) return {};
    std::vector<cv::Point2f> plane_m;
    points_on_plane_m(contour, camera, d_mm / 1000.0, plane_m);  // diem tren mat z = d
    const SizeMm s = rect_size_mm(plane_m);
    return {s.W_mm, s.L_mm, s.yaw_deg};
}

Fusion::Fusion(FusionConfig cfg, CameraModel camera)
    : cfg_(std::move(cfg)), camera_(std::move(camera)), rng_(cfg_.ransac_seed) {
    // truc quay LD06 (z_L) trong he camera = phap tuyen mat quet; chieu xuong mat anh -> huong w
    const cv::Matx33d& R = cfg_.extrinsics.R_CL;
    const double nx = R(0, 2), ny = R(1, 2), len = std::hypot(nx, ny);
    w_dir_ = len > 1e-6 ? cv::Point2d(nx / len, ny / len) : cv::Point2d(0.0, 1.0);
    u_dir_ = {-w_dir_.y, w_dir_.x};
    scan_points_.reserve(1024);
    scan_t_.reserve(1024);
}

void Fusion::prepare(std::span<const MaskFrame> frames) {
    infos_.resize(frames.size());
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const SegObject& o = frames[i].obj;
        FrameInfo& f = infos_[i];
        eroded_mask(o.mask, cfg_.erode_px, f.mask);
        f.center_px = o.center_px;
        f.ray = undistort_px(o.center_px, camera_);
        f.roi_r = box_max_norm_radius(o.mask.x0_px - 1, o.mask.y0_px - 1, o.mask.mask.cols + 2, o.mask.mask.rows + 2,
                                      camera_);
        f.t_ns = frames[i].t_ns;
        f.index = i;
    }
    std::stable_sort(infos_.begin(), infos_.end(), [](const FrameInfo& a, const FrameInfo& b) { return a.t_ns < b.t_ns; });
}

std::size_t Fusion::nearest_frame(int64_t t_ns) const noexcept {
    const auto it = std::lower_bound(infos_.begin(), infos_.end(), t_ns,
                                     [](const FrameInfo& f, int64_t t) { return f.t_ns < t; });
    if (it == infos_.begin()) return 0;
    if (it == infos_.end()) return infos_.size() - 1;
    const auto k = static_cast<std::size_t>(it - infos_.begin());
    return t_ns - infos_[k - 1].t_ns <= infos_[k].t_ns - t_ns ? k - 1 : k;
}

// Tam vat (px) hoac tia qua tam noi suy tuyen tinh theo thoi gian giua 2 khung bao quanh t_ns.
cv::Point2d Fusion::interp(int64_t t_ns, bool ray) const noexcept {
    const auto pick = [ray](const FrameInfo& f) { return ray ? f.ray : f.center_px; };
    if (t_ns <= infos_.front().t_ns) return pick(infos_.front());
    if (t_ns >= infos_.back().t_ns) return pick(infos_.back());
    const auto it = std::lower_bound(infos_.begin(), infos_.end(), t_ns,
                                     [](const FrameInfo& f, int64_t t) { return f.t_ns < t; });
    const FrameInfo& b = *it;
    const FrameInfo& a = *(it - 1);
    const double w = static_cast<double>(t_ns - a.t_ns) / static_cast<double>(std::max<int64_t>(1, b.t_ns - a.t_ns));
    return pick(a) + w * (pick(b) - pick(a));
}

void Fusion::collect(std::span<const MaskFrame> frames, std::span<const Scan> scans, FusionResult& r) {
    const auto window_ns = static_cast<int64_t>(cfg_.lidar_window_s * 1e9);
    for (std::size_t si = 0; si < scans.size(); ++si) {
        scan_points_.clear();
        scan_t_.clear();
        scan_to_camera_mm(scans[si], cfg_.extrinsics, cfg_, scan_points_, scan_t_);
        if (scan_points_.empty()) continue;
        project_px(scan_points_, camera_, uv_);
        bool used = false;
        for (std::size_t i = 0; i < scan_points_.size(); ++i) {
            const FrameInfo& f = infos_[nearest_frame(scan_t_[i])];
            if (std::abs(scan_t_[i] - f.t_ns) > window_ns) continue;  // khong co mask gan luc quet
            const cv::Point3f& p = scan_points_[i];
            if (std::hypot(static_cast<double>(p.x), static_cast<double>(p.y)) / p.z > f.roi_r) continue;  // meo gap
            const cv::Point2d shift = interp(scan_t_[i], false) - f.center_px;  // mask dich theo vat
            const SegMask& seg = frames[f.index].obj.mask;
            const int col = round_half_even(uv_[i].x - shift.x) - seg.x0_px;
            const int row = round_half_even(uv_[i].y - shift.y) - seg.y0_px;
            if (col < 0 || col >= f.mask.cols || row < 0 || row >= f.mask.rows) continue;
            if (f.mask.ptr<uint8_t>(row)[col] == 0) continue;
            r.points_C_mm.push_back(p);
            r.points_t_ns.push_back(scan_t_[i]);
            r.points_frame.push_back(static_cast<uint16_t>(f.index));
            r.points_scan.push_back(static_cast<uint16_t>(si));
            used = true;
        }
        r.n_scans += used ? 1 : 0;
    }
}

void Fusion::compensate(std::span<const MaskFrame> frames, FusionResult& r) {
    r.pose_compensated = std::all_of(frames.begin(), frames.end(), [](const MaskFrame& f) { return f.pose.has_value(); });
    const FrameInfo& ref = infos_.back();
    r.points_ref_mm.resize(r.points_C_mm.size());
    if (r.pose_compensated) {
        const CamPose& pr = *frames[ref.index].pose;
        for (std::size_t i = 0; i < r.points_C_mm.size(); ++i) {
            const CamPose& pf = *frames[r.points_frame[i]].pose;
            const cv::Point3f& p = r.points_C_mm[i];
            const cv::Vec3d pw = pf.R_WC * cv::Vec3d(p.x, p.y, p.z) + pf.t_WC_mm;
            const cv::Vec3d q = pr.R_WC.t() * (pw - pr.t_WC_mm);
            r.points_ref_mm[i] = {static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2])};
        }
        return;
    }
    std::vector<float> z;
    for (const auto& p : r.points_C_mm) z.push_back(p.z);
    const double d0 = median_inplace(std::span<float>(z));
    for (std::size_t i = 0; i < r.points_C_mm.size(); ++i) {
        const cv::Point2d shift = d0 * (ref.ray - interp(r.points_t_ns[i], true));
        const cv::Point3f& p = r.points_C_mm[i];
        r.points_ref_mm[i] = {static_cast<float>(p.x + shift.x), static_cast<float>(p.y + shift.y), p.z};
    }
}

void Fusion::set_pose(const SegObject& ref, const PlaneFit& fit, double c_mm, FusionResult& r) const {
    const PlaneModel& m = fit.m;
    // giao tia (x, y, 1) voi mat z = a u + b w + c (dt = 0)
    const auto on_plane = [&](cv::Point2d ray) -> cv::Vec3d {
        const double ru = ray.dot(u_dir_), rw = ray.dot(w_dir_);
        const double denom = 1.0 - m.a * ru - m.b * rw;
        const double s = std::abs(denom) > 1e-6 ? c_mm / denom : kNaN;
        return {s * ray.x, s * ray.y, s};
    };
    const cv::Vec3d center = on_plane(undistort_px(ref.center_px, camera_));
    cv::Vec3d n(-m.a * u_dir_.x - m.b * w_dir_.x, -m.a * u_dir_.y - m.b * w_dir_.y, 1.0);
    n /= cv::norm(n);
    constexpr double kAxisPx = 30.0;  // diem thu 2 tren canh dai (px) de dung truc x
    const double yaw = ref.yaw_deg * CV_PI / 180.0;
    const cv::Vec3d p2 = on_plane(undistort_px(ref.center_px + kAxisPx * cv::Point2d(std::cos(yaw), std::sin(yaw)), camera_));
    cv::Vec3d x = p2 - center;
    x -= x.dot(n) * n;
    x /= cv::norm(x);
    r.p_obj_mm = {center[0], center[1], center[2]};
    r.axis_z = n;
    r.axis_x = x;
    r.axis_y = n.cross(x);
    r.tilt_deg = std::acos(std::clamp(n[2], -1.0, 1.0)) * 180.0 / CV_PI;
    if (std::isfinite(fit.c2)) {  // mat 2 tai tam (cung nghieng)
        const cv::Point2d ray = undistort_px(ref.center_px, camera_);
        r.d2_mm = fit.c2 / (1.0 - m.a * ray.dot(u_dir_) - m.b * ray.dot(w_dir_));
    }
}

void Fusion::fit(std::span<const MaskFrame> frames, FusionResult& r) {
    const FrameInfo& ref = infos_.back();
    data_.clear();
    for (std::size_t i = 0; i < r.points_ref_mm.size(); ++i) {
        const cv::Point3f& q = r.points_ref_mm[i];
        const cv::Point2d xy(q.x, q.y);
        data_.u.push_back(xy.dot(u_dir_));
        data_.w.push_back(xy.dot(w_dir_));
        data_.z.push_back(q.z);
        data_.dt.push_back(static_cast<double>(r.points_t_ns[i] - ref.t_ns) / 1e9);
        data_.scan.push_back(r.points_scan[i]);
    }
    std::vector<double> w = data_.w;
    r.spread_w_mm = percentile_inplace(std::span<double>(w), 90.0) - percentile_inplace(std::span<double>(w), 10.0);
    const auto [tmin, tmax] = std::minmax_element(data_.dt.begin(), data_.dt.end());
    PlaneForm form;
    form.use_e = !r.pose_compensated && *tmax - *tmin >= cfg_.drift_min_span_s;
    form.use_b = r.spread_w_mm >= cfg_.min_spread_mm &&
                 !(form.use_e && std::abs(pearson(data_.w, data_.dt)) > cfg_.max_motion_time_corr);
    const double max_slope = std::tan(cfg_.max_tilt_deg * CV_PI / 180.0);
    ransac_plane(data_, form, cfg_.plane_iters, cfg_.ransac_tau_mm, max_slope, rng_, fit_);
    if (fit_.n_inliers == 0) return;
    r.inliers = fit_.inliers;
    r.n_inliers = fit_.n_inliers;
    r.inlier_ratio = static_cast<double>(fit_.n_inliers) / r.n_points;
    r.rms_mm = fit_.rms_mm;
    r.n2_inliers = fit_.n2;
    r.ambiguous = fit_.n2 > 0 && fit_.n2 >= cfg_.ambiguity_ratio * fit_.n_inliers;
    r.normal_observable = form.use_b;
    r.drift_mm_s = form.use_e ? fit_.m.e : kNaN;
    const double c_mm = median_offset_by_scan(data_, fit_);  // c ben vung (median theo vong -> giua cac vong)
    const SegObject& obj = frames[ref.index].obj;
    set_pose(obj, fit_, c_mm, r);
    r.d_mm = r.p_obj_mm.z;
    r.dist_mm = std::sqrt(r.p_obj_mm.dot(r.p_obj_mm));
    r.status = r.inlier_ratio >= cfg_.min_inlier_ratio && !r.ambiguous ? FusionStatus::kOk : FusionStatus::kLowConfidence;
    const TopSize ts = top_size(obj.mask, r.d_mm, camera_);
    r.W_mm = ts.W_mm;
    r.L_mm = ts.L_mm;
    r.yaw_deg = ts.yaw_deg;
}

FusionResult Fusion::operator()(std::span<const MaskFrame> frames, std::span<const Scan> scans) {
    FusionResult r;
    r.n_frames = static_cast<int>(frames.size());
    if (frames.empty()) return r;
    prepare(frames);
    r.t_image_ns = infos_.back().t_ns;
    collect(frames, scans, r);
    r.n_points = static_cast<int>(r.points_C_mm.size());
    r.inliers.assign(r.points_C_mm.size(), 0);
    if (!r.points_t_ns.empty()) {
        const auto [lo, hi] = std::minmax_element(r.points_t_ns.begin(), r.points_t_ns.end());
        r.t_lidar_first_ns = *lo;
        r.t_lidar_last_ns = *hi;
    }
    if (r.n_points < cfg_.min_pts) return r;  // INSUFFICIENT, so do nan
    compensate(frames, r);
    fit(frames, r);
    return r;
}

}  // namespace uavp
