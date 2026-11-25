#include "uavp/support/video_diag.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "uavp/common/common.hpp"
#include "uavp/common/log.hpp"

namespace uavp {

namespace {

constexpr const char* kModule = "uavp.video";
constexpr const char* kHeader =
    "t_ns,dt_ms,exposure_us,gain,lens,luma,sharp,grad_x,grad_y,dx_px,dy_px,pc_resp,skew_x_px,skew_y_px,"
    "wx,wy,wz,blur_pred_px,rot_dx_px,rot_dy_px,armed,landed,vn,ve,vd,diag_ms\n";

cv::Mat gray_small(const cv::Mat& rgb, int width_px, double& scale) {
    scale = std::min(1.0, static_cast<double>(width_px) / rgb.cols);
    cv::Mat small, gray, f;
    cv::resize(rgb, small, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::cvtColor(small, gray, cv::COLOR_RGB2GRAY);
    gray.convertTo(f, CV_32F);
    return f;
}

void ensure_window(cv::Mat& win, cv::Size size) {
    if (win.size() != size) cv::createHanningWindow(win, size, CV_32F);
}

}  // namespace

SharpStats sharp_stats(const cv::Mat& image_rgb, int crop_px) {
    SharpStats s;
    const int w = std::min(crop_px, image_rgb.cols), h = std::min(crop_px, image_rgb.rows);
    const cv::Rect roi((image_rgb.cols - w) / 2, (image_rgb.rows - h) / 2, w, h);
    cv::Mat g;
    cv::extractChannel(image_rgb(roi), g, 1);
    cv::Mat lap, gx, gy;
    cv::Laplacian(g, lap, CV_32F, 3);
    cv::Scalar mean, sd;
    cv::meanStdDev(lap, mean, sd);
    s.sharp = sd[0] * sd[0];
    cv::Sobel(g, gx, CV_32F, 1, 0, 3);
    cv::Sobel(g, gy, CV_32F, 0, 1, 3);
    s.grad_x = cv::mean(gx.mul(gx))[0];
    s.grad_y = cv::mean(gy.mul(gy))[0];
    s.luma = cv::mean(g)[0];
    return s;
}

bool ShiftMeter::update(const cv::Mat& image_rgb, ImageShift& out) {
    cv::Mat cur = gray_small(image_rgb, width_px_, scale_);
    if (prev_.empty() || prev_.size() != cur.size()) {
        prev_ = cur;
        return false;
    }
    ensure_window(win_, cur.size());
    double response = 0.0;
    // phaseCorrelate(a, b) = dich cua b so voi a: noi dung khung nay dich bao nhieu so voi khung truoc
    const cv::Point2d d = cv::phaseCorrelate(prev_, cur, win_, &response);
    const int strip = cur.rows / 3;
    const cv::Rect top(0, 0, cur.cols, strip), bot(0, cur.rows - strip, cur.cols, strip);
    ensure_window(win_strip_, top.size());
    const cv::Point2d dt = cv::phaseCorrelate(prev_(top), cur(top), win_strip_);
    const cv::Point2d db = cv::phaseCorrelate(prev_(bot), cur(bot), win_strip_);
    out.dx_px = d.x / scale_;
    out.dy_px = d.y / scale_;
    out.response = response;
    out.skew_x_px = (dt.x - db.x) / scale_;
    out.skew_y_px = (dt.y - db.y) / scale_;
    prev_ = cur;
    return true;
}

cv::Vec3d angular_rate(const cv::Matx33d& R_WC0, const cv::Matx33d& R_WC1, double dt_s) {
    const cv::Matx33d dR = R_WC0.t() * R_WC1;  // quay trong he camera
    cv::Vec3d rvec;
    cv::Rodrigues(cv::Mat(dR), rvec);
    return rvec / dt_s;
}

cv::Point2d rotation_shift_px(const cv::Matx33d& R_WC0, const cv::Matx33d& R_WC1, const CameraModel& cam) {
    const cv::Vec3d d1 = R_WC1.t() * (R_WC0 * cv::Vec3d(0.0, 0.0, 1.0));  // tia tam anh khung truoc, nhin tu khung sau
    if (d1[2] <= 1e-6) return {kNaN, kNaN};
    return {cam.K_px(0, 0) * d1[0] / d1[2], cam.K_px(1, 1) * d1[1] / d1[2]};
}

double blur_pred_px(const cv::Vec3d& w_C, double exposure_us, double f_px) noexcept {
    if (!(exposure_us > 0.0)) return kNaN;
    return std::hypot(w_C[0], w_C[1]) * f_px * exposure_us * 1e-6;
}

DiagFrameSource::DiagFrameSource(IFrameSource& inner, const VideoDiagConfig& cfg, const CameraModel& camera,
                                 const FcPoseBuffer* fc, const BodyCamera& body_camera)
    : inner_(inner), cfg_(cfg), camera_(camera), fc_(fc), body_camera_(body_camera), shift_(cfg.motion_width_px) {
    rows_.reserve(1 << 20);
}

DiagFrameSource::~DiagFrameSource() {
    try {
        flush();
    } catch (...) {  // huy: khong nem
    }
}

bool DiagFrameSource::next_frame(cv::Mat& image_rgb, FrameMeta& meta, double timeout_s) {
    if (!inner_.next_frame(image_rgb, meta, timeout_s)) return false;
    if (counter_++ % cfg_.every_n == 0) measure(image_rgb, meta);
    return true;
}

void DiagFrameSource::set_output(const std::filesystem::path& csv) {
    out_ = csv;
    flush();
}

void DiagFrameSource::flush() {
    if (out_.empty()) return;
    std::ofstream f(out_, std::ios::app);
    if (!f) return;
    if (!header_written_) {
        f << kHeader;
        header_written_ = true;
    }
    f << rows_;
    rows_.clear();
    flushed_ns_ = mono_ns();
}

void DiagFrameSource::measure(const cv::Mat& image_rgb, const FrameMeta& meta) {
    const int64_t start = mono_ns();
    const SharpStats s = sharp_stats(image_rgb, cfg_.crop_px);
    ImageShift sh;
    const bool has_shift = shift_.update(image_rgb, sh);
    const double dt_ms = prev_t_ns_ > 0 ? static_cast<double>(meta.t_ns - prev_t_ns_) / 1e6 : kNaN;
    prev_t_ns_ = meta.t_ns;

    cv::Vec3d w{kNaN, kNaN, kNaN};
    cv::Point2d rot{kNaN, kNaN};
    double blur = kNaN;
    FcStatus st;
    bool has_fc = false;
    if (fc_ != nullptr) {
        const auto h = static_cast<int64_t>(cfg_.gyro_half_window_s * 1e9);
        FcPose a, b, c;
        if (fc_->pose_at(meta.t_ns - h, a) && fc_->pose_at(meta.t_ns + h, b)) {
            w = angular_rate(camera_pose(a, body_camera_).R_WC, camera_pose(b, body_camera_).R_WC,
                             2.0 * cfg_.gyro_half_window_s);
            blur = blur_pred_px(w, meta.exposure_us, camera_.K_px(0, 0));
        }
        if (fc_->pose_at(meta.t_ns, c)) {
            const cv::Matx33d R = camera_pose(c, body_camera_).R_WC;
            if (has_prev_R_ && has_shift) rot = rotation_shift_px(prev_R_WC_, R, camera_);
            prev_R_WC_ = R;
            has_prev_R_ = true;
        } else {
            has_prev_R_ = false;
        }
        st = fc_->status(mono_ns(), 0.5);
        has_fc = st.heartbeat_ns > 0;
    }
    const double ms = static_cast<double>(mono_ns() - start) / 1e6;
    const double nan = kNaN;
    char row[512];
    std::snprintf(row, sizeof(row),
                  "%lld,%.3f,%.1f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,"
                  "%d,%d,%.3f,%.3f,%.3f,%.2f\n",
                  static_cast<long long>(meta.t_ns), dt_ms, meta.exposure_us, meta.analogue_gain, meta.lens_position,
                  s.luma, s.sharp, s.grad_x, s.grad_y, has_shift ? sh.dx_px : nan, has_shift ? sh.dy_px : nan,
                  has_shift ? sh.response : nan, has_shift ? sh.skew_x_px : nan, has_shift ? sh.skew_y_px : nan, w[0],
                  w[1], w[2], blur, rot.x, rot.y, has_fc ? (st.armed ? 1 : 0) : -1,
                  has_fc ? static_cast<int>(st.landed_state) : -1, has_fc ? st.vel_ned_m_s[0] : nan,
                  has_fc ? st.vel_ned_m_s[1] : nan, has_fc ? st.vel_ned_m_s[2] : nan, ms);
    rows_ += row;
    ++frames_;
    ++sum_n_;
    sum_sharp_ += s.sharp;
    sum_ms_ += ms;
    if (has_shift) sum_shift2_ += sh.dx_px * sh.dx_px + sh.dy_px * sh.dy_px;
    if (std::isfinite(w[0])) sum_w2_ += w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if (std::isfinite(blur)) sum_blur_ += blur;
    if (std::isfinite(dt_ms)) sum_dt_ += dt_ms, max_dt_ = std::max(max_dt_, dt_ms);
    const int64_t now = mono_ns();
    if (static_cast<double>(now - flushed_ns_) / 1e9 >= cfg_.flush_period_s) flush();
    if (static_cast<double>(now - logged_ns_) / 1e9 >= cfg_.log_period_s) log_summary(now);
}

void DiagFrameSource::log_summary(int64_t now_ns) {
    if (sum_n_ > 0) {
        const double n = sum_n_;
        UAVP_LOG_INFO(kModule,
                      "VIDEO %d khung | net %.0f | dich anh rms %.1f px/khung | quay rms %.2f rad/s | mo du doan %.2f px | "
                      "khung %.1f ms (max %.1f) | do %.1f ms",
                      sum_n_, sum_sharp_ / n, std::sqrt(sum_shift2_ / n), std::sqrt(sum_w2_ / n), sum_blur_ / n,
                      sum_dt_ / n, max_dt_, sum_ms_ / n);
    }
    logged_ns_ = now_ns;
    sum_n_ = 0;
    sum_sharp_ = sum_shift2_ = sum_w2_ = sum_blur_ = sum_dt_ = max_dt_ = sum_ms_ = 0.0;
}

}  // namespace uavp
