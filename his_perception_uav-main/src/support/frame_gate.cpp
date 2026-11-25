#include "uavp/support/frame_gate.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace uavp {

namespace {

// Doan [a, b) cua mau i: w khong doi tu t_i toi t_{i+1}; mau cuoi khong co doan (chi la moc ket thuc).
bool covers(std::span<const RateSample> r, int64_t t_lo, int64_t t_hi) noexcept {
    return r.size() >= 2 && r.front().t_ns <= t_lo && r.back().t_ns >= t_hi && t_hi >= t_lo;
}

}  // namespace

std::optional<cv::Vec3d> trajectory_width_rad(std::span<const RateSample> rates, int64_t t_lo_ns, int64_t t_hi_ns) {
    if (!covers(rates, t_lo_ns, t_hi_ns)) return std::nullopt;
    cv::Vec3d theta{0.0, 0.0, 0.0};  // quy dao tich luy, = 0 tai t_lo
    cv::Vec3d lo = theta, hi = theta;
    for (std::size_t i = 0; i + 1 < rates.size(); ++i) {
        const int64_t a = std::max(rates[i].t_ns, t_lo_ns);
        const int64_t b = std::min(rates[i + 1].t_ns, t_hi_ns);
        if (b <= a) continue;
        theta += rates[i].w_C * (static_cast<double>(b - a) / 1e9);
        for (int k = 0; k < 3; ++k) {  // diem gay (hoac dau cua so): cuc tri cua ham tuyen tinh tung doan
            lo[k] = std::min(lo[k], theta[k]);
            hi[k] = std::max(hi[k], theta[k]);
        }
    }
    return hi - lo;
}

std::optional<double> blur_px(std::span<const RateSample> rates, int64_t t_lo_ns, int64_t t_hi_ns, double f_px) {
    const auto w = trajectory_width_rad(rates, t_lo_ns, t_hi_ns);
    if (!w) return std::nullopt;
    return f_px * std::hypot((*w)[1], (*w)[0]);  // quay quanh y -> dich x; quanh x -> dich y
}

std::optional<double> max_perp_rate(std::span<const RateSample> rates, int64_t t_lo_ns, int64_t t_hi_ns) {
    if (!covers(rates, t_lo_ns, t_hi_ns)) return std::nullopt;
    double m = 0.0;
    for (std::size_t i = 0; i + 1 < rates.size(); ++i) {
        if (rates[i + 1].t_ns <= t_lo_ns || rates[i].t_ns >= t_hi_ns) continue;  // doan khong cham cua so
        m = std::max(m, std::hypot(rates[i].w_C[0], rates[i].w_C[1]));
    }
    return m;
}

std::vector<RateSample> rates_from_poses(std::span<const CamPoseSample> poses) {
    std::vector<RateSample> out;
    if (poses.size() < 2) return out;
    out.reserve(poses.size());
    for (std::size_t i = 0; i + 1 < poses.size(); ++i) {
        const double dt = static_cast<double>(poses[i + 1].t_ns - poses[i].t_ns) / 1e9;
        cv::Vec3d rv{0.0, 0.0, 0.0};
        if (dt > 0.0) {
            const cv::Matx33d R_rel = poses[i].pose.R_WC.t() * poses[i + 1].pose.R_WC;  // C_i -> C_{i+1}, he C_i
            cv::Rodrigues(R_rel, rv);
            rv /= dt;
        }
        out.push_back({poses[i].t_ns, rv});
    }
    out.push_back({poses.back().t_ns, out.back().w_C});  // moc ket thuc
    return out;
}

double strip_sharpness(const cv::Mat& image_rgb, int rows, double band) {
    CV_Assert(image_rgb.type() == CV_8UC3 && rows >= 2);
    const int h = image_rgb.rows, w = image_rgb.cols;
    const int y0 = static_cast<int>(h * (1.0 - band) / 2.0), y1 = static_cast<int>(h * (1.0 + band) / 2.0) - 2;
    const int x0 = static_cast<int>(w * (1.0 - band) / 2.0), x1 = static_cast<int>(w * (1.0 + band) / 2.0) - 1;
    if (y1 <= y0 || x1 <= x0) return 0.0;
    uint64_t sum = 0, n = 0;
    for (int k = 0; k < rows; ++k) {
        const int y = y0 + static_cast<int>(static_cast<int64_t>(y1 - y0) * k / (rows - 1));
        const uint8_t* p = image_rgb.ptr<uint8_t>(y);
        const uint8_t* q = image_rgb.ptr<uint8_t>(y + 1);  // hang ke duoi: sai phan doc
        for (int x = x0; x < x1; ++x) {
            const int g = p[3 * x + 1];  // kenh G (gan mat Y nhat, khong doi mau)
            const int gx = p[3 * (x + 1) + 1] - g;
            const int gy = q[3 * x + 1] - g;
            sum += static_cast<uint64_t>(gx * gx + gy * gy);
            ++n;
        }
    }
    return n == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(n);
}

GateResult FrameGate::evaluate(const cv::Mat& image_rgb, int64_t t0_ns, double exposure_us, double f_px,
                               std::span<const RateSample> rates) {
    GateResult r;
    // --- Tang 1: gyro -------------------------------------------------------------------------------------------
    const auto t_ro = static_cast<int64_t>(cfg_.readout_ms * 1e6);
    const auto t_exp = static_cast<int64_t>(std::max(0.0, exposure_us) * 1e3);
    if (covers(rates, t0_ns, t0_ns + t_ro + t_exp)) {
        r.gyro = true;
        double blur = 0.0, roll_rad = 0.0;
        for (const double row : {0.0, 0.5, 1.0}) {  // hang tren / giua / duoi: moi hang mot cua so phoi sang
            const int64_t a = t0_ns + static_cast<int64_t>(row * static_cast<double>(t_ro));
            blur = std::max(blur, *blur_px(rates, a, a + t_exp, f_px));
            roll_rad = std::max(roll_rad, (*trajectory_width_rad(rates, a, a + t_exp))[2]);
        }
        r.blur_px = blur;
        r.roll_px = cfg_.roll_radius_px * roll_rad;
        r.rs_px = f_px * *max_perp_rate(rates, t0_ns, t0_ns + t_ro) * cfg_.readout_ms / 1e3;
    }
    // --- Tang 2: do net so voi trung vi `history` khung TRUOC ----------------------------------------------------
    r.sharp = strip_sharpness(image_rgb, cfg_.sharp_rows, cfg_.sharp_band);
    r.history = static_cast<int>(sharp_.size());
    if (!sharp_.empty()) {
        std::vector<double> v(sharp_.begin(), sharp_.end());
        r.sharp_median = median_inplace(std::span<double>(v));
        if (r.sharp_median > 0.0) r.sharp_ratio = r.sharp / r.sharp_median;
    }
    sharp_.push_back(r.sharp);
    while (sharp_.size() > static_cast<std::size_t>(cfg_.history)) sharp_.pop_front();
    // --- Quyet dinh (tang 1 truoc: nguyen nhan) ------------------------------------------------------------------
    if (r.gyro && r.blur_px > cfg_.max_blur_px) r.reason = "blur";
    else if (r.gyro && r.rs_px > cfg_.max_rs_px) r.reason = "rolling_shutter";
    else if (r.gyro && cfg_.roll_radius_px > 0.0 && r.roll_px > cfg_.max_roll_px) r.reason = "roll";
    else if (!r.gyro && cfg_.require_gyro) r.reason = "no_gyro";
    else if (r.history < cfg_.min_history) r.reason = "warmup";
    else if (!(r.sharp_ratio >= cfg_.min_ratio)) r.reason = "sharp";
    else r.reason = "ok";
    r.pass = r.reason == "ok";
    return r;
}

std::vector<RateSample> camera_rates(const FcPoseBuffer& fc, const BodyCamera& bc, int64_t t_lo_ns, int64_t t_hi_ns) {
    std::vector<int64_t> times;
    fc.attitude_times(t_lo_ns, t_hi_ns, times);  // + 1 mau truoc / sau de phu 2 dau
    std::vector<CamPoseSample> poses;
    poses.reserve(times.size());
    for (const int64_t t : times) {
        FcPose p;
        if (fc.pose_at(t, p)) poses.push_back({camera_pose(p, bc), t});
    }
    return rates_from_poses(poses);
}

}  // namespace uavp
