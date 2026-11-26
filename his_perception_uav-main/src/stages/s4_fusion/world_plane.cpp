// s4 world_plane: slerp tu the, RANSAC 3 diem + PCA, dieu kien PLANE, giao tia - mat phang, hinh chu nhat tren mat.
// Chay vai lan / giay tren vai nghin diem (cho am): uu tien de doc, buffer cua goi dung lai.
#include "uavp/stages/s4_fusion/world_plane.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>

#include "uavp/stages/s2_segment/mask_shape.hpp"
#include "uavp/stages/s3_select/measure.hpp"

namespace uavp {
namespace {

constexpr double kDegToRad = CV_PI / 180.0;
constexpr double kMinSinSample = 1e-3;  // 3 diem mau gan thang hang (sin goc < 1e-3) -> bo

}  // namespace

// --- Quaternion ---------------------------------------------------------------------------------------------------
cv::Vec4d quat_from_matrix(const cv::Matx33d& R) noexcept {
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);
    cv::Vec4d q;
    if (tr > 0.0) {
        const double s = 2.0 * std::sqrt(tr + 1.0);
        q = {0.25 * s, (R(2, 1) - R(1, 2)) / s, (R(0, 2) - R(2, 0)) / s, (R(1, 0) - R(0, 1)) / s};
    } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
        const double s = 2.0 * std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2));
        q = {(R(2, 1) - R(1, 2)) / s, 0.25 * s, (R(0, 1) + R(1, 0)) / s, (R(0, 2) + R(2, 0)) / s};
    } else if (R(1, 1) > R(2, 2)) {
        const double s = 2.0 * std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2));
        q = {(R(0, 2) - R(2, 0)) / s, (R(0, 1) + R(1, 0)) / s, 0.25 * s, (R(1, 2) + R(2, 1)) / s};
    } else {
        const double s = 2.0 * std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1));
        q = {(R(1, 0) - R(0, 1)) / s, (R(0, 2) + R(2, 0)) / s, (R(1, 2) + R(2, 1)) / s, 0.25 * s};
    }
    return q / cv::norm(q);
}

cv::Matx33d matrix_from_quat(const cv::Vec4d& q) noexcept {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    return {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
            2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)};
}

cv::Vec4d slerp(const cv::Vec4d& q0, cv::Vec4d q1, double w) noexcept {
    double dot = q0.dot(q1);
    if (dot < 0.0) {  // q va -q cung mot phep quay: di duong ngan
        q1 = -q1;
        dot = -dot;
    }
    if (dot > 0.9995) {  // goc rat nho: noi suy tuyen tinh roi chuan hoa (tranh chia cho sin ~ 0)
        const cv::Vec4d q = q0 + w * (q1 - q0);
        return q / cv::norm(q);
    }
    const double theta = std::acos(std::clamp(dot, -1.0, 1.0));
    const double s = std::sin(theta);
    return (std::sin((1.0 - w) * theta) / s) * q0 + (std::sin(w * theta) / s) * q1;
}

bool interpolate_pose(std::span<const CamPoseSample> samples, int64_t t_ns, int64_t max_gap_ns, CamPose& out) noexcept {
    if (samples.empty()) return false;
    const auto it = std::lower_bound(samples.begin(), samples.end(), t_ns,
                                     [](const CamPoseSample& s, int64_t t) { return s.t_ns < t; });
    if (it == samples.end()) {  // sau mau cuoi
        if (t_ns - samples.back().t_ns > max_gap_ns) return false;
        out = samples.back().pose;
        return true;
    }
    if (it->t_ns == t_ns || it == samples.begin()) {  // trung mau / truoc mau dau
        if (it->t_ns - t_ns > max_gap_ns) return false;
        out = it->pose;
        return true;
    }
    const CamPoseSample& a = *(it - 1);
    const CamPoseSample& b = *it;
    if (b.t_ns - a.t_ns > 2 * max_gap_ns) return false;  // lo du lieu tu the: khong noi suy qua
    const double w = static_cast<double>(t_ns - a.t_ns) / static_cast<double>(b.t_ns - a.t_ns);
    out.R_WC = matrix_from_quat(slerp(quat_from_matrix(a.pose.R_WC), quat_from_matrix(b.pose.R_WC), w));
    out.t_WC_mm = a.pose.t_WC_mm + w * (b.pose.t_WC_mm - a.pose.t_WC_mm);
    out.has_position = a.pose.has_position && b.pose.has_position;
    return true;
}

// --- Mat phang ----------------------------------------------------------------------------------------------------
bool pca_plane(std::span<const cv::Vec3d> points_mm, std::span<const uint8_t> mask, PlanePca& out) {
    const bool all = mask.empty();
    cv::Vec3d sum(0, 0, 0);
    int n = 0;
    for (std::size_t i = 0; i < points_mm.size(); ++i) {
        if (!all && mask[i] == 0) continue;
        sum += points_mm[i];
        ++n;
    }
    if (n < 3) return false;
    const cv::Vec3d c = sum / n;
    cv::Matx33d cov = cv::Matx33d::zeros();
    for (std::size_t i = 0; i < points_mm.size(); ++i) {
        if (!all && mask[i] == 0) continue;
        const cv::Vec3d d = points_mm[i] - c;
        cov += d * d.t();
    }
    cov *= 1.0 / n;
    cv::Matx31d values;
    cv::Matx33d vectors;  // hang k = vector rieng cua values(k), giam dan
    cv::eigen(cov, values, vectors);
    out.c = c;
    out.n = cv::Vec3d(vectors(2, 0), vectors(2, 1), vectors(2, 2));
    out.n /= cv::norm(out.n);
    out.lambda_mm2 = {std::max(values(0), 0.0), std::max(values(1), 0.0), std::max(values(2), 0.0)};
    out.n_inliers = n;
    return true;
}

namespace {

int count_plane_inliers(std::span<const cv::Vec3d> pts, const cv::Vec3d& n, const cv::Vec3d& p0, double tau,
                        std::vector<uint8_t>* mask) {
    int cnt = 0;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        const bool in = std::abs(n.dot(pts[i] - p0)) < tau;
        cnt += in ? 1 : 0;
        if (mask != nullptr) (*mask)[i] = in ? 1 : 0;
    }
    return cnt;
}

}  // namespace

void ransac_pca_plane(std::span<const cv::Vec3d> points_mm, int iters, double tau_mm, const cv::Vec3d& up,
                      double max_tilt_deg, std::mt19937_64& rng, PlanePca& out) {
    out = PlanePca{};
    const std::size_t n = points_mm.size();
    out.inliers.assign(n, 0);
    if (n < 3) return;
    const double cos_max = std::cos(max_tilt_deg * kDegToRad);
    int best = 0;
    cv::Vec3d best_n, best_p;
    for (int it = 0; it < iters; ++it) {
        const std::size_t i0 = rng() % n, i1 = rng() % n, i2 = rng() % n;
        if (i0 == i1 || i0 == i2 || i1 == i2) continue;
        const cv::Vec3d a = points_mm[i1] - points_mm[i0], b = points_mm[i2] - points_mm[i0];
        cv::Vec3d nn = a.cross(b);
        const double len = cv::norm(nn);
        if (len < kMinSinSample * cv::norm(a) * cv::norm(b)) continue;  // 3 diem thang hang (cung duong quet)
        nn /= len;
        if (std::abs(nn.dot(up)) < cos_max) continue;  // mat ben / san lan: nghieng qua
        const int cnt = count_plane_inliers(points_mm, nn, points_mm[i0], tau_mm, nullptr);
        if (cnt > best) best = cnt, best_n = nn, best_p = points_mm[i0];
    }
    if (best == 0) return;
    count_plane_inliers(points_mm, best_n, best_p, tau_mm, &out.inliers);
    std::vector<uint8_t> inl = out.inliers;
    for (int pass = 0; pass < 2; ++pass) {  // PCA tren inlier -> dem lai inlier voi mat PCA -> PCA lan cuoi
        PlanePca p;
        if (!pca_plane(points_mm, inl, p)) return;
        if (pass == 1) {
            p.inliers = std::move(inl);
            out = std::move(p);
            return;
        }
        if (count_plane_inliers(points_mm, p.n, p.c, tau_mm, &inl) < 3) return;
    }
}

double effective_sigma_mm(const WorldPlaneConfig& cfg, double rho_mm) noexcept {
    const double att = std::max(0.0, rho_mm) * cfg.attitude_sigma_deg * kDegToRad;
    return std::sqrt(cfg.lidar_sigma_mm * cfg.lidar_sigma_mm + att * att);
}

PlaneQuality accept_plane(const PlanePca& p, const WorldPlaneConfig& cfg, double rho_mm) noexcept {
    PlaneQuality q;
    if (p.n_inliers < 3 || !std::isfinite(p.lambda_mm2[1])) return q;
    q.sqrt_l2_mm = std::sqrt(p.lambda_mm2[1]);
    q.sigma_eff_mm = effective_sigma_mm(cfg, rho_mm);
    q.sigma_perp_mm = std::max(std::sqrt(p.lambda_mm2[2]), q.sigma_eff_mm);
    const double theta_max = cfg.max_normal_sigma_deg * kDegToRad;
    if (q.sqrt_l2_mm > 0.0) {
        q.sigma_theta_deg = q.sigma_perp_mm / std::sqrt(p.n_inliers * p.lambda_mm2[1]) / kDegToRad;
        const double k = q.sigma_perp_mm / (theta_max * q.sqrt_l2_mm);
        q.n_required = k * k;
    }
    q.accepted = p.n_inliers >= cfg.min_inliers && q.sqrt_l2_mm >= cfg.min_sqrt_l2_mm &&
                 std::isfinite(q.sigma_theta_deg) && q.sigma_theta_deg <= cfg.max_normal_sigma_deg;
    return q;
}

double height_1d(std::span<const cv::Vec3d> points_mm, const cv::Vec3d& up, int iters, double tau_mm,
                 std::mt19937_64& rng, std::vector<uint8_t>& inliers) {
    const std::size_t n = points_mm.size();
    inliers.assign(n, 0);
    if (n == 0) return kNaN;
    std::vector<double> h(n);
    for (std::size_t i = 0; i < n; ++i) h[i] = up.dot(points_mm[i]);
    int best = 0;
    double best_h = h[0];
    for (int it = 0; it < iters; ++it) {
        const double cand = h[rng() % n];
        const auto cnt = static_cast<int>(std::count_if(h.begin(), h.end(), [&](double v) { return std::abs(v - cand) < tau_mm; }));
        if (cnt > best) best = cnt, best_h = cand;
    }
    std::vector<double> sel;
    sel.reserve(static_cast<std::size_t>(best));
    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(h[i] - best_h) >= tau_mm) continue;
        inliers[i] = 1;
        sel.push_back(h[i]);
    }
    return median_inplace(std::span<double>(sel));
}

bool ray_plane(const cv::Vec3d& ray, const cv::Vec3d& n, const cv::Vec3d& c, cv::Vec3d& p) noexcept {
    const double denom = n.dot(ray);
    if (std::abs(denom) < 1e-6) return false;  // tia song song mat phang
    const double s = n.dot(c) / denom;
    if (!(s > 0.0)) return false;  // giao sau camera (hoac nan)
    p = s * ray;
    return true;
}

bool top_rect_on_plane(const SegMask& seg, const CameraModel& camera, const cv::Vec3d& n_C, const cv::Vec3d& c_C,
                       TopRect& out) {
    out = TopRect{};
    std::vector<cv::Point> contour;
    if (!largest_contour_px(seg, contour) || contour.size() < 3) return false;
    std::vector<cv::Point2d> px(contour.begin(), contour.end()), norm;
    cv::undistortPoints(px, norm, camera.K_px, camera.dist_coeffs);
    cv::Vec3d e1 = cv::Vec3d(1, 0, 0) - n_C[0] * n_C;  // truc x camera chieu len mat phang
    if (cv::norm(e1) < 1e-6) return false;
    e1 /= cv::norm(e1);
    const cv::Vec3d e2 = n_C.cross(e1);
    std::vector<cv::Point2f> plane_m;
    plane_m.reserve(norm.size());
    for (const auto& q : norm) {
        cv::Vec3d p;
        if (!ray_plane({q.x, q.y, 1.0}, n_C, c_C, p)) return false;
        const cv::Vec3d d = p - c_C;
        plane_m.emplace_back(static_cast<float>(d.dot(e1) / 1000.0), static_cast<float>(d.dot(e2) / 1000.0));
    }
    const SizeMm s = rect_size_mm(plane_m);
    const double yaw = s.yaw_deg * kDegToRad;
    out.W_mm = s.W_mm;
    out.L_mm = s.L_mm;
    out.yaw_deg = s.yaw_deg;
    out.axis_x_C = std::cos(yaw) * e1 + std::sin(yaw) * e2;
    return true;
}

void fill_world(FusionResult& r, const CamPose& pose) noexcept {
    if (!pose.has_position || !std::isfinite(r.p_obj_mm.z)) return;
    if (!std::isfinite(r.p_obj_W_mm.x)) {
        const cv::Vec3d p = pose.R_WC * cv::Vec3d(r.p_obj_mm.x, r.p_obj_mm.y, r.p_obj_mm.z) + pose.t_WC_mm;
        r.p_obj_W_mm = {p[0], p[1], p[2]};
    }
    if (!std::isfinite(r.normal_W[0]) && std::isfinite(r.axis_z[0])) r.normal_W = pose.R_WC * r.axis_z;
    if (!std::isfinite(r.axis_x[0])) return;
    r.axis_x_W = pose.R_WC * r.axis_x;
    double yaw = std::atan2(r.axis_x_W[1], r.axis_x_W[0]) * 180.0 / CV_PI;
    yaw = std::fmod(yaw + 360.0, 180.0);  // canh dai: huong +/- nhu nhau
    r.yaw_W_deg = yaw >= 180.0 ? 0.0 : yaw;
}

}  // namespace uavp
