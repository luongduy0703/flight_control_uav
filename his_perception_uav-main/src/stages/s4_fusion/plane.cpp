// RANSAC mat phang top (s4). Chay vai lan / giay tren vai tram diem: uu tien de doc.
#include "uavp/stages/s4_fusion/plane.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>

#include <opencv2/core.hpp>

namespace uavp {
namespace {

// Cot (u, w, 1, dt) da chia ti le (u, w theo 100 mm) de he phuong trinh khong lech thang do.
constexpr double kScaleMm = 100.0;
constexpr double kMinDet = 1e-9;

std::array<double, 4> row(const PlaneData& d, std::size_t i, PlaneForm f) noexcept {
    return {d.u[i] / kScaleMm, f.use_b ? d.w[i] / kScaleMm : 0.0, 1.0, f.use_e ? d.dt[i] : 0.0};
}

// Cot khong dung -> hang / cot don vi, ve phai 0 (an = 0): luon giai he 4 x 4.
void pad_unused(cv::Matx44d& A, cv::Vec4d& rhs, PlaneForm f) noexcept {
    for (const int k : {1, 3}) {
        if ((k == 1 && f.use_b) || (k == 3 && f.use_e)) continue;
        for (int j = 0; j < 4; ++j) A(k, j) = A(j, k) = 0.0;
        A(k, k) = 1.0;
        rhs[k] = 0.0;
    }
}

bool solve4(cv::Matx44d A, cv::Vec4d rhs, PlaneForm f, PlaneModel& m) {
    pad_unused(A, rhs, f);
    if (std::abs(cv::determinant(A)) < kMinDet) return false;
    const cv::Vec4d x = A.solve(rhs, cv::DECOMP_LU);
    m = {x[0] / kScaleMm, x[1] / kScaleMm, x[2], x[3]};
    return true;
}

int count_inliers(const PlaneData& d, const PlaneModel& m, double tau, std::vector<uint8_t>* mask) {
    int n = 0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        const bool in = std::abs(d.z[i] - m.slope_z(d.u[i], d.w[i], d.dt[i]) - m.c) < tau;
        n += in ? 1 : 0;
        if (mask != nullptr) (*mask)[i] = in ? 1 : 0;
    }
    return n;
}

bool refit(const PlaneData& d, const std::vector<uint8_t>& in, PlaneForm f, PlaneModel& m) {
    cv::Matx44d AtA = cv::Matx44d::zeros();
    cv::Vec4d Atz(0, 0, 0, 0);
    for (std::size_t i = 0; i < d.size(); ++i) {
        if (in[i] == 0) continue;
        const auto r = row(d, i, f);
        for (int a = 0; a < 4; ++a) {
            Atz[a] += r[static_cast<std::size_t>(a)] * d.z[i];
            for (int b = 0; b < 4; ++b) AtA(a, b) += r[static_cast<std::size_t>(a)] * r[static_cast<std::size_t>(b)];
        }
    }
    return solve4(AtA, Atz, f, m);
}

// Mat song song thu 2 tren diem ngoai: RANSAC 1 chieu tren do lech c.
void second_plane(const PlaneData& d, int iters, double tau, std::mt19937_64& rng, PlaneFit& out) {
    std::vector<double> off;
    for (std::size_t i = 0; i < d.size(); ++i)
        if (out.inliers[i] == 0) off.push_back(d.z[i] - out.m.slope_z(d.u[i], d.w[i], d.dt[i]));
    if (off.empty()) return;
    double best_c = kNaN;
    int best_n = 0;
    for (int k = 0; k < iters; ++k) {
        const double cand = off[static_cast<std::size_t>(rng() % off.size())];
        if (std::abs(cand - out.m.c) < 2.0 * tau) continue;
        const auto n = static_cast<int>(std::count_if(off.begin(), off.end(), [&](double o) { return std::abs(o - cand) < tau; }));
        if (n > best_n) best_n = n, best_c = cand;
    }
    if (best_n == 0) return;
    std::vector<double> sel;
    for (const double o : off)
        if (std::abs(o - best_c) < tau) sel.push_back(o);
    out.n2 = static_cast<int>(sel.size());
    out.c2 = median_inplace(std::span<double>(sel));
}

}  // namespace

void ransac_plane(const PlaneData& d, PlaneForm form, int iters, double tau_mm, double max_slope,
                  std::mt19937_64& rng, PlaneFit& out) {
    out = PlaneFit{};
    const std::size_t n = d.size();
    const std::size_t k = 2 + (form.use_b ? 1U : 0U) + (form.use_e ? 1U : 0U);
    out.inliers.assign(n, 0);
    if (n < k) return;
    const std::array<int, 4> cols{0, form.use_b ? 1 : -1, 2, form.use_e ? 3 : -1};
    int best_n = -1;
    PlaneModel best;
    std::array<std::size_t, 4> idx{};
    for (int it = 0; it < iters; ++it) {
        for (std::size_t j = 0; j < k; ++j) {  // k diem khac nhau
            do idx[j] = static_cast<std::size_t>(rng() % n);
            while (std::find(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(j), idx[j]) !=
                   idx.begin() + static_cast<std::ptrdiff_t>(j));
        }
        cv::Matx44d A = cv::Matx44d::eye();
        cv::Vec4d rhs(0, 0, 0, 0);
        std::size_t rr = 0;
        for (const int c : cols) {  // hang cua he = cac phuong trinh mau, dat vao vi tri cot dung
            if (c < 0) continue;
            const auto r = row(d, idx[rr], form);
            for (int j = 0; j < 4; ++j) A(c, j) = r[static_cast<std::size_t>(j)];
            rhs[c] = d.z[idx[rr]];
            ++rr;
        }
        PlaneModel m;
        if (!solve4(A, rhs, form, m) || std::hypot(m.a, m.b) > max_slope) continue;
        const int cnt = count_inliers(d, m, tau_mm, nullptr);
        if (cnt > best_n) best_n = cnt, best = m;  // bang nhau: giu gia thuyet truoc
    }
    if (best_n <= 0) return;
    count_inliers(d, best, tau_mm, &out.inliers);
    PlaneModel fitted = best;
    if (refit(d, out.inliers, form, fitted) && std::hypot(fitted.a, fitted.b) <= max_slope) best = fitted;
    out.m = best;
    out.n_inliers = count_inliers(d, best, tau_mm, &out.inliers);
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (out.inliers[i] == 0) continue;
        const double r = d.z[i] - best.slope_z(d.u[i], d.w[i], d.dt[i]) - best.c;
        ss += r * r;
    }
    out.rms_mm = out.n_inliers > 0 ? std::sqrt(ss / out.n_inliers) : kNaN;
    second_plane(d, iters, tau_mm, rng, out);
}

double median_offset_by_scan(const PlaneData& d, const PlaneFit& fit) {
    std::vector<std::pair<uint16_t, double>> off;  // (vong, z - nghieng)
    for (std::size_t i = 0; i < d.size(); ++i)
        if (fit.inliers[i] != 0) off.emplace_back(d.scan[i], d.z[i] - fit.m.slope_z(d.u[i], d.w[i], d.dt[i]));
    if (off.empty()) return kNaN;
    std::stable_sort(off.begin(), off.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    std::vector<double> per_scan, one;
    for (std::size_t i = 0; i < off.size();) {
        one.clear();
        const uint16_t s = off[i].first;
        for (; i < off.size() && off[i].first == s; ++i) one.push_back(off[i].second);
        per_scan.push_back(median_inplace(std::span<double>(one)));
    }
    return median_inplace(std::span<double>(per_scan));
}

}  // namespace uavp
