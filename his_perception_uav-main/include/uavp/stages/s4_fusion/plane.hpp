// s4 fusion / plane: RANSAC mat phang top tren diem LiDAR da bu chuyen dong (ham thuan, test duoc).
// Toa do khung tham chieu: u doc mat quet LiDAR, w vuong goc mat quet (ca hai nam trong mat anh x-y), z truc quang.
//   z = a u + b w + c + e dt     (dt = t_diem - t_tham_chieu, s; e = troi do cao UAV, mm/s)
// b chi dung khi diem trai du theo w (UAV troi vuong goc mat quet); e chi dung khi bu chuyen dong bang anh (khong co
// tu the FC) va diem trai du lau.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "uavp/common/common.hpp"

namespace uavp {

struct PlaneForm {
    bool use_b{true};
    bool use_e{false};
};

struct PlaneModel {
    double a{0.0}, b{0.0}, c{kNaN}, e{0.0};
    [[nodiscard]] double slope_z(double u, double w, double dt) const noexcept { return a * u + b * w + e * dt; }
};

// Diem dua vao (SoA). scan: chi so vong quet (de lay median theo vong).
struct PlaneData {
    std::vector<double> u, w, z, dt;
    std::vector<uint16_t> scan;
    void clear() noexcept {
        u.clear();
        w.clear();
        z.clear();
        dt.clear();
        scan.clear();
    }
    [[nodiscard]] std::size_t size() const noexcept { return z.size(); }
};

struct PlaneFit {
    PlaneModel m;
    std::vector<uint8_t> inliers;
    double rms_mm{kNaN};
    double c2{kNaN};  // mat song song thu 2 (cung a, b, e), nan neu khong co
    int n_inliers{0};
    int n2{0};
};

// RANSAC (mau toi thieu 2-4 diem tuy form) + fit lai binh phuong toi thieu tren inlier; bo gia thuyet nghieng
// > atan(max_slope). Mat thu 2: tren diem con lai, cung do nghieng, lech c >= 2 tau.
void ransac_plane(const PlaneData& d, PlaneForm form, int iters, double tau_mm, double max_slope,
                  std::mt19937_64& rng, PlaneFit& out);

// c ben vung: moi vong quet median (z - a u - b w - e dt) tren inlier, roi median giua cac vong. LD06 lam tron tam
// theo bac ~15.6 mm; UAV troi lam cac vong roi vao bac khac nhau -> median giua cac vong min hon mot bac.
[[nodiscard]] double median_offset_by_scan(const PlaneData& d, const PlaneFit& fit);

}  // namespace uavp
