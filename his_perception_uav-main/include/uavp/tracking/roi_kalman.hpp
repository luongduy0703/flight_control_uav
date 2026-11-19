// Kalman nho cho tam ROI tren anh: trang thai (u, v, vu, vv) px, px/s. Hai truc doc lap (2 trang thai moi truc).
//   du doan: u += vu dt + du_ctrl   (du_ctrl = dich chuyen tam do tu the FC giua 2 khung: dau vao dieu khien)
//   do:      u_do tu optical flow / NCC (nhieu meas_px)
// vu chi con giu phan troi KHONG giai thich duoc boi tu the (gio canh quat day vat, tu the tre...).
#pragma once

#include <cmath>
#include <cstdint>

#include <opencv2/core.hpp>

namespace uavp {

class RoiKalman {
public:
    RoiKalman(double accel_px_s2, double init_vel_px_s) noexcept : accel_(accel_px_s2), init_vel_(init_vel_px_s) {}

    void init(cv::Point2d center_px, int64_t t_ns) noexcept;
    void predict(int64_t t_ns, cv::Point2d control_px) noexcept;
    void update(cv::Point2d meas_px, double meas_sigma_px) noexcept;

    [[nodiscard]] cv::Point2d center_px() const noexcept { return {x_[0].p, x_[1].p}; }
    [[nodiscard]] cv::Point2d vel_px_s() const noexcept { return {x_[0].v, x_[1].v}; }
    [[nodiscard]] double sigma_px() const noexcept { return std::sqrt(0.5 * (x_[0].P00 + x_[1].P00)); }

private:
    struct Axis {
        double p{0.0}, v{0.0};
        double P00{0.0}, P01{0.0}, P11{0.0};
    };
    static void predict_axis(Axis& a, double dt, double ctrl, double q) noexcept;
    static void update_axis(Axis& a, double z, double r) noexcept;

    Axis x_[2]{};
    double accel_;
    double init_vel_;
    int64_t t_ns_{0};
};

}  // namespace uavp
