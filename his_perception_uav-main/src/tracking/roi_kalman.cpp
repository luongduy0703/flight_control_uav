#include "uavp/tracking/roi_kalman.hpp"

#include <cmath>

namespace uavp {

void RoiKalman::init(cv::Point2d center_px, int64_t t_ns) noexcept {
    const double pv = init_vel_ * init_vel_;
    x_[0] = {center_px.x, 0.0, 1.0, 0.0, pv};
    x_[1] = {center_px.y, 0.0, 1.0, 0.0, pv};
    t_ns_ = t_ns;
}

// Mo hinh van toc khong doi, nhieu gia toc trang (accel px/s^2).
void RoiKalman::predict_axis(Axis& a, double dt, double ctrl, double q) noexcept {
    a.p += a.v * dt + ctrl;
    const double dt2 = dt * dt;
    const double P00 = a.P00 + 2.0 * dt * a.P01 + dt2 * a.P11 + q * dt2 * dt2 / 4.0;
    const double P01 = a.P01 + dt * a.P11 + q * dt2 * dt / 2.0;
    a.P11 += q * dt2;
    a.P00 = P00;
    a.P01 = P01;
}

void RoiKalman::update_axis(Axis& a, double z, double r) noexcept {
    const double s = a.P00 + r;
    const double k0 = a.P00 / s, k1 = a.P01 / s;
    const double y = z - a.p;
    a.p += k0 * y;
    a.v += k1 * y;
    const double P00 = (1.0 - k0) * a.P00;
    const double P01 = (1.0 - k0) * a.P01;
    a.P11 -= k1 * a.P01;
    a.P00 = P00;
    a.P01 = P01;
}

void RoiKalman::predict(int64_t t_ns, cv::Point2d control_px) noexcept {
    const double dt = std::max(0.0, static_cast<double>(t_ns - t_ns_) / 1e9);
    const double q = accel_ * accel_;
    predict_axis(x_[0], dt, control_px.x, q);
    predict_axis(x_[1], dt, control_px.y, q);
    t_ns_ = t_ns;
}

void RoiKalman::update(cv::Point2d meas_px, double meas_sigma_px) noexcept {
    const double r = meas_sigma_px * meas_sigma_px;
    update_axis(x_[0], meas_px.x, r);
    update_axis(x_[1], meas_px.y, r);
}

}  // namespace uavp
