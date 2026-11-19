// Mask -> tam, contour, minAreaRect (yaw). Giong mask_shape.py.
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "uavp/stages/s2_segment/mask_shape.hpp"

namespace uavp {

bool largest_contour_px(const SegMask& seg, std::vector<cv::Point>& contour_px) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(seg.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;
    std::size_t best = 0;  // max(): contour dau tien neu dien tich bang nhau
    double best_area = cv::contourArea(contours[0]);
    for (std::size_t i = 1; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area > best_area) {
            best_area = area;
            best = i;
        }
    }
    contour_px = std::move(contours[best]);
    for (auto& p : contour_px) p += cv::Point(seg.x0_px, seg.y0_px);
    return true;
}

RectSize min_area_rect(std::span<const cv::Point2f> points) {
    const cv::RotatedRect rect = cv::minAreaRect(std::vector<cv::Point2f>(points.begin(), points.end()));
    std::array<cv::Point2f, 4> corners{};
    rect.points(corners.data());  // cung thu tu voi cv2.boxPoints
    const cv::Point2f edge_a = corners[1] - corners[0];
    const cv::Point2f edge_b = corners[2] - corners[1];
    const float len_a = std::hypot(edge_a.x, edge_a.y);
    const float len_b = std::hypot(edge_b.x, edge_b.y);
    const cv::Point2f long_edge = len_a >= len_b ? edge_a : edge_b;
    const double yaw_deg = py_mod(std::atan2(static_cast<double>(long_edge.y), static_cast<double>(long_edge.x)) *
                                      180.0 / CV_PI, 180.0);
    return {std::min(len_a, len_b), std::max(len_a, len_b), yaw_deg};
}

bool seg_object(SegMask seg, SegObject& out) {
    const cv::Moments m = cv::moments(seg.mask, true);
    std::vector<cv::Point> contour;
    if (m.m00 < 0.5 || !largest_contour_px(seg, contour)) return false;  // m00 = so pixel mask (so nguyen)
    out.center_px = {m.m10 / m.m00 + seg.x0_px, m.m01 / m.m00 + seg.y0_px};
    out.area_px = m.m00;
    std::vector<cv::Point2f> contour_f(contour.begin(), contour.end());
    out.yaw_deg = min_area_rect(contour_f).yaw_deg;
    out.mask = std::move(seg);
    return true;
}

}  // namespace uavp
