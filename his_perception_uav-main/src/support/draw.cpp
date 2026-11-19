#include "uavp/support/draw.hpp"

#include <array>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "uavp/stages/s3_select.hpp"

namespace uavp {
namespace {

// Mau RGB (anh pipeline RGB), tra theo chi so enum / class (bang constexpr, CPP_RULES.md muc 7).
using Rgb = std::array<double, 3>;
constexpr std::array<Rgb, static_cast<std::size_t>(State::kCount)> kStateColors{
    Rgb{255, 215, 0}, Rgb{0, 200, 0}, Rgb{230, 30, 30}};  // DEFER, OK, ABORT
constexpr std::array<Rgb, 2> kClassColors{Rgb{0, 160, 255}, Rgb{255, 0, 200}};
constexpr Rgb kWhiteRgb{255, 255, 255};
constexpr Rgb kFinalContourRgb{0, 255, 255};
constexpr Rgb kFinalCenterRgb{255, 0, 255};
constexpr Rgb kLidarRgb{0, 255, 0};
constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
constexpr double kMaskAlpha = 0.5;

cv::Scalar sc(const Rgb& c) { return {c[0], c[1], c[2]}; }
cv::Scalar state_color(State s) { return sc(kStateColors[static_cast<std::size_t>(s)]); }
cv::Scalar class_color(int class_id) { return sc(kClassColors[static_cast<std::size_t>(class_id) % kClassColors.size()]); }
cv::Point pt(cv::Point2d p) { return {round_half_even(p.x), round_half_even(p.y)}; }

cv::Mat paint_masks(const cv::Mat& image, const std::vector<std::pair<const SegObject*, cv::Scalar>>& objects) {
    cv::Mat overlay = image.clone();
    for (const auto& [obj, color] : objects) {
        const SegMask& m = obj->mask;
        const cv::Rect r(m.x0_px, m.y0_px, m.mask.cols, m.mask.rows);
        const cv::Rect in = r & cv::Rect(0, 0, image.cols, image.rows);  // mask da doi co the lo ra ngoai anh
        if (in.area() > 0) overlay(in).setTo(color, m.mask(in - r.tl()));
    }
    cv::Mat out;
    cv::addWeighted(overlay, kMaskAlpha, image, 1.0 - kMaskAlpha, 0.0, out);
    return out;
}

void draw_pose(cv::Mat& canvas, const SegObject& obj, const cv::Scalar& color, const std::string& label, int thickness) {
    const SegMask& m = obj.mask;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(canvas, contours, -1, color, thickness, cv::LINE_8, cv::noArray(), INT_MAX,
                     cv::Point(m.x0_px, m.y0_px));
    const double len = 0.5 * std::sqrt(obj.area_px);
    const double yaw = obj.yaw_deg * CV_PI / 180.0;
    const cv::Point c = pt(obj.center_px);
    cv::drawMarker(canvas, c, color, cv::MARKER_CROSS, 24, 2);
    cv::line(canvas, c, pt({obj.center_px.x + len * std::cos(yaw), obj.center_px.y + len * std::sin(yaw)}), color, 2);
    cv::putText(canvas, label, {m.x0_px, std::max(m.y0_px - 8, 20)}, kFont, 0.7, color, 2);
}

std::string track_label(const Track& t, const char* prefix) {
    char buf[128];
    const std::string_view reason = name(t.reason);
    std::snprintf(buf, sizeof(buf), "%s#%d %s W %.0fmm%s%s", prefix, t.track_id,
                  t.state == State::kOk ? "OK" : (t.state == State::kDefer ? "DEFER" : "ABORT"), median_W_mm(t),
                  reason.empty() ? "" : " ", std::string(reason).c_str());
    return buf;
}

}  // namespace

void draw_title(cv::Mat& canvas, std::span<const std::string> lines) {
    for (std::size_t i = 0; i < lines.size(); ++i)
        cv::putText(canvas, lines[i], {20, 40 + 40 * static_cast<int>(i)}, kFont, 1.0, sc(kWhiteRgb), 3);
}

void draw_center(cv::Mat& canvas, cv::Point2d center_px) {
    cv::drawMarker(canvas, pt(center_px), sc(kWhiteRgb), cv::MARKER_CROSS, 40, 2);
}

cv::Mat draw_detections(const cv::Mat& image, std::span<const Detection> detections,
                        const std::vector<std::string>& classes, cv::Point2d center_px) {
    cv::Mat canvas = image.clone();
    for (const auto& d : detections) {
        const cv::Scalar color = class_color(d.class_id);
        const cv::Point p1 = pt({d.x1, d.y1}), p2 = pt({d.x2, d.y2});
        cv::rectangle(canvas, p1, p2, color, 2);
        char label[64];
        std::snprintf(label, sizeof(label), "%s %.2f", classes.at(d.class_id).c_str(), static_cast<double>(d.conf));
        cv::putText(canvas, label, {p1.x, std::max(p1.y - 8, 20)}, kFont, 0.7, color, 2);
    }
    draw_center(canvas, center_px);
    const std::array<std::string, 1> title{"s1 detect: " + std::to_string(detections.size()) + " vat"};
    draw_title(canvas, title);
    return canvas;
}

cv::Mat draw_segments(const cv::Mat& image, const std::vector<Track>& group, const std::vector<const SegObject*>& segs,
                      const std::vector<std::string>& seg_classes, cv::Point2d center_px) {
    std::vector<std::pair<const SegObject*, cv::Scalar>> painted;
    std::string missing;
    for (std::size_t i = 0; i < group.size(); ++i) {
        if (segs[i] != nullptr) painted.emplace_back(segs[i], class_color(group[i].detection.class_id));
        else missing += "#" + std::to_string(group[i].track_id) + " ";
    }
    cv::Mat canvas = paint_masks(image, painted);
    for (std::size_t i = 0; i < group.size(); ++i) {
        const Detection& d = group[i].detection;
        cv::rectangle(canvas, pt({d.x1, d.y1}), pt({d.x2, d.y2}), class_color(d.class_id), 1);
        if (segs[i] == nullptr) continue;
        char label[96];
        std::snprintf(label, sizeof(label), "#%d %s %.2f yaw %.0f", group[i].track_id,
                      seg_classes.at(segs[i]->mask.class_id).c_str(), static_cast<double>(segs[i]->mask.conf),
                      segs[i]->yaw_deg);
        draw_pose(canvas, *segs[i], class_color(d.class_id), label, 2);
    }
    draw_center(canvas, center_px);
    const std::array<std::string, 2> title{"s2 segment: top " + std::to_string(group.size()) + " gan tam",
                                           "khong co mask: " + (missing.empty() ? std::string("-") : missing)};
    draw_title(canvas, title);
    return canvas;
}

cv::Mat draw_track_masks(const cv::Mat& image, const std::vector<Track>& tracks, cv::Point2d center_px) {
    const std::vector<int> ranked = rank_ok(tracks, center_px);
    std::vector<std::pair<const SegObject*, cv::Scalar>> painted;
    for (const auto& t : tracks)
        if (const SegObject* s = last_seg(t)) painted.emplace_back(s, state_color(t.state));
    cv::Mat canvas = paint_masks(image, painted);
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        const SegObject* s = last_seg(tracks[i]);
        if (s == nullptr) continue;
        const bool selected = !ranked.empty() && ranked.front() == static_cast<int>(i);
        const bool waiting = !selected && std::find(ranked.begin(), ranked.end(), static_cast<int>(i)) != ranked.end();
        draw_pose(canvas, *s, state_color(tracks[i].state),
                  track_label(tracks[i], selected ? "SELECTED " : (waiting ? "WAITING " : "")), selected ? 5 : 2);
    }
    return canvas;
}

cv::Mat draw_debug(const cv::Mat& image, const std::vector<Track>& tracks, cv::Point2d center_px) {
    const std::vector<int> ranked = rank_ok(tracks, center_px);
    cv::Mat canvas = draw_track_masks(image, tracks, center_px);
    draw_center(canvas, center_px);
    std::string waiting, defer, abort;
    for (std::size_t i = 1; i < ranked.size(); ++i) waiting += std::to_string(tracks[static_cast<std::size_t>(ranked[i])].track_id) + " ";
    for (const auto& t : tracks) {
        if (t.state == State::kDefer) defer += std::to_string(t.track_id) + " ";
        if (t.state == State::kAbort) abort += "(" + std::to_string(t.track_id) + " " + std::string(name(t.reason)) + ") ";
    }
    const std::string sel = ranked.empty() ? "None" : "#" + std::to_string(tracks[static_cast<std::size_t>(ranked[0])].track_id);
    const std::array<std::string, 2> title{"s3 select: selected " + sel + " | waiting [" + waiting + "]",
                                           "defer [" + defer + "] | abort [" + abort + "]"};
    draw_title(canvas, title);
    return canvas;
}

namespace {

// Mask to mau ban trong suot + duong vien + dau tam vat (anh goc, RGB).
cv::Mat overlay_object(const cv::Mat& image, const SegObject& obj) {
    cv::Mat canvas = paint_masks(image, {{&obj, sc(kFinalContourRgb)}});
    const SegMask& m = obj.mask;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(canvas, contours, -1, sc(kFinalContourRgb), 3, cv::LINE_8, cv::noArray(), INT_MAX,
                     cv::Point(m.x0_px, m.y0_px));
    const cv::Point c = pt(obj.center_px);
    cv::drawMarker(canvas, c, sc(kFinalCenterRgb), cv::MARKER_CROSS, 40, 3);
    cv::circle(canvas, c, 6, sc(kFinalCenterRgb), -1);
    return canvas;
}

}  // namespace

cv::Mat draw_final(const cv::Mat& image, const SegObject& obj, std::span<const cv::Point2f> uv_px,
                   std::span<const uint8_t> inliers, std::span<const std::string> lines) {
    cv::Mat canvas = overlay_object(image, obj);
    for (std::size_t i = 0; i < uv_px.size(); ++i)
        cv::circle(canvas, pt(uv_px[i]), 4, inliers[i] != 0 ? state_color(State::kOk) : state_color(State::kAbort), -1);
    draw_title(canvas, lines);
    return canvas;
}

cv::Mat draw_object(const cv::Mat& image, const SegObject& obj, const FusionResult& r) {
    cv::Mat canvas = overlay_object(image, obj);
    char line[160];
    std::snprintf(line, sizeof(line), "tam (%.1f, %.1f) px", obj.center_px.x, obj.center_px.y);
    std::vector<std::string> text{line};
    if (std::isfinite(r.p_obj_mm.z)) {
        std::snprintf(line, sizeof(line), "p_obj (%.1f, %.1f, %.1f) mm | dist %.1f mm", r.p_obj_mm.x, r.p_obj_mm.y,
                      r.p_obj_mm.z, r.dist_mm);
        text.emplace_back(line);
    }
    const cv::Point c = pt(obj.center_px);
    for (std::size_t k = 0; k < text.size(); ++k) {  // chu canh tam vat (vien den de doc tren moi nen)
        const cv::Point at(c.x + 70, c.y - 10 + 34 * static_cast<int>(k));  // ben phai tam, tranh 3 truc
        cv::putText(canvas, text[k], at, kFont, 0.9, sc({0, 0, 0}), 5, cv::LINE_AA);
        cv::putText(canvas, text[k], at, kFont, 0.9, sc(kWhiteRgb), 2, cv::LINE_AA);
    }
    return canvas;
}

cv::Mat draw_live(const cv::Mat& image, std::span<const cv::Point2f> lidar_uv_px, cv::Point2d center_px,
                  std::span<const std::string> lines, const std::vector<Track>& tracks) {
    cv::Mat canvas = draw_track_masks(image, tracks, center_px);
    for (const auto& uv : lidar_uv_px) cv::circle(canvas, pt(uv), 3, sc(kLidarRgb), -1);
    draw_center(canvas, center_px);
    draw_title(canvas, lines);
    return canvas;
}

void draw_pose_axes(cv::Mat& canvas, const FusionResult& r, const CameraModel& camera, double axis_mm) {
    if (!std::isfinite(r.p_obj_mm.z) || !std::isfinite(r.axis_x[0])) return;
    const cv::Point3d c = r.p_obj_mm;
    const std::vector<cv::Point3d> pts{c, c + axis_mm * cv::Point3d(r.axis_x), c + axis_mm * cv::Point3d(r.axis_y),
                                       c + axis_mm * cv::Point3d(r.axis_z)};
    std::vector<cv::Point2d> uv;
    cv::projectPoints(pts, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera.K_px, camera.dist_coeffs, uv);
    constexpr std::array<Rgb, 3> kAxisRgb{Rgb{255, 0, 0}, Rgb{0, 255, 0}, Rgb{0, 120, 255}};  // x do, y luc, z lam
    for (std::size_t k = 0; k < 3; ++k) cv::arrowedLine(canvas, pt(uv[0]), pt(uv[k + 1]), sc(kAxisRgb[k]), 4, cv::LINE_AA, 0, 0.2);
}

}  // namespace uavp
