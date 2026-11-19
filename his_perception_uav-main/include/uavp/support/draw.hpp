// Ve anh kiem tra (RGB) giong src/uavp/draw.py: s1 detection, s2 mask, s3 trang thai track, man hinh cuoi s4,
// man hinh truc tiep. Chay o cho lanh (cuoi phien) hoac khi display.show = true.
#pragma once

#include <span>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

void draw_title(cv::Mat& canvas, std::span<const std::string> lines);
void draw_center(cv::Mat& canvas, cv::Point2d center_px);
[[nodiscard]] cv::Mat draw_detections(const cv::Mat& image, std::span<const Detection> detections,
                                      const std::vector<std::string>& classes, cv::Point2d center_px);
// s2: cac track cua nhom + SegObject tuong ung (nullptr = khong co mask).
[[nodiscard]] cv::Mat draw_segments(const cv::Mat& image, const std::vector<Track>& group,
                                    const std::vector<const SegObject*>& segs, const std::vector<std::string>& seg_classes,
                                    cv::Point2d center_px);
[[nodiscard]] cv::Mat draw_track_masks(const cv::Mat& image, const std::vector<Track>& tracks, cv::Point2d center_px);
[[nodiscard]] cv::Mat draw_debug(const cv::Mat& image, const std::vector<Track>& tracks, cv::Point2d center_px);
// Man hinh cuoi: duong bien mask, tam vat, diem LiDAR trong mask (xanh = inlier, do = outlier), chu.
[[nodiscard]] cv::Mat draw_final(const cv::Mat& image, const SegObject& obj, std::span<const cv::Point2f> uv_px,
                                 std::span<const uint8_t> inliers, std::span<const std::string> lines);
// Anh camera goc + mask vat to mau ban trong suot + vien + dau tam + toa do tam (px; p_obj, dist mm neu s4 co).
[[nodiscard]] cv::Mat draw_object(const cv::Mat& image, const SegObject& obj, const FusionResult& r);
// 3 truc cua vat (s4) tai tam mat top: x do, y luc, z (phap tuyen) lam; dai axis_mm.
void draw_pose_axes(cv::Mat& canvas, const FusionResult& r, const CameraModel& camera, double axis_mm);
// Man hinh truc tiep: mask cac track + duong quet LiDAR + (cx, cy) + chu.
[[nodiscard]] cv::Mat draw_live(const cv::Mat& image, std::span<const cv::Point2f> lidar_uv_px, cv::Point2d center_px,
                                std::span<const std::string> lines, const std::vector<Track>& tracks);

}  // namespace uavp
