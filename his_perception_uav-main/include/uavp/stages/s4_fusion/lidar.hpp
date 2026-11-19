// s4 fusion / lidar: vong quet LD06 -> diem he camera (T_CL), chieu len anh (co meo), loc diem trong mask.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

// --- LiDAR -> he camera ---------------------------------------------------------------------------------------
// Goc tho LD06 -> theta he L (rad, nguoc chieu kim dong ho tu truc x), trong [-pi, pi).
[[nodiscard]] double lidar_angle_rad(double angle_ld06_deg, bool clockwise, double offset_deg) noexcept;

// Scan -> diem he C (mm) + timestamp, da loc (tam, cuong do, goc quet, z_C > 0). Ghi them vao cuoi out.
void scan_to_camera_mm(const Scan& scan, const Extrinsics& ext, const FusionConfig& cfg,
                       std::vector<cv::Point3f>& points_C_mm, std::vector<int64_t>& t_ns);

// --- Chieu len anh ----------------------------------------------------------------------------------------------
void eroded_mask(const SegMask& seg, int erode_px, cv::Mat& out);
void project_px(std::span<const cv::Point3f> points_C_mm, const CameraModel& camera, std::vector<cv::Point2f>& uv_px);
// Ban kinh chuan hoa lon nhat cua 4 goc o chu nhat tren anh (da khu meo).
[[nodiscard]] double box_max_norm_radius(double x0_px, double y0_px, double w_px, double h_px, const CameraModel& camera);
// (u, v) cac diem nhin thay tren anh (z > 0, khong bi meo gap, trong anh).
void visible_px(std::span<const cv::Point3f> points_C_mm, const CameraModel& camera, int width_px, int height_px,
                std::vector<cv::Point2f>& uv_px);
// keep[i] = 1 neu diem i chieu len anh roi vao pixel mask (bo diem ngoai tam nhin cua ROI truoc khi chieu).
void in_mask(std::span<const cv::Point3f> points_C_mm, const cv::Mat& mask, const SegMask& seg,
             const CameraModel& camera, std::vector<uint8_t>& keep);

}  // namespace uavp
