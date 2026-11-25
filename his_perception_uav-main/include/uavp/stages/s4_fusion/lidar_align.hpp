// s4 fusion / lidar_align: dieu kien "LiDAR quet qua vat" truoc khi tich luy duong quet cho mat top.
//   Vong quet LD06 moi nhat -> he camera (T_CL) -> chieu len anh (co meo) = duong LiDAR tren anh (polyline).
//   e_px = khoang cach vuong goc tu tam vat (tam bbox / mask, px) toi duong LiDAR.
//   e_mm = e_px * d / f_x, d = do sau (z_C) cua diem LiDAR tai chan duong vuong goc.
//   Dat khi e_mm <= e_max_mm. delta_C_mm: vector tu diem LiDAR do -> tam vat (cung do sau d, he camera); UAV dich
//   ngang theo R_WC delta_C thi duong quet di qua tam vat (planA dung de chinh vi tri).
#pragma once

#include <span>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

struct LidarAlign {
    bool valid{false};       // co duong LiDAR tren anh (>= 1 diem nhin thay)
    bool ok{false};          // valid && e_mm <= e_max_mm
    double e_px{kNaN};
    double e_mm{kNaN};
    double d_mm{kNaN};
    cv::Point2d foot_px{kNaN, kNaN};          // chan duong vuong goc tren duong LiDAR
    cv::Vec3d foot_C_mm{kNaN, kNaN, kNaN};    // diem LiDAR (noi suy) tai chan
    cv::Vec3d delta_C_mm{kNaN, kNaN, kNaN};   // foot -> tam vat o do sau d (he camera)
    int n_visible{0};
};

// points_C_mm: diem 1 vong quet theo thu tu goc (scan_to_camera_mm). 2 diem lien tiep cach nhau > max_gap_px tren
// anh (vd mep vat -> san) khong noi thanh doan: chi tinh khoang cach toi tung diem.
[[nodiscard]] LidarAlign lidar_alignment(std::span<const cv::Point3f> points_C_mm, cv::Point2d center_px,
                                         const CameraModel& camera, int width_px, int height_px, double max_gap_px,
                                         double e_max_mm);

}  // namespace uavp
