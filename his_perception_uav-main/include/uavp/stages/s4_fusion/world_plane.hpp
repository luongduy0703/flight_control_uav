// s4 fusion / world_plane: huong mat top trong HE THE GIOI W (NED EKF Pixhawk, mm). Ham thuan, test duoc.
//   1. Dong bo + deskew: moi diem LiDAR i co t_i rieng trong vong quet ~100 ms. Tu the camera T_WC(t_i) noi suy giua
//      2 mau tu the FC: slerp quaternion cho goc quay, tuyen tinh cho vi tri.
//   2. Chieu + loc mask: p_W = T_WC(t_i) p_C (p_C = T_CL p_L, lidar.cpp; T_WC = T_WB T_BC). Dua p_W ve camera cua khung
//      co mask gan nhat (T_CW(t_f)), chieu len anh, giu diem trong mask da co erode_px (bo diem mep, bu sai hieu chinh).
//   3. Bu chuyen dong: trong W vat dung yen -> diem tu moi vong quet nam tren cung mot mat.
//   4. Tich luy trong cua so truot window_s (1-3 s) truoc khung co mask moi nhat.
//   5. RANSAC mat phang (mau 3 diem, nguong tau ~ 2 sigma LiDAR) -> PCA tren inlier: tam c, lambda1 >= lambda2 >=
//      lambda3; phap tuyen n = vector rieng cua lambda3.
//   6. Dieu kien (accept_plane): N >= min_inliers, sqrt(lambda2) >= min_sqrt_l2_mm, sigma_theta <= max_normal_sigma_deg
//      voi sigma_theta ~ sigma_perp / sqrt(N lambda2). Dat -> PLANE; khong -> HEIGHT_ONLY: n theo trong luc,
//      do cao = RANSAC 1D + median.
//   7. Tam mat top: tia qua tam mask (da khu meo) r_C = (x_n, y_n, 1) giao mat phang: p_C = (n.c / n.r) r.
//   8. Yaw + kich thuoc: chieu contour mask len mat phang, hinh chu nhat bao nho nhat trong toa do mat phang (m).
// Do trai: 1 vong quet = cac diem thang hang (lambda2 ~ 0, n quay tu do quanh duong quet). Trai chi co khi UAV nghieng
// / dich giua cac vong: duong quet dich delta ~ d tan(dphi) + dx; phu deu dai R -> sqrt(lambda2) ~ R / sqrt(12).
#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

// --- Tu the theo thoi gian ----------------------------------------------------------------------------------------
// Mot mau tu the camera (tu ATTITUDE + LOCAL_POSITION_NED / rangefinder, qua T_BC), dong ho chung mono_ns.
struct CamPoseSample {
    CamPose pose;
    int64_t t_ns{0};
};

// Quaternion (w, x, y, z), don vi.
[[nodiscard]] cv::Vec4d quat_from_matrix(const cv::Matx33d& R) noexcept;
[[nodiscard]] cv::Matx33d matrix_from_quat(const cv::Vec4d& q) noexcept;
// Slerp q0 -> q1 (duong ngan, w trong [0, 1]).
[[nodiscard]] cv::Vec4d slerp(const cv::Vec4d& q0, cv::Vec4d q1, double w) noexcept;
// Tu the camera luc t_ns: slerp goc quay + noi suy tuyen tinh vi tri giua 2 mau bao quanh (samples tang dan theo t).
// false neu t_ns ngoai [dau, cuoi] qua max_gap_ns hoac 2 mau bao quanh cach nhau > 2 max_gap_ns.
[[nodiscard]] bool interpolate_pose(std::span<const CamPoseSample> samples, int64_t t_ns, int64_t max_gap_ns,
                                    CamPose& out) noexcept;

// --- Mat phang: RANSAC 3 diem + PCA -------------------------------------------------------------------------------
struct PlanePca {
    cv::Vec3d c{kNaN, kNaN, kNaN};     // tam khoi inlier (mm)
    cv::Vec3d n{kNaN, kNaN, kNaN};     // phap tuyen (vector rieng cua lambda3), don vi
    std::array<double, 3> lambda_mm2{kNaN, kNaN, kNaN};  // lambda1 >= lambda2 >= lambda3 (hiep phuong sai, mm^2)
    std::vector<uint8_t> inliers;      // 0 / 1 theo diem dua vao
    int n_inliers{0};
};

// PCA tren cac diem co mask[i] != 0 (mask rong = moi diem). false neu < 3 diem.
[[nodiscard]] bool pca_plane(std::span<const cv::Vec3d> points_mm, std::span<const uint8_t> mask, PlanePca& out);
// RANSAC: mau 3 diem, inlier |n.(p - p0)| < tau_mm; bo gia thuyet phap tuyen lech `up` > max_tilt_deg. Fit lai PCA
// tren inlier, dem lai inlier voi mat PCA. n_inliers = 0 neu khong co gia thuyet hop le.
void ransac_pca_plane(std::span<const cv::Vec3d> points_mm, int iters, double tau_mm, const cv::Vec3d& up,
                      double max_tilt_deg, std::mt19937_64& rng, PlanePca& out);

struct PlaneQuality {
    double sqrt_l2_mm{kNaN};        // do trai vuong goc duong quet
    double sigma_perp_mm{kNaN};     // nhieu theo phap tuyen: max(sqrt(lambda3), lidar_sigma_mm)
    double sigma_theta_deg{kNaN};   // ~ sigma_perp / sqrt(N lambda2)
    double n_required{kNaN};        // N can de sigma_theta <= max_normal_sigma_deg
    bool accepted{false};
};
[[nodiscard]] PlaneQuality accept_plane(const PlanePca& p, const WorldPlaneConfig& cfg) noexcept;

// Do cao theo truc `up` (RANSAC 1D: |h - h_k| < tau_mm, roi median inlier). nan neu rong.
[[nodiscard]] double height_1d(std::span<const cv::Vec3d> points_mm, const cv::Vec3d& up, int iters, double tau_mm,
                               std::mt19937_64& rng, std::vector<uint8_t>& inliers);

// Tia r = (x_n, y_n, 1) giao mat phang (n, c) cung he: p = (n.c / n.r) r. false neu |n.r| < 1e-6 hoac p sau camera.
[[nodiscard]] bool ray_plane(const cv::Vec3d& ray, const cv::Vec3d& n, const cv::Vec3d& c, cv::Vec3d& p) noexcept;

// Contour mask (px, anh goc) -> mat phang (he camera) -> hinh chu nhat bao nho nhat trong toa do mat phang.
// Co so mat phang: e1 = truc x camera chieu len mat, e2 = n x e1 -> yaw (do, [0, 180)) cung quy uoc s3 khi mat song
// song mat anh. axis_x_C = huong canh dai (he camera). false neu contour / giao tia hong.
struct TopRect {
    cv::Vec3d axis_x_C{kNaN, kNaN, kNaN};
    double W_mm{kNaN}, L_mm{kNaN}, yaw_deg{kNaN};
};
[[nodiscard]] bool top_rect_on_plane(const SegMask& seg, const CameraModel& camera, const cv::Vec3d& n_C,
                                     const cv::Vec3d& c_C, TopRect& out);

// Ket qua (truc he camera tham chieu) -> he the gioi theo tu the camera luc khung tham chieu: dien p_obj_W_mm /
// normal_W neu con nan (method image), luon dien axis_x_W, yaw_W_deg = atan2(E, N) cua canh dai trong [0, 180).
// Khong lam gi neu pose khong co vi tri hoac ket qua chua co tam.
void fill_world(FusionResult& r, const CamPose& pose) noexcept;

}  // namespace uavp
