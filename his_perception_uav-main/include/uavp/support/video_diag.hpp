// support / video_diag: chan doan rung / mo video khi bay (planA). Chi ghi log, khong anh huong pipeline.
// DiagFrameSource boc camera (IFrameSource): moi khung lay tu camera -> do roi tra ve nguyen anh. Khong phai sua cac
// vong lap (cho lenh, chon vat, bam vat): moi khung o moi pha deu duoc do, ke ca luc tren san chua / da arm.
// Moi khung (video_diag.csv):
//   dt_ms                 khoang cach 2 khung (SensorTimestamp): rot khung / giat
//   exposure_us, gain     phoi sang dai -> mo theo chuyen dong; gain cao -> nhieu lam giam do net gia
//   sharp, grad_x/y       do net: phuong sai Laplacian, nang luong gradient x / y (o vuong giua anh, kenh G).
//                         Mo do rung mot huong -> gradient theo huong do giam manh hon huong kia.
//   dx/dy_px, pc_resp     anh dich giua 2 khung (phase correlation, px anh goc), do tin cay
//   skew_x/y_px           dich dai TREN - dai DUOI anh: rolling shutter doc tung hang trong ~20-30 ms -> rung
//                         tan so cao trong luc doc lam 2 dai dich khac nhau ("jello")
//   wx, wy, wz            toc do quay camera (rad/s, he camera) tu goc ATTITUDE luc t +- gyro_half_window_s
//   blur_pred_px          mo du doan tu chuyen dong quay doc duoc: |(wx, wy)| f t_phoisang
//   rot_dx/dy_px          dich anh du doan tu goc quay giua 2 khung (diem o xa, tam anh)
//   armed, landed         trang thai FC (landed: 1 san, 2 bay, 3 cat canh, 4 ha canh)
// So sanh: san tat motor / san da arm (motor quay, khong bay) / dang bay -> tach rung co khi (motor, canh quat ->
// gia camera) khoi lac than (dieu khien tu the) va troi (vi tri). Phan tich: planA/video_diag.py.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/sources/frame_source.hpp"
#include "uavp/sources/mavlink.hpp"
#include "uavp/tracking/world_pose.hpp"

namespace uavp {

// --- Ham thuan (test duoc) ------------------------------------------------------------------------------------
struct SharpStats {
    double sharp{0.0};   // phuong sai Laplacian
    double grad_x{0.0};  // trung binh (d/dx)^2 (Sobel)
    double grad_y{0.0};
    double luma{0.0};    // trung binh kenh G
};
// O vuong crop_px x crop_px giua anh RGB (kenh G).
[[nodiscard]] SharpStats sharp_stats(const cv::Mat& image_rgb, int crop_px);

struct ImageShift {
    double dx_px{0.0}, dy_px{0.0};      // noi dung anh dich (+x: sang phai, +y: xuong) tu khung truoc -> khung nay
    double response{0.0};
    double skew_x_px{0.0}, skew_y_px{0.0};  // dich dai tren - dich dai duoi
};
// Do dich giua 2 khung tren anh thu nho; giu anh truoc ben trong.
class ShiftMeter {
public:
    explicit ShiftMeter(int width_px) : width_px_(width_px) {}
    // false o khung dau (chua co khung truoc). Ket qua theo px anh goc.
    bool update(const cv::Mat& image_rgb, ImageShift& out);
    void reset() { prev_.release(); }

private:
    int width_px_;
    double scale_{1.0};  // anh nho / anh goc
    cv::Mat prev_, win_, win_strip_;
};

// Toc do quay camera (rad/s, he camera) tu 2 ma tran R_WC cach nhau dt_s: log(R0^T R1) / dt.
[[nodiscard]] cv::Vec3d angular_rate(const cv::Matx33d& R_WC0, const cv::Matx33d& R_WC1, double dt_s);
// Dich anh (px) cua diem o xa tren truc quang khi camera quay tu R_WC0 sang R_WC1.
[[nodiscard]] cv::Point2d rotation_shift_px(const cv::Matx33d& R_WC0, const cv::Matx33d& R_WC1, const CameraModel& cam);
// Mo du doan: |(wx, wy)| f t_phoisang (quay quanh truc quang wz gan nhu khong lam mo tam anh).
[[nodiscard]] double blur_pred_px(const cv::Vec3d& w_C, double exposure_us, double f_px) noexcept;

// --- Boc camera ---------------------------------------------------------------------------------------------------
class DiagFrameSource : public IFrameSource {
public:
    DiagFrameSource(IFrameSource& inner, const VideoDiagConfig& cfg, const CameraModel& camera, const FcPoseBuffer* fc,
                    const BodyCamera& body_camera);
    ~DiagFrameSource() override;
    DiagFrameSource(const DiagFrameSource&) = delete;
    DiagFrameSource& operator=(const DiagFrameSource&) = delete;

    [[nodiscard]] bool next_frame(cv::Mat& image_rgb, FrameMeta& meta, double timeout_s) override;
    // File csv (thu muc ket qua co sau khi mo camera): cac dong da do truoc do duoc ghi khi flush.
    void set_output(const std::filesystem::path& csv);
    void flush();
    [[nodiscard]] int frames() const noexcept { return frames_; }

private:
    void measure(const cv::Mat& image_rgb, const FrameMeta& meta);
    void log_summary(int64_t now_ns);

    IFrameSource& inner_;
    VideoDiagConfig cfg_;
    CameraModel camera_;
    const FcPoseBuffer* fc_;
    BodyCamera body_camera_;
    ShiftMeter shift_;
    std::filesystem::path out_;
    std::string rows_;
    bool header_written_{false};
    int frames_{0}, counter_{0};
    int64_t prev_t_ns_{0};
    bool has_prev_R_{false};
    cv::Matx33d prev_R_WC_;
    int64_t flushed_ns_{0}, logged_ns_{0};
    // tom tat tu lan log truoc
    int sum_n_{0};
    double sum_sharp_{0.0}, sum_shift2_{0.0}, sum_w2_{0.0}, sum_blur_{0.0}, sum_dt_{0.0}, max_dt_{0.0}, sum_ms_{0.0};
};

}  // namespace uavp
