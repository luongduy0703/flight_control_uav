// Cong chat luong khung TRUOC s1 (planA / uavp_real_time): khung chi duoc dua vao s1 khi qua 2 tang.
//
// Tang 1 - gyro (nguyen nhan vat ly, biet truoc khi doc pixel nao). Toc do quay camera w (he camera, rad/s):
//   d(t) = tich phan_{t0}^{t} w_perp dtau  (quy dao dich chuyen anh, px = f d)
//   nhoe  = f * be rong quy dao = f * hypot(ptp d_x, ptp d_y), ptp = max - min trong cua so phoi sang.
//   KHONG dung f |tich phan w| (dao dong tron chu ky -> 0 du anh nhoe) hay f tich phan |w| (di di lai lai tren cung
//   vet bi cong don). Vet nhoe = bao loi cua quy dao.
//   Meo rolling shutter: f * max|w_perp| * T_readout (lech hinh hoc co he thong, xau hon nhoe).
//   Quay quanh truc quang: r * ptp(theta_z) tai ban kinh r (goc anh khi xet ca khung).
//   Cua so phoi sang cua hang v: [t0 + (v / H) T_readout, + t_exp] (t0 = SensorTimestamp = hang dau bat dau phoi sang).
//   GIOI HAN: w lay tu ATTITUDE cua Pixhawk (~50 Hz) -> chi thay chuyen dong than (< ~20 Hz). Rung 100-200 Hz cua
//   motor nam ngoai Nyquist: tang 2 bat phan do.
//
// Tang 2 - do net tu anh (~1 ms): kenh G (thay mat Y) O DO PHAN GIAI GOC, it hang giua anh (KHONG thu nho anh: thu nho
//   xoa dung tan so cao can do). Chi so = trung binh (gx^2 + gy^2) voi sai phan ke nhau.
//   Moi chi so do net tron do nhoe voi do chi tiet cua canh -> KHONG dung nguong tuyet doi: so voi trung vi truot cua
//   chinh no tren `history` khung TRUOC (canh gan nhu khong doi giua cac khung lien tiep).
#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/common.hpp"
#include "uavp/common/config.hpp"
#include "uavp/sources/mavlink.hpp"
#include "uavp/stages/s4_fusion/world_plane.hpp"
#include "uavp/tracking/world_pose.hpp"

namespace uavp {

// Toc do quay camera (he camera C: x phai, y xuong, z truc quang), rad/s, khong doi tu t_ns toi mau ke tiep.
struct RateSample {
    int64_t t_ns{0};
    cv::Vec3d w_C{0.0, 0.0, 0.0};
};

// --- Tang 1: toan thuan ----------------------------------------------------------------------------------------
// Be rong quy dao goc (rad) trong [t_lo, t_hi]: (ptp theta_x, ptp theta_y, ptp theta_z), theta = tich phan w.
// w hang so tung doan -> theta tuyen tinh tung doan -> cuc tri o diem gay: xet diem gay + 2 dau cua so.
// nullopt neu mau khong phu het cua so.
[[nodiscard]] std::optional<cv::Vec3d> trajectory_width_rad(std::span<const RateSample> rates, int64_t t_lo_ns,
                                                            int64_t t_hi_ns);
// Nhoe (px): f * hypot(ptp theta_y, ptp theta_x)  (quay quanh y -> dich x, quanh x -> dich y).
[[nodiscard]] std::optional<double> blur_px(std::span<const RateSample> rates, int64_t t_lo_ns, int64_t t_hi_ns,
                                            double f_px);
// max |w_perp| (rad/s) trong [t_lo, t_hi].
[[nodiscard]] std::optional<double> max_perp_rate(std::span<const RateSample> rates, int64_t t_lo_ns, int64_t t_hi_ns);
// Toc do quay tu chuoi tu the camera R_WC (vd ATTITUDE x R_BC): w_i = log(R_i^T R_{i+1}) / dt, he camera.
[[nodiscard]] std::vector<RateSample> rates_from_poses(std::span<const CamPoseSample> poses);

// --- Tang 2 ---------------------------------------------------------------------------------------------------
// Trung binh (gx^2 + gy^2) kenh G tren `rows` cap hang deu nhau trong vung giua (ti le `band`), do phan giai goc.
[[nodiscard]] double strip_sharpness(const cv::Mat& image_rgb, int rows, double band);

// --- Ket qua + cong -------------------------------------------------------------------------------------------
struct GateResult {
    bool pass{false};
    bool gyro{false};            // co du tu the FC de xet tang 1
    double blur_px{kNaN};        // max tren hang tren / giua / duoi
    double rs_px{kNaN};
    double roll_px{kNaN};
    double sharp{kNaN};
    double sharp_median{kNaN};   // trung vi `history` khung truoc
    double sharp_ratio{kNaN};
    int history{0};
    std::string reason;          // "ok" | "blur" | "rolling_shutter" | "roll" | "no_gyro" | "sharp" | "warmup"
};

class FrameGate {
public:
    explicit FrameGate(const FrameGateConfig& cfg) : cfg_(cfg) {}
    // Xet mot khung. rates: toc do quay phu [t0, t0 + T_readout + t_exp] (rong: khong co FC). f_px: K(0,0).
    // Do net cua khung LUON vao lich su (ca khung bi loai) de trung vi theo kip canh.
    GateResult evaluate(const cv::Mat& image_rgb, int64_t t0_ns, double exposure_us, double f_px,
                        std::span<const RateSample> rates);
    [[nodiscard]] const FrameGateConfig& config() const noexcept { return cfg_; }

private:
    FrameGateConfig cfg_;
    std::deque<double> sharp_;  // `history` khung truoc
};

// Toc do quay camera trong [t_lo, t_hi] tu bo dem ATTITUDE (+ 1 mau truoc / sau). Rong neu khong co.
[[nodiscard]] std::vector<RateSample> camera_rates(const FcPoseBuffer& fc, const BodyCamera& bc, int64_t t_lo_ns,
                                                   int64_t t_hi_ns);

}  // namespace uavp
