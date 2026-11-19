// Dung chung: dong ho, lam tron giong Python, thong ke, tensor vao / ra model, interface suy luan.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace cv {
class Mat;
}  // namespace cv

namespace uavp {

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr int64_t kNsPerS = 1'000'000'000;
inline constexpr int64_t kNsPerMs = 1'000'000;

// --- Dong ho --------------------------------------------------------------------------------------------------
// Dong ho chung camera / LiDAR: CLOCK_MONOTONIC (= time.monotonic_ns() cua Python tren Pi).
inline int64_t mono_ns() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * kNsPerS + ts.tv_nsec;
}

// Lech CLOCK_BOOTTIME - CLOCK_MONOTONIC (thoi gian may da suspend; tren Pi thuong 0).
inline int64_t boottime_minus_monotonic_ns() noexcept {
    timespec boot{};
    clock_gettime(CLOCK_BOOTTIME, &boot);
    const int64_t mono = mono_ns();
    return static_cast<int64_t>(boot.tv_sec) * kNsPerS + boot.tv_nsec - mono;
}

inline double ms_since(int64_t start_ns) noexcept { return static_cast<double>(mono_ns() - start_ns) / 1e6; }

// --- Lam tron giong Python round() / np.round: nua ve so chan (CPP_RULES.md muc 6) ----------------------------
inline int round_half_even(double x) noexcept { return static_cast<int>(std::nearbyint(x)); }

// Goc ve [0, period) giong toan tu % cua Python voi so thuc.
inline double py_mod(double x, double period) noexcept {
    const double r = std::fmod(x, period);
    return r < 0.0 ? r + period : r;
}

// --- Thong ke giong numpy -------------------------------------------------------------------------------------
// np.median: so phan tu chan -> trung binh hai phan tu giua. Sap xep tai cho `v` (buffer cua nguoi goi).
template <typename T>
double median_inplace(std::span<T> v) noexcept {
    if (v.empty()) return kNaN;
    if constexpr (std::is_floating_point_v<T>) {  // np.median: co NaN -> NaN
        for (const T x : v)
            if (std::isnan(x)) return kNaN;
    }
    const std::size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    const double upper = static_cast<double>(v[mid]);
    if (v.size() % 2 == 1) return upper;
    const double lower = static_cast<double>(*std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid)));
    return 0.5 * (lower + upper);
}

// np.percentile (noi suy tuyen tinh). Sap xep tai cho `v`.
inline double percentile_inplace(std::span<double> v, double q) noexcept {
    if (v.empty()) return kNaN;
    std::sort(v.begin(), v.end());
    const double pos = q / 100.0 * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (pos - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

// --- Bang uint8 -> [0, 1] (tinh luc bien dich, giong np.multiply(u8, 1/255) roi ep float32) ---------------
constexpr std::array<float, 256> make_unit_lut() {
    std::array<float, 256> t{};
    for (int i = 0; i < 256; ++i) t[static_cast<std::size_t>(i)] = static_cast<float>(i * (1.0 / 255.0));
    return t;
}
inline constexpr auto kU8ToUnit = make_unit_lut();

// --- Tensor vao / ra model ------------------------------------------------------------------------------------
// Input model: float32 CHW (c, h, w), cap phat mot lan, ghi lai moi lan chay.
struct InputCHW {
    std::vector<float> data;
    int c{0}, h{0}, w{0};

    void reset(int channels, int height, int width) {
        c = channels;
        h = height;
        w = width;
        data.assign(static_cast<std::size_t>(c) * static_cast<std::size_t>(h) * static_cast<std::size_t>(w), 0.f);
    }
    [[nodiscard]] float* plane(int k) noexcept {
        return data.data() + static_cast<std::size_t>(k) * static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
    }
};

// Output model: float32 lien tuc, shape (rows, cols) hoac (c, h, w). `data` duoc dung lai giua cac lan chay.
struct Tensor {
    std::vector<float> data;
    std::array<int, 3> shape{};  // 2 chieu: {rows, cols, 1}
    int dims{0};

    [[nodiscard]] std::size_t size() const noexcept { return data.size(); }
    [[nodiscard]] int rows() const noexcept { return shape[0]; }
    [[nodiscard]] int cols() const noexcept { return dims == 2 ? shape[1] : shape[1] * shape[2]; }
    [[nodiscard]] const float* row(int r) const noexcept {
        return data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols());
    }
};

// Letterbox anh -> input model: resize anh nguon ve resized_*, dat o (pad_x, pad_y) trong khung out_* to pad_value.
struct ImagePrep {
    int resized_w_px{0};
    int resized_h_px{0};
    int pad_x_px{0};
    int pad_y_px{0};
    int out_w_px{0};
    int out_h_px{0};
    int pad_value{0};
};

// Suy luan model (NCNN that hoac ket qua ghi san). Goi mot lan moi tile / ROI: ham ao chap nhan duoc
// (CPP_RULES.md muc 8). `outputs` sap theo ten output (out0, out1, ...).
class IInfer {
public:
    virtual ~IInfer() = default;
    virtual void run(const InputCHW& input, std::vector<Tensor>& outputs) = 0;
    // Chay thang tu anh uint8 RGB (view HWC, stride bat ky): resize bilinear + dem + chuan hoa [0, 1] ben trong.
    // false = khong ho tro (model gia trong test) -> nguoi goi tu fill_input_chw + run (run_letterboxed).
    virtual bool run_rgb(const cv::Mat& /*src_rgb*/, const ImagePrep& /*prep*/, std::vector<Tensor>& /*outputs*/) {
        return false;
    }
};

}  // namespace uavp
