// Xem / dieu khien uavp_real_time tu may tinh (tools/remote_viewer.py) qua TCP (run/real_time.yaml: muc remote).
// Pi -> may tinh, moi tin: [loai u8][do dai u32 LE][noi dung]
//   'J'  anh: int64 t_ns (SensorTimestamp) + JPEG (anh thu nho preview_width_px, RGB -> JPEG)
//   'L'  LiDAR: int64 t_ns + uint32 n + n x {float u, float v (px anh thu nho), float z_mm}
//   'S'  trang thai mot khung (JSON: pha, thoi gian tung buoc, do tre, track, ROI...)
//   'R'  ket qua cuoi (result.json)
// May tinh -> Pi: 1 byte 's' (bat dau quet) | 'q' (dung / thoat).
// Luong rieng: nhan ket noi, doc lenh, ma hoa JPEG va gui; luong chinh chi thu nho anh (gioi han max_fps) roi tra
// ngay. Ket noi cham: chi giu anh moi nhat (bo anh cu), tin trang thai giu toi da kMaxQueued.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

namespace uavp {

struct RemoteConfig {
    int port{8100};
    int preview_width_px{960};
    int jpeg_quality{70};
    double max_fps{10.0};   // luc chay (s1 -> s4): it anh -> it CPU, do thoi gian sach
    double wait_fps{15.0};  // luc cho SPACE (can vat): nhieu anh -> stream muot
    bool enabled{false};
};

class RemoteView {
public:
    explicit RemoteView(const RemoteConfig& cfg);
    ~RemoteView();
    RemoteView(const RemoteView&) = delete;
    RemoteView& operator=(const RemoteView&) = delete;

    [[nodiscard]] bool connected() const noexcept { return connected_.load(); }
    // Lenh moi tu may tinh ('s', 'q'), -1 neu khong co.
    [[nodiscard]] int poll_key();
    // Den luot gui (max_fps)? Goi truoc khi chuan bi anh / LiDAR de khong ton cong khi chua can.
    [[nodiscard]] bool due() const noexcept;
    // Anh RGB goc + diem LiDAR he camera (mm) da chieu len anh goc (uv_px) -> thu nho, xep hang gui.
    void send_frame(const cv::Mat& image_rgb, int64_t t_ns, std::span<const cv::Point2f> lidar_uv_px,
                    std::span<const float> lidar_z_mm, bool force = false);
    void send_status(const std::string& json);
    void send_result(const std::string& json);
    [[nodiscard]] double scale() const noexcept { return scale_; }
    // Doi tan so gui anh (wait_fps luc cho, max_fps luc chay).
    void set_fps(double fps) noexcept { fps_ = fps; }

private:
    void loop();
    void push(char type, std::vector<uint8_t> payload);
    bool send_all(const std::vector<uint8_t>& msg);
    void serve_client();

    RemoteConfig cfg_;
    int listen_fd_{-1};
    int client_fd_{-1};
    std::thread thread_;
    std::mutex mutex_;
    cv::Mat pending_rgb_;          // anh thu nho cho ma hoa (moi nhat)
    int64_t pending_t_ns_{0};
    bool has_pending_{false};
    std::vector<uint8_t> pending_lidar_;
    std::deque<std::vector<uint8_t>> queue_;  // tin S / R da dong goi
    std::deque<int> keys_;
    double scale_{0.5};
    double fps_{10.0};
    int64_t last_frame_ns_{0};
    std::atomic<bool> running_{true};
    std::atomic<bool> connected_{false};
};

}  // namespace uavp
