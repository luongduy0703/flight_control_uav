// Gui anh / LiDAR / trang thai sang may tinh, nhan lenh s / q (xem remote_view.hpp).
#include "uavp/support/remote_view.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "uavp/common/common.hpp"
#include "uavp/common/log.hpp"

namespace uavp {
namespace {

constexpr const char* kModule = "uavp.remote";
constexpr std::size_t kMaxQueued = 64;
constexpr int kPollMs = 20;

template <typename T>
void put(std::vector<uint8_t>& b, T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);  // Pi / PC little-endian
    b.insert(b.end(), p, p + sizeof(T));
}

std::vector<uint8_t> frame_msg(char type, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> m;
    m.reserve(payload.size() + 5);
    m.push_back(static_cast<uint8_t>(type));
    put(m, static_cast<uint32_t>(payload.size()));
    m.insert(m.end(), payload.begin(), payload.end());
    return m;
}

}  // namespace

RemoteView::RemoteView(const RemoteConfig& cfg) : cfg_(cfg), fps_(cfg.max_fps) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error(std::string("remote: socket loi: ") + std::strerror(errno));
    const int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(cfg.port));
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || ::listen(listen_fd_, 1) != 0) {
        ::close(listen_fd_);
        throw std::runtime_error("remote: khong mo duoc cong TCP " + std::to_string(cfg.port) + " (" + std::strerror(errno) +
                                 "): chuong trinh khac dang dung cong?");
    }
    thread_ = std::thread([this] { loop(); });
    UAVP_LOG_INFO(kModule, "Cho may tinh ket noi: python tools/remote_viewer.py --host <ip-pi> --port %d", cfg.port);
}

RemoteView::~RemoteView() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    if (client_fd_ >= 0) ::close(client_fd_);
    if (listen_fd_ >= 0) ::close(listen_fd_);
}

int RemoteView::poll_key() {
    std::lock_guard lock(mutex_);
    if (keys_.empty()) return -1;
    const int k = keys_.front();
    keys_.pop_front();
    return k;
}

bool RemoteView::due() const noexcept {
    return connected_.load() && fps_ > 0.0 && static_cast<double>(mono_ns() - last_frame_ns_) / 1e9 >= 1.0 / fps_;
}

void RemoteView::send_frame(const cv::Mat& image_rgb, int64_t t_ns, std::span<const cv::Point2f> lidar_uv_px,
                            std::span<const float> lidar_z_mm, bool force) {
    if (!connected_.load() || (!force && !due())) return;
    last_frame_ns_ = mono_ns();
    scale_ = static_cast<double>(cfg_.preview_width_px) / image_rgb.cols;
    cv::Mat small;
    cv::resize(image_rgb, small, cv::Size(), scale_, scale_, cv::INTER_AREA);  // ~2 ms tren Pi 5, chi khi den luot
    std::vector<uint8_t> lidar;
    put(lidar, t_ns);
    put(lidar, static_cast<uint32_t>(lidar_uv_px.size()));
    for (std::size_t i = 0; i < lidar_uv_px.size(); ++i) {
        put(lidar, static_cast<float>(lidar_uv_px[i].x * scale_));
        put(lidar, static_cast<float>(lidar_uv_px[i].y * scale_));
        put(lidar, lidar_z_mm[i]);
    }
    std::lock_guard lock(mutex_);
    pending_rgb_ = std::move(small);
    pending_t_ns_ = t_ns;
    pending_lidar_ = std::move(lidar);
    has_pending_ = true;
}

void RemoteView::push(char type, std::vector<uint8_t> payload) {
    if (!connected_.load()) return;
    std::lock_guard lock(mutex_);
    if (queue_.size() >= kMaxQueued) queue_.pop_front();  // may tinh cham: bo tin cu nhat
    queue_.push_back(frame_msg(type, payload));
}

void RemoteView::send_status(const std::string& json) { push('S', std::vector<uint8_t>(json.begin(), json.end())); }
void RemoteView::send_result(const std::string& json) { push('R', std::vector<uint8_t>(json.begin(), json.end())); }

bool RemoteView::send_all(const std::vector<uint8_t>& msg) {
    std::size_t off = 0;
    while (off < msg.size()) {
        const ssize_t n = ::send(client_fd_, msg.data() + off, msg.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return false;
        off += static_cast<std::size_t>(n);
    }
    return true;
}

void RemoteView::serve_client() {
    pollfd pfd{client_fd_, POLLIN, 0};
    std::vector<uint8_t> jpeg;
    while (running_) {
        if (::poll(&pfd, 1, kPollMs) > 0) {
            char buf[16];
            const ssize_t n = ::recv(client_fd_, buf, sizeof(buf), 0);
            if (n <= 0) return;  // may tinh ngat
            std::lock_guard lock(mutex_);
            for (ssize_t i = 0; i < n; ++i) keys_.push_back(buf[i]);
        }
        cv::Mat img;
        int64_t t_ns = 0;
        std::vector<uint8_t> lidar;
        std::deque<std::vector<uint8_t>> msgs;
        {
            std::lock_guard lock(mutex_);
            if (has_pending_) std::swap(img, pending_rgb_), std::swap(lidar, pending_lidar_), t_ns = pending_t_ns_;
            has_pending_ = false;
            std::swap(msgs, queue_);
        }
        for (const auto& m : msgs)
            if (!send_all(m)) return;
        if (img.empty()) continue;
        cv::Mat bgr;
        cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);  // imencode can BGR
        std::vector<uint8_t> payload;
        put(payload, t_ns);
        cv::imencode(".jpg", bgr, jpeg, {cv::IMWRITE_JPEG_QUALITY, cfg_.jpeg_quality});
        payload.insert(payload.end(), jpeg.begin(), jpeg.end());
        if (!send_all(frame_msg('J', payload)) || !send_all(frame_msg('L', lidar))) return;
    }
}

void RemoteView::loop() {
    pollfd pfd{listen_fd_, POLLIN, 0};
    while (running_) {
        if (::poll(&pfd, 1, kPollMs * 5) <= 0) continue;
        client_fd_ = ::accept(listen_fd_, nullptr, nullptr);
        if (client_fd_ < 0) continue;
        const int one = 1;
        ::setsockopt(client_fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        UAVP_LOG_INFO(kModule, "May tinh da ket noi");
        connected_ = true;
        serve_client();
        connected_ = false;
        ::close(client_fd_);
        client_fd_ = -1;
        UAVP_LOG_WARN(kModule, "May tinh ngat ket noi (cho ket noi lai)");
    }
}

}  // namespace uavp
