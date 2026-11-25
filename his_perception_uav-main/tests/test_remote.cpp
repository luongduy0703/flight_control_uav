// RemoteView: may tinh ket noi, gui phim, nhan anh JPEG + LiDAR + trang thai dung khung tin.
#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <thread>

#include "uavp/support/remote_view.hpp"

using namespace uavp;

namespace {

bool recv_n(int fd, std::vector<uint8_t>& out, std::size_t n) {
    out.resize(n);
    std::size_t off = 0;
    while (off < n) {
        const ssize_t k = ::recv(fd, out.data() + off, n - off, 0);
        if (k <= 0) return false;
        off += static_cast<std::size_t>(k);
    }
    return true;
}

}  // namespace

TEST(Remote, KeysFramesAndStatusOverTcp) {
    RemoteConfig cfg;
    cfg.port = 18100 + static_cast<int>(::getpid() % 1000);
    cfg.preview_width_px = 96;
    cfg.max_fps = 1000.0;
    cfg.wait_fps = 1000.0;
    RemoteView view(cfg);
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(cfg.port));
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ASSERT_EQ(::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)), 0);
    for (int i = 0; i < 100 && !view.connected(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_TRUE(view.connected());
    ASSERT_EQ(::send(fd, "s", 1, 0), 1);
    int key = -1;
    for (int i = 0; i < 100 && key < 0; ++i, std::this_thread::sleep_for(std::chrono::milliseconds(10))) key = view.poll_key();
    EXPECT_EQ(key, 's');
    view.send_status("{\"phase\": \"select\"}");
    const cv::Mat img(108, 192, CV_8UC3, cv::Scalar(10, 200, 30));
    const std::vector<cv::Point2f> uv{{100.f, 50.f}};
    const std::vector<float> z{640.f};
    view.send_frame(img, 123, uv, z, true);
    std::vector<char> kinds;
    std::vector<uint8_t> hdr, body;
    for (int m = 0; m < 3; ++m) {
        ASSERT_TRUE(recv_n(fd, hdr, 5));
        uint32_t len = 0;
        std::memcpy(&len, hdr.data() + 1, 4);
        ASSERT_TRUE(recv_n(fd, body, len));
        kinds.push_back(static_cast<char>(hdr[0]));
        if (hdr[0] == 'L') {
            uint32_t n = 0;
            float u = 0.f;
            std::memcpy(&n, body.data() + 8, 4);
            std::memcpy(&u, body.data() + 12, 4);
            EXPECT_EQ(n, 1U);
            EXPECT_NEAR(u, 50.f, 1e-4);  // 100 px anh goc x (96 / 192)
        }
        if (hdr[0] == 'J') {
            EXPECT_EQ(body[8], 0xFF);  // JPEG bat dau FF D8
        }
    }
    EXPECT_EQ(kinds, (std::vector<char>{'S', 'J', 'L'}));
    ::close(fd);
}
