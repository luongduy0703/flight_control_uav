// uavp_camera_view: CHI xem camera Pi tu may tinh (khong model, khong LD06, khong Pixhawk).
//   ./build/uavp_camera_view --cfg configs/run/real_time.yaml [--seconds 0] [--set remote.wait_fps=10]
// May tinh: python tools/remote_viewer.py --host <ip-pi> --port 8100   (q tren viewer: dung). Ctrl+C cung dung.
// Camera mo dung nhu uavp_real_time (sensor_mode cua file K: 1920x1080, lat 180 do, net khoa); anh gui ve thu nho
// remote.preview_width_px, toi da remote.wait_fps anh / s. Khong ghi file nao.
#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/common.hpp"
#include "uavp/common/config.hpp"
#include "uavp/common/log.hpp"
#include "uavp/sources/frame_source.hpp"
#include "uavp/sources/pi_camera.hpp"
#include "uavp/support/remote_view.hpp"

using namespace uavp;

namespace {

constexpr const char* kModule = "uavp.camera_view";
std::atomic<bool> g_stop{false};  // Ctrl+C

void on_signal(int /*sig*/) { g_stop.store(true); }

struct Args {
    std::string cfg;
    std::vector<std::string> overrides;
    double seconds{0.0};  // 0 = toi khi dung
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) a.cfg = argv[++i];
        else if (std::strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) a.seconds = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--set") == 0)
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.overrides.emplace_back(argv[++i]);
    }
    if (a.cfg.empty()) throw std::invalid_argument("Dung: uavp_camera_view --cfg configs/run/real_time.yaml [--seconds 0]");
    return a;
}

int run_main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const YAML::Node raw = load_raw(args.cfg, args.overrides);
    const SourcesConfig src = sources_config(raw["sources"]);
    const CaptureConfig cap = load_capture(raw["calib_K"].as<std::string>(), src);
    const YAML::Node r = raw["remote"];
    const RemoteConfig rcfg{r["port"].as<int>(), r["preview_width_px"].as<int>(), r["jpeg_quality"].as<int>(),
                            r["max_fps"].as<double>(), r["wait_fps"].as<double>(), true};
    auto logger = std::make_unique<Logger>(Level::kInfo);
    set_logger(logger.get());
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    PiCamera camera(cap, src.camera_buffer_count, src.sensor_bit_depth);
    RemoteView remote(rcfg);
    remote.set_fps(rcfg.wait_fps);
    cv::Mat image(cap.height_px, cap.width_px, CV_8UC3);
    UAVP_LOG_INFO(kModule, "Camera %dx%d. Cho may tinh ket noi: python tools/remote_viewer.py --host <ip-pi> --port %d",
                  cap.width_px, cap.height_px, rcfg.port);
    UAVP_LOG_INFO(kModule, "Gui toi da %.0f anh / s, rong %d px. q tren viewer hoac Ctrl+C: dung", rcfg.wait_fps,
                  rcfg.preview_width_px);
    const int64_t t_end = args.seconds > 0.0 ? mono_ns() + static_cast<int64_t>(args.seconds * 1e9) : 0;
    int64_t t_tick = mono_ns();
    int frames = 0, sent = 0;
    double age_ms_sum = 0.0;
    bool was_connected = false;
    while (!g_stop.load() && (t_end == 0 || mono_ns() < t_end)) {
        FrameMeta meta;
        if (!camera.next_frame(image, meta, src.frame_timeout_s)) throw std::runtime_error("Camera: khong co khung moi");
        ++frames;
        age_ms_sum += static_cast<double>(mono_ns() - meta.t_ns) / 1e6;
        if (remote.poll_key() == 'q') {
            UAVP_LOG_INFO(kModule, "May tinh bam q: dung");
            break;
        }
        const bool connected = remote.connected();
        if (connected != was_connected) {
            UAVP_LOG_INFO(kModule, connected ? "May tinh da ket noi" : "May tinh ngat ket noi");
            was_connected = connected;
        }
        if (connected && remote.due()) {
            remote.send_frame(image, meta.t_ns, {}, {});
            ++sent;
        }
        const double dt_s = static_cast<double>(mono_ns() - t_tick) / 1e9;
        if (dt_s >= 5.0) {
            UAVP_LOG_INFO(kModule, "camera %.1f khung / s, tre %.0f ms | gui %.1f anh / s | %s", frames / dt_s,
                          age_ms_sum / std::max(frames, 1), sent / dt_s, connected ? "co may tinh" : "chua co may tinh");
            frames = sent = 0;
            age_ms_sum = 0.0;
            t_tick = mono_ns();
        }
    }
    UAVP_LOG_INFO(kModule, "Dung");
    set_logger(nullptr);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& e) {
        set_logger(nullptr);
        std::cerr << "LOI: " << e.what() << '\n';
        return 1;
    }
}
