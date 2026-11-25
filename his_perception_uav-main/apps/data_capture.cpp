// uavp_data_capture: THU ANH khi bay. Camera Pi 1080p30 (giong uavp_real_time), stream thu nho ve may tinh qua
// RemoteView (giong uavp_camera_view); bam SPACE tren remote_viewer -> luu KHUNG GOC 1920x1080 tren Pi.
//   ./build/uavp_data_capture --cfg configs/run/real_time.yaml [--out runs] [--format jpg|png] [--quality 95]
// May tinh: python tools/remote_viewer.py --host <ip-pi> --port 8100   (SPACE: chup, q: dung). Ctrl+C cung dung.
// Ket qua: <out>/<ngay-gio>_capture/images/000001.jpg ... + frames.csv (thoi diem, phoi sang, gain, lens moi anh).
// Khong model, khong LD06. Ghi file o luong rieng (JPEG 1080p ~30-60 ms, PNG ~200+ ms tren Pi 4):
// vong camera khong bi chan; hang doi toi da kMaxQueue anh (~6 MB / anh), day thi bo lenh chup + canh bao.
//
// --marker: do ArUco DIEM CAT CANH duoi chan UAV (muc launch_marker trong config: id, canh, period_s) + tu the Pixhawk
// (uavp_fc_bridge -> /dev/shm). Moi lan do hop le -> dong "MARKER seen=1 ..." qua UDP 127.0.0.1:<--marker-udp> cho
// pi/console.py --marker-hold (CONSOLE moi la noi gui setpoint; chuong trinh nay KHONG gui gi xuong Pixhawk).
// Ket qua do: <run>/launch_marker.csv; vi tri camera so voi marker + do cao hien tren remote_viewer.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "uavp/common/common.hpp"
#include "uavp/common/config.hpp"
#include "uavp/common/log.hpp"
#include "uavp/common/run_files.hpp"
#include "uavp/sources/frame_source.hpp"
#include "uavp/sources/mavlink.hpp"
#include "uavp/sources/pi_camera.hpp"
#include "uavp/support/launch_marker.hpp"
#include "uavp/support/remote_view.hpp"
#include "uavp/tracking/world_pose.hpp"

using namespace uavp;
namespace fs = std::filesystem;

namespace {

constexpr const char* kModule = "uavp.capture";
constexpr std::size_t kMaxQueue = 8;  // 8 x 6 MB = ~50 MB RAM toi da
std::atomic<bool> g_stop{false};      // Ctrl+C

void on_signal(int /*sig*/) { g_stop.store(true); }

struct Args {
    std::string cfg;
    std::vector<std::string> overrides;
    std::string out{"runs"};
    std::string format{"jpg"};
    int quality{95};
    bool marker{false};     // do ArUco diem cat canh + gui UDP cho console
    int marker_udp{14590};  // cong UDP 127.0.0.1 ma pi/console.py --marker-hold nghe
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) a.cfg = argv[++i];
        else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) a.out = argv[++i];
        else if (std::strcmp(argv[i], "--format") == 0 && i + 1 < argc) a.format = argv[++i];
        else if (std::strcmp(argv[i], "--quality") == 0 && i + 1 < argc) a.quality = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--marker") == 0) a.marker = true;
        else if (std::strcmp(argv[i], "--marker-udp") == 0 && i + 1 < argc) a.marker_udp = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--set") == 0)
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.overrides.emplace_back(argv[++i]);
    }
    if (a.cfg.empty())
        throw std::invalid_argument(
            "Dung: uavp_data_capture --cfg configs/run/real_time.yaml [--out runs] [--format jpg|png] [--quality 95] "
            "[--marker [--marker-udp 14590]]");
    if (a.format != "jpg" && a.format != "png") throw std::invalid_argument("--format chi nhan jpg hoac png");
    if (a.marker_udp <= 0 || a.marker_udp > 65535) throw std::invalid_argument("--marker-udp: cong 1..65535");
    a.quality = std::clamp(a.quality, 50, 100);
    return a;
}

struct Shot {
    int index{0};
    cv::Mat rgb;  // ban sao khung goc (slot camera duoc dung lai ngay)
    FrameMeta meta;
};

// Luong ghi anh: nhan Shot tu vong camera, RGB -> BGR, ma hoa, ghi file + mot dong frames.csv.
class Writer {
public:
    Writer(fs::path dir, std::string format, int quality)
        : dir_(std::move(dir)), format_(std::move(format)), quality_(quality) {
        fs::create_directories(dir_ / "images");
        csv_.open(dir_ / "frames.csv");
        if (!csv_) throw std::runtime_error("Khong tao duoc " + (dir_ / "frames.csv").string());
        csv_ << "index,file,t_ns,exposure_us,analogue_gain,lens_position,encode_ms\n";
        thread_ = std::thread([this] { loop(); });
    }
    ~Writer() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable()) thread_.join();  // ghi het hang doi truoc khi thoat
    }
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    // false: hang doi day (the nho / CPU cham) -> bo anh nay.
    bool push(Shot shot) {
        {
            std::lock_guard lock(mutex_);
            if (queue_.size() >= kMaxQueue) return false;
            queue_.push_back(std::move(shot));
        }
        cv_.notify_one();
        return true;
    }
    [[nodiscard]] int saved() const noexcept { return saved_.load(); }
    [[nodiscard]] std::size_t queued() {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

private:
    void loop() {
        std::vector<int> params = format_ == "jpg" ? std::vector<int>{cv::IMWRITE_JPEG_QUALITY, quality_}
                                                   : std::vector<int>{cv::IMWRITE_PNG_COMPRESSION, 1};  // nen nhe: nhanh
        cv::Mat bgr;
        while (true) {
            Shot shot;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;  // stop_ va da ghi het
                shot = std::move(queue_.front());
                queue_.pop_front();
            }
            const int64_t t0 = mono_ns();
            char name[32];
            std::snprintf(name, sizeof(name), "%06d.%s", shot.index, format_.c_str());
            const fs::path path = dir_ / "images" / name;
            cv::cvtColor(shot.rgb, bgr, cv::COLOR_RGB2BGR);  // pipeline RGB, OpenCV ghi BGR
            if (!cv::imwrite(path.string(), bgr, params)) {
                UAVP_LOG_ERROR(kModule, "Khong ghi duoc %s (the nho day?)", path.c_str());
                continue;
            }
            const double ms = static_cast<double>(mono_ns() - t0) / 1e6;
            csv_ << shot.index << ",images/" << name << ',' << shot.meta.t_ns << ',' << shot.meta.exposure_us << ','
                 << shot.meta.analogue_gain << ',' << shot.meta.lens_position << ',' << ms << '\n';
            csv_.flush();  // mat dien / rot UAV: khong mat dong da ghi
            saved_.fetch_add(1);
            UAVP_LOG_INFO(kModule, "Da luu %s (%.0f ms)", name, ms);
        }
    }

    fs::path dir_;
    std::string format_;
    int quality_;
    std::ofstream csv_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Shot> queue_;
    bool stop_{false};
    std::atomic<int> saved_{0};
    std::thread thread_;
};

// Gui dong MARKER (text) toi 127.0.0.1:port. Khong ai nghe -> goi tin bi bo, khong loi.
class MarkerUdp {
public:
    explicit MarkerUdp(int port) {
        fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd_ < 0) throw std::runtime_error(std::string("marker UDP socket loi: ") + std::strerror(errno));
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(static_cast<uint16_t>(port));
        addr_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    }
    ~MarkerUdp() {
        if (fd_ >= 0) ::close(fd_);
    }
    MarkerUdp(const MarkerUdp&) = delete;
    MarkerUdp& operator=(const MarkerUdp&) = delete;
    void send(const std::string& line) const noexcept {
        (void)::sendto(fd_, line.data(), line.size(), MSG_DONTWAIT, reinterpret_cast<const sockaddr*>(&addr_),
                       sizeof(addr_));
    }

private:
    int fd_{-1};
    sockaddr_in addr_{};
};

struct MarkerState {
    bool any{false};
    MarkerObservation obs;
    int64_t t_ns{0};  // luc nhan (mono)
};

std::string status_json(int saved, int requested, std::size_t queued, int dropped, const std::string& dir,
                        const MarkerState* marker) {
    std::string m = "null";
    if (marker != nullptr && marker->any) {
        const auto& p = marker->obs.p_cam_S_mm;  // camera trong he marker (z am: tren san)
        char b[200];
        std::snprintf(b, sizeof(b), "{\"age_s\": %.2f, \"x_m\": %.3f, \"y_m\": %.3f, \"h_m\": %.3f, \"world\": %d}",
                      static_cast<double>(mono_ns() - marker->t_ns) / 1e9, p[0] / 1000.0, p[1] / 1000.0,
                      -p[2] / 1000.0, marker->obs.has_world ? 1 : 0);
        m = b;
    } else if (marker != nullptr) {
        m = "{}";  // dang do nhung chua thay marker
    }
    return "{\"phase\": \"capture\", \"saved\": " + std::to_string(saved) + ", \"requested\": " +
           std::to_string(requested) + ", \"queued\": " + std::to_string(queued) + ", \"dropped\": " +
           std::to_string(dropped) + ", \"dir\": \"" + dir + "\", \"marker\": " + m + "}";
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

    const fs::path run_dir = make_run_dir(args.out, "capture");

    // --- ArUco diem cat canh + tu the Pixhawk (tuy chon) ---
    std::unique_ptr<IFcSource> fc_src;
    const FcPoseBuffer* fc = nullptr;
    std::unique_ptr<CameraModel> cam_model;
    std::unique_ptr<BodyCamera> body_camera;
    std::unique_ptr<LaunchMarkerTracker> marker;
    std::unique_ptr<MarkerUdp> marker_udp;
    MarkerState marker_state;
    if (args.marker) {
        LaunchMarkerConfig mcfg = launch_marker_config(raw["launch_marker"]);
        mcfg.enabled = true;
        cam_model = std::make_unique<CameraModel>(load_camera(raw["calib_K"].as<std::string>()));
        body_camera = std::make_unique<BodyCamera>(load_body_camera(lock_config(raw["lock_track"]).calib_T_BC));
        const MavlinkConfig mav = mavlink_config(raw["sources"]["mavlink"]);
        if (mav.enabled) {
            try {
                fc_src = open_fc_source(mav);
                const int64_t t0 = mono_ns();
                while (fc_src->poses().n_attitude() == 0 &&
                       static_cast<double>(mono_ns() - t0) / 1e9 < src.connect_timeout_s)
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (fc_src->poses().n_attitude() > 0) fc = &fc_src->poses();
            } catch (const std::exception& e) {
                UAVP_LOG_WARN(kModule, "Khong mo duoc MAVLink: %s", e.what());
            }
        }
        if (fc == nullptr)
            UAVP_LOG_WARN(kModule, "ArUco: KHONG co tu the Pixhawk (uavp_fc_bridge?) -> chi do camera so voi marker, "
                          "console --marker-hold se KHONG chinh vi tri (can huong marker trong NED)");
        marker = std::make_unique<LaunchMarkerTracker>(mcfg, *cam_model, *body_camera, run_dir / "launch_marker.csv");
        marker_udp = std::make_unique<MarkerUdp>(args.marker_udp);
        UAVP_LOG_INFO(kModule, "ArUco diem cat canh: id %d, canh %.1f mm, moi %.2f s -> UDP 127.0.0.1:%d (%s)",
                      mcfg.marker_id, mcfg.side_mm, mcfg.period_s, args.marker_udp,
                      fc != nullptr ? "co tu the Pixhawk" : "KHONG co tu the Pixhawk");
    }

    PiCamera camera(cap, src.camera_buffer_count, src.sensor_bit_depth);
    RemoteView remote(rcfg);
    remote.set_fps(rcfg.wait_fps);
    auto writer_ptr = std::make_unique<Writer>(run_dir, args.format, args.quality);
    Writer& writer = *writer_ptr;
    cv::Mat image(cap.height_px, cap.width_px, CV_8UC3);
    UAVP_LOG_INFO(kModule, "Camera %dx%d. Luu vao %s (%s%s)", cap.width_px, cap.height_px, run_dir.c_str(),
                  args.format.c_str(), args.format == "jpg" ? (" q" + std::to_string(args.quality)).c_str() : "");
    UAVP_LOG_INFO(kModule, "May tinh: python tools/remote_viewer.py --host <ip-pi> --port %d | SPACE: chup, q: dung",
                  rcfg.port);

    int requested = 0, dropped = 0, frames = 0;
    int64_t t_tick = mono_ns();
    bool was_connected = false;
    while (!g_stop.load()) {
        FrameMeta meta;
        if (!camera.next_frame(image, meta, src.frame_timeout_s)) throw std::runtime_error("Camera: khong co khung moi");
        ++frames;
        if (marker) {  // tu gioi han theo launch_marker.period_s (ArUco tren anh 1080p ton CPU)
            marker->update(image, meta.t_ns, fc);
            if (const auto o = marker->take_new()) {
                marker_udp->send(marker_line(*o));
                marker_state = {true, *o, mono_ns()};
            }
        }
        bool quit = false, shoot = false;
        for (int key = remote.poll_key(); key != -1; key = remote.poll_key()) {
            if (key == 'q') quit = true;
            else if (key == 's' || key == ' ') shoot = true;  // nhieu lan bam trong 1 khung = 1 anh
        }
        if (quit) {
            UAVP_LOG_INFO(kModule, "May tinh bam q: dung");
            break;
        }
        if (shoot) {
            ++requested;
            if (!writer.push(Shot{requested, image.clone(), meta})) {
                ++dropped;
                UAVP_LOG_WARN(kModule, "Hang doi ghi day (%zu anh): bo anh #%d - bam cham lai / the nho cham", kMaxQueue,
                              requested);
            }
        }
        const bool connected = remote.connected();
        if (connected != was_connected) {
            UAVP_LOG_INFO(kModule, connected ? "May tinh da ket noi" : "May tinh ngat ket noi (van chay, cho ket noi lai)");
            was_connected = connected;
        }
        if (connected && (shoot || remote.due())) {
            remote.send_frame(image, meta.t_ns, {}, {}, shoot);
            remote.send_status(status_json(writer.saved(), requested, writer.queued(), dropped, run_dir.string(),
                                           marker ? &marker_state : nullptr));
        }
        const double dt_s = static_cast<double>(mono_ns() - t_tick) / 1e9;
        if (dt_s >= 5.0) {
            UAVP_LOG_INFO(kModule, "camera %.1f khung / s | da luu %d / %d anh, bo %d | %s", frames / dt_s,
                          writer.saved(), requested, dropped, connected ? "co may tinh" : "chua co may tinh");
            frames = 0;
            t_tick = mono_ns();
        }
    }
    UAVP_LOG_INFO(kModule, "Dung: dang ghi not %zu anh trong hang doi...", writer.queued());
    writer_ptr.reset();  // ghi het hang doi (luong ghi con log) truoc khi go logger
    UAVP_LOG_INFO(kModule, "Xong: yeu cau %d anh, bo %d | %s", requested, dropped, run_dir.c_str());
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
