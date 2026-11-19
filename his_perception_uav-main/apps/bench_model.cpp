// uavp_bench_model: do thoi gian s1 (detect ca anh) va s2 (segment 1 ROI) tren Pi, dung code + config that
// (tien xu ly ncnn, pool allocator, so luong, nhan CPU). So fp16 storage bat / tat trong cung mot lan chay.
//   ./build/uavp_bench_model --cfg configs/run/real_time_perception.yaml --image runs/<phien>/frame.jpg
//   ./build/uavp_bench_model --cfg ... --image ... --loops 30 --fp16 both --set s1_detect.model.num_threads=4
// --fp16: off | on | both (mac dinh both). Khong co --image: anh xam 1920 x 1080 (s2 do tren ROI giua anh).
// Ket qua in ra terminal: median / p90 / min (ms) cua s1 va s2 cho tung cau hinh.
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/log.hpp"
#include "uavp/common/run_files.hpp"
#include "uavp/model/ncnn_model.hpp"
#include "uavp/stages/s1_detect.hpp"
#include "uavp/stages/s2_segment.hpp"
#include "uavp/support/cpu_affinity.hpp"

using namespace uavp;

namespace {

constexpr int kWarmup = 3;

struct Args {
    std::string cfg, image, fp16{"both"};
    int loops{20};
    std::vector<std::string> overrides;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--cfg" && i + 1 < argc) a.cfg = argv[++i];
        else if (k == "--image" && i + 1 < argc) a.image = argv[++i];
        else if (k == "--loops" && i + 1 < argc) a.loops = std::stoi(argv[++i]);
        else if (k == "--fp16" && i + 1 < argc) a.fp16 = argv[++i];
        else if (k == "--set")
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.overrides.emplace_back(argv[++i]);
    }
    if (a.cfg.empty() || a.loops < 1 || (a.fp16 != "on" && a.fp16 != "off" && a.fp16 != "both"))
        throw std::invalid_argument(
            "Dung: uavp_bench_model --cfg <yaml> [--image <anh>] [--loops N] [--fp16 off|on|both] [--set k=v ...]");
    return a;
}

struct Stat {
    double median{0.0}, p90{0.0}, min{0.0};
};

Stat stat_of(std::vector<double> v) {
    Stat s;
    s.min = *std::min_element(v.begin(), v.end());
    s.p90 = percentile_inplace(v, 90.0);
    s.median = median_inplace(std::span<double>(v));
    return s;
}

// Mot cau hinh fp16 storage: nap model moi, chay warmup + loops lan s1 va s2.
void bench(const Config& base, const cv::Mat& image, bool fp16, int loops) {
    Config cfg = base;
    cfg.detector.model.use_fp16_storage = fp16;
    cfg.segmenter.model.use_fp16_storage = fp16;
    Detector det(cfg.detector, std::make_unique<NcnnModel>(cfg.detector.model));
    Segmenter seg(cfg.segmenter, std::make_unique<NcnnModel>(cfg.segmenter.model));
    pin_ncnn_threads(cfg.cpu.cores);
    std::vector<Detection> dets;
    std::vector<double> t1, t2;
    for (int i = 0; i < kWarmup + loops; ++i) {
        const int64_t start = mono_ns();
        det(image, dets);
        if (i >= kWarmup) t1.push_back(ms_since(start));
    }
    // s2: vat gan tam anh nhat (nhu pipeline); khong co vat -> ROI 100 x 100 giua anh (chi do thoi gian)
    const cv::Point2d c = cfg.camera.center_px();
    Detection target{static_cast<float>(c.x - 50), static_cast<float>(c.y - 50), static_cast<float>(c.x + 50),
                     static_cast<float>(c.y + 50), 1.f, 0, {}};
    double best = 1e18;
    for (const auto& d : dets) {
        const double dist = std::hypot(d.center_px().x - c.x, d.center_px().y - c.y);
        if (dist < best) {
            best = dist;
            target = d;
        }
    }
    SegObject obj;
    bool found = false;
    for (int i = 0; i < kWarmup + loops; ++i) {
        const int64_t start = mono_ns();
        found = seg(image, target, obj);
        if (i >= kWarmup) t2.push_back(ms_since(start));
    }
    const Stat s1 = stat_of(t1), s2 = stat_of(t2);
    std::printf("fp16_storage=%-3s | s1 %dx%d %d luong: median %7.1f  p90 %7.1f  min %7.1f ms | %zu vat\n",
                fp16 ? "on" : "off", cfg.detector.input_w_px, cfg.detector.input_h_px, cfg.detector.model.num_threads,
                s1.median, s1.p90, s1.min, dets.size());
    std::printf("                 | s2 %dx%d %d luong: median %7.1f  p90 %7.1f  min %7.1f ms | mask: %s\n",
                cfg.segmenter.input_px, cfg.segmenter.input_px, cfg.segmenter.model.num_threads, s2.median, s2.p90,
                s2.min, found ? "co" : "khong");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const Config cfg = config_from_raw(load_raw(args.cfg, args.overrides));
        const std::string cores = pin_process_to_cores(cfg.cpu.cores);
        const cv::Mat image = args.image.empty() ? cv::Mat(1080, 1920, CV_8UC3, cv::Scalar::all(128))
                                                 : read_image_rgb(args.image);
        std::printf("CPU %s | ncnn %s | s1 %s | s2 %s | %d lan (+%d khoi dong)\n", cores.c_str(), ncnn_version().c_str(),
                    cfg.detector.model.dir.c_str(), cfg.segmenter.model.dir.c_str(), args.loops, kWarmup);
        if (args.fp16 != "on") bench(cfg, image, false, args.loops);
        if (args.fp16 != "off") bench(cfg, image, true, args.loops);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "LOI: " << e.what() << '\n';
        return 1;
    }
}
