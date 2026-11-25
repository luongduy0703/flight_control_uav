// uavp_offline: mot phien chon vat tren thu muc anh / video / mot anh lap lai (ban C++ cua run_session.py va
// test_stages.py). Dung de so ket qua va thoi gian voi ban Python tren cung du lieu.
//   ./build/uavp_offline --cfg configs/run/offline.yaml --source <thu muc anh | video> [--set k=v ...]
//   ./build/uavp_offline --cfg configs/run/offline.yaml --image <anh>     # them s1/s2/s3 _*.jpg
// Ket qua: runs/<ngay-gio>_offline/: run.log, config.yaml, result.json, debug.jpg, selected_mask.png.
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "uavp/common/config.hpp"
#include "uavp/support/draw.hpp"
#include "uavp/sources/frame_source.hpp"
#include "uavp/common/log.hpp"
#include "uavp/model/ncnn_model.hpp"
#include "uavp/common/run_files.hpp"
#include "uavp/stages/s3_select.hpp"
#include "uavp/pipeline/session.hpp"
#include "uavp/support/cpu_affinity.hpp"

using namespace uavp;

namespace {

constexpr const char* kModule = "uavp.offline";

struct Args {
    std::string cfg, source, image;
    std::vector<std::string> overrides;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--cfg" && i + 1 < argc) a.cfg = argv[++i];
        else if (k == "--source" && i + 1 < argc) a.source = argv[++i];
        else if (k == "--image" && i + 1 < argc) a.image = argv[++i];
        else if (k == "--set")
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.overrides.emplace_back(argv[++i]);
    }
    if (a.cfg.empty() || (a.source.empty() == a.image.empty()))
        throw std::invalid_argument("Dung: uavp_offline --cfg <yaml> (--source <thu muc | video> | --image <anh>) [--set ...]");
    return a;
}

// s1, s2 chay lai tren anh cuoi (detect + segment vat gan tam) -> anh kiem tra tung buoc.
void write_stage_images(const std::filesystem::path& dir, const cv::Mat& image, const Session& s, const Config& cfg,
                        Detector& det, Segmenter& seg) {
    std::vector<Detection> dets;
    det(image, dets);
    std::vector<Track> tmp;
    const std::vector<int> group = select_group(tmp, dets, cfg.camera.center_px(), kTargets, cfg.match_radius_px, 0);
    std::vector<SegObject> objs(group.size());
    std::vector<const SegObject*> segs;
    for (std::size_t i = 0; i < group.size(); ++i)
        segs.push_back(seg(image, tmp[static_cast<std::size_t>(group[i])].detection, objs[i]) ? &objs[i] : nullptr);
    write_image_rgb(dir / "s1_detect.jpg", draw_detections(image, dets, cfg.detector.classes, cfg.camera.center_px()), kJpegQuality);
    write_image_rgb(dir / "s2_segment.jpg", draw_segments(image, tmp, segs, cfg.segmenter.classes, cfg.camera.center_px()),
                    kJpegQuality);
    write_image_rgb(dir / "s3_select.jpg", draw_debug(image, s.tracks, cfg.camera.center_px()), kJpegQuality);
}

int run_main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const YAML::Node raw = load_raw(args.cfg, args.overrides);
    const Config cfg = config_from_raw(raw);
    const std::string cores = pin_process_to_cores(cfg.cpu.cores);  // truoc moi luong (log, NCNN)
    auto logger = std::make_unique<Logger>(Level::kInfo);
    set_logger(logger.get());
    const std::filesystem::path dir = make_run_dir(cfg.runs_root, "offline");
    logger->open_file(dir / "run.log");
    write_config(dir, raw, args.overrides);
    UAVP_LOG_INFO(kModule, "CPU: chay tren %s", cores.c_str());
    Detector det(cfg.detector, std::make_unique<NcnnModel>(cfg.detector.model));
    Segmenter seg(cfg.segmenter, std::make_unique<NcnnModel>(cfg.segmenter.model));
    pin_ncnn_threads(cfg.cpu.cores);
    const bool single = !args.image.empty();
    ReplaySource source(single ? args.image : args.source, single);
    cv::Mat image(source.frame_size(), CV_8UC3);
    FrameMeta meta;
    Session s;
    Timings timings;
    int frames = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(cfg.max_frames) && source.next_frame(image, meta, 0.0); ++i) {
        const StepTimes t = session_step(s, i, meta.t_ns, image, cfg, det, seg);
        if (t.detected) timings.add("s1_detect", t.detect_ms);
        timings.add("s2_segment", t.segment_ms);
        timings.add("s3_select", t.select_ms);
        frames = static_cast<int>(i) + 1;
        if (s.done) break;
    }
    if (frames == 0) throw std::runtime_error("Khong co khung nao");
    timings.log_summary();
    JsonWriter w;
    write_session_result(w, s.tracks, cfg, frames);
    write_text(dir / "result.json", w.str());
    write_image_rgb(dir / (single ? "s3_select.jpg" : "debug.jpg"), draw_debug(image, s.tracks, cfg.camera.center_px()),
                    kJpegQuality);
    if (const Track* t = selected_track(s.tracks, cfg.camera.center_px()))
        write_mask_png(dir / kSelectedMaskFile, last_seg(*t)->mask, image.rows, image.cols);
    if (single) write_stage_images(dir, image, s, cfg, det, seg);
    UAVP_LOG_INFO(kModule, "Ket thuc sau %d khung. Ket qua: %s", frames, dir.c_str());
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
