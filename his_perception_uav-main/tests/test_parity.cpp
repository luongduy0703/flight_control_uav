// Parity C++ <-> Python (CPP_RULES.md muc 0.1): phat lai output NCNN that do tools/dump_parity.py ghi tu ban Python
// qua code C++ (tien xu ly, giai ma, NMS, ghep tile, mask, tam, yaw, W) va so voi ket qua Python:
//   bbox <= 1 px, mask IoU >= 0.99, W <= 1 mm (+ input model gan nhu bang nhau, tam <= 1 px, yaw <= 1 do).
// Chua co du lieu (parity_data/manifest.json) -> bo qua.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>

#include <opencv2/imgcodecs.hpp>

#include "helpers.hpp"
#include "uavp/stages/s1_detect.hpp"
#include "uavp/common/run_files.hpp"
#include "uavp/stages/s2_segment.hpp"
#include "uavp/stages/s3_select.hpp"
#include "uavp/common/tensor_io.hpp"

using namespace uavp;
namespace fs = std::filesystem;

namespace {

const fs::path kDir = UAVP_PARITY_DIR;

// Sai lech lon nhat (in ra cuoi test).
struct Worst {
    double input{0}, bbox{0}, conf{0}, center{0}, yaw{0}, W{0};
    double iou{1};
    int dets{0}, segs{0};
} g_worst;  // NOLINT: chi dung trong file test

std::vector<std::vector<fs::path>> calls_of(const YAML::Node& calls) {
    std::vector<std::vector<fs::path>> out;
    for (const auto& c : calls) {
        out.emplace_back();
        for (const auto& f : c) out.back().push_back(kDir / f.as<std::string>());
    }
    return out;
}

double max_abs_diff(const InputCHW& in, const fs::path& python_input) {
    Tensor ref;
    read_tensor(python_input, ref);
    EXPECT_EQ(ref.data.size(), in.data.size());
    double m = 0.0;
    for (std::size_t i = 0; i < std::min(ref.data.size(), in.data.size()); ++i)
        m = std::max(m, static_cast<double>(std::abs(ref.data[i] - in.data[i])));
    return m;
}

Detection det_of(const YAML::Node& d) {
    return {d[0].as<float>(), d[1].as<float>(), d[2].as<float>(), d[3].as<float>(), d[4].as<float>(),
            static_cast<uint8_t>(d[5].as<int>()), {}};
}

void check_detections(const cv::Mat& image, const YAML::Node& entry, const Config& cfg) {
    auto infer = std::make_unique<ReplayInfer>(calls_of(entry["det_calls"]));
    const ReplayInfer& rec = *infer;
    Detector det(cfg.detector, std::move(infer));
    std::vector<Detection> out;
    det(image, out);
    const double din = max_abs_diff(rec.first_input(), kDir / entry["det_input"].as<std::string>());
    g_worst.input = std::max(g_worst.input, din);
    EXPECT_LE(din, 1e-6) << "input tile dau";
    const YAML::Node py = entry["detections"];
    ASSERT_EQ(out.size(), py.size()) << "so detection";
    for (std::size_t k = 0; k < out.size(); ++k) {
        const Detection p = det_of(py[k]);
        EXPECT_EQ(out[k].class_id, p.class_id);
        EXPECT_NEAR(out[k].conf, p.conf, 1e-5);
        g_worst.conf = std::max(g_worst.conf, static_cast<double>(std::abs(out[k].conf - p.conf)));
        ++g_worst.dets;
        for (const auto& [a, b] : {std::pair{out[k].x1, p.x1}, {out[k].y1, p.y1}, {out[k].x2, p.x2}, {out[k].y2, p.y2}})
        {
            EXPECT_LE(std::abs(a - b), 1.f) << "bbox lech > 1 px";
            g_worst.bbox = std::max(g_worst.bbox, static_cast<double>(std::abs(a - b)));
        }
    }
}

double mask_iou(const SegMask& m, const fs::path& python_mask) {
    const cv::Mat py = cv::imread(python_mask.string(), cv::IMREAD_GRAYSCALE) > 0;
    const cv::Mat cpp = m.mask > 0;
    if (py.size() != cpp.size()) return 0.0;
    const double inter = cv::countNonZero(py & cpp), uni = cv::countNonZero(py | cpp);
    return uni > 0 ? inter / uni : 1.0;
}

void check_segment(const cv::Mat& image, const YAML::Node& s, const Config& cfg, double D_m) {
    auto infer = std::make_unique<ReplayInfer>(calls_of(s["calls"]));
    const ReplayInfer& rec = *infer;
    Segmenter seg(cfg.segmenter, std::move(infer));
    SegObject obj;
    const bool found = seg(image, det_of(s["det"]), obj);
    const double din = max_abs_diff(rec.first_input(), kDir / s["input"].as<std::string>());
    g_worst.input = std::max(g_worst.input, din);
    EXPECT_LE(din, 1e-6) << "input ROI";
    ASSERT_EQ(found, s["found"].as<bool>());
    if (!found) return;
    EXPECT_EQ(obj.mask.x0_px, s["x0"].as<int>());
    EXPECT_EQ(obj.mask.y0_px, s["y0"].as<int>());
    const double iou = mask_iou(obj.mask, kDir / s["mask"].as<std::string>());
    g_worst.iou = std::min(g_worst.iou, iou);
    EXPECT_GE(iou, 0.99) << "mask IoU";
    const double dc = std::hypot(obj.center_px.x - s["center"][0].as<double>(), obj.center_px.y - s["center"][1].as<double>());
    g_worst.center = std::max(g_worst.center, dc);
    ++g_worst.segs;
    EXPECT_LE(dc, 1.0);
    const double dyaw = std::abs(obj.yaw_deg - s["yaw"].as<double>());
    g_worst.yaw = std::max(g_worst.yaw, std::min(dyaw, 180.0 - dyaw));
    EXPECT_LE(std::min(dyaw, 180.0 - dyaw), 1.0) << "yaw";
    const Observation o = measure(obj, s["det"][4].as<float>(), 0, cfg.camera, D_m);
    if (!s["W_mm"].IsNull()) {
        g_worst.W = std::max(g_worst.W, std::abs(o.W_mm - s["W_mm"].as<double>()));
        EXPECT_LE(std::abs(o.W_mm - s["W_mm"].as<double>()), 1.0) << "W";
    }
}

}  // namespace

TEST(Parity, DetectSegmentMeasureMatchPython) {
    if (!fs::exists(kDir / "manifest.json")) GTEST_SKIP() << "Chua co parity_data: chay cpp/tools/dump_parity.py";
    const fs::path old = fs::current_path();
    fs::current_path(test::kRoot);  // duong dan trong config tinh tu thu muc goc
    const YAML::Node man = YAML::LoadFile((kDir / "manifest.json").string());
    // du lieu dump tu ban Python o che do 8 tile 640 (model tile): kiem dung cach chay do. Che do ca anh 960 x 544 dung
    // chung letterbox / giai ma (test_detect: FullModeRectInputRunsModelOnce); can dump lai khi Python co mode full.
    // Ban Python cat ROI s2 thang tren anh goc -> roi_image_px = [0, 0] (che do 960 x 544: test_segment).
    const Config cfg =
        config_from_raw(load_raw("configs/run/offline.yaml", {"s1_detect.mode=tile", "s1_detect.model.input_px=640",
                                                     "s1_detect.model.dir=models/det_yolo26n_v3ft_s2_ncnn_fp16",
                                                     "s2_segment.roi_image_px=[0, 0]"}));
    if (man["models"]) {  // du lieu dump phai tu cung model voi configs/models/*.yaml
        EXPECT_EQ(cfg.detector.model.dir.filename().string(), man["models"]["s1"].as<std::string>());
        EXPECT_EQ(cfg.segmenter.model.dir.filename().string(), man["models"]["s2"].as<std::string>());
    }
    int n_seg = 0;
    for (const auto& entry : man["images"]) {
        SCOPED_TRACE(entry["source"].as<std::string>());
        const cv::Mat image = read_image_rgb(kDir / entry["image"].as<std::string>());
        check_detections(image, entry, cfg);
        for (const auto& s : entry["segs"]) {
            check_segment(image, s, cfg, man["D_m"].as<double>());
            ++n_seg;
        }
    }
    fs::current_path(old);
    EXPECT_GT(n_seg, 0);
    std::printf("Parity: %d detection, %d mask | lech lon nhat: input %.2g, bbox %.4f px, conf %.2g, tam %.4f px, "
                "yaw %.4f do, W %.4f mm | mask IoU nho nhat %.5f\n",
                g_worst.dets, g_worst.segs, g_worst.input, g_worst.bbox, g_worst.conf, g_worst.center, g_worst.yaw,
                g_worst.W, g_worst.iou);
}
