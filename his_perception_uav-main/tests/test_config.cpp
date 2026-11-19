// Doc config that cua du an (duong dan trong config tinh tu thu muc goc uav_perception_pi/).
#include <gtest/gtest.h>

#include <filesystem>

#include "helpers.hpp"
#include "uavp/common/config.hpp"

using namespace uavp;

namespace {

struct ChdirRoot {  // RAII: chay test trong thu muc goc, tra lai thu muc cu
    std::filesystem::path old = std::filesystem::current_path();
    ChdirRoot() { std::filesystem::current_path(test::kRoot); }
    ~ChdirRoot() { std::filesystem::current_path(old); }
};

}  // namespace

TEST(Config, StageConfigsLoadWithOverrides) {
    const ChdirRoot cd;
    const YAML::Node raw = load_raw("configs/run/offline.yaml", {"s3_select.n_min=3", "s4_fusion.min_pts=8"});
    const Config c = config_from_raw(raw);
    EXPECT_EQ(c.decide.n_min, 3);
    EXPECT_EQ(c.fusion.min_pts, 8);
    EXPECT_EQ(c.detector.classes.size(), 2U);
    EXPECT_NEAR(c.fusion.extrinsics.t_CL_mm[1], 38.8, 1e-6);
    EXPECT_GT(c.camera.K_px(0, 0), 800.0);
    EXPECT_THROW(load_raw("configs/run/offline.yaml", {"s3_select.khong_co=1"}), std::runtime_error);
}

TEST(Config, ModelFilesLoadedIntoStages) {
    const ChdirRoot cd;
    const Config c = config_from_raw(load_raw("configs/run/offline.yaml", {}));
    EXPECT_EQ(c.detector.input_w_px, 960);  // configs/models/s1_detect_ncnn.yaml: [960, 544]
    EXPECT_EQ(c.detector.input_h_px, 544);
    EXPECT_EQ(c.detector.mode, DetectMode::kFull);
    EXPECT_EQ(c.detector.model.num_threads, 2);
    EXPECT_EQ(c.cpu.cores, (std::vector<int>{2, 3}));
    EXPECT_EQ(c.segmenter.roi_image_w_px, 960);  // configs/stages/s2_segment.yaml: ROI tren anh 960 x 544
    EXPECT_EQ(c.segmenter.roi_image_h_px, 544);
    EXPECT_FALSE(c.detector.model.use_fp16_storage);
    EXPECT_EQ(c.detector.pad_value, 114);
    EXPECT_EQ(c.segmenter.input_px, 320);  // configs/models/s2_segment_ncnn.yaml
    EXPECT_EQ(c.segmenter.classes, (std::vector<std::string>{"box_top", "bottle_top"}));
    EXPECT_EQ(c.detector.model.dir, "models/det_yolo26n_960x544_s0_ncnn_fp32");
    const Config o = config_from_raw(
        load_raw("configs/run/offline.yaml", {"s1_detect.model.input_px=640", "s2_segment.model.num_threads=2"}));
    EXPECT_EQ(o.detector.input_w_px, 640);  // mot so = input vuong
    EXPECT_EQ(o.detector.input_h_px, 640);
    EXPECT_THROW((void)config_from_raw(load_raw("configs/run/offline.yaml", {"s1_detect.model.input_px=[960,540]"})),
                 std::runtime_error);  // 540 khong chia het cho 32
    EXPECT_EQ(o.segmenter.model.num_threads, 2);
}

TEST(Config, RealTimeSections) {
    const ChdirRoot cd;
    const YAML::Node raw = load_raw("configs/run/real_time.yaml", {"display.show=false", "aruco.check_aruco=true"});
    const SourcesConfig s = sources_config(raw["sources"]);
    EXPECT_EQ(s.lidar_baudrate, 230400);
    EXPECT_GT(s.lidar_buffer_scans, 0);
    EXPECT_FALSE(display_config(raw["display"]).show);
    EXPECT_TRUE(aruco_config(raw["aruco"]).check_aruco);
    EXPECT_GT(monitor_config(raw["monitor"]).period_s, 0.0);
    const MavlinkConfig mav = mavlink_config(raw["sources"]["mavlink"]);
    EXPECT_EQ(mav.baudrate, 921600);
    const LockConfig lock = lock_config(raw["lock_track"]);
    EXPECT_GE(lock.flow_min_points, 6);
    EXPECT_EQ(lock.calib_T_BC, "configs/calib/T_BC.yaml");
    const CaptureConfig cap = load_capture(raw["calib_K"].as<std::string>());
    EXPECT_EQ(cap.width_px, 1920);
    EXPECT_EQ(cap.rotate_deg, 180);
    const YAML::Node bad = load_raw("configs/run/real_time.yaml", {"display.show=yes"});
    EXPECT_THROW((void)display_config(bad["display"]), std::runtime_error);  // phai la true / false
}
