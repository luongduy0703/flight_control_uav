// Doc / ghi file cua mot lan chay (giong run_files.py): thu muc run, anh, mask, CSV, config, bao cao JSON.
#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "uavp/support/aruco_check.hpp"
#include "uavp/common/config.hpp"
#include "uavp/common/json_writer.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

inline constexpr int kJpegQuality = 90;
inline constexpr const char* kSelectedMaskFile = "selected_mask.png";

[[nodiscard]] std::filesystem::path make_run_dir(const std::filesystem::path& runs_root, const std::string& name);
[[nodiscard]] cv::Mat read_image_rgb(const std::filesystem::path& path);
void write_image_rgb(const std::filesystem::path& path, const cv::Mat& image_rgb, int jpeg_quality);
// Mask vat (kich thuoc anh goc), 255 = mat tren.
void write_mask_png(const std::filesystem::path& path, const SegMask& seg, int height_px, int width_px);
// Diem LiDAR trong mask: X, Y, Z (mm, he camera luc do), Xref.. (da bu ve khung tham chieu), u, v (px), t_ns, khung,
// vong quet, inlier.
void write_points_csv(const std::filesystem::path& path, const FusionResult& r, std::span<const cv::Point2f> uv_px);
void write_text(const std::filesystem::path& path, const std::string& text);
// config.yaml: config phien + moi stage (da gop, da ghi de).
void write_config(const std::filesystem::path& run_dir, const YAML::Node& raw, const std::vector<std::string>& overrides);

// --- Thoi gian tung buoc (ten theo thu tu them vao, giong dict cua Python) --------------------------------------
class Timings {
public:
    void add(const std::string& name, double ms);
    [[nodiscard]] const std::vector<double>* get(const std::string& name) const;
    // {"<ten>": {"median_ms", "p95_ms", "n"}}
    void write_json(JsonWriter& w) const;
    void log_summary() const;

private:
    std::vector<std::pair<std::string, std::vector<double>>> items_;
};

// --- Muc result.json (cung khoa voi ban Python) -------------------------------------------------------------------
void write_track_summary(JsonWriter& w, const Track& t, const Config& cfg);
void write_fusion_summary(JsonWriter& w, const FusionResult& r);
void write_aruco_summary(JsonWriter& w, const ArucoPose& pose, cv::Point3d p_aruco_mm, const FusionResult& r);
// Ket qua phien offline: selected, waiting, defer, abort.
void write_session_result(JsonWriter& w, const std::vector<Track>& tracks, const Config& cfg, int frames);

}  // namespace uavp
