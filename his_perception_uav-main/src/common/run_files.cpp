#include "uavp/common/run_files.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace uavp {

std::filesystem::path make_run_dir(const std::filesystem::path& runs_root, const std::string& name) {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&now, &tm);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm);
    const std::filesystem::path dir = runs_root / (std::string(stamp) + "_" + name);
    std::filesystem::create_directories(runs_root);
    if (!std::filesystem::create_directory(dir)) throw std::runtime_error("Thu muc run da ton tai: " + dir.string());
    return dir;
}

cv::Mat read_image_rgb(const std::filesystem::path& path) {
    const cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("Khong doc duoc anh: " + path.string());
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);  // OpenCV doc BGR -> RGB
    return rgb;
}

void write_image_rgb(const std::filesystem::path& path, const cv::Mat& image_rgb, int jpeg_quality) {
    cv::Mat bgr;
    cv::cvtColor(image_rgb, bgr, cv::COLOR_RGB2BGR);  // OpenCV ghi BGR
    if (!cv::imwrite(path.string(), bgr, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality}))
        throw std::runtime_error("Khong ghi duoc anh: " + path.string());
}

void write_mask_png(const std::filesystem::path& path, const SegMask& seg, int height_px, int width_px) {
    cv::Mat full = cv::Mat::zeros(height_px, width_px, CV_8U);
    const cv::Rect m(seg.x0_px, seg.y0_px, seg.mask.cols, seg.mask.rows);
    const cv::Rect in = m & cv::Rect(0, 0, width_px, height_px);  // mask da doi (bam vat) co the lo ra ngoai anh
    if (in.area() > 0) full(in).setTo(255, seg.mask(in - m.tl()));
    if (!cv::imwrite(path.string(), full)) throw std::runtime_error("Khong ghi duoc mask: " + path.string());
}

void write_points_csv(const std::filesystem::path& path, const FusionResult& r, std::span<const cv::Point2f> uv_px) {
    std::string text = "X_mm,Y_mm,Z_mm,Xref_mm,Yref_mm,Zref_mm,u_px,v_px,t_ns,frame,scan,inlier\n";
    char line[256];
    const bool has_ref = r.points_ref_mm.size() == r.points_C_mm.size();
    for (std::size_t i = 0; i < r.points_C_mm.size(); ++i) {
        const cv::Point3f& p = r.points_C_mm[i];
        const cv::Point3f q = has_ref ? r.points_ref_mm[i] : p;  // da bu chuyen dong ve khung tham chieu
        std::snprintf(line, sizeof(line), "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%lld,%d,%d,%d\n",
                      static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z),
                      static_cast<double>(q.x), static_cast<double>(q.y), static_cast<double>(q.z),
                      static_cast<double>(uv_px[i].x), static_cast<double>(uv_px[i].y),
                      static_cast<long long>(r.points_t_ns[i]), static_cast<int>(r.points_frame[i]),
                      static_cast<int>(r.points_scan[i]), static_cast<int>(r.inliers[i]));
        text += line;
    }
    write_text(path, text);
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Khong ghi duoc " + path.string());
    f << text;
}

void write_config(const std::filesystem::path& run_dir, const YAML::Node& raw, const std::vector<std::string>& overrides) {
    YAML::Emitter emitter;
    emitter << raw;
    std::string header = "# --set [";
    for (std::size_t i = 0; i < overrides.size(); ++i) header += (i != 0U ? ", '" : "'") + overrides[i] + "'";
    write_text(run_dir / "config.yaml", header + "]\n" + emitter.c_str() + "\n");
}

}  // namespace uavp
