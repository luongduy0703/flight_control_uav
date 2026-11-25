#include <algorithm>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "uavp/common/common.hpp"
#include "uavp/sources/frame_source.hpp"
#include "uavp/common/run_files.hpp"

namespace uavp {

ReplaySource::ReplaySource(const std::filesystem::path& source, bool repeat) : repeat_(repeat) {
    if (std::filesystem::is_directory(source)) {
        for (const auto& e : std::filesystem::directory_iterator(source)) {
            std::string ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") files_.push_back(e.path());
        }
        std::sort(files_.begin(), files_.end());
        if (files_.empty()) throw std::runtime_error("Thu muc khong co anh: " + source.string());
        size_ = read_image_rgb(files_.front()).size();
    } else if (std::filesystem::is_regular_file(source) &&
               (source.extension() == ".jpg" || source.extension() == ".png" || source.extension() == ".jpeg")) {
        files_.push_back(source);
        size_ = read_image_rgb(source).size();
    } else {
        if (!capture_.open(source.string())) throw std::runtime_error("Khong mo duoc video: " + source.string());
        size_ = {static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT))};
    }
}

bool ReplaySource::next_frame(cv::Mat& image_rgb, FrameMeta& meta, double /*timeout_s*/) {
    meta = FrameMeta{};
    meta.t_ns = mono_ns();
    if (repeat_ && !first_.empty()) {
        first_.copyTo(image_rgb);
        return true;
    }
    if (!files_.empty()) {
        if (index_ >= files_.size()) return false;
        const cv::Mat bgr = cv::imread(files_[index_++].string(), cv::IMREAD_COLOR);
        if (bgr.empty()) return false;
        cv::cvtColor(bgr, image_rgb, cv::COLOR_BGR2RGB);  // OpenCV doc BGR -> RGB
    } else {
        if (!capture_.read(bgr_)) return false;
        cv::cvtColor(bgr_, image_rgb, cv::COLOR_BGR2RGB);  // OpenCV doc BGR -> RGB
    }
    if (repeat_) image_rgb.copyTo(first_);
    return true;
}

}  // namespace uavp
