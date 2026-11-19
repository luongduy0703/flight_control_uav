// Nguon khung anh (camera Pi that / replay thu muc anh, video) va kho khung dung chung (FramePool).
// Moi khung chi nam o mot slot cua FramePool; cac buoc chi trao chi so slot, khong chep pixel (CPP_RULES.md muc 3).
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace uavp {

struct FrameMeta {
    int64_t t_ns{0};             // thoi diem khung (dong ho chung mono_ns)
    double lens_position{-1.0};  // < 0: khong co
    double exposure_us{-1.0};
    double analogue_gain{-1.0};
};

// Goi mot lan moi khung: ham ao chap nhan duoc (CPP_RULES.md muc 8).
class IFrameSource {
public:
    virtual ~IFrameSource() = default;
    // Ghi khung moi nhat (RGB uint8) vao `image_rgb` (buffer co san, dung kich thuoc). false neu het timeout / het nguon.
    [[nodiscard]] virtual bool next_frame(cv::Mat& image_rgb, FrameMeta& meta, double timeout_s) = 0;
};

// Kho slot anh cap phat mot lan. acquire() lay slot trong; retain/release dem so noi dang giu slot.
class FramePool {
public:
    FramePool(int slots, cv::Size size) : images_(static_cast<std::size_t>(slots)), refs_(static_cast<std::size_t>(slots), 0),
                                          t_ns_(static_cast<std::size_t>(slots), 0) {
        for (auto& m : images_) m.create(size, CV_8UC3);
    }
    FramePool(const FramePool&) = delete;
    FramePool& operator=(const FramePool&) = delete;

    [[nodiscard]] int acquire() {
        for (std::size_t i = 0; i < refs_.size(); ++i) {
            if (refs_[i] != 0) continue;
            refs_[i] = 1;
            return static_cast<int>(i);
        }
        throw std::logic_error("FramePool het slot: tang so slot");
    }
    void retain(int slot) noexcept { ++refs_[static_cast<std::size_t>(slot)]; }
    void release(int slot) noexcept {
        if (slot >= 0 && refs_[static_cast<std::size_t>(slot)] > 0) --refs_[static_cast<std::size_t>(slot)];
    }
    [[nodiscard]] cv::Mat& image(int slot) noexcept { return images_[static_cast<std::size_t>(slot)]; }
    [[nodiscard]] int64_t& t_ns(int slot) noexcept { return t_ns_[static_cast<std::size_t>(slot)]; }

private:
    std::vector<cv::Mat> images_;
    std::vector<int> refs_;
    std::vector<int64_t> t_ns_;
};

// Replay: thu muc anh (sap theo ten) hoac file video; `repeat`: lap lai mot anh (gia lap hover).
class ReplaySource final : public IFrameSource {
public:
    explicit ReplaySource(const std::filesystem::path& source, bool repeat = false);
    [[nodiscard]] bool next_frame(cv::Mat& image_rgb, FrameMeta& meta, double timeout_s) override;
    [[nodiscard]] cv::Size frame_size() const noexcept { return size_; }

private:
    std::vector<std::filesystem::path> files_;
    cv::VideoCapture capture_;
    cv::Mat bgr_;
    cv::Mat first_;  // anh dau (repeat)
    std::size_t index_{0};
    cv::Size size_;
    bool repeat_{false};
};

}  // namespace uavp
