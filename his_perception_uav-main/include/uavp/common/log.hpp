// Log qua luong rieng (CPP_RULES.md muc 9): luong goi chi dinh dang vao buffer tren stack roi chep vao mot slot
// cua ring buffer co san; luong log ghi ra stderr + run.log, xa buffer moi 1 s. Khong cap phat khi log.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <thread>

namespace uavp {

enum class Level : uint8_t { kError, kWarn, kInfo, kDebug, kCount };

class Logger {
public:
    static constexpr std::size_t kSlots = 1024;  // luy thua 2
    static constexpr std::size_t kLineBytes = 384;
    static_assert((kSlots & (kSlots - 1)) == 0, "ring buffer can la luy thua 2");

    explicit Logger(Level level);
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // Them file log (run.log). Goi tu luong chinh truoc khi chay pipeline.
    void open_file(const std::filesystem::path& path);
    [[nodiscard]] bool enabled(Level level) const noexcept { return level <= level_; }
    void write(Level level, const char* module, const char* text) noexcept;

private:
    void loop();
    void drain(std::FILE* file);

    std::array<std::array<char, kLineBytes>, kSlots> ring_{};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    std::FILE* file_{nullptr};
    std::size_t head_{0};  // so dong da ghi vao ring (luong goi)
    std::size_t tail_{0};  // so dong da xuat (luong log)
    std::size_t dropped_{0};
    Level level_;
    bool running_{true};
};

// Logger dang dung (tao trong main, gan bang set_logger). nullptr -> in thang ra stderr (vd khi chay test).
void set_logger(Logger* logger) noexcept;

void log(Level level, const char* module, const char* fmt, ...) noexcept __attribute__((format(printf, 3, 4)));

#define UAVP_LOG_INFO(module, ...) ::uavp::log(::uavp::Level::kInfo, module, __VA_ARGS__)
#define UAVP_LOG_WARN(module, ...) ::uavp::log(::uavp::Level::kWarn, module, __VA_ARGS__)
#define UAVP_LOG_ERROR(module, ...) ::uavp::log(::uavp::Level::kError, module, __VA_ARGS__)
#define UAVP_LOG_DEBUG(module, ...) ::uavp::log(::uavp::Level::kDebug, module, __VA_ARGS__)

}  // namespace uavp
