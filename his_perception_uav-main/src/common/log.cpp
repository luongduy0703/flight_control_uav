#include "uavp/common/log.hpp"

#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <stdexcept>

namespace uavp {
namespace {

std::atomic<Logger*> g_logger{nullptr};  // con tro tam thuong, khong co constructor phuc tap
constexpr std::array<const char*, static_cast<std::size_t>(Level::kCount)> kLevelNames{"ERROR", "WARNING", "INFO",
                                                                                        "DEBUG"};

// "2026-11-12 10:20:30,123 INFO module: text" giong dinh dang log cua ban Python.
int format_prefix(char* buf, std::size_t n, Level level, const char* module) noexcept {
    const auto now = std::chrono::system_clock::now();
    const std::time_t secs = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
    localtime_r(&secs, &tm);
    char date[32]{};
    std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", &tm);
    return std::snprintf(buf, n, "%s,%03d %s %s: ", date, static_cast<int>(ms),
                         kLevelNames[static_cast<std::size_t>(level)], module);
}

}  // namespace

Logger::Logger(Level level) : level_(level) { thread_ = std::thread([this] { loop(); }); }

Logger::~Logger() {
    {
        std::lock_guard lock(mutex_);
        running_ = false;
    }
    cv_.notify_one();
    if (thread_.joinable()) thread_.join();
    if (file_ != nullptr) std::fclose(file_);
}

void Logger::open_file(const std::filesystem::path& path) {
    std::FILE* f = std::fopen(path.c_str(), "a");
    if (f == nullptr) throw std::runtime_error("Khong mo duoc file log " + path.string());
    std::lock_guard lock(mutex_);
    if (file_ != nullptr) std::fclose(file_);
    file_ = f;
}

void Logger::write(Level level, const char* module, const char* text) noexcept {
    char line[kLineBytes]{};
    const int n = format_prefix(line, sizeof(line), level, module);
    if (n > 0 && static_cast<std::size_t>(n) < sizeof(line)) {
        std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "%s", text);
    }
    {
        std::lock_guard lock(mutex_);
        if (head_ - tail_ >= kSlots) {
            ++dropped_;  // ring day: bo dong moi, khong chan luong tinh toan
            return;
        }
        std::memcpy(ring_[head_ & (kSlots - 1)].data(), line, sizeof(line));
        ++head_;
    }
    cv_.notify_one();
}

void Logger::drain(std::FILE* file) {
    for (;;) {
        std::array<char, kLineBytes> line{};
        {
            std::lock_guard lock(mutex_);
            if (tail_ == head_) return;
            line = ring_[tail_ & (kSlots - 1)];
            ++tail_;
        }
        std::fprintf(stderr, "%s\n", line.data());
        if (file != nullptr) std::fprintf(file, "%s\n", line.data());
    }
}

void Logger::loop() {
    auto last_flush = std::chrono::steady_clock::now();
    for (;;) {
        std::FILE* file = nullptr;
        bool running = true;
        {
            std::unique_lock lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(200), [this] { return head_ != tail_ || !running_; });
            file = file_;
            running = running_;
        }
        drain(file);
        const auto now = std::chrono::steady_clock::now();
        if (!running || now - last_flush > std::chrono::seconds(1)) {
            std::fflush(stderr);
            if (file != nullptr) std::fflush(file);
            last_flush = now;
        }
        if (!running) return;
    }
}

void set_logger(Logger* logger) noexcept { g_logger.store(logger); }

void log(Level level, const char* module, const char* fmt, ...) noexcept {
    Logger* logger = g_logger.load();
    if (logger != nullptr && !logger->enabled(level)) return;  // log tat: khong dinh dang chuoi
    char text[Logger::kLineBytes]{};
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    if (logger != nullptr) {
        logger->write(level, module, text);
        return;
    }
    std::fprintf(stderr, "%s %s: %s\n", kLevelNames[static_cast<std::size_t>(level)], module, text);
}

}  // namespace uavp
