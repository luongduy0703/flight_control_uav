// Theo doi phan cung tren Pi (giong src/uavp/hw_monitor.py): CPU (tong, tung nhan, tien trinh), RAM, nhiet do,
// tan so, co throttle (vcgencmd). Lay mau o luong rieng moi period_s; luong chinh chi doi `phase` (atomic).
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "uavp/common/json_writer.hpp"

namespace uavp {

enum class Phase : uint8_t { kStartup, kWaitKey, kS1S3, kS4, kLock, kSave, kCount };
inline constexpr std::array<std::string_view, static_cast<std::size_t>(Phase::kCount)> kPhaseNames{
    "khoi_dong", "cho_bam_s", "s1_s3", "s4", "khoa_bam", "luu"};

// --- Ham thuan: phan tich /proc, /sys, vcgencmd -----------------------------------------------------------------
using Jiffies = std::array<int64_t, 2>;  // [ban, tong]
// /proc/stat -> dong 0 = tong, dong i = cpu(i-1).
void parse_cpu_jiffies(std::string_view proc_stat, std::vector<Jiffies>& out);
// % ban giua hai lan doc (tong + tung nhan).
void cpu_percent(const std::vector<Jiffies>& prev, const std::vector<Jiffies>& cur, std::vector<double>& out);
// (tong MB, dang dung MB = tong - MemAvailable)
[[nodiscard]] std::pair<double, double> parse_meminfo_mb(std::string_view meminfo);
[[nodiscard]] int64_t parse_self_cpu_jiffies(std::string_view self_stat);  // utime + stime
[[nodiscard]] double parse_rss_mb(std::string_view self_status);
struct Throttled {
    std::string hex;  // rong = khong doc duoc
    std::vector<std::string> flags;
};
[[nodiscard]] Throttled decode_throttled(std::string_view text);

struct HwSample {
    double t_s{0.0};
    double cpu_total_pct{0.0};
    double cpu_core_max_pct{0.0};
    double proc_cpu_pct{0.0};  // % mot nhan (4 nhan -> toi da 400)
    double proc_rss_mb{0.0};
    double mem_used_mb{0.0};
    double temp_c{0.0};
    double freq_mhz_mean{0.0};
    double freq_mhz_min{0.0};
    Phase phase{Phase::kStartup};
};

// Tom tat theo phase (va "tat_ca") -> JSON (cung khoa voi ban Python).
void write_phase_summary(JsonWriter& w, const std::vector<HwSample>& samples);

class HwMonitor {
public:
    HwMonitor(double period_s, bool enabled);
    ~HwMonitor();
    HwMonitor(const HwMonitor&) = delete;
    HwMonitor& operator=(const HwMonitor&) = delete;

    void set_phase(Phase p) noexcept { phase_.store(p); }
    void stop();  // dung lay mau, doc throttle cuoi
    void write_csv(const std::filesystem::path& path) const;
    void write_summary(JsonWriter& w) const;  // muc hw_usage
    // Mau moi nhat (gui ve may tinh). false neu chua co.
    [[nodiscard]] bool latest(HwSample& out);

private:
    void loop();

    std::vector<HwSample> samples_;
    Throttled throttled_start_, throttled_end_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    double period_s_;
    std::atomic<Phase> phase_{Phase::kStartup};
    bool available_{false};
    bool stop_{false};
};

// Muc device: model Pi, OS, kernel, so nhan, RAM + cac cap (khoa, gia tri) phan mem cua ung dung.
void write_device_info(JsonWriter& w, const std::vector<std::pair<std::string, std::string>>& extra);

}  // namespace uavp
