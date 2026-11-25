// HwMonitor: luong lay mau + ghi hw_usage.csv + tom tat theo phase + thong tin thiet bi.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>

#include <sys/utsname.h>
#include <unistd.h>

#include "uavp/common/common.hpp"
#include "uavp/support/hw_monitor.hpp"
#include "uavp/common/log.hpp"
#include "uavp/common/run_files.hpp"

namespace uavp {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Co ha xung / thieu nguon: vcgencmd (Raspberry Pi OS); khong co (Ubuntu) -> file sysfs cua firmware Pi.
Throttled read_throttled_cmd() {
    if (std::FILE* p = popen("vcgencmd get_throttled 2>/dev/null", "r")) {
        char buf[128]{};
        const bool ok = std::fgets(buf, sizeof(buf), p) != nullptr;
        pclose(p);
        if (ok) {
            Throttled t = decode_throttled(buf);
            if (!t.hex.empty()) return t;
        }
    }
    const std::string sysfs = read_file("/sys/devices/platform/soc/soc:firmware/get_throttled");  // vd "0\n" (hex)
    return sysfs.empty() ? Throttled{} : decode_throttled("throttled=0x" + sysfs);
}

double read_number(const std::string& path, double scale) {
    const std::string t = read_file(path);
    return t.empty() ? kNaN : std::strtod(t.c_str(), nullptr) / scale;
}

// Tan so tung nhan (MHz): (trung binh, nho nhat).
std::pair<double, double> read_freq_mhz(int n_cores) {
    double sum = 0.0, lo = kNaN;
    int n = 0;
    for (int i = 0; i < n_cores; ++i) {
        const double f = read_number("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/scaling_cur_freq", 1000.0);
        if (std::isnan(f)) continue;
        sum += f;
        lo = std::isnan(lo) ? f : std::min(lo, f);
        ++n;
    }
    return {n > 0 ? sum / n : kNaN, lo};
}

void write_throttled(JsonWriter& w, const Throttled& t) {
    w.begin_object().key("hex");
    if (t.hex.empty()) w.null();
    else w.value(t.hex);
    w.key("flags").begin_array();
    for (const auto& f : t.flags) w.value(f);
    w.end_array().end_object();
}

template <typename F>
double reduce(const std::vector<const HwSample*>& g, F get, bool take_max, bool take_min = false) {
    double acc = take_min ? 1e300 : (take_max ? -1e300 : 0.0);
    int n = 0;
    for (const HwSample* s : g) {
        const double v = get(*s);
        if (std::isnan(v)) continue;
        acc = take_min ? std::min(acc, v) : (take_max ? std::max(acc, v) : acc + v);
        ++n;
    }
    if (n == 0) return kNaN;
    return take_max || take_min ? acc : acc / n;
}

}  // namespace

void write_phase_summary(JsonWriter& w, const std::vector<HwSample>& samples) {
    std::vector<std::pair<std::string_view, std::vector<const HwSample*>>> groups{{"tat_ca", {}}};
    for (const auto& s : samples) {
        groups[0].second.push_back(&s);
        const std::string_view n = kPhaseNames[static_cast<std::size_t>(s.phase)];
        auto it = std::find_if(groups.begin() + 1, groups.end(), [&](const auto& g) { return g.first == n; });
        if (it == groups.end()) groups.emplace_back(n, std::vector<const HwSample*>{&s});
        else it->second.push_back(&s);
    }
    w.begin_object();
    for (const auto& [name, g] : groups) {
        if (g.empty()) continue;
        w.key(name).begin_object().key("n").value(g.size()).key("duration_s").num(g.back()->t_s - g.front()->t_s);
        w.key("cpu_total_pct_mean").num(reduce(g, [](const HwSample& s) { return s.cpu_total_pct; }, false));
        w.key("cpu_total_pct_max").num(reduce(g, [](const HwSample& s) { return s.cpu_total_pct; }, true));
        w.key("cpu_core_max_pct_max").num(reduce(g, [](const HwSample& s) { return s.cpu_core_max_pct; }, true));
        w.key("proc_cpu_pct_mean").num(reduce(g, [](const HwSample& s) { return s.proc_cpu_pct; }, false));
        w.key("proc_rss_mb_max").num(reduce(g, [](const HwSample& s) { return s.proc_rss_mb; }, true));
        w.key("mem_used_mb_max").num(reduce(g, [](const HwSample& s) { return s.mem_used_mb; }, true));
        w.key("temp_c_mean").num(reduce(g, [](const HwSample& s) { return s.temp_c; }, false));
        w.key("temp_c_max").num(reduce(g, [](const HwSample& s) { return s.temp_c; }, true));
        w.key("freq_mhz_mean").num(reduce(g, [](const HwSample& s) { return s.freq_mhz_mean; }, false));
        w.key("freq_mhz_min").num(reduce(g, [](const HwSample& s) { return s.freq_mhz_min; }, false, true));
        w.end_object();
    }
    w.end_object();
}

HwMonitor::HwMonitor(double period_s, bool enabled) : period_s_(period_s) {
    available_ = enabled && !read_file("/proc/stat").empty();
    if (!available_) {
        if (enabled) UAVP_LOG_WARN("uavp.hw_monitor", "Khong co /proc/stat (khong phai Linux): khong theo doi phan cung");
        return;
    }
    throttled_start_ = read_throttled_cmd();
    thread_ = std::thread([this] { loop(); });
}

HwMonitor::~HwMonitor() { stop(); }

void HwMonitor::stop() {
    {
        std::lock_guard lock(mutex_);
        if (stop_) return;
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    if (available_) throttled_end_ = read_throttled_cmd();
}

void HwMonitor::loop() {
    const auto start = std::chrono::steady_clock::now();
    const int n_cores = static_cast<int>(std::max(1L, sysconf(_SC_NPROCESSORS_ONLN)));
    const double clk_tck = static_cast<double>(sysconf(_SC_CLK_TCK));
    std::vector<Jiffies> prev, cur;
    std::vector<double> pct;
    parse_cpu_jiffies(read_file("/proc/stat"), prev);
    int64_t prev_self = parse_self_cpu_jiffies(read_file("/proc/self/stat"));
    auto prev_t = std::chrono::steady_clock::now();
    for (;;) {
        {
            std::unique_lock lock(mutex_);
            if (cv_.wait_for(lock, std::chrono::duration<double>(period_s_), [this] { return stop_; })) return;
        }
        parse_cpu_jiffies(read_file("/proc/stat"), cur);
        const int64_t self = parse_self_cpu_jiffies(read_file("/proc/self/stat"));
        const auto now = std::chrono::steady_clock::now();
        cpu_percent(prev, cur, pct);
        const double dt = std::chrono::duration<double>(now - prev_t).count();
        const auto [f_mean, f_min] = read_freq_mhz(n_cores);
        HwSample s;
        s.t_s = std::round(std::chrono::duration<double>(now - start).count() * 1000.0) / 1000.0;
        s.phase = phase_.load();
        s.cpu_total_pct = pct.empty() ? kNaN : pct[0];
        s.cpu_core_max_pct = pct.size() > 1 ? *std::max_element(pct.begin() + 1, pct.end()) : kNaN;
        s.proc_cpu_pct = 100.0 * static_cast<double>(self - prev_self) / clk_tck / std::max(dt, 1e-9);
        s.proc_rss_mb = parse_rss_mb(read_file("/proc/self/status"));
        s.mem_used_mb = parse_meminfo_mb(read_file("/proc/meminfo")).second;
        s.temp_c = read_number("/sys/class/thermal/thermal_zone0/temp", 1000.0);
        s.freq_mhz_mean = f_mean;
        s.freq_mhz_min = f_min;
        {
            std::lock_guard lock(mutex_);
            samples_.push_back(s);
        }
        std::swap(prev, cur);
        prev_self = self;
        prev_t = now;
    }
}

void HwMonitor::write_csv(const std::filesystem::path& path) const {
    std::string text = "t_s,phase,cpu_total_pct,cpu_core_max_pct,proc_cpu_pct,proc_rss_mb,mem_used_mb,temp_c,"
                       "freq_mhz_mean,freq_mhz_min\n";
    char line[256];
    for (const auto& s : samples_) {
        std::snprintf(line, sizeof(line), "%.2f,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", s.t_s,
                      std::string(kPhaseNames[static_cast<std::size_t>(s.phase)]).c_str(), s.cpu_total_pct,
                      s.cpu_core_max_pct, s.proc_cpu_pct, s.proc_rss_mb, s.mem_used_mb, s.temp_c, s.freq_mhz_mean,
                      s.freq_mhz_min);
        text += line;
    }
    write_text(path, text);
}

void HwMonitor::write_summary(JsonWriter& w) const {
    w.begin_object().key("period_s").value(period_s_).key("n_samples").value(samples_.size()).key("phases");
    write_phase_summary(w, samples_);
    w.key("throttled_start");
    write_throttled(w, throttled_start_);
    w.key("throttled_end");
    write_throttled(w, throttled_end_);
    w.end_object();
}

void write_device_info(JsonWriter& w, const std::vector<std::pair<std::string, std::string>>& extra) {
    std::string model = read_file("/proc/device-tree/model");
    model.erase(std::remove(model.begin(), model.end(), '\0'), model.end());
    std::string os = "?";
    std::istringstream rel(read_file("/etc/os-release"));
    for (std::string line; std::getline(rel, line);)
        if (line.rfind("PRETTY_NAME=", 0) == 0) os = line.substr(12);
    os.erase(std::remove(os.begin(), os.end(), '"'), os.end());
    utsname un{};
    uname(&un);
    const auto [mem_total, mem_used] = parse_meminfo_mb(read_file("/proc/meminfo"));
    w.begin_object().key("model").value(model.empty() ? std::string(un.machine) : model).key("os").value(os);
    w.key("kernel").value(un.release).key("impl").value("cpp").key("cpu_count").value(static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN)));
    w.key("mem_total_mb").value(static_cast<int>(std::round(mem_total)));
    (void)mem_used;
    for (const auto& [k, v] : extra) w.key(k).value(v);
    w.end_object();
}

bool HwMonitor::latest(HwSample& out) {
    std::lock_guard lock(mutex_);
    if (samples_.empty()) return false;
    out = samples_.back();
    return true;
}

}  // namespace uavp
