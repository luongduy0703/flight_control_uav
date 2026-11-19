// Ham thuan phan tich /proc, /sys, vcgencmd (test duoc tren may bat ky).
#include "uavp/support/hw_monitor.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>

#include "uavp/common/common.hpp"

namespace uavp {
namespace {

constexpr double kKbPerMb = 1024.0;
constexpr std::array<std::pair<int, std::string_view>, 8> kThrottleBits{{
    {0, "under_voltage"}, {1, "freq_capped"}, {2, "throttled"}, {3, "soft_temp_limit"},
    {16, "under_voltage_occurred"}, {17, "freq_capped_occurred"}, {18, "throttled_occurred"},
    {19, "soft_temp_limit_occurred"}}};

std::vector<std::string_view> split_ws(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
        const std::size_t j = s.find_first_of(" \t", i);
        if (i < s.size()) out.push_back(s.substr(i, j == std::string_view::npos ? s.size() - i : j - i));
        i = j == std::string_view::npos ? s.size() : j;
    }
    return out;
}

int64_t to_i64(std::string_view s) {
    int64_t v = 0;
    std::from_chars(s.data(), s.data() + s.size(), v);
    return v;
}

}  // namespace

void parse_cpu_jiffies(std::string_view text, std::vector<Jiffies>& out) {
    out.clear();
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t end = std::min(text.find('\n', pos), text.size());
        const auto parts = split_ws(text.substr(pos, end - pos));
        pos = end + 1;
        if (parts.empty() || parts[0].substr(0, 3) != "cpu") continue;
        int64_t total = 0;
        for (std::size_t i = 1; i < parts.size() && i <= 8; ++i) total += to_i64(parts[i]);  // user..steal
        const int64_t idle = (parts.size() > 4 ? to_i64(parts[4]) : 0) + (parts.size() > 5 ? to_i64(parts[5]) : 0);
        out.push_back({total - idle, total});
    }
}

void cpu_percent(const std::vector<Jiffies>& prev, const std::vector<Jiffies>& cur, std::vector<double>& out) {
    out.clear();
    for (std::size_t i = 0; i < std::min(prev.size(), cur.size()); ++i) {
        const auto busy = static_cast<double>(cur[i][0] - prev[i][0]);
        const auto total = static_cast<double>(cur[i][1] - prev[i][1]);
        out.push_back(total > 0.0 ? 100.0 * busy / total : 0.0);
    }
}

std::pair<double, double> parse_meminfo_mb(std::string_view text) {
    const auto field = [&](std::string_view key) -> double {
        const std::size_t p = text.find(key);
        if (p == std::string_view::npos) return 0.0;
        const auto parts = split_ws(text.substr(p + key.size(), 40));
        return parts.empty() ? 0.0 : static_cast<double>(to_i64(parts[0])) / kKbPerMb;
    };
    const double total = field("MemTotal:");
    return {total, total - field("MemAvailable:")};
}

int64_t parse_self_cpu_jiffies(std::string_view text) {
    const std::size_t close = text.rfind(')');  // ten tien trinh co the chua dau cach
    if (close == std::string_view::npos) return 0;
    const auto parts = split_ws(text.substr(close + 1));
    return parts.size() > 12 ? to_i64(parts[11]) + to_i64(parts[12]) : 0;  // truong 14, 15
}

double parse_rss_mb(std::string_view text) {
    const std::size_t p = text.find("VmRSS:");
    if (p == std::string_view::npos) return kNaN;
    const auto parts = split_ws(text.substr(p + 6, 40));
    return parts.empty() ? kNaN : static_cast<double>(to_i64(parts[0])) / kKbPerMb;
}

Throttled decode_throttled(std::string_view text) {
    const std::size_t eq = text.find('=');
    if (eq == std::string_view::npos) return {};
    std::string_view v = text.substr(eq + 1);
    while (!v.empty() && (v.back() == '\n' || v.back() == ' ')) v.remove_suffix(1);
    if (v.substr(0, 2) == "0x") v.remove_prefix(2);
    uint32_t bits = 0;
    const auto res = std::from_chars(v.data(), v.data() + v.size(), bits, 16);
    if (res.ec != std::errc() || v.empty()) return {};
    Throttled t;
    char hex[16];
    std::snprintf(hex, sizeof(hex), "0x%x", bits);
    t.hex = hex;
    for (const auto& [bit, name] : kThrottleBits)
        if ((bits >> bit) & 1U) t.flags.emplace_back(name);
    return t;
}

}  // namespace uavp
