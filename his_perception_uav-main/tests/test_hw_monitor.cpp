// Phan tich /proc, /sys, vcgencmd (chuoi mau) va tom tat theo phase (giong tests/test_hw_monitor.py).
#include <gtest/gtest.h>

#include "uavp/support/hw_monitor.hpp"
#include "uavp/common/json_writer.hpp"

using namespace uavp;

TEST(HwMonitor, CpuPercentTotalAndPerCore) {
    std::vector<Jiffies> a, b;
    parse_cpu_jiffies("cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 50 0 50 400 0 0 0 0 0 0\ncpu1 50 0 50 400 0 0 0 0 0 0\nintr 1 2\n", a);
    parse_cpu_jiffies("cpu  250 0 150 900 0 0 0 0 0 0\ncpu0 150 0 100 450 0 0 0 0 0 0\ncpu1 100 0 50 450 0 0 0 0 0 0\n", b);
    ASSERT_EQ(a.size(), 3U);
    std::vector<double> pct;
    cpu_percent(a, b, pct);
    EXPECT_NEAR(pct[0], 200.0 / 3.0, 1e-9);  // ban +200 / tong +300
    EXPECT_NEAR(pct[1], 75.0, 1e-9);
    EXPECT_NEAR(pct[2], 50.0, 1e-9);
}

TEST(HwMonitor, MeminfoRssSelfStat) {
    const auto [total, used] = parse_meminfo_mb("MemTotal:  8192000 kB\nMemFree: 100 kB\nMemAvailable:  6144000 kB\n");
    EXPECT_NEAR(total, 8000.0, 1e-9);
    EXPECT_NEAR(used, 2000.0, 1e-9);
    EXPECT_NEAR(parse_rss_mb("Name:\tpython\nVmRSS:\t  512000 kB\n"), 500.0, 1e-9);
    EXPECT_EQ(parse_self_cpu_jiffies("1234 (uavp real time) S 1 1 1 0 -1 0 0 0 0 0 700 300 0 0 20 0 8 0 100"), 1000);
}

TEST(HwMonitor, DecodeThrottled) {
    EXPECT_EQ(decode_throttled("throttled=0x0\n").hex, "0x0");
    const Throttled t = decode_throttled("throttled=0x50005");
    EXPECT_EQ(t.flags, (std::vector<std::string>{"under_voltage", "throttled", "under_voltage_occurred", "throttled_occurred"}));
    EXPECT_TRUE(decode_throttled("").hex.empty());
}

TEST(HwMonitor, SummaryByPhase) {
    const auto sample = [](double t, Phase ph, double cpu, double temp) {
        HwSample s;
        s.t_s = t;
        s.phase = ph;
        s.cpu_total_pct = cpu;
        s.temp_c = temp;
        s.freq_mhz_min = 1800.0;
        return s;
    };
    JsonWriter w;
    write_phase_summary(w, {sample(0.0, Phase::kWaitKey, 5, 50), sample(0.5, Phase::kS1S3, 80, 60),
                            sample(1.0, Phase::kS1S3, 90, 65)});
    const std::string& j = w.str();
    EXPECT_NE(j.find("\"s1_s3\": {\n    \"n\": 2"), std::string::npos);
    EXPECT_NE(j.find("\"cpu_total_pct_mean\": 85"), std::string::npos);
    EXPECT_NE(j.find("\"temp_c_max\": 65"), std::string::npos);
}

TEST(HwMonitor, DisabledMonitorHasEmptySummary) {
    HwMonitor m(0.5, false);
    m.stop();
    JsonWriter w;
    m.write_summary(w);
    EXPECT_NE(w.str().find("\"n_samples\": 0"), std::string::npos);
}
