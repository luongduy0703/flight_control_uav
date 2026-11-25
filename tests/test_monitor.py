from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import unittest
from monitor_pi import cpu_percent, decode_throttled, read_cpu_times, read_meminfo

STAT_A = """cpu  100 0 100 700 100 0 0 0 0 0
cpu0 50 0 50 350 50 0 0 0 0 0
cpu1 50 0 50 350 50 0 0 0 0 0
intr 1 2 3
"""
STAT_B = """cpu  200 0 200 800 100 0 0 0 0 0
cpu0 150 0 50 350 50 0 0 0 0 0
cpu1 50 0 150 450 50 0 0 0 0 0
"""


class MonitorTests(unittest.TestCase):
    def test_cpu_percent_per_core_excludes_idle_and_iowait(self):
        pct = cpu_percent(read_cpu_times(STAT_A), read_cpu_times(STAT_B))
        self.assertAlmostEqual(pct['cpu'], 200 / 300 * 100)
        self.assertAlmostEqual(pct['cpu0'], 100.0)
        self.assertAlmostEqual(pct['cpu1'], 50.0)

    def test_unchanged_counters_give_zero_not_division_error(self):
        times = read_cpu_times(STAT_A)
        self.assertEqual(cpu_percent(times, times)['cpu0'], 0.0)

    def test_meminfo_uses_available_not_free(self):
        mem = read_meminfo('MemTotal: 4096000 kB\nMemFree: 100000 kB\n'
                           'MemAvailable: 3072000 kB\nSwapTotal: 2048000 kB\nSwapFree: 2047000 kB\n')
        self.assertEqual(mem['ram_used_mib'], 1000.0)
        self.assertEqual(mem['ram_pct'], 25.0)
        self.assertEqual(mem['swap_used_mib'], 1.0)

    def test_throttled_reports_only_current_bits(self):
        self.assertEqual(decode_throttled(0x50005), ['undervolt', 'throttled'])
        self.assertEqual(decode_throttled(0x0), [])


if __name__ == '__main__':
    unittest.main()
