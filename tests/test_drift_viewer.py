from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import unittest
from drift_viewer import Track, decimate, ned_to_xyz, padded_limits, stats_text


class DriftViewerTests(unittest.TestCase):
    def test_ned_down_becomes_up(self):
        self.assertEqual(ned_to_xyz(1.0, 2.0, -3.0), (1.0, 2.0, 3.0))

    def test_drift_and_path_length(self):
        track = Track()
        for i in range(5):
            self.assertEqual(track.add(i * 0.1, 0.1 * i, 0.0, 0.0), '')
        t, x, y, z, jumps, boot, max_h, path = track.snapshot()
        self.assertEqual(len(t), 5)
        self.assertAlmostEqual(max_h, 0.4)
        self.assertAlmostEqual(path, 0.4)
        self.assertEqual((jumps, boot), ([], 0))

    def test_jump_is_marked_and_not_counted_as_path(self):
        track = Track(jump_m=0.5)
        track.add(1.0, 0.0, 0.0, 0.0)
        self.assertEqual(track.add(1.03, 2.0, 0.0, 0.0), 'jump')
        self.assertEqual(track.snapshot()[4], [1])
        self.assertEqual(track.snapshot()[7], 0.0)

    def test_large_step_after_a_long_gap_is_not_a_jump(self):
        track = Track(jump_m=0.5, jump_window_s=0.5)
        track.add(1.0, 0.0, 0.0, 0.0)
        self.assertEqual(track.add(5.0, 2.0, 0.0, 0.0), '')

    def test_reboot_restarts_trail_with_new_origin(self):
        track = Track()
        track.add(100.0, 3.0, 4.0, 0.0)
        self.assertEqual(track.add(2.0, 0.0, 0.0, 0.0), 'reboot')
        t, _, _, _, _, boot, max_h, _ = track.snapshot()
        self.assertEqual((t, boot, max_h), ([2.0], 1, 0.0))

    def test_duplicate_or_late_packet_is_ignored(self):
        track = Track()
        track.add(10.0, 0.0, 0.0, 0.0)
        self.assertEqual(track.add(9.9, 5.0, 0.0, 0.0), 'stale')
        self.assertEqual(len(track.snapshot()[0]), 1)

    def test_decimate_keeps_first_and_last(self):
        values = list(range(10001))
        out = decimate(values, limit=100)
        self.assertLessEqual(len(out), 102)
        self.assertEqual((out[0], out[-1]), (0, 10000))
        self.assertEqual(decimate([1, 2, 3], limit=100), [1, 2, 3])

    def test_stats_text_reports_horizontal_drift(self):
        text = stats_text([1.0], [3.0], [4.0], [1.0], 0, 5.0, 5.0, [], 0.1, 30.0)
        self.assertIn('horizontal 5.00 m', text)
        self.assertIn('Waiting', stats_text([], [], [], [], 0, 0, 0, [], 0, 0))

    def test_padded_limits_keep_noise_small(self):
        lo, hi = padded_limits([0.001, 0.002], 0.5)
        self.assertAlmostEqual(hi - lo, 0.5)
        lo, hi = padded_limits([0.0, 2.0], 0.5)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 2.0)


if __name__ == '__main__':
    unittest.main()
