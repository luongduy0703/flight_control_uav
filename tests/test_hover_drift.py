from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import unittest
from hover_drift import IN_AIR, LANDING, ON_GROUND, HoverMonitor, slope
from drift_viewer import Track


def fly(monitor, drift_n_per_s=0.01, hover_s=20.0, launch_z=0.15, dt=0.1):
    """Ground 2 s -> climb to launch_z + 0.8 -> hover with drift -> LANDING -> ground."""
    events, i, t = [], 0, 0.0

    def step(x, y, z, landed, v=(0.0, 0.0, 0.0), r=None):
        nonlocal i, t
        events.append(monitor.update(i, t, x, y, z, *v, landed=landed, range_m=r))
        i += 1
        t += dt
    for _ in range(20):
        step(0.0, 0.0, launch_z, ON_GROUND, r=0.10)
    for k in range(1, 21):
        step(0.0, 0.0, launch_z + 0.04 * k, IN_AIR, v=(0, 0, 0.4), r=0.10 + 0.04 * k)
    start = t
    while t - start < hover_s:
        step(drift_n_per_s * (t - start), 0.0, launch_z + 0.8, IN_AIR, r=0.90)
    xh = drift_n_per_s * hover_s
    for k in range(1, 21):
        step(xh, 0.0, launch_z + 0.8 - 0.04 * k, LANDING, v=(0, 0, -0.4))
    step(xh + 0.05, 0.02, launch_z, ON_GROUND)
    return events


class HoverDriftTests(unittest.TestCase):
    def test_slope(self):
        self.assertAlmostEqual(slope([0, 1, 2, 3], [1, 3, 5, 7]), 2.0)
        self.assertEqual(slope([1], [1]), 0.0)

    def test_hover_measured_from_hover_start_after_settle(self):
        m = HoverMonitor(target_m=0.8, settle_s=2.0)
        events = fly(m, drift_n_per_s=0.01, hover_s=20.0)
        self.assertEqual(events.count('hover_start'), 1)
        self.assertEqual(events.count('hover_end'), 1)
        self.assertEqual(len(m.results), 1)
        r = m.results[0]
        self.assertEqual(r['reason_end'], 'landing')
        # Measuring starts 2 s into the 20 s hover: about 18 s at 0.01 m/s.
        self.assertAlmostEqual(r['duration_s'], 18.0, delta=0.3)
        self.assertAlmostEqual(r['final_drift_m']['horizontal'], 0.18, delta=0.01)
        self.assertAlmostEqual(r['drift_rate_m_per_min'], 0.6, delta=0.01)
        self.assertAlmostEqual(r['height_mean_m'], 0.8, delta=0.01)
        self.assertAlmostEqual(r['range_height_mean_m'], 0.8, delta=0.01)

    def test_landing_offset_is_relative_to_launch(self):
        m = HoverMonitor()
        fly(m, drift_n_per_s=0.01, hover_s=10.0)
        self.assertEqual(len(m.landings), 1)
        o = m.landings[0]['offset_from_launch_m']
        self.assertAlmostEqual(o['x_north'], 0.15, delta=0.01)
        self.assertAlmostEqual(o['y_east'], 0.02, delta=0.001)

    def test_height_uses_launch_point_not_boot_origin(self):
        m = HoverMonitor(target_m=0.8)
        fly(m, launch_z=0.5)  # EKF z already +0.5 m on the ground
        self.assertAlmostEqual(m.results[0]['height_mean_m'], 0.8, delta=0.01)

    def test_wrong_height_never_starts_a_hover(self):
        m = HoverMonitor(target_m=1.5)
        events = fly(m)
        self.assertNotIn('hover_start', events)
        self.assertEqual(m.results, [])

    def test_no_landed_state_no_measurement(self):
        m = HoverMonitor()
        for i in range(100):
            self.assertEqual(m.update(i, i * 0.1, 0.0, 0.0, 0.8, landed=None), '')
        self.assertIn('waiting for ON_GROUND', m.status_line())

    def test_leaving_height_band_ends_hover(self):
        m = HoverMonitor(settle_s=1.0, exit_m=0.3, exit_s=1.0)
        m.update(0, 0.0, 0, 0, 0, landed=ON_GROUND)
        t, i = 0.1, 1
        while t < 5:
            m.update(i, t, 0, 0, 0.8, landed=IN_AIR)
            t, i = t + 0.1, i + 1
        self.assertEqual(m.state, 'HOVER')
        events = []
        while t < 7:
            events.append(m.update(i, t, 0, 0, 1.5, landed=IN_AIR))
            t, i = t + 0.1, i + 1
        self.assertIn('hover_end', events)
        self.assertEqual(m.results[-1]['reason_end'], 'left height band')

    def test_reboot_mid_hover_closes_measurement(self):
        track = Track(hover=HoverMonitor(settle_s=0.5))
        track.add(10.0, 0, 0, 0, landed=ON_GROUND)
        for k in range(1, 20):
            track.add(10.0 + 0.1 * k, 0, 0, 0.8, landed=IN_AIR)
        self.assertEqual(track.hover.state, 'HOVER')
        self.assertIn('reboot', track.add(1.0, 0, 0, 0))
        self.assertEqual(track.hover.results[-1]['reason_end'], 'pixhawk reboot')
        self.assertEqual(track.hover.state, 'WAIT_LANDED_STATE')

    def test_rejects_bad_tolerances(self):
        with self.assertRaises(ValueError):
            HoverMonitor(tol_m=0.4, exit_m=0.3)


if __name__ == '__main__':
    unittest.main()
