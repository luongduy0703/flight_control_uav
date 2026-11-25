from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi"))

import math
import unittest
from unittest.mock import Mock

from console import execute, parse_command
from marker_hold import HoldConfig, MarkerFeed, MarkerHold, correction_ned, estimate
import test_offboard_control as controller_fixture


def marker_line(t_s, cam_mm=(0.0, 0.0, -1000.0), world=1, p_M_W_mm=(400.0, -700.0, 6231.0), yaw_deg=0.0):
    """Same format as marker_line() in his_perception_uav-main/src/support/launch_marker.cpp."""
    return (f'[INFO uavp.launch_marker] MARKER seen=1 t_ns={int(t_s * 1e9)} reproj_px=0.40 tilt_cam_deg=3.0 '
            f'p_cam_S_mm={cam_mm[0]:.1f},{cam_mm[1]:.1f},{cam_mm[2]:.1f} world={world} '
            f'p_M_W_mm={p_M_W_mm[0]:.1f},{p_M_W_mm[1]:.1f},{p_M_W_mm[2]:.1f} yaw_M_W_deg={yaw_deg:.2f} '
            f'tilt_W_deg=2.0 yaw_fc_deg=37.00 yaw_rel_deg=37.00')


def quiet_socket():
    sock = Mock()
    sock.recv.side_effect = BlockingIOError
    return sock


class MathTests(unittest.TestCase):
    def test_correction_points_camera_back_to_marker(self):
        # Arrow north: camera 10 cm along the arrow (north of the marker) -> move 10 cm south.
        dn, de = correction_ned({'cam_S_m': (0.1, 0.0, -1.0), 'yaw_deg': 0.0})
        self.assertAlmostEqual(dn, -0.1)
        self.assertAlmostEqual(de, 0.0)
        # Arrow east: x_S = east, y_S (right of the arrow) = south.
        dn, de = correction_ned({'cam_S_m': (0.1, 0.05, -1.0), 'yaw_deg': 90.0})
        self.assertAlmostEqual(dn, 0.05)
        self.assertAlmostEqual(de, -0.1)

    def test_estimate_median_rejects_outlier_and_reports_height(self):
        feed = MarkerFeed(sock=quiet_socket(), clock=lambda: 0.0)
        for i, x in enumerate((20.0, 22.0, 21.0, 400.0, 19.0)):
            self.assertTrue(feed.feed_line(marker_line(1.0 + 0.1 * i, cam_mm=(x, 0.0, -1250.0))))
        est = estimate(feed.since(0.0))
        self.assertEqual(est['n'], 5)
        self.assertAlmostEqual(est['cam_S_m'][0], 0.021)
        self.assertAlmostEqual(est['height_m'], 1.25)

    def test_feed_ignores_other_lines(self):
        feed = MarkerFeed(sock=quiet_socket(), clock=lambda: 0.0)
        self.assertFalse(feed.feed_line('MARKER seen=0 n=3 rejected=1 reason=khong_thay'))
        self.assertFalse(feed.feed_line('OBJARUCO seen=1 t_ns=1 p_cam_S_mm=1,2,3'))
        self.assertEqual(feed.received, 0)

    def test_config_limits(self):
        for kw in ({'max_step_m': 0.5}, {'max_total_m': 2.0}, {'deadband_m': 0.0}, {'min_samples': 0}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                HoldConfig(**kw)

    def test_parse_hold_command(self):
        self.assertEqual(parse_command('hold on'), ('hold', True))
        self.assertEqual(parse_command('HOLD off'), ('hold', False))
        with self.assertRaises(ValueError):
            parse_command('hold')


class HoldTests(unittest.TestCase):
    def setUp(self):
        # Controller fixture (fake MAVSDK + clock) from test_offboard_control, used by composition so its own
        # tests do not run again here.
        self.c = controller_fixture.ControllerTests('test_arm_then_takeoff_does_not_arm_twice')
        self.c.setUp()
        self.addCleanup(self.c.doCleanups)
        self.feed = MarkerFeed(sock=quiet_socket(), clock=lambda: self.c.now)
        self.hold = MarkerHold(self.feed, HoldConfig(), clock=lambda: self.c.now)
        self.out = []

    def see(self, n=6, dt=0.2, **kw):
        for _ in range(n):
            self.c.advance(dt)
            self.feed.feed_line(marker_line(self.c.now, **kw))

    def step(self):
        return self.hold.step(self.c.fc, output=self.out.append)

    def test_moves_toward_marker_with_step_limit(self):
        self.c.held()
        self.step()  # HOLD seen: window starts now
        self.see(cam_mm=(200.0, 0.0, -1000.0))  # camera 20 cm north of the marker
        target = dict(self.c.fc.target)
        self.assertIsNotNone(self.step())
        self.assertEqual(self.c.fc.phase, 'FLY')
        self.assertAlmostEqual(self.c.fc.target['north_m'], target['north_m'] - 0.10)  # max_step_m
        self.assertAlmostEqual(self.c.fc.target['east_m'], target['east_m'])
        self.assertAlmostEqual(self.c.fc.target['down_m'], target['down_m'])  # height untouched
        self.assertEqual(self.hold.moves, 1)

    def test_deadband_no_move(self):
        self.c.held()
        self.step()
        self.see(cam_mm=(20.0, -10.0, -1000.0))
        self.assertIsNone(self.step())
        self.assertEqual(self.c.fc.phase, 'HOLD')

    def test_samples_before_hold_are_ignored(self):
        self.c.takeoff()
        self.see(cam_mm=(300.0, 0.0, -1000.0))  # climbing: not HOLD
        self.assertIsNone(self.step())
        self.c.arrive()
        self.assertIsNone(self.step())  # HOLD just started: old samples do not count
        self.assertEqual(self.c.fc.phase, 'HOLD')

    def test_refuses_beyond_total_limit(self):
        self.c.held()
        self.step()
        self.c.fc.target['north_m'] = self.c.fc.launch_ned['north_m'] - 0.35
        self.see(cam_mm=(200.0, 0.0, -1000.0))
        self.assertIsNone(self.step())
        self.assertEqual(self.c.fc.phase, 'HOLD')
        self.assertIn('refused', self.hold.note)

    def test_no_fc_pose_measures_only(self):
        self.c.held()
        self.step()
        self.see(cam_mm=(200.0, 0.0, -1000.0), world=0)
        self.assertIsNone(self.step())
        self.assertEqual(self.c.fc.phase, 'HOLD')
        m = self.hold.measurement()
        self.assertAlmostEqual(m['horizontal_offset_m'], 0.2)
        self.assertAlmostEqual(m['camera_height_m'], 1.0)

    def test_stale_marker_no_move(self):
        self.c.held()
        self.step()
        self.see(cam_mm=(200.0, 0.0, -1000.0))
        self.c.advance(1.0)
        self.assertIsNone(self.step())
        self.assertEqual(self.c.fc.phase, 'HOLD')

    def test_scattered_marker_not_trusted(self):
        self.c.held()
        self.step()
        for i in range(6):
            self.c.advance(0.2)
            self.feed.feed_line(marker_line(self.c.now, cam_mm=(200.0, 0.0, -1000.0),
                                            p_M_W_mm=(400.0 + 300.0 * (i % 2), -700.0, 6231.0)))
        self.assertIsNone(self.step())
        self.assertIn('scattered', self.hold.note)

    def test_hold_off_command_and_status(self):
        self.c.held()
        execute(self.c.fc, 'hold off', output=self.out.append, marker=self.hold)
        self.step()
        self.see(cam_mm=(200.0, 0.0, -1000.0))
        self.assertIsNone(self.step())
        lines = []
        execute(self.c.fc, 'status', output=lines.append, marker=self.hold)
        self.assertIn('"aruco"', lines[-1])
        with self.assertRaises(ValueError):
            execute(self.c.fc, 'hold on', output=self.out.append)  # console without --marker-hold

    def test_period_between_corrections(self):
        self.c.held()
        self.step()
        self.see(cam_mm=(200.0, 0.0, -1000.0))
        self.step()
        self.c.arrive()
        self.step()
        self.see(n=3, cam_mm=(150.0, 0.0, -1000.0))
        self.assertIsNone(self.step())  # < period_s since the last correction
        self.assertEqual(self.hold.moves, 1)
        self.see(n=10, cam_mm=(150.0, 0.0, -1000.0))
        self.assertIsNotNone(self.step())
        self.assertEqual(self.hold.moves, 2)
        self.assertTrue(math.isfinite(self.c.fc.target['north_m']))


if __name__ == '__main__':
    unittest.main()
