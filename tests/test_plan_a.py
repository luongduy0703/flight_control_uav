from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'planA'))

import json
import math
import tempfile
import unittest

import plan_a
from run_plan_a import PlanA

ROOT = Path(__file__).resolve().parents[1]


def cfg(**changes):
    c = plan_a.validate(plan_a.merge(plan_a.DEFAULTS, changes))
    c['flight']['hover_stabilize_s'] = 0.0
    c['approach']['wait_after_arrival_s'] = 0.0
    c['approach']['align_wait_s'] = 0.0
    return c


def lift_off(plan):
    """Operator: arm + yes, takeoff + yes (nothing moves without both)."""
    for word in ('arm', 'yes', 'takeoff', 'yes'):
        plan.command(word)


LAUNCH = {'north_m': 1.0, 'east_m': 2.0, 'down_m': -0.05}


class ConfigTests(unittest.TestCase):
    def test_scenario_file_loads(self):
        c = plan_a.load_config(ROOT / 'planA' / 'config' / 'scenario.yaml')
        self.assertEqual(c['flight']['hover_height_m'], 1.0)
        self.assertEqual((c['approach']['final_height_m'], c['perception']['auto_start']), (0.8, False))
        self.assertEqual((c['video_diag']['enabled'], c['imu_log']['enabled'], c['perception']['fp16']),
                         (False, False, True))
        self.assertTrue(all(c['control'][k] for k in ('confirm_arm', 'confirm_takeoff', 'confirm_moves',
                                                      'require_arm_before_takeoff')))
        self.assertEqual((c['ground_truth']['x_m'], c['ground_truth']['y_m']), (0.635, 0.255))

    def test_bad_values_are_rejected(self):
        for change in ({'frame': {'axes': 'enu'}}, {'flight': {'hover_height_m': 5.0}},
                       {'approach': {'body_offset_m': [1.0]}}, {'nope': 1},
                       {'approach': {'final_height_m': 0.9}},  # above hover_height_m 0.8: not a descent
                       {'approach': {'final_height_m': 0.1}}, {'perception': {'auto_start': 'yes'}},
                       {'perception': {'detect_timeout_s': 0}}):
            with self.assertRaises(ValueError):
                plan_a.validate(plan_a.merge(plan_a.DEFAULTS, change))

    def test_perception_overrides(self):
        o = plan_a.perception_overrides(cfg(perception={'e_max_mm': 12.0, 'set': ['remote.max_fps=5']}))
        self.assertIn('lock_track.align_gate=true', o)
        self.assertIn('lock_track.align_e_max_mm=12.0', o)
        self.assertIn('s4_fusion.method=world', o)
        self.assertIn('sources.mavlink.use_rangefinder=false', o)
        self.assertEqual(o[-1], 'remote.max_fps=5')
        for stage in ('s1_detect', 's2_segment'):
            self.assertIn(f'{stage}.model.use_fp16_storage=true', o)  # fp16 mac dinh
        self.assertFalse(any('video_diag' in x for x in o))  # chan doan video tat mac dinh
        o = plan_a.perception_overrides(cfg(perception={'fp16': False}, video_diag={'enabled': True}))
        self.assertFalse(any('fp16' in x for x in o))
        self.assertIn('video_diag.enabled=true', o)
        with self.assertRaises(ValueError):
            cfg(perception={'fp16': 'yes'})

    def test_marker_ground_truth_from_scenario(self):
        import yaml
        c = plan_a.validate(plan_a.merge(plan_a.DEFAULTS, yaml.safe_load(
            (Path(__file__).resolve().parents[1] / 'planA' / 'config' / 'scenario.yaml').read_text())))
        gt = c['marker']['ground_truth']
        self.assertEqual((gt['x_m'], gt['y_m']), (0.425, -0.22))  # 42.5 cm theo mui ten, 22 cm ben trai
        self.assertIsNotNone(gt['z_m'])  # co z -> tinh sai so Euclid 3D

    def test_match_radius_override(self):
        self.assertIn('s3_select.match_radius_px=80.0', plan_a.perception_overrides(cfg()))
        o = plan_a.perception_overrides(cfg())
        self.assertIn('s3_select.world_track=true', o)
        self.assertIn('s3_select.match_radius_mm=80.0', o)
        self.assertIn('s3_select.same_object_mm=100.0', o)
        with self.assertRaises(ValueError):
            cfg(perception={'same_object_mm': 5.0})
        with self.assertRaises(ValueError):
            cfg(perception={'match_radius_px': 5.0})

    def test_single_shot_and_frame_gate_overrides(self):
        o = plan_a.perception_overrides(cfg())
        self.assertIn('s3_select.single_shot=true', o)
        self.assertIn('frame_gate.enabled=true', o)
        o = plan_a.perception_overrides(cfg(perception={'single_shot': False, 'frame_gate': False}))
        self.assertIn('s3_select.single_shot=false', o)
        self.assertIn('frame_gate.enabled=false', o)
        with self.assertRaises(ValueError):
            cfg(perception={'single_shot': 'yes'})

    def test_candidate_line_is_not_a_target(self):
        import perception_process
        line = ('[INFO uavp.live] CANDIDATE rank=1 id=2 class=box_top conf=0.912 world=1 '
                'p_W_mm=120.5,-340.0,912.0 center_px=1500.0,300.0')
        kind, values = perception_process.parse_marked(line)
        self.assertEqual(kind, 'candidate')
        self.assertEqual(values['p_W_mm'], (120.5, -340.0, 912.0))
        self.assertEqual(values['class'], 'box_top')

    def test_size_abort_locked_by_default(self):
        self.assertIn('s3_select.size_abort=false', plan_a.perception_overrides(cfg()))
        self.assertIn('s3_select.size_abort=true', plan_a.perception_overrides(cfg(perception={'size_abort': True})))
        with self.assertRaises(ValueError):
            cfg(perception={'size_abort': 'no'})

    def test_aruco_distance_error(self):
        import perception_process
        line = ('2026-11-22 18:40:01,000 INFO uavp.live: ARUCO_S4 found=1 world=1 p_W_mm=1.0,2.0,3.0 '
                'M_W_mm=1.0,2.0,3.0 p_C_mm=10.0,5.0,700.0 d_err_mm=4.00 p_err_mm=4.50 reproj_px=0.40 tilt_deg=3.0 '
                'd_s4_mm=704.0 d_aruco_mm=700.0 dist_s4_mm=710.0 dist_aruco_mm=705.0 plane_s4_mm=nan '
                'plane_aruco_mm=698.0 marker_dist_mm=702.0')
        kind, values = perception_process.parse_marked(line)
        self.assertEqual(kind, 'aruco')
        rep = plan_a.evaluate_aruco_distance(values)
        self.assertTrue(rep['found'])
        d, dist, plane = rep['rows']
        self.assertAlmostEqual(d['err_mm'], 4.0)
        self.assertAlmostEqual(d['err_pct'], 100 * 4.0 / 700.0)
        self.assertAlmostEqual(dist['err_mm'], 5.0)
        self.assertIsNone(plane['err_mm'])  # s4 khong co mat phang: khong tinh
        text = plan_a.format_aruco_distance(rep)
        self.assertIn('sai so +4.0 mm', text)
        self.assertIn('Lech tam vat (3D) 4.5 mm', text)
        missing = plan_a.evaluate_aruco_distance({'found': 0.0, 'd_s4_mm': 704.0, 'dist_s4_mm': 710.0,
                                                  'plane_s4_mm': 700.0})
        self.assertFalse(missing['found'])
        self.assertIn('KHONG thay ArUco', plan_a.format_aruco_distance(missing))
        self.assertIn('Khong co dong ARUCO_S4', plan_a.format_aruco_distance(None))

    def test_scan_gate_and_adaptive_window_overrides(self):
        o = plan_a.perception_overrides(cfg())
        self.assertIn('lock_track.wait_scan_command=true', o)  # tich luy chi sau lenh QUET (UAV o 0.8 m)
        self.assertIn('s4_fusion.world.max_window_s=10.0', o)
        self.assertIn('lock_track.scan_max_duration_s=12.0', o)
        for bad in ({'max_window_s': 1.0}, {'max_window_s': 20.0}, {'max_window_s': 10.0, 'scan_max_s': 8.0}):
            with self.assertRaises(ValueError):
                cfg(perception=bad)


class FrameTests(unittest.TestCase):
    def test_launch_heading_axes(self):
        # Nose to the East (yaw 90): scenario x = East, y = South.
        p = plan_a.ned_to_scenario((1.0, 3.0, -0.25), LAUNCH, 90.0, 'launch_heading')
        self.assertAlmostEqual(p[0], 1.0)
        self.assertAlmostEqual(p[1], 0.0)
        self.assertAlmostEqual(p[2], -0.2)
        n, e = plan_a.scenario_to_ned_xy(1.0, 0.5, LAUNCH, 90.0, 'launch_heading')
        self.assertAlmostEqual(n, 0.5)
        self.assertAlmostEqual(e, 3.0)
        self.assertAlmostEqual(plan_a.yaw_to_scenario(100.0, 90.0, 'launch_heading'), 10.0)
        self.assertAlmostEqual(plan_a.yaw_to_scenario(10.0, 90.0, 'launch_heading'), 100.0)

    def test_ned_axes_and_rangefinder_z(self):
        p = plan_a.ned_to_scenario((1.635, 2.255, -0.12), LAUNCH, 37.0, 'ned', 'rangefinder')
        self.assertAlmostEqual(p[0], 0.635)
        self.assertAlmostEqual(p[1], 0.255)
        self.assertAlmostEqual(p[2], -0.12)  # rangefinder: floor = 0 already

    def test_tilt(self):
        self.assertAlmostEqual(plan_a.tilt_deg((0.0, 0.0, 1.0)), 0.0)
        self.assertAlmostEqual(plan_a.tilt_deg((math.sin(0.1), 0.0, math.cos(0.1))), math.degrees(0.1))


class DecisionTests(unittest.TestCase):
    def test_approach_and_safety_limit(self):
        c = cfg(approach={'body_offset_m': [-0.1, 0.0]}, frame={'axes': 'ned'})
        (n, e), (x, y) = plan_a.approach_setpoint((1635.0, 2255.0, -120.0), LAUNCH, 0.0, c)
        self.assertAlmostEqual(x, 0.535)
        self.assertAlmostEqual((n, e)[0], 1.535)
        with self.assertRaises(ValueError):
            plan_a.approach_setpoint((5000.0, 2000.0, 0.0), LAUNCH, 0.0, c)

    def test_correction(self):
        c = cfg()
        self.assertIsNone(plan_a.correction_setpoint((1.5, 2.2), (5.0, 5.0, 0.0), LAUNCH, c))  # < 2 cm
        self.assertIsNone(plan_a.correction_setpoint((1.5, 2.2), (math.nan, 1.0, 0.0), LAUNCH, c))
        n, e = plan_a.correction_setpoint((1.5, 2.2), (0.0, 50.0, 0.0), LAUNCH, c)
        self.assertAlmostEqual((n, e)[1], 2.25)
        n, e = plan_a.correction_setpoint((1.5, 2.2), (0.0, 900.0, 0.0), LAUNCH, c)  # clipped to 0.30 m
        self.assertAlmostEqual(e, 2.5)

    def test_speed_window(self):
        w = plan_a.SpeedWindow(1.0)
        self.assertIsNone(w.speed())
        for i in range(11):
            w.add(i * 0.1, (0.05 * i, 0.0, 0.0))
        self.assertAlmostEqual(w.speed(), 0.5)


class ReportTests(unittest.TestCase):
    OBJECT = {'p_W_mm': (1640.0, 2250.0, -170.0), 'normal_W': (0.0, 0.0, 1.0), 'yaw_W_deg': 30.0,
              'mode': 'plane', 'd_mm': 598.0}

    def test_object_prefers_log_line_then_result_json(self):
        result = {'fusion': {'status': 'ok', 'mode': 'plane', 'd_mm': 600.0,
                             'world': {'p_obj_W_mm': [1.0, 2.0, 3.0], 'normal_W': [0, 0, 1], 'yaw_W_deg': None}},
                  'lock': {'p_W_mm': [9.0, 9.0, 9.0]}}
        self.assertEqual(plan_a.object_from(result, self.OBJECT)['source'], 'log OBJECT')
        o = plan_a.object_from(result, None)
        self.assertEqual((o['p_W_mm'], o['source']), ([1.0, 2.0, 3.0], 'result.json fusion.world'))
        result['fusion']['world']['p_obj_W_mm'] = [None, None, None]
        self.assertEqual(plan_a.object_from(result, None)['p_W_mm'], [9.0, 9.0, 9.0])
        self.assertIsNone(plan_a.object_from(None, None)['p_W_mm'])

    def test_errors_2d_and_3d(self):
        c = cfg(frame={'axes': 'ned'})
        rep = plan_a.evaluate(plan_a.object_from(None, self.OBJECT), LAUNCH, 0.0, c)
        self.assertAlmostEqual(rep['position_m'][0], 0.64)
        self.assertAlmostEqual(rep['error_2d_m'], math.hypot(0.005, -0.005))
        self.assertIsNone(rep['error_3d_m'])
        c['ground_truth']['z_m'] = -0.35
        rep = plan_a.evaluate(plan_a.object_from(None, self.OBJECT), LAUNCH, 0.0, c)
        # floor origin: z = -0.17 - (launch -0.05 + FC 0.25 above the floor) = -0.37
        self.assertAlmostEqual(rep['position_m'][2], -0.37)
        self.assertAlmostEqual(rep['height_above_floor_m'], 0.37)
        self.assertAlmostEqual(rep['error_3d_m'], math.sqrt(0.005 ** 2 + 0.005 ** 2 + 0.02 ** 2))
        text = plan_a.format_report(rep, c, {'to_target_s': 1.5, 'to_result_s': 9.0})
        self.assertIn('SAI SO EUCLID 3D: 2.1 cm', text)
        self.assertIn('-> ket qua cuoi 9.00 s', text)
        self.assertIn('KHONG co toa do', plan_a.format_report(plan_a.evaluate({}, LAUNCH, 0.0, c), c, {}))


# ---- whole scenario with fakes -------------------------------------------------------------------
class FakeFc:
    def __init__(self):
        self.phase = 'IDLE'
        self.takeoff_complete = self.target_reached = False
        self.launch_ned = self.target = None
        self.last_error = None
        self.pos = (1.0, 2.0, -0.05)
        self.flown = []
        self.landed = False
        self.calls = []

    def state(self, key, age=None):
        if key == 'armed':
            return self.phase not in ('IDLE', 'LANDED')
        if key == 'landed_state':
            return 'ON_GROUND' if self.phase in ('IDLE', 'ARMED', 'LANDED') else 'IN_AIR'
        if key == 'flight_mode':
            return 'OFFBOARD'
        if key != 'local_ned':
            raise ValueError(key)
        return dict(north_m=self.pos[0], east_m=self.pos[1], down_m=self.pos[2])

    def preflight_ok(self):
        return True

    def arm(self):
        self.calls.append('arm')
        self.phase = 'ARMED'

    def disarm(self):
        self.calls.append('disarm')
        self.phase = 'IDLE'

    def takeoff(self):
        assert self.phase == 'ARMED'
        self.calls.append('takeoff')
        self.launch_ned = dict(north_m=1.0, east_m=2.0, down_m=-0.05)
        self.target = dict(north_m=1.0, east_m=2.0, down_m=-0.85, yaw_deg=0.0)
        self.phase = 'TAKEOFF'

    def ready_for_fly(self):
        return self.phase == 'HOLD'

    def fly(self, n, e, down=None):
        assert self.phase == 'HOLD'
        self.flown.append((n, e) if down is None else (n, e, down))
        self.phase, self.target_reached = 'FLY', False

    def arrive(self):
        self.phase, self.target_reached, self.takeoff_complete = 'HOLD', True, True
        if self.flown:
            f = self.flown[-1]
            self.pos = (f[0], f[1], f[2] if len(f) == 3 else -0.85)

    def land(self):
        self.landed = True
        self.phase = 'LANDING'


class FakePerception:
    ids = {'companion_sysid': 1, 'companion_compid': 191, 'command_id': 31010}

    def __init__(self, run_dir):
        self.dir = run_dir.parent
        self.state, self.result, self.exit_code = 'STARTING', None, None
        self.queue = []
        self.closed = False

    def status(self):
        return {'state': self.state, 'last_result': self.result, 'last_error': None, 'exit_code': self.exit_code}

    def take_marked(self):
        q, self.queue = self.queue, []
        return q

    def close(self):
        self.closed = True


class FloorOriginTests(unittest.TestCase):
    def test_floor_below_fc_at_takeoff_is_zero(self):
        launch = {'north_m': 0.0, 'east_m': 0.0, 'down_m': -0.30}  # EKF z of the FC on the floor (range height ref)
        c = cfg(frame={'axes': 'ned'})
        self.assertEqual(plan_a.to_scenario((0.0, 0.0, -0.30), launch, 0.0, c)[2], 0.0 - 0.25)  # FC 25 cm up
        self.assertAlmostEqual(plan_a.to_scenario((0.5, 0.2, -0.05), launch, 0.0, c)[2], 0.0)  # floor
        self.assertAlmostEqual(plan_a.to_scenario((0.5, 0.2, -0.11), launch, 0.0, c)[2], -0.06)  # 6 cm top
        self.assertAlmostEqual(plan_a.to_scenario((0.5, 0.2, -0.11), launch, 0.0, c, 'rangefinder')[2], -0.11)
        c['frame']['fc_above_ground_m'] = 0.0  # old behaviour: origin at the FC
        self.assertAlmostEqual(plan_a.to_scenario((0.5, 0.2, -0.11), launch, 0.0, c)[2], 0.19)

    def test_fc_height_is_validated(self):
        for bad in (-0.1, 1.5, float('nan'), 'x'):
            with self.assertRaises(ValueError):
                cfg(frame={'fc_above_ground_m': bad})


class ScenarioTests(unittest.TestCase):
    def test_full_flow_with_one_alignment_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run1'
            run_dir.mkdir()
            (run_dir / 'detect.jpg').write_bytes(b'x')
            (run_dir / 'result.json').write_text(json.dumps({'fusion': {'status': 'ok', 'mode': 'plane'}}))
            log_dir = Path(tmp) / 'log'
            log_dir.mkdir()
            out, commands = [], []
            fc, per = FakeFc(), FakePerception(run_dir)
            c = cfg(frame={'axes': 'ned'})
            c['ground_truth']['z_m'] = -0.371  # top 37.1 cm above the floor (EKF z -0.171, FC 0.25 above the floor)
            now = [0.0]
            plan = PlanA(c, fc, per, log_dir, out.append,
                         lambda f, ids, start, **kw: commands.append(start) or True, clock=lambda: now[0])
            plan.step()
            self.assertEqual(plan.state, 'WAIT_READY')
            per.state = 'WAITING'
            plan.step()
            self.assertEqual(plan.state, 'WAIT_TAKEOFF')
            with self.assertRaises(ValueError):
                plan.command('start_perception')  # not hovering yet
            lift_off(plan)
            self.assertEqual(plan.state, 'TAKEOFF')
            fc.pos = (1.0, 2.0, -0.85)
            fc.arrive()
            plan.step()
            self.assertEqual(plan.state, 'STABILIZE')
            for _ in range(12):  # steady hover for 1.2 s
                now[0] += 0.1
                plan.step()
            self.assertEqual(plan.state, 'WAIT_START')
            plan.command('start_perception')
            self.assertEqual((plan.state, commands), ('SCANNING', [True]))
            per.state = 'SCANNING'
            per.queue.append(('target', {'p_W_mm': (1600.0, 2250.0, -120.0)}, 0.0))
            plan.step()
            self.assertEqual((plan.state, fc.flown), ('SCANNING', []))  # proposal only: waits for "go"
            plan.command('go')
            self.assertEqual(plan.state, 'APPROACH')
            self.assertAlmostEqual(fc.flown[0][0], 1.6)
            fc.arrive()
            plan.step()
            self.assertEqual(plan.state, 'ALIGN')
            per.queue.append(('align', {'ok': 0.0, 'e_mm': 40.0, 'e_max_mm': 15.0, 'dW_mm': (35.0, 0.0, 0.0)}, 0.0))
            plan.step()
            self.assertEqual((plan.state, len(fc.flown)), ('ALIGN', 1))  # correction proposed, not flown
            plan.command('go')
            self.assertEqual(plan.state, 'APPROACH')  # correction toward the object
            self.assertAlmostEqual(fc.flown[1][0], 1.635)
            fc.arrive()
            per.queue.append(('align', {'ok': 1.0, 'e_mm': 4.0, 'e_max_mm': 15.0}, 0.0))
            plan.step()
            self.assertIn('to_aligned_s', plan.times)
            per.queue.append(('object', {'p_W_mm': (1636.0, 2254.0, -171.0), 'normal_W': (0.0, 0.0, 1.0),
                                         'yaw_W_deg': 5.0, 'mode': 'plane', 'd_mm': 700.0}, 0.0))
            per.result, per.state, per.exit_code = 'run1', 'STOPPED', 0
            plan.step()
            self.assertEqual(plan.state, 'DONE')
            self.assertTrue(per.closed)
            self.assertFalse(fc.landed)  # auto_land false: waits for "land"
            self.assertLess(plan.report['error_3d_m'], 0.002)
            self.assertTrue((log_dir / 'perception' / 'detect.jpg').is_file())
            summary = json.loads((log_dir / 'summary.json').read_text())
            self.assertEqual(summary['corrections'], 1)
            self.assertTrue(any('SAI SO EUCLID 3D' in line for line in out))
            plan.command('land')
            self.assertTrue(fc.landed)

    def test_takeover_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            fc, per = FakeFc(), FakePerception(Path(tmp) / 'r')
            per.state = 'WAITING'
            plan = PlanA(cfg(), fc, per, Path(tmp), [].append, lambda *a, **kw: True)
            plan.step()
            lift_off(plan)
            fc.phase = 'EXTERNAL'
            plan.step()
            self.assertEqual(plan.state, 'ABORTED')


class SafetyGateTests(unittest.TestCase):
    """Nothing arms, takes off or moves unless the operator types it (and confirms)."""

    def _plan(self, tmp, **changes):
        fc, per = FakeFc(), FakePerception(Path(tmp) / 'r')
        out, now = [], [0.0]
        per.state = 'WAITING'
        plan = PlanA(cfg(**changes), fc, per, Path(tmp), out.append, lambda *a, **kw: True, clock=lambda: now[0])
        plan.step()
        return plan, fc, per, out, now

    def test_ready_does_nothing_by_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, per, out, now = self._plan(tmp)
            for _ in range(50):
                now[0] += 1.0
                plan.step()
            self.assertEqual((plan.state, fc.phase, fc.calls), ('WAIT_TAKEOFF', 'IDLE', []))

    def test_takeoff_needs_arm_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, *_ = self._plan(tmp)
            with self.assertRaises(ValueError):
                plan.command('takeoff')
            self.assertEqual(fc.calls, [])

    def test_no_and_timeout_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, per, out, now = self._plan(tmp)
            plan.command('arm')
            plan.command('no')
            self.assertEqual(fc.calls, [])
            with self.assertRaises(ValueError):
                plan.command('yes')  # nothing pending any more
            plan.command('arm')
            now[0] += 9.0  # > confirm_timeout_s 8
            with self.assertRaises(ValueError):
                plan.command('yes')
            self.assertEqual(fc.calls, [])
            self.assertTrue(any('KIEM TRA' in line for line in out))  # checklist shown before asking

    def test_takeoff_refused_after_px4_auto_disarm(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, *_ = self._plan(tmp)
            plan.command('arm')
            plan.command('yes')
            plan.command('takeoff')
            fc.phase = 'IDLE'  # PX4 disarmed by itself while the operator thought
            with self.assertRaises(ValueError):
                plan.command('yes')
            self.assertEqual(fc.calls, ['arm'])

    def _hovering_scan(self, tmp):
        run_dir = Path(tmp) / 'run1'
        run_dir.mkdir()
        (run_dir / 'result.json').write_text(json.dumps({'fusion': {'status': 'ok', 'mode': 'plane'}}))
        plan, fc, per, out, now = self._plan(tmp, frame={'axes': 'ned'},
                                             approach={'lidar_line': False, 'final_height_m': 0.8})
        lift_off(plan)
        fc.pos = (1.0, 2.0, -0.85)
        fc.arrive()
        plan.step()
        for _ in range(12):
            now[0] += 0.1
            plan.step()
        self.assertEqual(plan.state, 'WAIT_START')  # auto_start false: waits for the operator
        plan.command('s_p')  # lenh ngan
        per.state = 'SCANNING'
        per.queue.append(('target', {'p_W_mm': (1600.0, 2250.0, -60.0)}, 0.0))
        plan.step()
        return plan, fc, per, out, now

    def test_garbled_command_is_rejected_and_echoed(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, per, out, now = self._hovering_scan(tmp)
            with self.assertRaises(ValueError) as e:
                plan.command('start_perceptiomstart_land')  # dong go bi dinh chu (2026-11-22): KHONG doan, KHONG land
            self.assertIn('"start_perceptiomstart_land"', str(e.exception))
            self.assertIn('s_p', str(e.exception))
            self.assertFalse(fc.landed)

    def test_move_is_only_a_proposal_until_go(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, per, out, now = self._hovering_scan(tmp)
            for _ in range(5):
                now[0] += 0.5
                plan.step()
            self.assertEqual((plan.state, fc.flown), ('SCANNING', []))
            self.assertTrue(any('DE XUAT BAY TOI VAT' in line for line in out))
            with self.assertRaises(ValueError):
                plan.command('yes')  # "yes" is not "go"
            plan.command('skip')
            self.assertEqual((plan.state, fc.flown, plan.proposal), ('SCANNING', [], None))
            with self.assertRaises(ValueError):
                plan.command('go')  # nothing left to fly to

    def test_go_waits_for_hold_and_land_clears_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, per, out, now = self._hovering_scan(tmp)
            fc.phase = 'FLY'
            with self.assertRaises(ValueError):
                plan.command('go')
            self.assertIsNotNone(plan.proposal)
            fc.phase = 'HOLD'
            plan.command('land')
            self.assertEqual((plan.proposal, fc.landed, fc.flown), (None, True, []))
            with self.assertRaises(ValueError):
                plan.command('go')

    def test_skip_descend_scans_at_current_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, per, out, now = self._hovering_scan(tmp)
            plan.command('go')  # approach
            fc.arrive()
            plan.step()
            self.assertEqual(plan.state, 'DESCEND')  # descend BEFORE scanning
            plan.step()
            self.assertEqual(plan.proposal['kind'], 'descend')
            plan.command('skip')
            self.assertEqual((plan.state, len(fc.flown), fc.landed), ('ALIGN', 1, False))
            self.assertTrue(plan.scan_sent)
            per.queue.append(('object', {'p_W_mm': (1600.0, 2250.0, -60.0), 'mode': 'plane'}, 0.0))
            per.result, per.state, per.exit_code = 'run1', 'STOPPED', 0
            plan.step()
            self.assertEqual((plan.state, fc.landed), ('DONE', False))
            self.assertTrue(any('Go "land"' in line for line in out))

    def test_exit_refused_while_armed_or_flying(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, *_ = self._plan(tmp)
            self.assertFalse(plan.command('exit'))  # on the ground, disarmed: ok
            plan.command('arm')
            plan.command('yes')
            with self.assertRaises(ValueError):
                plan.command('exit')
            self.assertFalse(plan.command('exit force'))

    def test_confirmations_can_be_turned_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, *_ = self._plan(tmp, control={'confirm_arm': False, 'confirm_takeoff': False})
            plan.command('arm')
            plan.command('takeoff')
            self.assertEqual((fc.calls, plan.state), (['arm', 'takeoff'], 'TAKEOFF'))

    def test_control_block_is_validated(self):
        for bad in ({'confirm_timeout_s': 30.0}, {'confirm_timeout_s': 0.5}, {'confirm_arm': 'yes'}):
            with self.assertRaises(ValueError):
                cfg(control=bad)


if __name__ == '__main__':
    unittest.main()


# ---- launch ArUco (evaluation only) ---------------------------------------------------------------
def mk(p=(1000.0, 2000.0, 0.0), yaw=90.0, cam=(5.0, -3.0, -150.0), rel=1.0):
    """Values of one perception MARKER line (marker 1 m N / 2 m E, arrow pointing East)."""
    return {'seen': 1.0, 'world': 1.0, 'p_M_W_mm': p, 'yaw_M_W_deg': yaw, 'p_cam_S_mm': cam, 'yaw_rel_deg': rel}


# Ground samples: camera (5, -3) mm from the marker centre, nose along the arrow -> marker centre in the scenario
# frame = t_BC - camera = (0.125, -0.027). Object at scenario (0.636, 0.254, -0.371): tape 1 mm short in x.
MARKER_GT = {'x_m': 0.510, 'y_m': 0.281, 'z_m': -0.371}


class MarkerTests(unittest.TestCase):
    def test_overrides_only_when_enabled(self):
        self.assertFalse(any(o.startswith('launch_marker') for o in plan_a.perception_overrides(cfg())))
        o = plan_a.perception_overrides(cfg(marker={'enabled': True, 'marker_id': 3, 'side_mm': 98.5}))
        self.assertIn('launch_marker.enabled=true', o)
        self.assertIn('launch_marker.marker_id=3', o)
        self.assertIn('launch_marker.side_mm=98.5', o)

    def test_bad_marker_values_rejected(self):
        with self.assertRaises(ValueError):
            cfg(marker={'enabled': True, 'side_mm': 0})
        with self.assertRaises(ValueError):
            cfg(marker={'ground_truth': {'x_m': 'abc'}})

    def test_frame_estimate_is_robust(self):
        samples = [mk(), mk(p=(1002.0, 2000.0, 0.0), yaw=91.0), mk(p=(1500.0, 2000.0, 0.0), yaw=89.0),
                   {'seen': 1.0, 'world': 0.0, 'p_M_W_mm': (float('nan'),) * 3, 'yaw_M_W_deg': float('nan')}]
        f = plan_a.estimate_marker_frame(samples)
        self.assertEqual(f['n'], 3)  # the sample without FC pose is ignored
        self.assertAlmostEqual(f['p_W_m'][0], 1.002)  # median: the 0.5 m outlier does not pull it
        self.assertAlmostEqual(f['yaw_deg'], 90.0, places=6)
        self.assertIsNone(plan_a.estimate_marker_frame([]))
        wrap = plan_a.estimate_marker_frame([mk(yaw=179.0), mk(yaw=-179.0)])
        self.assertAlmostEqual(abs(wrap['yaw_deg']), 180.0, places=6)

    def test_ned_to_marker_frame(self):
        f = {'p_W_m': (1.0, 2.0, 0.0), 'yaw_deg': 90.0}
        x, y, z = plan_a.ned_to_marker((1.636, 2.254, -0.171), f)  # arrow East: x = East, y = South
        self.assertAlmostEqual(x, 0.254)
        self.assertAlmostEqual(y, -0.636)
        self.assertAlmostEqual(z, -0.171)

    def test_placement_ground_truth(self):
        """Tape (marker frame) -> scenario frame through the UAV placement (camera vs marker + t_BC), not EKF."""
        gt = {'x_m': -0.266, 'y_m': 0.560, 'z_m': -0.061}
        c = cfg(marker={'enabled': True, 'ground_truth': gt})
        t_BC = (0.13, -0.03, 0.11)
        g = plan_a.placement_ground_truth({'offset_xy_m': (0.02, -0.01), 'yaw_rel_deg': 0.0}, t_BC, 0.0, c)
        for a, b in zip(g['marker_m'], (0.11, -0.02, 0.0)):
            self.assertAlmostEqual(a, b)
        for a, b in zip(g['gt_m'], (-0.156, 0.54, -0.061)):
            self.assertAlmostEqual(a, b)
        # nose 90 deg right of the arrow: arrow = nose left, tape "right of arrow" = forward
        g = plan_a.placement_ground_truth({'offset_xy_m': (0.02, -0.01), 'yaw_rel_deg': 90.0}, t_BC, 0.0, c)
        for a, b in zip(g['gt_m'], (0.70, 0.256, -0.061)):
            self.assertAlmostEqual(a, b)
        c_ned = cfg(frame={'axes': 'ned'}, marker={'enabled': True, 'ground_truth': gt})
        g = plan_a.placement_ground_truth({'offset_xy_m': (0.02, -0.01), 'yaw_rel_deg': 90.0}, t_BC, 90.0, c_ned)
        for a, b in zip(g['gt_m'], (-0.256, 0.70, -0.061)):  # heading East at takeoff
            self.assertAlmostEqual(a, b)
        self.assertIsNone(plan_a.placement_ground_truth({'offset_xy_m': None}, t_BC, 0.0, c))
        self.assertIsNone(plan_a.placement_ground_truth({'offset_xy_m': (0, 0), 'yaw_rel_deg': 0}, None, 0.0, c))

    def test_evaluate_marker_errors(self):
        gt = {'x_m': -0.266, 'y_m': 0.560, 'z_m': -0.061}
        c = cfg(marker={'enabled': True, 'ground_truth': gt})
        check = {'offset_xy_m': (0.02, -0.01), 'yaw_rel_deg': 0.0, 'offset_m': 0.022, 'height_m': 0.15, 'n': 3,
                 'ok': True}
        launch = {'north_m': 1.0, 'east_m': 2.0, 'down_m': -0.30}  # FC 0.25 m above the floor
        obj = {'p_W_mm': [1000.0 - 156.0 + 1.0, 2000.0 + 540.0, -50.0 - 61.0]}  # 1 mm ahead of the tape point
        rep = plan_a.evaluate_marker(obj, launch, 0.0, check, (0.13, -0.03, 0.11), c)
        self.assertAlmostEqual(rep['error_3d_m'], 0.001, places=6)
        self.assertAlmostEqual(rep['height_above_floor_m'], 0.061)
        ground = {'p_W_m': (1.0, 2.0, 0.0), 'yaw_deg': 0.0}
        hover = {'p_W_m': (1.03, 2.04, 0.0), 'yaw_deg': 1.0}
        text = plan_a.format_marker_report(rep, check, ground, hover, c)
        self.assertIn('SAI SO EUCLID 3D (ArUco): 0.1 cm', text)
        self.assertIn('EKF TROI (marker dat -> hover): 5.0 cm', text)
        no_gt = plan_a.evaluate_marker(obj, launch, 0.0, check, (0.13, -0.03, 0.11), cfg(marker={'enabled': True}))
        self.assertIsNone(no_gt['error_m'])
        self.assertIn('chua nhap marker.ground_truth', plan_a.format_marker_report(no_gt, None, None, None, c))
        no_place = plan_a.evaluate_marker(obj, launch, 0.0, None, (0.13, -0.03, 0.11), c)
        self.assertIn('khong co vi tri dat UAV', no_place['reason'])
        self.assertFalse(plan_a.evaluate_marker({'p_W_mm': None}, launch, 0.0, check, None, c)['ok'])

    def test_launch_check(self):
        c = cfg(marker={'enabled': True})
        ok = plan_a.launch_check([mk(), mk(cam=(7.0, -1.0, -152.0), rel=2.0)], c)
        self.assertTrue(ok['ok'])
        self.assertAlmostEqual(ok['height_m'], 0.151)
        self.assertIn('-> OK', plan_a.format_launch_check(ok, c))
        far = plan_a.launch_check([mk(cam=(40.0, 0.0, -150.0))], c)
        self.assertFalse(far['ok'])
        self.assertIn('VUOT NGUONG', plan_a.format_launch_check(far, c))
        self.assertFalse(plan_a.launch_check([mk(rel=12.0)], c)['ok'])
        self.assertIn('KHONG thay', plan_a.format_launch_check(plan_a.launch_check([], c), c))

    def test_marker_lines_never_change_setpoints(self):
        """Full flow with the marker on: setpoints equal the EKF-only flow; the marker only adds an evaluation."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run1'
            run_dir.mkdir()
            (run_dir / 'launch_marker.csv').write_text('t_ns\n')
            (run_dir / 'result.json').write_text(json.dumps({'fusion': {'status': 'ok', 'mode': 'plane'}}))
            log_dir = Path(tmp) / 'log'
            log_dir.mkdir()
            out = []
            fc, per = FakeFc(), FakePerception(run_dir)
            c = cfg(frame={'axes': 'ned'}, approach={'lidar_line': False},
                    marker={'enabled': True, 'require_at_takeoff': True, 'ground_truth': MARKER_GT})
            now = [0.0]
            g = {'t_BC': (0.13, -0.03, 0.11)}  # only T_BC is used with lidar_line off
            plan = PlanA(c, fc, per, log_dir, out.append, lambda f, ids, start, **kw: True, clock=lambda: now[0],
                         geometry=g)
            per.state = 'WAITING'
            plan.step()
            plan.command('arm')
            plan.command('yes')
            plan.command('takeoff')
            with self.assertRaises(ValueError):
                plan.command('yes')  # require_at_takeoff: marker not seen yet
            self.assertEqual(fc.phase, 'ARMED')
            for _ in range(3):
                per.queue.append(('marker', mk(p=(990.0, 2000.0, 0.0), rel=0.0), now[0]))
            plan.command('takeoff')
            plan.command('yes')
            self.assertTrue(plan.launch_marker_check['ok'])
            fc.pos = (1.0, 2.0, -0.85)
            fc.arrive()
            plan.step()
            for _ in range(12):
                now[0] += 0.1
                per.queue.append(('marker', mk(cam=(10.0, 0.0, -800.0)), now[0]))
                plan.step()
            self.assertEqual(plan.state, 'WAIT_START')
            self.assertIn('ArUco: camera x 0.010', plan.status_line())
            plan.command('start_perception')
            self.assertEqual(plan.marker_frame['source'], 'hover 3 s cuoi')
            per.state = 'SCANNING'
            per.queue.append(('target', {'p_W_mm': (1600.0, 2250.0, -120.0)}, now[0]))
            plan.step()
            plan.command('go')
            self.assertEqual(fc.flown, [(1.6, 2.25)])  # EKF target, untouched by the marker
            fc.arrive()
            plan.step()
            per.queue.append(('align', {'ok': 1.0, 'e_mm': 4.0, 'e_max_mm': 15.0}, now[0]))
            per.queue.append(('object', {'p_W_mm': (1636.0, 2254.0, -171.0), 'mode': 'plane'}, now[0]))
            per.result, per.state, per.exit_code = 'run1', 'STOPPED', 0
            plan.step()
            self.assertEqual(plan.state, 'DONE')
            self.assertAlmostEqual(plan.marker_report['error_3d_m'], 0.001, places=6)
            summary = json.loads((log_dir / 'summary.json').read_text())
            self.assertAlmostEqual(summary['marker']['ekf_drift_ground_to_hover']['distance_m'], 0.01)
            self.assertEqual(len(summary['marker']['samples']), 15)
            self.assertTrue((log_dir / 'perception' / 'launch_marker.csv').is_file())
            self.assertTrue(any('SAI SO EUCLID 3D (ArUco): 0.1 cm' in line for line in out))
            self.assertTrue(any('EKF TROI (marker dat -> hover): 1.0 cm' in line for line in out))
            self.assertTrue(any('EKF TROI (marker luc dat -> hover): 1.0 cm' in line for line in out))


# ---- fly to the bbox centre with the LiDAR line across it ----------------------------------------
GEOMETRY = plan_a.load_lidar_geometry(ROOT / 'his_perception_uav-main', cfg())


def _rot_yaw(deg):
    a = math.radians(deg)
    return [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]


def _mat(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def scan_plane_distance_mm(body_ned_m, yaw_deg, target_m, g=GEOMETRY):
    """Distance (mm) of the target to the LiDAR scan plane with the UAV level at body_ned_m, heading yaw_deg."""
    R_WB = _rot_yaw(yaw_deg)
    R_WC = _mat(R_WB, g['R_BC'])
    cam = [b + o for b, o in zip(body_ned_m, plan_a._matvec(R_WB, g['t_BC']))]
    lidar = [c + o for c, o in zip(cam, plan_a._matvec(R_WC, g['t_CL']))]
    n = plan_a._matvec(R_WC, [g['R_CL'][i][2] for i in range(3)])
    return 1000.0 * abs(sum(n[i] * (target_m[i] - lidar[i]) for i in range(3)))


class LidarLineTests(unittest.TestCase):
    def test_foot_point_of_project_calibration(self):
        x, y = plan_a.lidar_foot_body(GEOMETRY, 0.75)
        # T_BC v1: camera 13 cm ahead / 3 cm left of the FC; scan line ~2.4 cm behind the camera -> x_B ~ +10.6 cm
        self.assertAlmostEqual(x, 0.106, delta=0.003)
        self.assertAlmostEqual(y, -0.029, delta=0.003)
        self.assertIsNone(plan_a.lidar_foot_body(GEOMETRY, float('nan')))
        self.assertAlmostEqual(plan_a.camera_depth_to([0.0, 0.0, -60.0], -0.85, GEOMETRY), 0.68)  # camera 11 cm below FC

    def test_setpoint_puts_the_target_on_the_scan_plane(self):
        for yaw in (0.0, 30.0, 167.7, -120.0):
            for axes in ('ned', 'launch_heading'):
                c = cfg(frame={'axes': axes})
                target_mm = [1636.0, 2254.0, -60.0]
                d = plan_a.camera_depth_to(target_mm, -0.85, GEOMETRY)
                (n, e), _ = plan_a.approach_setpoint(target_mm, LAUNCH, yaw, c, plan_a.lidar_foot_body(GEOMETRY, d))
                dist = scan_plane_distance_mm((n, e, -0.85), yaw, [v / 1000.0 for v in target_mm])
                self.assertLess(dist, 0.5, f'yaw {yaw} axes {axes}')
                # without the offset the line misses by the ~11 cm lever (T_CL + T_BC v1)
                (n0, e0), _ = plan_a.approach_setpoint(target_mm, LAUNCH, yaw, c)
                self.assertGreater(scan_plane_distance_mm((n0, e0, -0.85), yaw, [v / 1000.0 for v in target_mm]), 15)

    def test_flies_to_bbox_centre_not_mask_centre(self):
        with tempfile.TemporaryDirectory() as tmp:
            fc, per = FakeFc(), FakePerception(Path(tmp) / 'r')
            out = []
            c = cfg(frame={'axes': 'ned'})
            now = [0.0]
            plan = PlanA(c, fc, per, Path(tmp), out.append, lambda *a, **kw: True, clock=lambda: now[0],
                         geometry=GEOMETRY)
            per.state = 'WAITING'
            plan.step()
            lift_off(plan)
            fc.pos = (1.0, 2.0, -0.85)
            fc.arrive()
            plan.step()
            for _ in range(12):
                now[0] += 0.1
                plan.step()
            plan.command('start_perception')
            bbox = (1650.0, 2240.0, -60.0)
            per.queue.append(('target', {'p_W_mm': (1600.0, 2250.0, -60.0), 'bbox_p_W_mm': bbox}, 0.0))
            plan.step()
            plan.command('go')
            self.assertEqual(plan.state, 'APPROACH')
            n, e = fc.flown[0]
            self.assertLess(scan_plane_distance_mm((n, e, -0.85), 0.0, [v / 1000.0 for v in bbox]), 0.5)
            fx, fy = plan_a.lidar_foot_body(GEOMETRY, 0.68)  # camera 0.68 m above the 6 cm top
            self.assertAlmostEqual(n, 1.65 - fx, delta=0.003)  # body ~10.5 cm behind: line crosses the bbox centre
            self.assertAlmostEqual(e, 2.24 - fy, delta=0.003)
            self.assertTrue(any('tam bbox' in line for line in out))
            self.assertTrue(any('Duong LiDAR o do sau' in line for line in out))


# ---- flight scenario: hover 1 m -> auto start -> object -> descend 0.8 m / no object -> keep hovering ----
def _auto_plan(tmp, **changes):
    run_dir = Path(tmp) / 'run1'
    run_dir.mkdir()
    (run_dir / 'result.json').write_text(json.dumps({'fusion': {'status': 'ok', 'mode': 'plane'}}))
    log_dir = Path(tmp) / 'log'
    log_dir.mkdir()
    out, commands, now = [], [], [0.0]
    fc, per = FakeFc(), FakePerception(run_dir)
    base = {'frame': {'axes': 'ned'}, 'flight': {'hover_height_m': 1.0},
            'approach': {'lidar_line': False, 'final_height_m': 0.8},
            'perception': {'auto_start': True, 'detect_timeout_s': 10.0}}
    for key, value in changes.items():
        base[key] = {**base.get(key, {}), **value}
    sent_kw = []
    plan = PlanA(cfg(**base), fc, per, log_dir, out.append,
                 lambda f, ids, start, **kw: commands.append(start) or sent_kw.append(kw) or True, clock=lambda: now[0])
    plan.sent_kw = sent_kw
    per.state = 'WAITING'
    plan.step()
    lift_off(plan)
    fc.pos = (1.0, 2.0, -1.05)
    fc.arrive()
    plan.step()
    for _ in range(12):  # steady hover -> perception starts by itself
        now[0] += 0.1
        plan.step()
    return plan, fc, per, out, commands, now


class AutoFlowTests(unittest.TestCase):
    def test_auto_start_descend_then_scan_with_lidar_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, per, out, commands, now = _auto_plan(tmp)
            self.assertEqual((plan.state, commands), ('SCANNING', [True]))  # no "start_perception" typed
            # muc san EKF gui kem lenh bat dau: z FC luc cat canh (-0.05) + fc_above_ground_m (0.25)
            self.assertAlmostEqual(plan.sent_kw[0]['floor_down_m'], -0.05 + 0.25)
            per.state = 'SCANNING'
            per.queue.append(('target', {'p_W_mm': (1600.0, 2250.0, 140.0)}, 0.0))
            plan.step()
            plan.command('go')
            self.assertEqual(fc.flown, [(1.6, 2.25)])  # approach keeps the hover altitude
            fc.arrive()
            plan.step()
            self.assertEqual(plan.state, 'DESCEND')  # over the object: descend first, no scan yet
            plan.step()
            self.assertTrue(any('DE XUAT HA DO CAO' in line for line in out))
            self.assertEqual(commands, [True])
            plan.command('go')
            n, e, d = fc.flown[-1]
            self.assertEqual((n, e), (1.6, 2.25))  # same place, lower
            self.assertAlmostEqual(d, -0.05 - 0.8)  # takeoff FC z - final_height_m
            self.assertEqual(plan.state, 'DESCEND')
            fc.arrive()
            plan.step()
            self.assertEqual(plan.state, 'ALIGN')  # at 0.8 m: SCAN command -> perception accumulates
            self.assertEqual(commands, [True, 'scan'])
            self.assertTrue(any('DA HA DO CAO' in line for line in out))
            per.queue.append(('align', {'ok': 1.0, 'e_mm': 4.0, 'e_max_mm': 15.0}, 0.0))
            per.queue.append(('object', {'p_W_mm': (1600.0, 2250.0, 139.0), 'mode': 'plane', 'plane_dist_mm': 693.0,
                                         'n_mask_pts': 240.0, 'window_s': 4.0, 't_plane_s': 3.6}, 0.0))
            per.queue.append(('aruco', {'found': 1.0, 'd_s4_mm': 693.0, 'd_aruco_mm': 690.0, 'dist_s4_mm': 700.0,
                                        'dist_aruco_mm': 698.0, 'plane_s4_mm': 692.0, 'plane_aruco_mm': 689.5}, 0.0))
            per.result, per.state, per.exit_code = 'run1', 'STOPPED', 0
            plan.step()
            self.assertEqual((plan.state, fc.landed), ('DONE', False))  # no second descend
            self.assertAlmostEqual(plan.aruco_report['rows'][0]['err_mm'], 3.0)
            summary = json.loads((plan.log_dir / 'summary.json').read_text())
            self.assertAlmostEqual(summary['aruco_object_distance']['rows'][2]['err_mm'], 2.5)
            self.assertTrue(any('SAI SO KHOANG CACH TUYET DOI' in line for line in out))
            self.assertAlmostEqual(plan.report['height_above_floor_m'], 0.061)  # EKF camera pose + LiDAR distance
            self.assertAlmostEqual(plan.report['lidar_dist_m'], 0.693)
            self.assertTrue(any('240' in line and 'trong mask' in line for line in out))
            self.assertEqual((plan.report['window_s'], plan.report['t_plane_s']), (4.0, 3.6))
            self.assertTrue(any('ra PLANE sau 3.6 s' in line for line in out))

    def test_no_object_keeps_hovering(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, per, out, commands, now = _auto_plan(tmp)
            per.state = 'SCANNING'
            now[0] += 5.0
            plan.step()
            self.assertEqual(plan.state, 'SCANNING')
            now[0] += 6.0  # > detect_timeout_s without TARGET
            plan.step()
            self.assertEqual((plan.state, commands), ('WAIT_RESULT', [True, False]))
            per.state, per.exit_code = 'STOPPED', 0
            plan.step()
            self.assertEqual(plan.state, 'DONE')
            self.assertEqual((fc.flown, fc.landed), ([], False))  # no setpoint, no landing: still hovering
            self.assertTrue(any('KHONG THAY VAT' in line for line in out))

    def test_arm_disarm_only_before_takeoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            fc, per = FakeFc(), FakePerception(Path(tmp) / 'r')
            per.state = 'WAITING'
            plan = PlanA(cfg(), fc, per, Path(tmp), [].append, lambda *a, **kw: True)
            plan.step()
            plan.command('arm')
            self.assertEqual(fc.calls, [])  # asks first
            plan.command('yes')
            plan.command('disarm')
            self.assertEqual(fc.calls, ['arm', 'disarm'])
            lift_off(plan)
            for word in ('arm', 'disarm'):
                with self.assertRaises(ValueError):
                    plan.command(word)

    def test_manual_start_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, *_ = _auto_plan(tmp, perception={'auto_start': False})
            self.assertEqual(plan.state, 'WAIT_START')
            plan.command('start_perception')
            self.assertEqual(plan.state, 'SCANNING')


# ---- ArUco LAN DAU: he local danh gia + giu vi tri theo marker (khong vao EKF Pixhawk) --------------------------
class FirstMarkerTests(unittest.TestCase):
    def test_reference_is_the_first_samples_and_never_moves(self):
        early = [mk(p=(1000.0 + i, 2000.0, 0.0)) for i in range(5)]
        late = [mk(p=(1500.0, 2000.0, 0.0)) for _ in range(20)]
        ref = plan_a.first_marker_reference(early + late, 5)
        self.assertAlmostEqual(ref['p_W_m'][0], 1.002)
        self.assertIsNone(plan_a.first_marker_reference(early[:3] + [{'world': 0.0}], 5))

    def test_error_in_marker_frame(self):
        c = cfg(marker={'enabled': True, 'ground_truth': {'x_m': 0.5, 'y_m': 0.2, 'z_m': -0.06}})
        ref = {'n': 5, 'p_W_m': (1.0, 2.0, 0.0), 'yaw_deg': 90.0, 'spread_m': 0.0}  # mui ten huong Dong
        n, e, d = plan_a.marker_gt_world(ref, c)
        self.assertAlmostEqual(n, 1.0 - 0.2)  # ben phai mui ten Dong = Nam
        self.assertAlmostEqual(e, 2.5)
        self.assertAlmostEqual(d, -0.06)
        rep = plan_a.evaluate_marker_ref({'p_W_mm': (810.0, 2530.0, -50.0)}, ref, c)
        ex, ey, ez = rep['error_m']
        self.assertAlmostEqual(ex, 0.03)   # 3 cm xa hon theo mui ten
        self.assertAlmostEqual(ey, -0.01)  # y phai = Nam; N 0.81 > 0.80 -> lech 1 cm sang TRAI
        self.assertAlmostEqual(ez, 0.01)
        self.assertIn('SAI SO EUCLID 3D (ArUco lan dau)', plan_a.format_marker_ref_report(rep, c))
        self.assertFalse(plan_a.evaluate_marker_ref({'p_W_mm': None}, None, c)['ok'])

    def test_hold_step_deadband_step_and_limit(self):
        h = cfg()['marker']['hold']
        ref = {'p_W_m': (1.0, 2.0, 0.0)}
        self.assertIsNone(plan_a.hold_step(ref, {'p_W_m': (1.03, 2.0, 0.0)}, (0.0, 0.0), h)[0])  # < 5 cm
        new, _ = plan_a.hold_step(ref, {'p_W_m': (1.16, 2.0, 0.0)}, (0.0, 0.0), h)
        self.assertAlmostEqual(new[0], 0.10)  # buoc toi da 10 cm
        new, _ = plan_a.hold_step(ref, {'p_W_m': (1.16, 2.0, 0.0)}, new, h)
        self.assertAlmostEqual(new[0], 0.16)
        none, why = plan_a.hold_step(ref, {'p_W_m': (1.4, 2.0, 0.0)}, (0.0, 0.0), h)
        self.assertIsNone(none)
        self.assertIn('KHONG chinh', why)
        with self.assertRaises(ValueError):
            cfg(marker={'enabled': True, 'hold': {'max_total_m': 2.0}})

    def test_overrides_keep_marker_measured_and_excluded_during_scan(self):
        o = plan_a.perception_overrides(cfg(marker={'enabled': True, 'exclude_radius_mm': 120.0}))
        self.assertIn('launch_marker.track_during_scan=true', o)
        self.assertIn('launch_marker.exclude_radius_mm=120.0', o)

    def _hover(self, tmp, **marker):
        out, now = [], [0.0]
        fc, per = FakeFc(), FakePerception(Path(tmp) / 'r')
        per.state = 'WAITING'
        c = cfg(frame={'axes': 'ned'}, marker={'enabled': True, 'min_samples': 3, **marker})
        plan = PlanA(c, fc, per, Path(tmp), out.append, lambda *a, **kw: True, clock=lambda: now[0])
        plan.step()
        lift_off(plan)
        fc.pos = (1.0, 2.0, -0.85)
        fc.arrive()
        return plan, fc, per, out, now

    def _see(self, plan, per, now, north_mm, n=4):
        for _ in range(n):
            now[0] += 0.2
            per.queue.append(('marker', mk(p=(north_mm, 2000.0, 0.0)), now[0]))
            plan.step()

    def test_hover_follows_marker_when_ekf_drifts(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, per, out, now = self._hover(tmp)
            self._see(plan, per, now, 1000.0)
            self.assertAlmostEqual(plan.marker_ref['p_W_m'][0], 1.0)
            self.assertTrue(any('ArUco LAN DAU' in line for line in out))
            self.assertEqual(fc.flown, [])  # khong troi: khong chinh
            now[0] += 3.0
            self._see(plan, per, now, 1080.0, n=8)  # marker "dich" 8 cm trong EKF = EKF troi 8 cm
            self.assertEqual(len(fc.flown), 1)
            self.assertAlmostEqual(fc.flown[0][0], 1.0 + 0.08, places=3)  # setpoint hover theo marker
            self.assertAlmostEqual(fc.flown[0][1], 2.0)
            self.assertIn(plan.state, plan_a.HOLD_STATES)  # van hover, khong phai bay toi vat

    def test_hold_refuses_large_drift_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, per, out, now = self._hover(tmp)
            self._see(plan, per, now, 1000.0)
            now[0] += 3.0
            self._see(plan, per, now, 1450.0, n=8)  # 45 cm > max_total_m 30 cm: nghi sai, khong bay
            self.assertEqual(fc.flown, [])
            self.assertTrue(any('KHONG chinh' in line for line in out))
        with tempfile.TemporaryDirectory() as tmp:
            plan, fc, per, out, now = self._hover(tmp, hold={'enabled': False})
            self._see(plan, per, now, 1000.0)
            now[0] += 3.0
            self._see(plan, per, now, 1080.0, n=8)
            self.assertEqual(fc.flown, [])
