from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi"))

import json
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from console import execute, parse_command, poll_perception, main, pin_control_cores
from perception_process import (PerceptionProcess, read_perception_ids, send_perception_command,
                                MAX_FAILED_STARTS, parse_marked)

ROOT = Path(__file__).resolve().parents[1]
PERCEPTION_CONFIG = ROOT / 'his_perception_uav-main' / 'configs' / 'run' / 'real_time.yaml'
CONFIG = '''sources:
  mavlink:
    companion_sysid: 1 # Pi
    companion_compid: 191
    command_id: 31010
'''
READY = 'INFO uavp.live: >>> Cho lenh bat dau tu may tinh: gcs_command.py start (khong can hover) <<<\n'
SCAN = 'INFO uavp.live: BAT DAU: lenh gcs_command.py qua Pixhawk (do cao cam bien 1.000 m)\n'
RESULT = 'INFO uavp.live: Ket qua: runs/20261117_real_time\n'


class FakeChild:
    def __init__(self, lines, code):
        self.stderr = iter(lines)
        self.code = code
        self.pid = 4242

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_project(tmp, config=CONFIG):
    project = Path(tmp) / 'perception'
    (project / 'build').mkdir(parents=True)
    (project / 'build' / 'uavp_real_time').write_text('')
    (project / 'configs' / 'run').mkdir(parents=True)
    (project / 'configs' / 'run' / 'real_time.yaml').write_text(config)
    return project


def wait_state(process, state, timeout=2.0):
    deadline = time.monotonic() + timeout
    while process.status()['state'] != state and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.01)
    return process.status()['state']


class PerceptionIdsTests(unittest.TestCase):
    @unittest.skipUnless(PERCEPTION_CONFIG.exists(), 'perception project not present')
    def test_ids_come_from_the_real_perception_config(self):
        self.assertEqual(read_perception_ids(PERCEPTION_CONFIG),
                         {'companion_sysid': 1, 'companion_compid': 191, 'command_id': 31010})

    def test_missing_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'c.yaml'
            path.write_text('companion_sysid: 1\ncommand_id: 31010\n')
            with self.assertRaises(ValueError):
                read_perception_ids(path)

    def test_unbuilt_perception_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(tmp)
            (project / 'build' / 'uavp_real_time').unlink()
            with self.assertRaises(ValueError):
                PerceptionProcess(project)


class PlanALineTests(unittest.TestCase):
    TARGET = 'INFO uavp.live: TARGET p_W_mm=635.0,255.5,-120.0 p_C_mm=1.0,2.0,600.0\n'
    ALIGN = ('INFO uavp.live: ALIGN ok=0 valid=1 e_px=41.2 e_mm=30.1 e_max_mm=15.0 d_mm=600.0 '
             'dW_mm=10.0,-30.0,0.0 n=120\n')
    OBJECT = ('INFO uavp.live: OBJECT p_W_mm=640.0,250.0,-118.0 normal_W=0.0100,0.0200,0.9990 '
              'yaw_W_deg=12.5 mode=plane d_mm=598.0\n')

    def test_lines_are_parsed(self):
        kind, v = parse_marked(self.TARGET)
        self.assertEqual(kind, 'target')
        self.assertEqual(v['p_W_mm'], (635.0, 255.5, -120.0))
        kind, v = parse_marked(self.ALIGN)
        self.assertEqual(kind, 'align')
        self.assertEqual((v['ok'], v['e_mm'], v['dW_mm']), (0.0, 30.1, (10.0, -30.0, 0.0)))
        kind, v = parse_marked(self.OBJECT)
        self.assertEqual((kind, v['mode'], v['yaw_W_deg']), ('object', 'plane', 12.5))
        self.assertIsNone(parse_marked(READY))
        kind, v = parse_marked('TARGET p_W_mm=1600.0,2250.0,-60.0 p_C_mm=1.0,2.0,700.0 bbox_px=960.5,540.0 '
                               'bbox_p_W_mm=1650.0,2240.0,-60.0')
        self.assertEqual((kind, v['bbox_p_W_mm'], v['bbox_px']), ('target', (1650.0, 2240.0, -60.0), (960.5, 540.0)))
        kind, v = parse_marked('INFO uavp.marker: MARKER seen=1 t_ns=12 reproj_px=0.40 tilt_cam_deg=2.0 '
                               'p_cam_S_mm=5.0,-3.0,-800.0 world=1 p_M_W_mm=1000.0,2000.0,0.0 yaw_M_W_deg=90.00 '
                               'tilt_W_deg=1.0 yaw_fc_deg=91.00 yaw_rel_deg=1.00')
        self.assertEqual(kind, 'marker')
        self.assertEqual(v['p_M_W_mm'], (1000.0, 2000.0, 0.0))
        self.assertEqual((v['seen'], v['world'], v['yaw_rel_deg']), (1.0, 1.0, 1.0))
        self.assertEqual(parse_marked('MARKER seen=0 n=4 rejected=1 reason=nghieng')[1]['reason'], 'nghieng')
        _, v = parse_marked('ALIGN ok=0 valid=0 e_px=nan e_mm=nan dW_mm=nan,nan,nan n=0')
        self.assertNotEqual(v['e_mm'], v['e_mm'])  # nan

    def test_reader_keeps_latest_values_and_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = FakeChild([READY, SCAN, 'INFO uavp.live: THU MUC KET QUA runs/x\n', self.TARGET,
                               self.ALIGN, self.OBJECT, RESULT], None)
            process = PerceptionProcess(make_project(tmp), log_dir=tmp, popen=Mock(return_value=child))
            process.start()
            process.reader.join(1)
            self.assertEqual(process.run_dir, 'runs/x')
            self.assertEqual(process.last_target['p_W_mm'][0], 635.0)
            self.assertEqual([k for k, _, _ in process.take_marked()], ['target', 'align', 'object'])
            self.assertEqual(process.take_marked(), [])
            self.assertEqual(process.last_result, 'runs/20261117_real_time')


class PerceptionLifecycleTests(unittest.TestCase):
    def test_child_is_isolated_from_console_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            popen = Mock(return_value=FakeChild([], None))
            process = PerceptionProcess(make_project(tmp), log_dir=tmp, popen=popen)
            process.start()
            kwargs = popen.call_args.kwargs
            self.assertIs(kwargs['stdin'], subprocess.DEVNULL)  # must not steal console keys
            self.assertTrue(kwargs['start_new_session'])
            command = popen.call_args.args[0]
            self.assertIn('start.accept_viewer_key=false', command)
            self.assertIn('display.show=false', command)
            process.start()  # Already running: no second child.
            self.assertEqual(popen.call_count, 1)

    def test_scan_result_then_automatic_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = Clock()
            first = FakeChild([READY], None)
            popen = Mock(side_effect=[first, FakeChild([READY], None)])
            process = PerceptionProcess(make_project(tmp), log_dir=tmp, popen=popen, clock=clock)
            process.start()
            self.assertEqual(wait_state(process, 'WAITING'), 'WAITING')
            first.stderr = iter([SCAN, RESULT])
            process.reader = None
            # Simulate the reader seeing scan + result from the same run.
            process._read_log(first, open(Path(tmp) / 'x.log', 'a'), process.generation)
            self.assertEqual(process.status()['state'], 'SCANNING')
            first.code = 0
            process.poll()
            self.assertEqual(process.status()['state'], 'STOPPED')
            self.assertEqual(process.status()['last_result'], 'runs/20261117_real_time')
            self.assertTrue(any('DONE' in e for e in process.take_events()))
            process.poll()
            self.assertEqual(popen.call_count, 1)  # Restart waits restart_delay_s.
            clock.now += 2.5
            process.poll()
            self.assertEqual(popen.call_count, 2)
            self.assertEqual(wait_state(process, 'WAITING'), 'WAITING')

    def test_repeated_startup_errors_end_in_failed_without_restart_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = Clock()
            popen = Mock(side_effect=lambda *a, **k: FakeChild(['LOI: camera khong mo duoc\n'], 1))
            process = PerceptionProcess(make_project(tmp), log_dir=tmp, popen=popen, clock=clock)
            process.start()
            for _ in range(MAX_FAILED_STARTS + 2):
                process.reader.join(1)
                process.poll()
                clock.now += 5
                process.poll()
            self.assertEqual(process.status()['state'], 'FAILED')
            self.assertEqual(popen.call_count, MAX_FAILED_STARTS)
            self.assertIn('camera', process.status()['last_error'])
            process.restart()
            self.assertEqual(popen.call_count, MAX_FAILED_STARTS + 1)


class FakeDirect:
    """MAVSDK MavlinkDirect stand-in: the bridge ACKs every COMMAND_LONG."""

    def __init__(self, ack=True):
        self.sent, self.ack, self.callback = [], ack, None

    def subscribe_message(self, name, callback):
        self.callback = callback
        return 'handle'

    def unsubscribe_message(self, handle):
        self.callback = None

    def send_message(self, message):
        self.sent.append(message)
        if self.ack:
            fields = json.loads(message.fields_json)
            self.callback(Mock(system_id=1, component_id=191,
                               fields_json=json.dumps({'command': fields['command'], 'result': 0})))


IDS = {'companion_sysid': 1, 'companion_compid': 191, 'command_id': 31010}


class PerceptionCommandTests(unittest.TestCase):
    def test_command_long_targets_perception_component(self):
        fc = Mock(mavlink_direct=FakeDirect())
        self.assertTrue(send_perception_command(fc, IDS, start=True))
        message = fc.mavlink_direct.sent[0]
        fields = json.loads(message.fields_json)
        self.assertEqual((message.message_name, message.target_system_id, message.target_component_id),
                         ('COMMAND_LONG', 1, 191))
        self.assertEqual((fields['command'], fields['param1']), (31010, 1.0))
        self.assertTrue(send_perception_command(fc, IDS, start=False))
        self.assertEqual(json.loads(fc.mavlink_direct.sent[1].fields_json)['param1'], 0.0)

    def test_no_ack_is_reported_after_retries(self):
        fc = Mock(mavlink_direct=FakeDirect(ack=False))
        self.assertFalse(send_perception_command(fc, IDS, start=True, timeout_s=0.01))
        self.assertEqual(len(fc.mavlink_direct.sent), 3)

    def test_parse(self):
        self.assertEqual(parse_command('perception'), ('perception', 'start'))
        self.assertEqual(parse_command('Perception STOP'), ('perception', 'stop'))
        self.assertEqual(parse_command('perception restart'), ('perception', 'restart'))
        with self.assertRaises(ValueError):
            parse_command('perception go')

    def perception(self, state):
        perception = Mock(ids=IDS, generation=1, auto_stopped=None)
        perception.status.return_value = {'state': state}
        perception.take_events.return_value = []
        return perception

    def test_scan_requires_hold_and_ready_perception(self):
        fc = Mock()
        with patch('console.send_perception_command', return_value=True) as send:
            fc.ready_for_fly.return_value = False
            with self.assertRaisesRegex(ValueError, 'HOLD'):
                execute(fc, 'perception', output=Mock(), perception=self.perception('WAITING'))
            fc.ready_for_fly.return_value = True
            with self.assertRaisesRegex(ValueError, 'STARTING'):
                execute(fc, 'perception', output=Mock(), perception=self.perception('STARTING'))
            send.assert_not_called()
            execute(fc, 'perception', output=Mock(), perception=self.perception('WAITING'))
            send.assert_called_once_with(fc, IDS, start=True)

    def bench_fc(self, armed=False, landed='ON_GROUND', connected=True):
        fc = Mock()
        fc.ready_for_fly.return_value = False
        fc.drone.is_connected.return_value = connected
        fc.state.side_effect = lambda key, *a: {'armed': armed, 'landed_state': landed}[key]
        return fc

    def test_ground_scan_rejected_without_bench_flag(self):
        with patch('console.send_perception_command', return_value=True) as send:
            with self.assertRaisesRegex(ValueError, '--perception-bench'):
                execute(self.bench_fc(), 'perception', output=Mock(), perception=self.perception('WAITING'))
        send.assert_not_called()

    def test_bench_scan_only_when_disarmed_on_ground_with_fresh_telemetry(self):
        def stale(key, *a):
            raise ValueError('Missing/stale armed telemetry')
        stale_fc = self.bench_fc()
        stale_fc.state.side_effect = stale
        for fc in (self.bench_fc(armed=True), self.bench_fc(landed='IN_AIR'),
                   self.bench_fc(connected=False), stale_fc):
            with self.subTest(fc=fc), patch('console.send_perception_command', return_value=True) as send:
                with self.assertRaisesRegex(ValueError, 'DISARMED'):
                    execute(fc, 'perception', output=Mock(), perception=self.perception('WAITING'),
                            perception_bench=True)
                send.assert_not_called()
        perception = self.perception('WAITING')
        with patch('console.send_perception_command', return_value=True) as send:
            execute(self.bench_fc(), 'perception', output=Mock(), perception=perception,
                    perception_bench=True)
        send.assert_called_once()
        self.assertEqual(perception.scan_mode, 'bench')

    def test_bench_scan_stops_when_vehicle_arms(self):
        perception = self.perception('SCANNING')
        perception.events, perception.scan_mode = [], 'bench'
        with patch('console.send_perception_command', return_value=True) as send:
            poll_perception(self.bench_fc(), perception, Mock(), output=Mock())
            send.assert_not_called()  # Still disarmed on ground: scan continues.
            poll_perception(self.bench_fc(armed=True), perception, Mock(), output=Mock())
            poll_perception(self.bench_fc(armed=True), perception, Mock(), output=Mock())
        send.assert_called_once_with(unittest.mock.ANY, IDS, start=False)

    def test_bench_scan_stopped_before_arm_or_takeoff_is_sent(self):
        for command in ('arm', 'takeoff'):
            perception = self.perception('SCANNING')
            perception.scan_mode = 'bench'
            fc = self.bench_fc()
            order = []
            getattr(fc, command).side_effect = lambda: order.append(command)
            with self.subTest(command=command), \
                    patch('console.send_perception_command',
                          side_effect=lambda *a, **k: order.append('stop') or True):
                execute(fc, command, output=Mock(), perception=perception, perception_bench=True)
            self.assertEqual(order, ['stop', command])

    def test_bench_flag_needs_perception_dir(self):
        with patch('sys.argv', ['console.py', '--profile', 'hardware', '--perception-bench',
                                '--connection', 'udpin://127.0.0.1:14540']), \
                patch('console.OffboardControl') as factory, patch('sys.stderr'):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        factory.assert_not_called()

    def test_perception_command_without_perception_dir(self):
        with self.assertRaisesRegex(ValueError, '--perception-dir'):
            execute(Mock(), 'perception', output=Mock(), perception=None)

    def test_leaving_hold_stops_scan_once(self):
        fc = Mock()
        fc.ready_for_fly.return_value = False
        perception = self.perception('SCANNING')
        perception.events = []
        with patch('console.send_perception_command', return_value=True) as send:
            poll_perception(fc, perception, Mock(), output=Mock())
            poll_perception(fc, perception, Mock(), output=Mock())
        send.assert_called_once_with(fc, IDS, start=False)

    def test_perception_errors_never_reach_flight_control(self):
        fc = Mock()
        perception = self.perception('WAITING')
        perception.poll.side_effect = RuntimeError('broken')
        logger = Mock()
        poll_perception(fc, perception, logger, output=Mock())
        logger.exception.assert_called_once()
        fc.land.assert_not_called()

    def test_perception_needs_router_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(tmp)
            with patch('sys.argv', ['console.py', '--profile', 'hardware', '--perception-dir', str(project),
                                    '--connection', 'serial:///dev/ttyAMA0:921600']), \
                    patch('console.validate_connection', return_value='/dev/ttyAMA0'), \
                    patch('console.OffboardControl') as factory, patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)
            factory.assert_not_called()


class CpuPinningTests(unittest.TestCase):
    def test_control_cores_parsed_validated_and_applied(self):
        with patch('console.os.cpu_count', return_value=4), patch('console.os.sched_setaffinity') as setaff:
            self.assertEqual(pin_control_cores('0,1'), [0, 1])
            setaff.assert_called_once_with(0, {0, 1})
            self.assertIsNone(pin_control_cores(''))
            for bad in ('0,4', 'a', '0,1,2,3', '-1'):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    pin_control_cores(bad)
            self.assertEqual(setaff.call_count, 1)

    def test_perception_child_starts_with_all_cpus_then_console_mask_is_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            masks = []
            popen = Mock(side_effect=lambda *a, **k: masks.append(set(current[0])) or FakeChild([], 0))
            current = [{0, 1}]
            with patch('perception_process.os.sched_getaffinity', side_effect=lambda pid: set(current[0])), \
                    patch('perception_process.os.sched_setaffinity',
                          side_effect=lambda pid, cpus: current.__setitem__(0, set(cpus))), \
                    patch('perception_process.os.cpu_count', return_value=4):
                process = PerceptionProcess(make_project(tmp), log_dir=tmp, popen=popen)
                process.start()
                process.close()
            self.assertEqual(masks, [{0, 1, 2, 3}])  # Child forked with every CPU allowed.
            self.assertEqual(current[0], {0, 1})     # Console thread back on its control cores.


if __name__ == '__main__':
    unittest.main()
