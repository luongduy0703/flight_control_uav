from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pi'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'planA'))

import csv
import json
import math
import random
import tempfile
import types
import unittest

import imu_diag
from imu_logger import ImuLogger, FILES, MSG_IDS

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class FakeDirect:
    def __init__(self):
        self.subs = {}

    def subscribe_message(self, name, cb):
        self.subs[name] = cb
        return name

    def unsubscribe_message(self, handle):
        self.subs.pop(handle, None)

    def send(self, name, fields, sysid=1, compid=1):
        msg = types.SimpleNamespace(message_name=name, system_id=sysid, component_id=compid,
                                    fields_json=json.dumps(fields))
        self.subs[name](msg)


class ImuLoggerTests(unittest.TestCase):
    def test_records_streams_state_and_restores_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            direct, intervals, now = FakeDirect(), [], [1000]
            log = ImuLogger(direct, tmp, 1, {'HIGHRES_IMU': 200.0, 'VIBRATION': 10.0, 'SERVO_OUTPUT_RAW': 0},
                            params=lambda name, kind: 30.0 if kind == 'float' else 1,
                            set_interval=lambda d, sysid, msgid, us: intervals.append((msgid, us)),
                            clock=lambda: now[0])
            log.start()
            self.assertIn((MSG_IDS['HIGHRES_IMU'], 5000), intervals)
            self.assertIn((MSG_IDS['VIBRATION'], 100000), intervals)
            self.assertNotIn(MSG_IDS['SERVO_OUTPUT_RAW'], [m for m, _ in intervals])  # 0 Hz: PX4 rate kept
            direct.send('HIGHRES_IMU', {'time_usec': 5, 'xacc': 0.1, 'yacc': 0.2, 'zacc': -9.8, 'xgyro': 0.01,
                                        'ygyro': 0.02, 'zgyro': 0.03})
            direct.send('HIGHRES_IMU', {'time_usec': 6, 'xacc': 0.0}, sysid=2)  # other system: ignored
            direct.send('HEARTBEAT', {'base_mode': 128 + 1})
            direct.send('HEARTBEAT', {'base_mode': 128 + 1})  # unchanged state: no new row
            direct.send('EXTENDED_SYS_STATE', {'landed_state': 2})
            log.poll()
            log.stop()
            self.assertIn((MSG_IDS['HIGHRES_IMU'], 0), intervals)  # PX4 default restored
            rows = (Path(tmp) / FILES['HIGHRES_IMU']).read_text().splitlines()
            self.assertEqual(rows[1], '1000,5,0.1,0.2,-9.8,0.01,0.02,0.03')
            self.assertEqual(len(rows), 2)
            state = (Path(tmp) / 'imu_state.csv').read_text().splitlines()
            self.assertEqual(state[1:], ['1000,1,-1', '1000,1,2'])
            meta = json.loads((Path(tmp) / 'imu_meta.json').read_text())
            self.assertEqual(meta['params']['IMU_GYRO_CUTOFF'], 30.0)
            self.assertEqual(meta['samples']['HIGHRES_IMU'], 1)
            log.stop()  # twice: no error


def write(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_log(d, vib_hz=60.0, vib_amp=(0.0, 0.05, 0.4), sharp=(1000, 900, 500), fs=200.0, seconds=(8, 8, 16)):
    """Phases ground_off / ground_armed / air; gyro = body sway 1 Hz + vibration at vib_hz (amplitude per phase)."""
    rng = random.Random(2)
    hr, vib, state, video, mot = [], [], [], [], []
    t = 0.0
    for k, (armed, landed) in enumerate(((0, 1), (1, 1), (1, 2))):
        state.append({'rx_ns': int(t * 1e9), 'armed': armed, 'landed': landed})
        end = t + seconds[k]
        while t < end:
            w = vib_amp[k] * math.sin(2 * math.pi * vib_hz * t)
            sway = (0.05 if k == 2 else 0.0) * math.sin(2 * math.pi * 1.0 * t)
            hr.append({'rx_ns': int(t * 1e9) + 3_000_000, 'time_usec': int(t * 1e6), 'xacc': rng.gauss(0, 0.05),
                       'yacc': rng.gauss(0, 0.05), 'zacc': -9.8, 'xgyro': w + sway + rng.gauss(0, 0.002),
                       'ygyro': rng.gauss(0, 0.002), 'zgyro': rng.gauss(0, 0.002)})
            if len(hr) % 20 == 0:
                level = (2.0, 15.0, 45.0)[k]
                vib.append({'rx_ns': int(t * 1e9), 'time_usec': int(t * 1e6), 'vibration_x': level,
                            'vibration_y': level * 0.8, 'vibration_z': level * 0.5, 'clipping_0': 0,
                            'clipping_1': 0, 'clipping_2': 3 * len(vib) if k == 2 else 0})
                mot.append({'rx_ns': int(t * 1e9), 'time_usec': int(t * 1e6), **{f'servo{i}_raw': (900, 1150, 1500)[k]
                                                                                for i in range(1, 9)}})
            if len(hr) % 7 == 0:  # ~30 fps video
                video.append({'t_ns': int(t * 1e9), 'exposure_us': 880.0,
                              'sharp': sharp[k] * (1 - 0.2 * abs(w) / max(vib_amp)) * rng.uniform(0.95, 1.05)})
            t += 1.0 / fs
    write(d / 'imu_highres.csv', list(hr[0]), hr)
    write(d / 'imu_vibration.csv', list(vib[0]), vib)
    write(d / 'imu_motors.csv', list(mot[0]), mot)
    write(d / 'imu_state.csv', ['rx_ns', 'armed', 'landed'], state)
    write(d / 'video_diag.csv', ['t_ns', 'exposure_us', 'sharp'], video)
    (d / 'imu_meta.json').write_text(json.dumps({'params': {'IMU_GYRO_CUTOFF': 40.0}}))


def finding(rep, prefix):
    return next(i for i in rep['findings'] if i['cause'].startswith(prefix))


class ImuDiagTests(unittest.TestCase):
    def test_phases_vibration_metric_and_motor(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_log(Path(tmp))
            rep, text = imu_diag.analyze_and_save(tmp)
            self.assertEqual(list(rep['phases']), ['ground_off', 'ground_armed', 'air'])
            air = rep['phases']['air']
            self.assertAlmostEqual(air['vibration_max_axis'], 45.0)
            self.assertGreater(air['clipping_increase'], 0)
            self.assertAlmostEqual(air['motor_us_mean'], 1500.0)
            self.assertEqual(finding(rep, 'Muc rung PX4 (VIBRATION) khi bay')['verdict'], 'CAO')
            self.assertEqual(rep['options']['exposure_us'], 880.0)  # from video_diag.csv
            self.assertIn('CHAN DOAN RUNG TU IMU', text)
            self.assertTrue((Path(tmp) / 'imu_diag_report.json').is_file())

    @unittest.skipUnless(HAVE_NUMPY, 'numpy needed for spectra')
    def test_spectrum_finds_vibration_frequency_and_camera_impact(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_log(Path(tmp))
            rep = imu_diag.analyze(imu_diag.load(tmp))
            air = rep['phases']['air']
            self.assertAlmostEqual(air['sample_hz'], 200.0, delta=1.0)
            pk = air['gyro_peaks'][0]
            self.assertAlmostEqual(pk['hz'], 60.0, delta=1.0)
            self.assertAlmostEqual(pk['amplitude'], 0.4, delta=0.06)
            # blur = 0.4 rad/s * 823.3 px * 880 us
            self.assertAlmostEqual(pk['camera']['blur_px'], pk['amplitude'] * 823.3 * 880e-6, places=6)
            theta = pk['amplitude'] / (2 * math.pi * pk['hz'])
            self.assertAlmostEqual(pk['camera']['skew_px'], 2 * theta * 823.3, places=6)  # 60 Hz: > 1 cycle / readout
            self.assertGreater(air['gyro_vib_rms'], air['gyro_body_rms'])
            self.assertEqual(finding(rep, 'Rung do motor')['verdict'], 'CO')
            self.assertEqual(finding(rep, 'Rung IMU lien quan')['verdict'], 'CO')

    def test_missing_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep, text = imu_diag.analyze_and_save(tmp)
            self.assertIsNone(rep)
            self.assertIn('chua co imu_', text)


if __name__ == '__main__':
    unittest.main()
