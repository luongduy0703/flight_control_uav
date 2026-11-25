from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'planA'))

import csv
import math
import random
import tempfile
import unittest

import video_diag
from run_plan_a import VideoDiagCollector

FIELDS = ('t_ns,dt_ms,exposure_us,gain,lens,luma,sharp,grad_x,grad_y,dx_px,dy_px,pc_resp,skew_x_px,skew_y_px,'
          'wx,wy,wz,blur_pred_px,rot_dx_px,rot_dy_px,armed,landed,vn,ve,vd,diag_ms').split(',')


def make_rows(segments, fps=30.0, seed=1, drop_every=0):
    """segments: list of (seconds, armed, landed, sharp, skew_px, wobble_px, wobble_hz, extra_px)."""
    rng = random.Random(seed)
    rows, t, k = [], 0.0, 0
    for seconds, armed, landed, sharp, skew, wobble, hz, extra in segments:
        for _ in range(int(seconds * fps)):
            dt = 1.0 / fps
            if drop_every and k % drop_every == drop_every - 1:
                dt *= 2
            t += dt
            k += 1
            rot = wobble * math.sin(2 * math.pi * hz * t)  # image motion from attitude
            w = rot / (900.0 * dt)                          # rad/s that makes this motion (f = 900 px)
            rows.append({
                't_ns': int(t * 1e9), 'dt_ms': dt * 1000, 'exposure_us': 880.0, 'gain': 2.0, 'lens': 1.0,
                'luma': 120 + rng.uniform(-2, 2), 'sharp': sharp * rng.uniform(0.9, 1.1), 'grad_x': 100.0,
                'grad_y': 110.0, 'dx_px': rot + extra * rng.uniform(-1, 1), 'dy_px': rng.uniform(-0.3, 0.3),
                'pc_resp': 0.5, 'skew_x_px': skew * rng.uniform(-1, 1), 'skew_y_px': 0.0, 'wx': 0.0, 'wy': w,
                'wz': 0.0, 'blur_pred_px': abs(w) * 900 * 880e-6, 'rot_dx_px': rot, 'rot_dy_px': 0.0,
                'armed': armed, 'landed': landed, 'vn': 0.0, 've': 0.0, 'vd': 0.0, 'diag_ms': 3.0})
    return rows


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def verdict(rep, prefix):
    return next(i['verdict'] for i in rep['findings'] if i['cause'].startswith(prefix))


class AnalysisTests(unittest.TestCase):
    def test_mechanical_vibration_shows_on_the_ground_with_motors_on(self):
        rows = make_rows([(5, 0, 1, 1000, 0.2, 0, 1, 0.2), (5, 1, 1, 450, 2.0, 0, 1, 0.2),
                          (10, 1, 2, 400, 2.5, 0.5, 2, 0.5)])
        rep = video_diag.analyze(rows)
        self.assertEqual(set(rep['phases']), {'ground_off', 'ground_armed', 'air'})
        self.assertEqual(verdict(rep, 'Rung co khi'), 'CO')
        self.assertEqual(verdict(rep, 'Rung trong luc doc'), 'CO')
        self.assertEqual(verdict(rep, 'Lac than'), 'KHONG')
        self.assertEqual(verdict(rep, 'Camera rot khung'), 'KHONG')
        self.assertIn('KET LUAN: Rung co khi', video_diag.format_report(rep))

    def test_attitude_oscillation_is_explained_by_the_fc_rates(self):
        rows = make_rows([(5, 0, 1, 1000, 0.2, 0, 1, 0.2), (5, 1, 1, 980, 0.2, 0, 1, 0.2),
                          (20, 1, 2, 700, 0.3, 10.0, 2.5, 0.5)])
        rep = video_diag.analyze(rows)
        self.assertEqual(verdict(rep, 'Rung co khi'), 'KHONG')
        self.assertEqual(verdict(rep, 'Lac than'), 'CO')
        self.assertGreater(rep['rotation_fit_air']['r2'], 0.9)
        self.assertEqual(verdict(rep, 'Troi vi tri'), 'KHONG')
        spec = rep['spectrum_air']
        if spec is not None:  # numpy present
            self.assertAlmostEqual(spec['dx_px']['peak_hz'], 2.5, delta=0.2)

    def test_frame_drops_and_unexplained_motion(self):
        rows = make_rows([(5, 0, 1, 1000, 0.2, 0, 1, 0.2), (10, 1, 2, 950, 0.2, 0.0, 1, 8.0)], drop_every=10)
        rep = video_diag.analyze(rows)
        self.assertEqual(verdict(rep, 'Camera rot khung'), 'CO')
        self.assertEqual(verdict(rep, 'Troi vi tri'), 'CO THE')

    def test_no_fc_and_tiny_input_do_not_crash(self):
        rows = make_rows([(1, 0, 1, 1000, 0.2, 0, 1, 0.2)])
        for r in rows:
            r['armed'] = r['landed'] = -1
            r['wx'] = r['wy'] = r['blur_pred_px'] = r['rot_dx_px'] = math.nan
        rep = video_diag.analyze(rows)
        self.assertEqual(list(rep['phases']), ['no_fc'])
        self.assertEqual(verdict(rep, 'Rung co khi'), 'CHUA DU DU LIEU')
        video_diag.format_report(rep)
        self.assertEqual(video_diag.analyze([])['frames'], 0)


class FakePerception:
    def __init__(self, root):
        self.dir = root
        self.run_dir = None
        self.closed = 0

    def close(self):
        self.closed += 1


class CollectorTests(unittest.TestCase):
    def test_end_of_run_joins_runs_and_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, log = Path(tmp) / 'perception', Path(tmp) / 'log'
            log.mkdir()
            rows = make_rows([(5, 0, 1, 1000, 0.2, 0, 1, 0.2), (5, 1, 1, 450, 2.0, 0, 1, 0.2)])
            for name, part in (('runs/a', rows[:150]), ('runs/b', rows[150:])):
                (root / name).mkdir(parents=True)
                write_csv(root / name / 'video_diag.csv', part)
            out = []
            per = FakePerception(root)
            c = VideoDiagCollector(per, log, out.append, True)
            per.run_dir = 'runs/a'
            c.poll()
            per.run_dir = 'runs/b'
            c.finalize()
            c.finalize()  # once only
            self.assertEqual(per.closed, 1)
            self.assertEqual(len((log / 'video_diag.csv').read_text().splitlines()), len(rows) + 1)
            self.assertTrue((log / 'video_diag_report.txt').is_file())
            self.assertTrue((log / 'video_diag_report.json').is_file())
            self.assertTrue(any('CHAN DOAN RUNG' in line for line in out))
            self.assertEqual(video_diag.find_csv(log), log / 'video_diag.csv')

    def test_missing_csv_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = []
            c = VideoDiagCollector(FakePerception(Path(tmp)), Path(tmp), out.append, True)
            c.finalize()
            self.assertTrue(any('chua co video_diag.csv' in line for line in out))


if __name__ == '__main__':
    unittest.main()
