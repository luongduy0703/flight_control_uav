"""IMU vibration analysis for planA (end of each run; also standalone on the laptop).

    python3 planA/imu_diag.py planA/logs/<run>     -> imu_diag_report.txt / .json (+ imu_diag.png)

Input: imu_*.csv from pi/imu_logger.py (+ video_diag.csv from perception when present, same clock).
Per phase (ground_off / ground_armed / air, from HEARTBEAT armed + EXTENDED_SYS_STATE landed_state):
  * gyro / accel spectrum of HIGHRES_IMU (Welch, Hann): main vibration frequencies + amplitude
  * PX4 VIBRATION metric (m/s^2, unfiltered accel) + accelerometer clipping
  * motor outputs (us): prop vibration frequency follows motor speed
  * camera impact of each gyro peak with the real exposure: blur_px = w_amp f_px t_exp,
    rolling-shutter skew_px = min(w_amp t_readout, 2 theta_amp) f_px  (theta_amp = w_amp / (2 pi f_vib))
  * per-second correlation of IMU vibration with video sharpness (video_diag.csv)
Limits: HIGHRES_IMU is low-pass filtered by PX4 (IMU_GYRO_CUTOFF / IMU_ACCEL_CUTOFF) and sampled at the link rate;
content above rate / 2 folds back (alias). Prop vibration (often 80-250 Hz) is seen mainly through the VIBRATION
metric; its true frequency needs the Pixhawk SD log with raw IMU FIFO (SDLOG_PROFILE).
numpy is needed for spectra (otherwise only metrics); matplotlib optional for the plot.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import video_diag

PHASES = ('ground_off', 'ground_armed', 'takeoff', 'air', 'landing', 'unknown')
VIB_GOOD, VIB_BAD = 30.0, 60.0  # PX4 / QGC guidance for the VIBRATION metric (m/s^2)
DEFAULTS = {'camera_f_px': 823.3, 'readout_ms': 20.0, 'exposure_us': 900.0, 'segment': 256,
            'peak_min_hz': 3.0, 'body_band_hz': 5.0}


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _read(path):
    if not path.is_file():
        return []
    with open(path, newline='') as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load(log_dir):
    d = Path(log_dir)
    data = {k: _read(d / f) for k, f in (('highres', 'imu_highres.csv'), ('vibration', 'imu_vibration.csv'),
                                          ('attitude', 'imu_attitude.csv'), ('motors', 'imu_motors.csv'),
                                          ('state', 'imu_state.csv'), ('video', 'video_diag.csv'))}
    try:
        data['meta'] = json.loads((d / 'imu_meta.json').read_text())
    except (OSError, ValueError):
        data['meta'] = {}
    return data


# ---- phases --------------------------------------------------------------------------------------------
class Timeline:
    """Phase at a monotonic time from the armed / landed changes in imu_state.csv."""

    def __init__(self, state_rows):
        self.rows = sorted(state_rows, key=lambda r: r['rx_ns'])

    def at(self, t_ns):
        phase = 'unknown'
        for r in self.rows:
            if r['rx_ns'] > t_ns:
                break
            phase = self.phase(r['armed'], r['landed'])
        return phase

    @staticmethod
    def phase(armed, landed):
        if armed == 0:
            return 'ground_off'
        if armed == 1:
            return {1: 'ground_armed', 2: 'air', 3: 'takeoff', 4: 'landing'}.get(int(landed), 'air')
        return 'unknown'

    def split(self, rows, key='rx_ns'):
        out = {}
        for r in rows:
            out.setdefault(self.at(r[key]), []).append(r)
        return out


# ---- spectra (numpy) -------------------------------------------------------------------------------------
def _segments(t, v, dt, gap_factor=3.0):
    """Contiguous runs (no gap > gap_factor x dt)."""
    runs, start = [], 0
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > gap_factor * dt:
            runs.append((start, i))
            start = i
    runs.append((start, len(t)))
    return [(t[a:b], v[a:b]) for a, b in runs]


def welch(rows, keys, n_seg):
    """Summed one-sided PSD over keys ((unit)^2 / Hz), frequencies, sample rate. None without numpy / data."""
    try:
        import numpy as np
    except ImportError:
        return None
    rows = [r for r in rows if math.isfinite(r.get('time_usec', math.nan))]
    if len(rows) < n_seg:
        return None
    t = np.array([r['time_usec'] for r in rows]) / 1e6
    order = np.argsort(t)
    t = t[order]
    dt = float(np.median(np.diff(t)))
    if not dt > 0:
        return None
    fs = 1.0 / dt
    win = np.hanning(n_seg)
    scale = 1.0 / (fs * (win ** 2).sum())
    psd, count = np.zeros(n_seg // 2 + 1), 0
    for key in keys:
        v = np.array([rows[i][key] for i in order], dtype=float)
        for ts, vs in _segments(t, v, dt):
            if len(vs) < n_seg or not np.isfinite(vs).all():
                continue
            for s in range(0, len(vs) - n_seg + 1, n_seg // 2):
                x = vs[s:s + n_seg]
                x = (x - x.mean()) * win
                p = np.abs(np.fft.rfft(x)) ** 2 * scale
                p[1:-1] *= 2
                psd += p
                count += 1
    if count == 0:
        return None
    psd /= count / len(keys)  # average over segments, summed over axes
    return np.fft.rfftfreq(n_seg, dt), psd, fs


def peaks(freqs, psd, min_hz, top=3):
    """Strongest local maxima above min_hz: frequency, amplitude (sinusoid peak from band power)."""
    df = freqs[1] - freqs[0]
    out = []
    for i in range(1, len(psd) - 1):
        if freqs[i] < min_hz or not (psd[i] >= psd[i - 1] and psd[i] >= psd[i + 1]):
            continue
        band_power = float(psd[max(0, i - 2):i + 3].sum() * df)  # variance in +-2 bins
        out.append({'hz': float(freqs[i]), 'amplitude': math.sqrt(2.0 * band_power), 'power': band_power})
    out.sort(key=lambda p: -p['power'])
    return out[:top]


def band_rms(freqs, psd, lo, hi):
    df = freqs[1] - freqs[0]
    sel = [(f >= lo) and (f < hi) for f in freqs]
    return math.sqrt(sum(p for p, s in zip(psd, sel) if s) * df)


# ---- per-phase analysis ----------------------------------------------------------------------------------
def camera_impact(peak, f_px, exposure_us, readout_ms):
    w = peak['amplitude']  # rad/s
    theta = w / (2 * math.pi * peak['hz']) if peak['hz'] > 0 else 0.0
    return {'blur_px': w * f_px * exposure_us * 1e-6,
            'skew_px': min(w * readout_ms * 1e-3, 2 * theta) * f_px,
            'cycles_per_frame_readout': peak['hz'] * readout_ms * 1e-3}


def _mean(values):
    v = [x for x in values if math.isfinite(x)]
    return sum(v) / len(v) if v else None


def phase_report(rows_by, phase, opts):
    hr = rows_by['highres'].get(phase, [])
    vib = rows_by['vibration'].get(phase, [])
    mot = rows_by['motors'].get(phase, [])
    rep = {'highres_samples': len(hr), 'vibration_samples': len(vib)}
    if vib:
        per_axis = [_mean([r[f'vibration_{a}'] for r in vib]) for a in 'xyz']
        rep['vibration_metric'] = {a: v for a, v in zip('xyz', per_axis)}
        rep['vibration_max_axis'] = max(v for v in per_axis if v is not None) if any(per_axis) else None
        clip = [r['clipping_0'] + r['clipping_1'] + r['clipping_2'] for r in vib
                if all(math.isfinite(r[f'clipping_{i}']) for i in range(3))]
        rep['clipping_increase'] = (max(clip) - min(clip)) if clip else None
    if mot:
        outs = [r[f'servo{i}_raw'] for r in mot for i in range(1, 5)
                if math.isfinite(r[f'servo{i}_raw']) and 900 <= r[f'servo{i}_raw'] <= 2200]
        rep['motor_us_mean'] = _mean(outs)
    gyro = welch(hr, ('xgyro', 'ygyro', 'zgyro'), opts['segment'])
    acc = welch(hr, ('xacc', 'yacc', 'zacc'), opts['segment'])
    if gyro is not None:
        f, p, fs = gyro
        rep['sample_hz'] = fs
        rep['nyquist_hz'] = fs / 2
        rep['gyro_body_rms'] = band_rms(f, p, 0.3, opts['body_band_hz'])
        rep['gyro_vib_rms'] = band_rms(f, p, opts['body_band_hz'], fs / 2 + 1)
        rep['gyro_peaks'] = peaks(f, p, opts['peak_min_hz'])
        for pk in rep['gyro_peaks']:
            pk['camera'] = camera_impact(pk, opts['camera_f_px'], opts['exposure_us'], opts['readout_ms'])
            pk['near_nyquist'] = pk['hz'] > 0.8 * fs / 2
        rep['spectrum_gyro'] = {'hz': [float(x) for x in f], 'psd': [float(x) for x in p]}
    if acc is not None:
        f, p, _ = acc
        rep['accel_vib_rms'] = band_rms(f, p, opts['body_band_hz'], f[-1] + 1)
        rep['accel_peaks'] = peaks(f, p, opts['peak_min_hz'])
    return rep


def window_correlation(data, timeline, window_s=1.0):
    """Per-second IMU vibration (VIBRATION metric, gyro high-frequency proxy) vs video sharpness (median)."""
    video = data['video']
    if not video:
        return None
    win = int(window_s * 1e9)
    buckets = {}

    def put(t, key, value):
        if math.isfinite(t) and math.isfinite(value):
            buckets.setdefault(int(t // win), {}).setdefault(key, []).append(value)

    for r in video:
        put(r['t_ns'], 'sharp', r['sharp'])
    for r in data['vibration']:
        put(r['rx_ns'], 'vib', max(r['vibration_x'], r['vibration_y'], r['vibration_z']))
    hr = data['highres']
    for prev, cur in zip(hr, hr[1:]):  # first difference ~ high-pass: emphasises vibration over body motion
        dt = (cur['time_usec'] - prev['time_usec']) / 1e6
        if dt > 0:
            put(cur['rx_ns'], 'gyro_hf', math.sqrt(sum((cur[k] - prev[k]) ** 2 for k in ('xgyro', 'ygyro'))) / dt)
    keys = sorted(buckets)
    sharp = [video_diag.median(buckets[k].get('sharp', [])) for k in keys]
    vib = [_mean(buckets[k].get('vib', [])) for k in keys]
    ghf = [video_diag.rms(buckets[k].get('gyro_hf', [])) for k in keys]
    clean = lambda xs: [math.nan if x is None else x for x in xs]  # noqa: E731
    return {'windows': len(keys),
            'sharp_vs_vibration_metric': video_diag.spearman(clean(sharp), clean(vib)),
            'sharp_vs_gyro_hf': video_diag.spearman(clean(sharp), clean(ghf))}


def analyze(data, options=None):
    opts = {**DEFAULTS, **(options or {})}
    if data['video']:
        exp = video_diag.median([r['exposure_us'] for r in data['video'] if r['exposure_us'] > 0])
        if exp:
            opts['exposure_us'] = exp
    tl = Timeline(data['state'])
    by = {k: tl.split(data[k]) for k in ('highres', 'vibration', 'motors')}
    present = [p for p in PHASES if any(p in by[k] for k in by)]
    rep = {'options': opts, 'meta': data['meta'],
           'phases': {p: phase_report(by, p, opts) for p in present},
           'video_correlation': window_correlation(data, tl)}
    rep['findings'] = findings(rep)
    return rep


def findings(rep):
    st, out = rep['phases'], []
    air, armed, off = st.get('air'), st.get('ground_armed'), st.get('ground_off')

    def add(cause, verdict, evidence):
        out.append({'cause': cause, 'verdict': verdict, 'evidence': evidence})

    for name, s in (('khi bay', air), ('arm tren san', armed)):
        if s and s.get('vibration_max_axis') is not None:
            v = s['vibration_max_axis']
            level = 'TOT' if v < VIB_GOOD else 'CAO' if v < VIB_BAD else 'NGUY HIEM'
            add(f'Muc rung PX4 (VIBRATION) {name}', level,
                f'{v:.1f} m/s^2 (tot < {VIB_GOOD:g}, xau > {VIB_BAD:g}); tat motor: '
                f'{_f(off.get("vibration_max_axis") if off else None, ".1f")} m/s^2; clipping tang '
                f'{_f(s.get("clipping_increase"), ".0f")}')
    if air and air.get('gyro_peaks'):
        pk = air['gyro_peaks'][0]
        cam = pk['camera']
        alias = ' (gan Nyquist: co the la rung tan so cao hon bi gap lai)' if pk['near_nyquist'] else ''
        add('Tan so rung chinh (gyro) khi bay', 'DO DUOC',
            f'{pk["hz"]:.1f} Hz, bien do {pk["amplitude"]:.3f} rad/s{alias}; lay mau {air["sample_hz"]:.0f} Hz '
            f'(Nyquist {air["nyquist_hz"]:.0f} Hz)')
        add('Anh huong len camera cua tan so do', 'CO' if cam['blur_px'] > 1.0 or cam['skew_px'] > 2.0 else 'NHO',
            f'mo trong {rep["options"]["exposure_us"]:.0f} us phoi sang ~{cam["blur_px"]:.2f} px; lech rolling shutter '
            f'~{cam["skew_px"]:.1f} px ({cam["cycles_per_frame_readout"]:.1f} chu ky trong {rep["options"]["readout_ms"]:g} ms '
            'doc anh)')
        if air.get('gyro_body_rms') is not None:
            add('Lac than (< 5 Hz) so voi rung (>= 5 Hz)', 'SO SANH',
                f'gyro < 5 Hz rms {air["gyro_body_rms"]:.3f} rad/s, >= 5 Hz rms {air["gyro_vib_rms"]:.3f} rad/s')
    if armed and off and armed.get('gyro_vib_rms') is not None and off.get('gyro_vib_rms') is not None:
        r = armed['gyro_vib_rms'] / off['gyro_vib_rms'] if off['gyro_vib_rms'] > 0 else None
        add('Rung do motor (arm tren san / tat motor)', 'CO' if r is not None and r > 3 else 'KHONG RO',
            f'gyro >= 5 Hz tang {_f(r, ".1f")} lan khi motor quay; motor {_f(armed.get("motor_us_mean"), ".0f")} us')
    corr = rep['video_correlation']
    if corr and corr['windows'] >= 10:
        r = corr['sharp_vs_vibration_metric']
        g = corr['sharp_vs_gyro_hf']
        strong = any(x is not None and x < -0.4 for x in (r, g))
        add('Rung IMU lien quan do net video (tung giay)', 'CO' if strong else 'YEU',
            f'Spearman do net - VIBRATION {_f(r, ".2f")}, do net - gyro tan so cao {_f(g, ".2f")} '
            f'({corr["windows"]} cua so 1 s; am = rung tang thi anh mo)')
    params = rep['meta'].get('params') or {}
    if params:
        add('Bo loc / log PX4', 'THONG TIN',
            ', '.join(f'{k}={v}' for k, v in params.items())
            + ' (HIGHRES_IMU da qua loc; muon pho that > Nyquist: SD log raw IMU FIFO)')
    return out


def _f(v, spec='.2f'):
    return 'n/a' if v is None else format(v, spec)


def format_report(rep):
    lines = ['', '=' * 72, ' CHAN DOAN RUNG TU IMU PIXHAWK', '=' * 72,
             f' {"pha":<13}{"IMU":>6}{"Hz":>6}{"VIB":>7}{"clip":>6}{"motor":>7}{"w<5Hz":>8}{"w>=5Hz":>8}  dinh gyro (Hz: rad/s)']
    for p, s in rep['phases'].items():
        pk = ', '.join(f'{x["hz"]:.0f}: {x["amplitude"]:.3f}' for x in s.get('gyro_peaks', []))
        lines.append(f' {p:<13}{s["highres_samples"]:>6}{_f(s.get("sample_hz"), ".0f"):>6}'
                     f'{_f(s.get("vibration_max_axis"), ".1f"):>7}{_f(s.get("clipping_increase"), ".0f"):>6}'
                     f'{_f(s.get("motor_us_mean"), ".0f"):>7}{_f(s.get("gyro_body_rms"), ".3f"):>8}'
                     f'{_f(s.get("gyro_vib_rms"), ".3f"):>8}  {pk or "-"}')
    lines.append(' (VIB: chi so rung PX4 m/s^2, truc lon nhat; w: gyro rms rad/s theo dai tan; motor: us trung binh)')
    lines.append('')
    for item in rep['findings']:
        lines.append(f' [{item["verdict"]:^11}] {item["cause"]}')
        lines.append(f'      {item["evidence"]}')
    lines.append('=' * 72)
    return '\n'.join(lines)


def plot(rep, data, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(2, 1, figsize=(12, 9))
    for phase, s in rep['phases'].items():
        spec = s.get('spectrum_gyro')
        if spec:
            ax[0].semilogy(spec['hz'][1:], spec['psd'][1:], lw=0.8, label=phase)
    ax[0].set_xlabel('Hz')
    ax[0].set_ylabel('gyro PSD (rad/s)^2/Hz')
    ax[0].set_title('Pho gyro HIGHRES_IMU theo pha (da qua loc PX4; tren Nyquist bi gap)')
    ax[0].legend()
    vib = data['vibration']
    if vib:
        t0 = vib[0]['rx_ns']
        ax[1].plot([(r['rx_ns'] - t0) / 1e9 for r in vib],
                   [max(r['vibration_x'], r['vibration_y'], r['vibration_z']) for r in vib], lw=0.8,
                   color='tab:red', label='VIBRATION (m/s^2)')
        ax[1].axhline(VIB_GOOD, color='tab:orange', ls='--', lw=0.6)
        ax[1].axhline(VIB_BAD, color='tab:red', ls='--', lw=0.6)
        ax[1].set_ylabel('m/s^2')
        if data['video']:
            tw = ax[1].twinx()
            tw.plot([(r['t_ns'] - t0) / 1e9 for r in data['video']], [r['sharp'] for r in data['video']], lw=0.5,
                    color='tab:blue', alpha=0.6, label='do net video')
            tw.set_ylabel('do net video')
        ax[1].set_xlabel('t (s)')
        ax[1].legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def analyze_and_save(log_dir, options=None):
    """Returns (report, text) or (None, message) when there is no IMU log."""
    data = load(log_dir)
    if not data['highres'] and not data['vibration']:
        return None, 'Chan doan IMU: chua co imu_*.csv (imu_log tat / chua ket noi Pixhawk).'
    rep = analyze(data, options)
    text = format_report(rep)
    out = Path(log_dir)
    (out / 'imu_diag_report.txt').write_text(text + '\n')
    slim = json.loads(json.dumps(rep))
    for s in slim['phases'].values():
        s.pop('spectrum_gyro', None)  # spectrum only in the plot
    (out / 'imu_diag_report.json').write_text(json.dumps(slim, indent=2))
    rep['plot'] = str(out / 'imu_diag.png') if plot(rep, data, out / 'imu_diag.png') else None
    return rep, text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log_dir', help='planA log folder (imu_*.csv, video_diag.csv)')
    parser.add_argument('--f-px', type=float, default=DEFAULTS['camera_f_px'])
    parser.add_argument('--readout-ms', type=float, default=DEFAULTS['readout_ms'])
    args = parser.parse_args()
    rep, text = analyze_and_save(args.log_dir, {'camera_f_px': args.f_px, 'readout_ms': args.readout_ms})
    print(text)
    if rep is not None:
        print(f'Luu: {args.log_dir} (imu_diag_report.txt / .json' + (', imu_diag.png)' if rep.get('plot') else ')'))


if __name__ == '__main__':
    main()
