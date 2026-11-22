"""Video vibration / blur diagnosis for planA (end of each run; also standalone on the laptop).

    python3 planA/video_diag.py planA/logs/<run>            # or a video_diag.csv directly
    -> video_diag_report.txt / .json (+ video_diag.png if matplotlib) next to the csv's run folder

Input: video_diag.csv written by perception (his_perception_uav-main/src/support/video_diag.cpp), one row per
camera frame in every phase (waiting on the ground, armed, takeoff, hover, scan). Phases from the FC state:
    ground_off    disarmed on the ground: reference sharpness (no vibration)
    ground_armed  motors spinning, not flying: vibration from motors / props reaching the camera
    air           flying: + attitude oscillation + position drift
The verdict compares these phases and checks how much of the image motion the measured attitude explains.
Pure Python; numpy (spectrum) and matplotlib (plot) are optional.
"""
import argparse
import csv
import json
import math
from pathlib import Path

PHASES = ('ground_off', 'ground_armed', 'takeoff', 'air', 'landing', 'no_fc')
LIMITS = {
    'drop_factor': 1.5,          # dt > 1.5 x median dt = dropped frame
    'drop_ratio': 0.02,          # > 2 % dropped frames: camera / CPU problem
    'armed_sharp_ratio': 0.75,   # motors on (ground) sharpness < 75 % of motors off: mechanical vibration
    'air_sharp_ratio': 0.6,      # flying sharpness < 60 % of ground reference: blurred
    'blur_pred_small_px': 0.5,   # measured attitude rates would blur < 0.5 px: they cannot explain the blur
    'blur_pred_large_px': 1.5,   # >= 1.5 px: attitude motion alone blurs the image
    'skew_factor': 2.0,          # rolling-shutter skew in the air > 2 x ground: vibration during readout
    'skew_min_px': 1.0,
    'rot_r2': 0.5,               # attitude explains > 50 % of the frame-to-frame image motion
    'shift_rms_px': 3.0,         # frame-to-frame image motion worth discussing
    'exposure_long_us': 4000.0,
    'gain_ratio': 1.5,           # gain / brightness change between phases biases the sharpness comparison
    'lens_range': 0.05,
    'diag_ms': 8.0,
    'min_frames': 15,
}


# ---- input -------------------------------------------------------------------------------------------
def _num(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return math.nan
    return v


def load_csv(path):
    with open(path, newline='') as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def phase_of(row):
    armed, landed = row.get('armed', -1), row.get('landed', -1)
    if not math.isfinite(armed) or armed < 0:
        return 'no_fc'
    if armed == 0:
        return 'ground_off'
    return {1: 'ground_armed', 2: 'air', 3: 'takeoff', 4: 'landing'}.get(int(landed), 'air')


# ---- small statistics (no numpy) ---------------------------------------------------------------------
def _finite(values):
    return [v for v in values if v is not None and math.isfinite(v)]


def median(values):
    return percentile(values, 50)


def percentile(values, q):
    v = sorted(_finite(values))
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def rms(values):
    v = _finite(values)
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else None


def pearson(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 5:
        return None
    mx = sum(p[0] for p in pairs) / len(pairs)
    my = sum(p[1] for p in pairs) / len(pairs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    return None if sxx <= 0 or syy <= 0 else sxy / math.sqrt(sxx * syy)


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    for rank, i in enumerate(order):
        r[i] = float(rank)
    return r


def spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 5:
        return None
    return pearson(_ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs]))


def ratio(a, b):
    return None if a is None or b is None or b <= 0 else a / b


# ---- analysis ----------------------------------------------------------------------------------------
def phase_stats(rows, dt_nominal_ms, limits):
    col = lambda key: [r[key] for r in rows]  # noqa: E731
    dts = _finite(col('dt_ms'))
    drops = sum(1 for d in dts if d > limits['drop_factor'] * dt_nominal_ms) if dt_nominal_ms else 0
    shift = [math.hypot(r['dx_px'], r['dy_px']) for r in rows]
    w_perp = [math.hypot(r['wx'], r['wy']) for r in rows]
    skew = [math.hypot(r['skew_x_px'], r['skew_y_px']) for r in rows]
    ts = _finite(col('t_ns'))
    lens = _finite([v for v in col('lens') if v >= 0])
    gx, gy = col('grad_x'), col('grad_y')
    return {
        'frames': len(rows),
        'duration_s': (max(ts) - min(ts)) / 1e9 if len(ts) > 1 else 0.0,
        'sharp_median': median(col('sharp')), 'sharp_p10': percentile(col('sharp'), 10),
        'grad_x_over_y': ratio(median(gx), median(gy)),
        'luma_median': median(col('luma')), 'gain_median': median(col('gain')),
        'exposure_us_median': median(col('exposure_us')), 'exposure_us_max': percentile(col('exposure_us'), 100),
        'dt_ms_median': median(dts), 'dt_ms_p99': percentile(dts, 99), 'dt_ms_max': percentile(dts, 100),
        'drop_ratio': drops / len(dts) if dts else None,
        'shift_rms_px': rms(shift), 'skew_rms_px': rms(skew),
        'w_perp_rms_rad_s': rms(w_perp), 'wz_rms_rad_s': rms(col('wz')),
        'blur_pred_median_px': median(col('blur_pred_px')), 'blur_pred_p90_px': percentile(col('blur_pred_px'), 90),
        'pc_response_median': median(col('pc_resp')),
        'lens_range': (max(lens) - min(lens)) if lens else None,
        'diag_ms_median': median(col('diag_ms')),
    }


def rotation_fit(rows):
    """How much of the measured frame-to-frame image motion the FC attitude predicts (R^2 over dx, dy)."""
    meas, resid = [], []
    for r in rows:
        if all(math.isfinite(r[k]) for k in ('dx_px', 'dy_px', 'rot_dx_px', 'rot_dy_px')):
            meas += [r['dx_px'], r['dy_px']]
            resid += [r['dx_px'] - r['rot_dx_px'], r['dy_px'] - r['rot_dy_px']]
    if len(meas) < 20:
        return {'r2': None, 'residual_rms_px': None, 'pairs': len(meas) // 2}
    mean = sum(meas) / len(meas)
    var = sum((m - mean) ** 2 for m in meas)
    ss = sum(e * e for e in resid)
    return {'r2': None if var <= 0 else 1.0 - ss / var, 'residual_rms_px': rms(resid), 'pairs': len(meas) // 2}


def spectrum_peaks(rows, keys=('dx_px', 'dy_px')):
    """Dominant frequency of the image motion (Hz). Needs numpy; frame rate limits it to < fps / 2."""
    try:
        import numpy as np
    except ImportError:
        return None
    t = np.array([r['t_ns'] for r in rows], dtype=float) / 1e9
    out = {}
    if len(t) < 64:
        return None
    dt = np.median(np.diff(t))
    if not dt > 0:
        return None
    grid = np.arange(t[0], t[-1], dt)
    for key in keys:
        v = np.array([r[key] for r in rows], dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 64:
            continue
        y = np.interp(grid, t[ok], v[ok])
        y = (y - y.mean()) * np.hanning(len(y))
        p = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), dt)
        band = f >= 0.3
        if not band.any():
            continue
        i = int(np.argmax(p[band]))
        total = p[band].sum()
        low = p[band & (f < 3.0)].sum()
        out[key] = {'peak_hz': float(f[band][i]), 'peak_share': float(p[band][i] / total) if total else None,
                    'below_3hz_share': float(low / total) if total else None, 'nyquist_hz': float(0.5 / dt)}
    return out or None


def analyze(rows, limits=None):
    limits = {**LIMITS, **(limits or {})}
    by_phase = {}
    for r in rows:
        by_phase.setdefault(phase_of(r), []).append(r)
    dt_nominal = median([r['dt_ms'] for r in rows])
    stats = {p: phase_stats(by_phase[p], dt_nominal, limits) for p in PHASES if p in by_phase}
    air = by_phase.get('air', [])
    rep = {'frames': len(rows), 'dt_nominal_ms': dt_nominal,
           'fps': 1000.0 / dt_nominal if dt_nominal else None, 'phases': stats,
           'rotation_fit_air': rotation_fit(air) if air else None,
           'sharp_vs_blur_pred_air': spearman([r['sharp'] for r in air], [r['blur_pred_px'] for r in air]),
           'sharp_vs_rate_air': spearman([r['sharp'] for r in air],
                                         [math.hypot(r['wx'], r['wy']) for r in air]),
           'spectrum_air': spectrum_peaks(air) if air else None}
    rep['findings'] = findings(rep, limits)
    return rep


def _ok(stats, phase, limits):
    s = stats.get(phase)
    return s if s is not None and s['frames'] >= limits['min_frames'] else None


def findings(rep, limits):
    """List of {cause, verdict: CO / CO THE / KHONG / CHUA DU DU LIEU, evidence, fix}."""
    st = rep['phases']
    ref = _ok(st, 'ground_off', limits) or _ok(st, 'no_fc', limits)
    armed, air = _ok(st, 'ground_armed', limits), _ok(st, 'air', limits)
    out = []

    def add(cause, verdict, evidence, fix):
        out.append({'cause': cause, 'verdict': verdict, 'evidence': evidence, 'fix': fix})

    # 1. Camera timing
    all_drop = [s['drop_ratio'] for s in st.values() if s['drop_ratio'] is not None]
    worst = max(all_drop) if all_drop else None
    add('Camera rot khung / giat thoi gian', 'CO' if worst is not None and worst > limits['drop_ratio'] else 'KHONG',
        f'ti le rot khung lon nhat {_pct(worst)}, khung {_f(rep["dt_nominal_ms"], ".1f")} ms '
        f'({_f(rep["fps"], ".1f")} fps)',
        'Giam tai CPU (perception cores, video_diag.every_n), kiem tra nguon Pi (under-voltage), buffer camera')

    # 2. Mechanical vibration (motors / props -> camera mount)
    if ref and armed:
        r = ratio(armed['sharp_median'], ref['sharp_median'])
        add('Rung co khi tan so cao (motor / canh quat -> gia camera)',
            'CO' if r is not None and r < limits['armed_sharp_ratio'] else 'KHONG',
            f'do net khi arm tren san (motor quay, khong bay) = {_pct(r)} so voi tat motor; '
            f'lech tren/duoi anh {_f(armed["skew_rms_px"])} px vs {_f(ref["skew_rms_px"])} px',
            'Dem cao su / gel cho camera, siet gia, can bang canh quat, kiem tra truc motor')
    elif ref and air:
        r = ratio(air['sharp_median'], ref['sharp_median'])
        small = air['blur_pred_median_px'] is not None and air['blur_pred_median_px'] < limits['blur_pred_small_px']
        blurred = r is not None and r < limits['air_sharp_ratio']
        add('Rung co khi tan so cao (motor / canh quat -> gia camera)', 'CO THE' if blurred and small else 'KHONG',
            f'khong co doan arm tren san; khi bay do net {_pct(r)} nhung mo du doan tu tu the chi '
            f'{_f(air["blur_pred_median_px"])} px (tu the doc duoc khong giai thich duoc do mo)',
            'Lan sau: ARM va cho 5-10 s tren san truoc takeoff de tach rieng rung motor')
    else:
        add('Rung co khi tan so cao (motor / canh quat -> gia camera)', 'CHUA DU DU LIEU',
            'thieu doan tat motor hoac doan arm / bay', 'Chay perception tu luc UAV con tat motor tren san')

    # 3. Rolling-shutter jello
    if ref and air:
        k = ratio(air['skew_rms_px'], ref['skew_rms_px'])
        yes = (air['skew_rms_px'] or 0) > limits['skew_min_px'] and k is not None and k > limits['skew_factor']
        add('Rung trong luc doc cam bien (rolling shutter, "jello")', 'CO' if yes else 'KHONG',
            f'lech tren - duoi anh khi bay {_f(air["skew_rms_px"])} px = {_f(k, ".1f")} x tren san',
            'Giam rung co khi (nhu tren); rung tan so cao tren gia camera')

    # 4. Attitude oscillation (control loop / gusts)
    if air:
        fit = rep['rotation_fit_air'] or {}
        r2 = fit.get('r2')
        big = (air['shift_rms_px'] or 0) > limits['shift_rms_px']
        blur_big = (air['blur_pred_p90_px'] or 0) >= limits['blur_pred_large_px']
        spec = (rep['spectrum_air'] or {}).get('dx_px') or (rep['spectrum_air'] or {}).get('dy_px')
        verdict = 'CO' if (r2 is not None and r2 > limits['rot_r2'] and big) or blur_big else 'KHONG'
        add('Lac than UAV (vong dieu khien tu the, gio)', verdict,
            f'dich anh khi bay {_f(air["shift_rms_px"])} px/khung, tu the giai thich R2 = {_f(r2, ".2f")}; '
            f'toc do quay rms {_f(air["w_perp_rms_rad_s"], ".2f")} rad/s -> mo du doan p90 '
            f'{_f(air["blur_pred_p90_px"])} px; tuong quan do net - mo du doan '
            f'{_f(rep["sharp_vs_blur_pred_air"], ".2f")}'
            + ('' if not spec else f'; tan so troi {spec["peak_hz"]:.1f} Hz (Nyquist {spec["nyquist_hz"]:.0f} Hz)'),
            'Chinh PID tu the (MC_ROLLRATE_*, MC_PITCHRATE_*), loc gyro (IMU_GYRO_CUTOFF), giam phoi sang')

    # 5. Translation drift (residual motion not explained by attitude)
    if air:
        fit = rep['rotation_fit_air'] or {}
        res = fit.get('residual_rms_px')
        add('Troi vi tri (tinh tien, khong phai quay)',
            'CO THE' if res is not None and res > limits['shift_rms_px'] else 'KHONG',
            f'dich anh con lai sau khi tru phan do quay: {_f(res)} px/khung',
            'Giu vi tri: chat luong optical flow, EKF; giam toc do / gio')

    # 6. Exposure
    exp_ref = air or ref
    if exp_ref:
        long_exp = (exp_ref['exposure_us_max'] or 0) > limits['exposure_long_us']
        add('Phoi sang dai', 'CO' if long_exp else 'KHONG',
            f'phoi sang trung vi {_f(exp_ref["exposure_us_median"], ".0f")} us, max {_f(exp_ref["exposure_us_max"], ".0f")} us',
            'Khoa phoi sang ngan (ExposureTime) + tang gain / them den')

    # 7. Comparison biases
    if ref and air:
        g = ratio(air['gain_median'], ref['gain_median'])
        lum = ratio(air['luma_median'], ref['luma_median'])
        biased = (g is not None and (g > limits['gain_ratio'] or g < 1 / limits['gain_ratio'])) or \
                 (lum is not None and abs(lum - 1.0) > 0.25)
        add('Anh sang / gain khac nhau giua cac pha (so sanh do net bi lech)', 'CO' if biased else 'KHONG',
            f'gain bay / san {_f(g, ".2f")}, do sang bay / san {_f(lum, ".2f")}; canh nhin cung thay doi khi bay '
            'len cao (chi tiet nho hon) -> do net giam mot phan vi canh', 'So sanh them anh hover cung do cao')
    lens = [s['lens_range'] for s in st.values() if s['lens_range'] is not None]
    if lens:
        add('Lay net thay doi', 'CO' if max(lens) > limits['lens_range'] else 'KHONG',
            f'vi tri lens dao dong {max(lens):.3f}', 'Khoa LensPosition (file K: sensor_mode)')
    load = [s['diag_ms_median'] for s in st.values() if s['diag_ms_median'] is not None]
    if load and max(load) > limits['diag_ms']:
        add('Chan doan ton CPU', 'CO', f'{max(load):.1f} ms / khung', 'Tang video_diag.every_n')
    return out


def _f(v, spec='.2f'):
    return 'n/a' if v is None else format(v, spec)


def _pct(v):
    return 'n/a' if v is None else f'{100 * v:.0f} %'


def format_report(rep):
    lines = ['', '=' * 72, ' CHAN DOAN RUNG / MO VIDEO', '=' * 72,
             f' {rep["frames"]} khung, {_f(rep["fps"], ".1f")} fps']
    head = f' {"pha":<13}{"khung":>6}{"s":>6}{"net":>8}{"net p10":>8}{"gx/gy":>6}{"dich":>6}{"lech":>6}' \
           f'{"w":>6}{"mo du":>6}{"phoi":>6}{"gain":>6}{"rot":>6}'
    lines += [head, ' ' + '-' * (len(head) - 1)]
    for p, s in rep['phases'].items():
        lines.append(f' {p:<13}{s["frames"]:>6}{s["duration_s"]:>6.0f}{_f(s["sharp_median"], ".0f"):>8}'
                     f'{_f(s["sharp_p10"], ".0f"):>8}{_f(s["grad_x_over_y"], ".2f"):>6}{_f(s["shift_rms_px"], ".1f"):>6}'
                     f'{_f(s["skew_rms_px"], ".1f"):>6}{_f(s["w_perp_rms_rad_s"], ".2f"):>6}'
                     f'{_f(s["blur_pred_p90_px"], ".1f"):>6}{_f(s["exposure_us_median"], ".0f"):>6}'
                     f'{_f(s["gain_median"], ".1f"):>6}{_pct(s["drop_ratio"]):>6}')
    lines.append(' (net: phuong sai Laplacian; gx/gy < 1: mo theo x; dich / lech: px moi khung; w: rad/s; '
                 'mo du: mo du doan p90 px; phoi: us; rot: rot khung)')
    lines.append('')
    for item in rep['findings']:
        lines.append(f' [{item["verdict"]:^15}] {item["cause"]}')
        lines.append(f'      {item["evidence"]}')
        if item['verdict'] in ('CO', 'CO THE'):
            lines.append(f'      -> {item["fix"]}')
    main = [i['cause'] for i in rep['findings'] if i['verdict'] == 'CO']
    lines.append('')
    lines.append(' KET LUAN: ' + ('; '.join(main) if main else 'khong thay nguyen nhan ro rang tu so lieu nay'))
    lines.append('=' * 72)
    return '\n'.join(lines)


def plot(rows, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not rows:
        return False
    t0 = rows[0]['t_ns']
    t = [(r['t_ns'] - t0) / 1e9 for r in rows]
    colors = {'ground_off': 'tab:green', 'ground_armed': 'tab:orange', 'takeoff': 'tab:purple', 'air': 'tab:red',
              'landing': 'tab:brown', 'no_fc': 'tab:gray'}
    c = [colors[phase_of(r)] for r in rows]
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    ax[0].scatter(t, [r['sharp'] for r in rows], c=c, s=3)
    ax[0].set_ylabel('do net (Laplacian)')
    ax[0].set_title('xanh: san tat motor, cam: arm tren san, tim: cat canh, do: bay')
    ax[1].plot(t, [math.hypot(r['dx_px'], r['dy_px']) for r in rows], lw=0.7, label='dich anh do duoc')
    ax[1].plot(t, [math.hypot(r['rot_dx_px'], r['rot_dy_px']) for r in rows], lw=0.7, label='du doan tu goc quay')
    ax[1].set_ylabel('px / khung')
    ax[1].legend(loc='upper right')
    ax[2].plot(t, [math.hypot(r['wx'], r['wy']) for r in rows], lw=0.7, label='|w| vuong goc truc quang')
    ax[2].plot(t, [r['blur_pred_px'] for r in rows], lw=0.7, label='mo du doan (px)')
    ax[2].set_ylabel('rad/s | px')
    ax[2].legend(loc='upper right')
    ax[3].plot(t, [math.hypot(r['skew_x_px'], r['skew_y_px']) for r in rows], lw=0.7)
    ax[3].set_ylabel('lech tren-duoi (px)')
    ax[4].plot(t, [r['dt_ms'] for r in rows], lw=0.7, label='khung (ms)')
    ax[4].plot(t, [r['exposure_us'] / 100.0 for r in rows], lw=0.7, label='phoi sang (x100 us)')
    ax[4].set_ylabel('ms')
    ax[4].set_xlabel('t (s)')
    ax[4].legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def analyze_and_save(csv_path, out_dir, limits=None):
    """End of a planA run: report text + json (+ png). Returns (report, text)."""
    rows = load_csv(csv_path)
    rep = analyze(rows, limits)
    text = format_report(rep)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'video_diag_report.txt').write_text(text + '\n')
    (out_dir / 'video_diag_report.json').write_text(json.dumps(rep, indent=2))
    rep['plot'] = str(out_dir / 'video_diag.png') if plot(rows, out_dir / 'video_diag.png') else None
    return rep, text


def find_csv(path):
    path = Path(path)
    if path.is_file():
        return path
    for candidate in (path / 'video_diag.csv', path / 'perception' / 'video_diag.csv'):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'no video_diag.csv in {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='planA log folder, perception run folder or video_diag.csv')
    parser.add_argument('--out', help='output folder (default: next to the csv)')
    args = parser.parse_args()
    csv_path = find_csv(args.path)
    rep, text = analyze_and_save(csv_path, args.out or csv_path.parent)
    print(text)
    print(f'Luu: {Path(args.out or csv_path.parent)} (video_diag_report.txt / .json'
          + (', video_diag.png)' if rep.get('plot') else ')'))


if __name__ == '__main__':
    main()
