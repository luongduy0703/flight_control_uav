"""Hover drift measurement pipeline for tools/drift_viewer.py (pure logic, no I/O).

States per Pixhawk boot:
    GROUND    landed_state ON_GROUND: launch point = latest position (as pi/console.py,
              whose TAKEOFF target is launch down_m - --height).
    AIR       airborne, waiting until height above launch is within target +- tol and
              horizontal and vertical speed <= max_speed for settle_s.
    HOVER     reference point P0 = position at hover start; every sample is stored as
              (t, dx, dy, height). Ends when the height leaves target +- exit_m for exit_s,
              on LANDING/ON_GROUND, or after duration_s (0 = no limit).
After each HOVER a summary dict is appended to `results`; after each landing the
estimated landing offset from launch is appended to `landings`.

LOCAL_POSITION_NED is PX4's own estimate. In a position-hold hover PX4 keeps that
estimate still, so this measures position-hold error. Estimator drift (vehicle moves,
estimate does not) is only visible against a real reference: compare `landings`
with the offset measured on the floor (tape) between launch and touchdown marks.
"""
import math

ON_GROUND, IN_AIR, TAKEOFF, LANDING = 1, 2, 3, 4  # MAV_LANDED_STATE
AIRBORNE = (IN_AIR, TAKEOFF, LANDING)


def slope(ts, values):
    """Least-squares slope d(values)/dt; 0 for fewer than 2 distinct times."""
    n = len(ts)
    if n < 2:
        return 0.0
    mt, mv = sum(ts) / n, sum(values) / n
    den = sum((t - mt) ** 2 for t in ts)
    return sum((t - mt) * (v - mv) for t, v in zip(ts, values)) / den if den > 0 else 0.0


def summarize(samples, target_m, reason, ref, start_t):
    """samples: [(t, dx, dy, height, range_height or None)] relative to the hover start P0."""
    ts = [s[0] for s in samples]
    dx = [s[1] for s in samples]
    dy = [s[2] for s in samples]
    h = [s[3] for s in samples]
    horiz = [math.hypot(a, b) for a, b in zip(dx, dy)]
    n = len(samples)
    mean_h = sum(h) / n
    rate = math.hypot(slope(ts, dx), slope(ts, dy))
    result = {
        'reason_end': reason,
        'start_t_boot_s': round(start_t, 3),
        'duration_s': round(ts[-1] - ts[0], 2),
        'samples': n,
        'target_height_m': target_m,
        'p0_boot_frame_m': {'x_north': round(ref[0], 3), 'y_east': round(ref[1], 3)},
        'final_drift_m': {'x_north': round(dx[-1], 3), 'y_east': round(dy[-1], 3),
                          'horizontal': round(horiz[-1], 3)},
        'max_horizontal_drift_m': round(max(horiz), 3),
        'rms_horizontal_m': round(math.sqrt(sum(v * v for v in horiz) / n), 3),
        'drift_rate_m_per_min': round(rate * 60, 3),
        'height_mean_m': round(mean_h, 3),
        'height_std_m': round(math.sqrt(sum((v - mean_h) ** 2 for v in h) / n), 3),
        'height_error_mean_m': round(mean_h - target_m, 3),
        'height_error_max_abs_m': round(max(abs(v - target_m) for v in h), 3),
    }
    ranged = [(s[3], s[4]) for s in samples if s[4] is not None]
    if ranged:
        result['range_height_mean_m'] = round(sum(r for _, r in ranged) / len(ranged), 3)
        result['ekf_minus_range_mean_m'] = round(sum(e - r for e, r in ranged) / len(ranged), 3)
    return result


def format_summary(r):
    text = (f"HOVER {r['target_height_m']:.2f} m, {r['duration_s']:.1f} s ({r['reason_end']}): "
            f"drift {r['final_drift_m']['horizontal']:.3f} m "
            f"(N {r['final_drift_m']['x_north']:+.3f}, E {r['final_drift_m']['y_east']:+.3f}), "
            f"max {r['max_horizontal_drift_m']:.3f} m, RMS {r['rms_horizontal_m']:.3f} m, "
            f"rate {r['drift_rate_m_per_min']:.3f} m/min, "
            f"height {r['height_mean_m']:.3f}+-{r['height_std_m']:.3f} m")
    if 'range_height_mean_m' in r:
        text += f", rangefinder {r['range_height_mean_m']:.3f} m"
    return text


class HoverMonitor:
    def __init__(self, target_m=0.8, tol_m=0.15, settle_s=2.0, max_speed=0.2,
                 exit_m=0.3, exit_s=1.0, duration_s=0.0):
        if not 0 < tol_m < exit_m:
            raise ValueError('need 0 < tol_m < exit_m')
        self.target_m, self.tol_m, self.settle_s = target_m, tol_m, settle_s
        self.max_speed, self.exit_m, self.exit_s = max_speed, exit_m, exit_s
        self.duration_s = duration_s
        self.results, self.landings = [], []
        self.segments = []  # (boot, first index, last index or None) into the Track arrays
        self.boot = 0
        self.samples = []
        self.reset_boot()

    def reset_boot(self, boot=None):
        """New Pixhawk boot: launch point and any open hover are meaningless now."""
        if boot is not None:
            self.boot = boot
        if self.samples:  # rebooted mid-hover
            self._finish('pixhawk reboot', None)
        self.launch = None        # (x, y, z) while ON_GROUND
        self.launch_range = None  # rangefinder reading on the ground
        self.state = 'WAIT_LANDED_STATE'
        self.settle_since = None
        self.out_since = None
        self.height = None

    def update(self, i, t, x, y, z, vx=0.0, vy=0.0, vz=0.0, landed=None, range_m=None):
        """Feed one sample (index i in the Track). Returns '', 'hover_start', 'hover_end' or 'landed'."""
        if landed == ON_GROUND:
            event = ''
            if self.samples:
                self._finish('landed', i)
                event = 'hover_end'
            if self.state in ('AIR', 'HOVER') and self.launch is not None:
                self.landings.append({'t_boot_s': round(t, 3),
                                      'offset_from_launch_m': {
                                          'x_north': round(x - self.launch[0], 3),
                                          'y_east': round(y - self.launch[1], 3),
                                          'horizontal': round(math.hypot(x - self.launch[0],
                                                                         y - self.launch[1]), 3)}})
                event = event or 'landed'
            self.launch, self.launch_range = (x, y, z), range_m
            self.state, self.settle_since, self.height = 'GROUND', None, 0.0
            return event
        if self.launch is None or landed not in AIRBORNE:
            return ''
        self.height = z - self.launch[2]
        range_h = (range_m - self.launch_range
                   if range_m is not None and self.launch_range is not None else None)
        if self.state != 'HOVER':
            self.state = 'AIR'
            steady = (abs(self.height - self.target_m) <= self.tol_m
                      and math.hypot(vx, vy) <= self.max_speed and abs(vz) <= self.max_speed)
            if not steady:
                self.settle_since = None
                return ''
            if self.settle_since is None:
                self.settle_since = t
            if t - self.settle_since < self.settle_s:
                return ''
            self.state, self.ref, self.start_t, self.out_since = 'HOVER', (x, y), t, None
            self.samples = []
            self.segments.append((self.boot, i, None))
        self.samples.append((t, x - self.ref[0], y - self.ref[1], self.height, range_h))
        if landed == LANDING:
            self._finish('landing', i)
            return 'hover_end'
        if abs(self.height - self.target_m) > self.exit_m:
            self.out_since = self.out_since if self.out_since is not None else t
            if t - self.out_since >= self.exit_s:
                self._finish('left height band', i)
                return 'hover_end'
        else:
            self.out_since = None
        if self.duration_s > 0 and t - self.start_t >= self.duration_s:
            self._finish('duration reached', i)
            return 'hover_end'
        return 'hover_start' if len(self.samples) == 1 else ''

    def _finish(self, reason, i):
        boot, first, _ = self.segments[-1]
        self.segments[-1] = (boot, first, i)
        self.results.append(summarize(self.samples, self.target_m, reason, self.ref, self.start_t))
        self.samples = []
        self.state = 'AIR'
        self.settle_since = None

    def status_line(self):
        if self.state == 'WAIT_LANDED_STATE':
            return 'hover: waiting for ON_GROUND (EXTENDED_SYS_STATE) to set the launch point'
        if self.state == 'GROUND':
            return f'hover: on ground, launch point set; target {self.target_m:.2f} m above launch'
        if self.state == 'AIR':
            return (f'hover: airborne, height {self.height:+.2f} m above launch; waiting for '
                    f'{self.target_m:.2f}+-{self.tol_m:.2f} m steady {self.settle_s:.0f} s')
        t, dx, dy, h, _ = self.samples[-1]
        return (f'HOVER {t - self.start_t:5.1f} s  drift N {dx:+.3f} E {dy:+.3f} '
                f'= {math.hypot(dx, dy):.3f} m  height {h:.3f} m '
                f'(err {h - self.target_m:+.3f})')
