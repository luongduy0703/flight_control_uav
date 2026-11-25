"""Hold the UAV above the launch ArUco under it, and measure flight offset/height from the marker (not the EKF).

uavp_data_capture --marker measures the launch ArUco (his_perception_uav-main, launch_marker section) and sends one
"MARKER seen=1 ..." line per valid measurement over UDP 127.0.0.1:<port>. This module (inside pi/console.py, the
only process that sends flight commands) turns them into small, bounded HOLD setpoint corrections:

    camera position in the marker frame S (ArUco, metres)  --rotate by the marker heading in NED-->  error e (NED)
    new hover setpoint = current setpoint - e   (step <= max_step_m, total offset from launch <= max_total_m)

Moving the body moves the camera by the same amount (yaw unchanged), so T_BC is not needed. The EKF is only used for
the short move itself; where the UAV should be comes from the marker. Same safety style as planA hold_step:
deadband, step limit, total limit, minimum period, sample count/spread/age checks, and only from confirmed HOLD.
"""
import logging
import math
import socket
import time
from collections import deque
from dataclasses import dataclass

from perception_process import parse_marked

DEFAULT_PORT = 14590


@dataclass(frozen=True)
class HoldConfig:
    deadband_m: float = 0.05     # camera within 5 cm of the marker centre: no correction (ArUco noise ~1-2 cm)
    max_step_m: float = 0.10     # each correction moves the setpoint at most 10 cm
    max_total_m: float = 0.40    # setpoint never further than 40 cm from the launch point (else: refuse, warn)
    period_s: float = 3.0        # at least 3 s between corrections (each one: HOLD -> Offboard -> arrive -> HOLD)
    window_s: float = 1.5        # estimate from the last 1.5 s of measurements, taken after HOLD was confirmed
    min_samples: int = 3
    max_spread_m: float = 0.10   # marker position in the EKF scattered > 10 cm in the window: do not trust
    max_age_s: float = 0.6       # newest measurement older than this: marker lost, no correction

    def __post_init__(self):
        for key in ('deadband_m', 'max_step_m', 'max_total_m', 'period_s', 'window_s', 'max_spread_m', 'max_age_s'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'marker hold {key} must be positive')
        if self.max_step_m > 0.2 or self.max_total_m > 1.0:
            raise ValueError('marker hold: max_step_m <= 0.2 and max_total_m <= 1.0')
        if self.deadband_m >= self.max_total_m or self.min_samples < 1:
            raise ValueError('marker hold: deadband_m < max_total_m and min_samples >= 1')


def _median(values):
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _circular_mean_deg(angles):
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(s, c))


def _vec3(value):
    if isinstance(value, tuple) and len(value) == 3 and all(math.isfinite(v) for v in value):
        return value
    return None


def estimate(samples):
    """Robust estimate from MARKER values. None without a usable camera-in-marker measurement.

    cam_S_m: camera in the marker frame (x along the printed arrow, y right, z down; m). height_m = -z.
    yaw_deg / p_M_W_m / spread_m: marker heading and position in the EKF NED frame (None without FC pose).
    """
    cams = [c for c in (_vec3(v.get('p_cam_S_mm')) for v in samples) if c is not None]
    if not cams:
        return None
    cam = tuple(_median([c[i] for c in cams]) / 1000.0 for i in range(3))
    out = {'n': len(cams), 'cam_S_m': cam, 'height_m': -cam[2], 'offset_m': math.hypot(cam[0], cam[1]),
           'yaw_deg': None, 'p_M_W_m': None, 'spread_m': None}
    world = [(_vec3(v.get('p_M_W_mm')), v.get('yaw_M_W_deg')) for v in samples if v.get('world') == 1.0]
    world = [(p, y) for p, y in world if p is not None and isinstance(y, float) and math.isfinite(y)]
    if world:
        p = tuple(_median([w[0][i] for w in world]) / 1000.0 for i in range(3))
        out['p_M_W_m'] = p
        out['yaw_deg'] = _circular_mean_deg([w[1] for w in world])
        out['spread_m'] = max(math.hypot(w[0][0] / 1000.0 - p[0], w[0][1] / 1000.0 - p[1]) for w in world)
        out['n_world'] = len(world)
    return out


def correction_ned(est):
    """Setpoint move (dN, dE) in metres that brings the camera over the marker centre."""
    ex, ey = est['cam_S_m'][0], est['cam_S_m'][1]  # camera minus desired (marker centre), marker frame
    a = math.radians(est['yaw_deg'])
    c, s = math.cos(a), math.sin(a)
    return -(c * ex - s * ey), -(s * ex + c * ey)


class MarkerFeed:
    """Non-blocking UDP listener for MARKER lines (127.0.0.1 only)."""

    def __init__(self, port=DEFAULT_PORT, clock=time.monotonic, sock=None):
        self.clock = clock
        self.samples = deque(maxlen=400)  # (t_s, values); t_s = frame time (same CLOCK_MONOTONIC on the Pi)
        self.received = 0
        if sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('127.0.0.1', port))
        sock.setblocking(False)
        self.sock = sock

    def feed_line(self, line, rx_time=None):
        parsed = parse_marked(line)
        if parsed is None or parsed[0] != 'marker' or parsed[1].get('seen') != 1.0:
            return False
        values = parsed[1]
        t_ns = values.get('t_ns')
        t = t_ns / 1e9 if isinstance(t_ns, float) and t_ns > 0 else (self.clock() if rx_time is None else rx_time)
        self.samples.append((t, values))
        self.received += 1
        return True

    def poll(self):
        while True:
            try:
                data = self.sock.recv(2048)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            self.feed_line(data.decode('utf-8', 'replace'))

    def since(self, t0):
        return [v for t, v in self.samples if t >= t0]

    def newest_age(self, now):
        return now - self.samples[-1][0] if self.samples else None

    def close(self):
        self.sock.close()


class MarkerHold:
    def __init__(self, feed, cfg=None, enabled=True, clock=time.monotonic):
        self.feed, self.cfg, self.enabled, self.clock = feed, cfg or HoldConfig(), enabled, clock
        self.last_move_at = -math.inf
        self.hold_since = None   # when the current confirmed HOLD started (samples before it describe motion)
        self.moves = 0
        self.note = 'waiting for HOLD after takeoff'
        self.note_key = None
        self.log = logging.getLogger(__name__)

    def _say(self, key, text, output):
        """Print once per kind of situation (key); numbers inside the text may change every call."""
        if key != self.note_key or key == 'move':
            self.log.info('Marker hold: %s', text)
            if output is not None:
                output(f'[ArUco hold] {text}')
        self.note, self.note_key = text, key

    def measurement(self):
        """ArUco measurement of the last window (status / flight-distance report), independent of the EKF."""
        now = self.clock()
        est = estimate(self.feed.since(now - self.cfg.window_s))
        age = self.feed.newest_age(now)
        out = {'enabled': self.enabled, 'received': self.feed.received, 'moves': self.moves, 'note': self.note,
               'newest_age_s': None if age is None else round(age, 2)}
        if est is not None:
            out.update({'n': est['n'],
                        'camera_from_marker_m': [round(est['cam_S_m'][0], 3), round(est['cam_S_m'][1], 3)],
                        'horizontal_offset_m': round(est['offset_m'], 3),
                        'camera_height_m': round(est['height_m'], 3),
                        'marker_spread_in_ekf_m': None if est['spread_m'] is None else round(est['spread_m'], 3)})
        return out

    def step(self, fc, output=print):
        """Call often (console loop). Sends at most one bounded HOLD correction through fc.fly()."""
        self.feed.poll()
        now = self.clock()
        in_hold = (fc.phase == 'HOLD' and getattr(fc, 'takeoff_complete', False)
                   and fc.launch_ned is not None and fc.target is not None and fc.ready_for_fly())
        if not in_hold:
            self.hold_since = None
            return None
        if self.hold_since is None:
            self.hold_since = now
        if not self.enabled or now - self.last_move_at < self.cfg.period_s:
            return None
        start = max(self.hold_since, now - self.cfg.window_s)
        est = estimate(self.feed.since(start))
        age = self.feed.newest_age(now)
        if est is None or est['n'] < self.cfg.min_samples or age is None or age > self.cfg.max_age_s:
            self._say('lost', 'marker not seen enough in HOLD: no correction', output)
            return None
        if est['yaw_deg'] is None:
            self._say('nopose', 'no Pixhawk pose in MARKER lines (uavp_fc_bridge?): measuring only, no correction', output)
            return None
        if est['spread_m'] > self.cfg.max_spread_m:
            self._say('spread', f'marker scattered {est["spread_m"] * 100:.0f} cm in the EKF: not trusted', output)
            return None
        dn, de = correction_ned(est)
        dist = math.hypot(dn, de)
        if dist < self.cfg.deadband_m:
            self._say('ok', f'camera {est["offset_m"] * 100:.1f} cm from marker, height {est["height_m"]:.2f} m: OK', output)
            return None
        k = min(1.0, self.cfg.max_step_m / dist)
        north = fc.target['north_m'] + k * dn
        east = fc.target['east_m'] + k * de
        total = math.hypot(north - fc.launch_ned['north_m'], east - fc.launch_ned['east_m'])
        if total > self.cfg.max_total_m:
            self._say('limit', f'correction would put the setpoint {total * 100:.0f} cm from launch '
                      f'(> {self.cfg.max_total_m * 100:.0f} cm): refused, check marker/EKF', output)
            return None
        self.last_move_at = now
        try:
            fc.fly(north, east, explicit=True)
        except Exception as error:  # noqa: BLE001 - the controller refuses: report and keep HOLD
            self._say('refused', f'setpoint refused: {error}', output)
            return None
        self.moves += 1
        self.hold_since = None
        self._say('move', f'camera {est["offset_m"] * 100:.1f} cm from marker -> move {k * dist * 100:.1f} cm, '
                  f'setpoint NED ({north:.3f}, {east:.3f}), {total * 100:.1f} cm from launch', output)
        return north, east
