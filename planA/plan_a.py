"""planA: pure logic (no MAVSDK, no process). Used by run_plan_a.py and tests/test_plan_a.py.

Frames:
  W  perception world = PX4 local NED (EKF), mm. x/y from LOCAL_POSITION_NED; z from EKF or rangefinder
     (perception sources.mavlink.use_rangefinder).
  S  scenario frame, metres: origin = EKF position at takeoff (optical flow), z down.
     axes 'launch_heading': x = vehicle nose at takeoff, y = right. axes 'ned': x = North, y = East.
  A  launch ArUco frame (EVALUATION ONLY), metres: origin = centre of the printed marker on the floor,
     x = printed '+x' arrow, y = right of it (seen from above), z down (floor 0). Ground truth measured with a
     tape from the marker. Perception reports the marker pose in W; nothing here feeds a setpoint.
"""
import copy
import math
from pathlib import Path

DEFAULTS = {
    'flight': {
        'connection': 'udpin://127.0.0.1:14540', 'profile': 'hardware', 'system_id': 1,
        'position_source': 'flow', 'site_origin_file': 'site_origin.json', 'hover_height_m': 0.8,
        'xy_tolerance_m': 0.10, 'z_tolerance_m': 0.15, 'yaw_tolerance_deg': 10.0, 'settle_time_s': 2.0,
        'target_timeout_s': 30.0, 'hover_stabilize_s': 5.0, 'hover_max_speed_m_s': 0.10,
        'max_target_distance_m': 1.5, 'control_cores': '0,1', 'auto_land': False,
    },
    'control': {  # operator gates (run_plan_a.py): nothing moves without a command
        'confirm_arm': True, 'confirm_takeoff': True, 'confirm_moves': True, 'require_arm_before_takeoff': True,
        'confirm_timeout_s': 8.0,
    },
    'frame': {'axes': 'launch_heading', 'z_source': 'ekf', 'fc_above_ground_m': 0.25},
    'approach': {
        'target': 'bbox', 'lidar_line': True, 'calib_T_CL': 'configs/calib/T_CL.yaml',
        'calib_T_BC': 'configs/calib/T_BC.yaml',
        'body_offset_m': [0.0, 0.0], 'wait_after_arrival_s': 1.0, 'align_wait_s': 3.0,
        'max_corrections': 3, 'min_correction_m': 0.02, 'max_correction_m': 0.30,
        'final_height_m': None,
    },
    'perception': {
        'dir': '~/uav_perception_pi', 'config': 'configs/run/real_time.yaml', 'ready_timeout_s': 90,
        'result_timeout_s': 180, 'e_max_mm': 15.0, 'lock_max_duration_s': 120, 'min_segments': 5,
        'window_s': 2.0, 'max_window_s': 10.0, 'scan_max_s': 12.0, 'require_hover': False,
        'set': [], 'auto_start': False, 'detect_timeout_s': 30.0,
        'size_abort': False,  # False: KHOA abort do kich thuoc (too_big / bbox_max_mm) o s3
        'match_radius_px': 80.0,  # ghep detection voi track khi perception KHONG co tu the FC (du phong)
        'match_radius_mm': 80.0,  # co tu the FC: ghep detection voi track theo diem tren san (mm)
        'same_object_mm': 100.0,  # 2 tam tren san cach <= 100 mm -> cung mot vat (ke ca vat da ABORT)
        'fp16': True,
    },
    'ground_truth': {'x_m': 0.635, 'y_m': 0.255, 'z_m': None},
    'marker': {  # launch ArUco: evaluation only (tape vs object), never used for control
        'enabled': False, 'dictionary': 'DICT_4X4_50', 'marker_id': 1, 'side_mm': 100.0, 'period_s': 0.2,
        'max_reproj_px': 2.0, 'max_tilt_deg': 15.0, 'hover_window_s': 3.0, 'min_samples': 5,
        'launch_window_s': 2.0, 'launch_max_offset_m': 0.03, 'launch_max_yaw_deg': 5.0, 'require_at_takeoff': False,
        'ground_truth': {'x_m': None, 'y_m': None, 'z_m': None},
        # perception do marker tiep khi quet / bam: KHONG chon to giay lam vat (vung exclude_radius_mm quanh marker)
        'track_during_scan': True, 'exclude_radius_mm': 150.0,
        # giu vi tri theo marker khi hover (STABILIZE / WAIT_START / SCANNING): EKF troi -> marker "dich" trong EKF ->
        # setpoint hover += do dich (tu dong, buoc nho, gioi han tong). Khong dua marker vao EKF cua Pixhawk.
        'hold': {'enabled': True, 'deadband_m': 0.05, 'max_step_m': 0.10, 'max_total_m': 0.30, 'period_s': 3.0,
                 'window_s': 1.5, 'min_samples': 3, 'max_spread_m': 0.05},
    },
    'video_diag': {'enabled': False, 'every_n': 1},  # perception video_diag.csv -> report at the end of the run
    'imu_log': {  # Pixhawk IMU / vibration telemetry (pi/imu_logger.py) -> imu_diag report at the end of the run
        'enabled': False, 'highres_hz': 200.0, 'vibration_hz': 10.0, 'motors_hz': 10.0,
        'camera_f_px': 823.3, 'readout_ms': 20.0,
    },
    'logging': {'dir': 'planA/logs'},
}
MARKER_PHASES = {'WAIT_READY': 'ground', 'WAIT_TAKEOFF': 'ground', 'TAKEOFF': 'takeoff',
                 'STABILIZE': 'hover', 'WAIT_START': 'hover', 'SCANNING': 'scan'}
HOLD_STATES = ('STABILIZE', 'WAIT_START', 'SCANNING')  # giu vi tri theo marker chi khi hover tai cho
AXES = ('launch_heading', 'ned')
TARGETS = ('bbox', 'mask')  # TARGET: bbox_p_W_mm (tam bbox detect) | p_W_mm (tam mask, s4 lan dau)
Z_SOURCES = ('ekf', 'rangefinder')


# ---- config ----------------------------------------------------------------------------------------
def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key not in out:
            raise ValueError(f'unknown config key: {key}')
        if isinstance(out[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f'{key} must be a mapping')
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def _positive(cfg, section, key):
    value = cfg[section][key]
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{section}.{key} must be a positive number')


def validate(cfg):
    if cfg['frame']['axes'] not in AXES:
        raise ValueError(f'frame.axes must be one of {AXES}')
    if cfg['frame']['z_source'] not in Z_SOURCES:
        raise ValueError(f'frame.z_source must be one of {Z_SOURCES}')
    h = cfg['frame']['fc_above_ground_m']
    if not isinstance(h, (int, float)) or not math.isfinite(h) or not 0.0 <= h <= 1.0:
        raise ValueError('frame.fc_above_ground_m must be a number in [0, 1] m')
    for key in ('hover_height_m', 'xy_tolerance_m', 'z_tolerance_m', 'hover_stabilize_s',
                'hover_max_speed_m_s', 'max_target_distance_m', 'target_timeout_s'):
        _positive(cfg, 'flight', key)
    if not 0.5 <= cfg['flight']['hover_height_m'] <= 3.0:
        raise ValueError('flight.hover_height_m must be 0.5..3 m for this scenario')
    if cfg['flight']['z_tolerance_m'] >= cfg['flight']['hover_height_m']:
        raise ValueError('flight.z_tolerance_m must be less than hover_height_m')
    for key in ('e_max_mm', 'lock_max_duration_s', 'result_timeout_s', 'ready_timeout_s', 'window_s',
                'detect_timeout_s', 'max_window_s', 'scan_max_s'):
        _positive(cfg, 'perception', key)
    pc = cfg['perception']
    if not pc['window_s'] <= pc['max_window_s'] <= 15.0:
        raise ValueError('perception.max_window_s phai trong [window_s, 15] s')
    if pc['scan_max_s'] < pc['max_window_s']:
        raise ValueError('perception.scan_max_s phai >= max_window_s (de du thoi gian tich luy)')
    if not isinstance(cfg['perception']['auto_start'], bool):
        raise ValueError('perception.auto_start must be true or false')
    if not isinstance(cfg['perception']['size_abort'], bool):
        raise ValueError('perception.size_abort phai la true / false')
    if not 10.0 <= cfg['perception']['match_radius_px'] <= 300.0:
        raise ValueError('perception.match_radius_px phai trong [10, 300] px')
    for k in ('match_radius_mm', 'same_object_mm'):
        if not 20.0 <= cfg['perception'][k] <= 500.0:
            raise ValueError(f'perception.{k} phai trong [20, 500] mm')
    if not isinstance(cfg['perception']['fp16'], bool):
        raise ValueError('perception.fp16 phai la true / false')
    final = cfg['approach']['final_height_m']
    if final is not None and (not isinstance(final, (int, float)) or not math.isfinite(final)
                              or not 0.3 <= final <= cfg['flight']['hover_height_m']):
        raise ValueError('approach.final_height_m must be null or 0.3..hover_height_m')
    if cfg['approach']['target'] not in TARGETS:
        raise ValueError(f'approach.target must be one of {TARGETS}')
    if not isinstance(cfg['approach']['lidar_line'], bool):
        raise ValueError('approach.lidar_line must be true or false')
    offset = cfg['approach']['body_offset_m']
    if len(offset) != 2 or not all(isinstance(v, (int, float)) for v in offset):
        raise ValueError('approach.body_offset_m must be [x, y]')
    if cfg['approach']['min_correction_m'] >= cfg['approach']['max_correction_m']:
        raise ValueError('approach.min_correction_m must be < max_correction_m')
    gt = cfg['ground_truth']
    for key in ('x_m', 'y_m'):
        if not isinstance(gt[key], (int, float)):
            raise ValueError(f'ground_truth.{key} must be a number')
    if gt['z_m'] is not None and not isinstance(gt['z_m'], (int, float)):
        raise ValueError('ground_truth.z_m must be a number or null')
    if not isinstance(cfg['perception']['set'], list):
        raise ValueError('perception.set must be a list of KEY=VALUE')
    m = cfg['marker']
    if not isinstance(m['enabled'], bool) or not isinstance(m['require_at_takeoff'], bool):
        raise ValueError('marker.enabled / marker.require_at_takeoff must be true or false')
    if m['enabled']:
        for key in ('side_mm', 'period_s', 'max_reproj_px', 'max_tilt_deg', 'hover_window_s', 'launch_window_s',
                    'launch_max_offset_m', 'launch_max_yaw_deg'):
            _positive(cfg, 'marker', key)
        if not isinstance(m['marker_id'], int) or m['marker_id'] < 0:
            raise ValueError('marker.marker_id must be an integer >= 0')
        if not isinstance(m['min_samples'], int) or m['min_samples'] < 1:
            raise ValueError('marker.min_samples must be an integer >= 1')
        if not isinstance(m['track_during_scan'], bool):
            raise ValueError('marker.track_during_scan must be true or false')
        if not isinstance(m['exclude_radius_mm'], (int, float)) or not 0 <= m['exclude_radius_mm'] <= 500:
            raise ValueError('marker.exclude_radius_mm must be in [0, 500] (0 = off)')
        h = m['hold']
        if not isinstance(h['enabled'], bool):
            raise ValueError('marker.hold.enabled must be true or false')
        for key in ('deadband_m', 'max_step_m', 'max_total_m', 'period_s', 'window_s', 'max_spread_m'):
            if not isinstance(h[key], (int, float)) or not math.isfinite(h[key]) or h[key] <= 0:
                raise ValueError(f'marker.hold.{key} must be > 0')
        if not isinstance(h['min_samples'], int) or h['min_samples'] < 1:
            raise ValueError('marker.hold.min_samples must be an integer >= 1')
        if h['max_step_m'] > 0.2 or h['max_total_m'] > 0.5:
            raise ValueError('marker.hold: max_step_m <= 0.2 m, max_total_m <= 0.5 m (an toan)')
        if h['deadband_m'] >= h['max_total_m']:
            raise ValueError('marker.hold.deadband_m must be < max_total_m')
    for key in ('x_m', 'y_m', 'z_m'):
        v = m['ground_truth'][key]
        if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)):
            raise ValueError(f'marker.ground_truth.{key} must be a number or null')
    for key in ('highres_hz', 'vibration_hz', 'motors_hz'):
        hz = cfg['imu_log'][key]
        if not isinstance(hz, (int, float)) or not 0 <= hz <= 500:
            raise ValueError(f'imu_log.{key} must be 0..500 Hz (0 = keep the PX4 rate)')
    for key in ('camera_f_px', 'readout_ms'):
        _positive(cfg, 'imu_log', key)
    for key in ('confirm_arm', 'confirm_takeoff', 'confirm_moves', 'require_arm_before_takeoff'):
        if not isinstance(cfg['control'][key], bool):
            raise ValueError(f'control.{key} must be true / false')
    t = cfg['control']['confirm_timeout_s']
    if not isinstance(t, (int, float)) or not 2.0 <= t <= 9.0:
        raise ValueError('control.confirm_timeout_s must be 2..9 s (PX4 disarms ~10 s after ARM on the ground)')
    every_n = cfg['video_diag']['every_n']
    if not isinstance(every_n, int) or every_n < 1:
        raise ValueError('video_diag.every_n must be an integer >= 1')
    return cfg


def load_config(path):
    import yaml  # python3-yaml (Ubuntu default)
    data = yaml.safe_load(Path(path).read_text()) or {}
    return validate(merge(DEFAULTS, data))


def perception_overrides(cfg):
    """--set for uavp_real_time: alignment gate, long lock (the UAV flies during it), world fusion."""
    p = cfg['perception']
    return [
        'lock_track.align_gate=true',
        f"lock_track.align_e_max_mm={p['e_max_mm']}",
        f"lock_track.max_duration_s={p['lock_max_duration_s']}",
        f"lock_track.min_segments={p['min_segments']}",
        'lock_track.stop_when_converged=true',
        's4_fusion.method=world',
        f"s4_fusion.world.window_s={p['window_s']}",
        f"s4_fusion.world.max_window_s={p['max_window_s']}",  # mo rong cua so toi khi ra vector huong
        'lock_track.wait_scan_command=true',  # chi tich luy / ket thuc sau lenh QUET (UAV da o do cao quet)
        f"lock_track.scan_max_duration_s={p['scan_max_s']}",
        f"s3_select.size_abort={'true' if p['size_abort'] else 'false'}",
        f"s3_select.match_radius_px={p['match_radius_px']}",
        's3_select.world_track=true',  # vat = diem tren san, du doan ROI bang tu the camera luc chup
        f"s3_select.match_radius_mm={p['match_radius_mm']}",
        f"s3_select.same_object_mm={p['same_object_mm']}",
        f"start.require_hover={'true' if p['require_hover'] else 'false'}",
        f"sources.mavlink.use_rangefinder={'true' if cfg['frame']['z_source'] == 'rangefinder' else 'false'}",
        *marker_overrides(cfg),
        *([f"{stage}.model.use_fp16_storage=true" for stage in ('s1_detect', 's2_segment')] if p['fp16'] else []),
        *(["video_diag.enabled=true", f"video_diag.every_n={cfg['video_diag']['every_n']}"]
          if cfg['video_diag']['enabled'] else []),
        *p['set'],
    ]


def marker_overrides(cfg):
    m = cfg['marker']
    if not m['enabled']:
        return []
    return ['launch_marker.enabled=true', f"launch_marker.dictionary={m['dictionary']}",
            f"launch_marker.marker_id={m['marker_id']}", f"launch_marker.side_mm={m['side_mm']}",
            f"launch_marker.period_s={m['period_s']}", f"launch_marker.max_reproj_px={m['max_reproj_px']}",
            f"launch_marker.max_tilt_deg={m['max_tilt_deg']}",
            f"launch_marker.track_during_scan={'true' if m['track_during_scan'] else 'false'}",
            f"launch_marker.exclude_radius_mm={m['exclude_radius_mm']}"]


# ---- frames ----------------------------------------------------------------------------------------
def _yaw(axes, yaw0_deg):
    return math.radians(yaw0_deg) if axes == 'launch_heading' else 0.0


def ned_vec_to_scenario(v, yaw0_deg, axes):
    """Rotate a NED vector into scenario axes (z unchanged)."""
    a = _yaw(axes, yaw0_deg)
    c, s = math.cos(a), math.sin(a)
    return (c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2])


def scenario_vec_to_ned(v, yaw0_deg, axes):
    a = _yaw(axes, yaw0_deg)
    c, s = math.cos(a), math.sin(a)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


def ned_to_scenario(p_ned_m, launch, yaw0_deg, axes, z_source='ekf', fc_above_ground_m=0.0):
    """World point (m, NED) -> scenario (m). launch = {'north_m', 'east_m', 'down_m'} of the FC at takeoff.

    Scenario origin = the FLOOR right below the FC at takeoff (as handoff_uav frames.world_origin_in_ekf):
    z_source 'ekf': z = z_EKF - (launch down + fc_above_ground_m), so the floor is z = 0 and a top surface h above
    the floor is z = -h (same convention as ground_truth.z_m). 'rangefinder': z is already floor-referenced.
    """
    if z_source == 'ekf':
        z = p_ned_m[2] - (launch['down_m'] + fc_above_ground_m)
    else:
        z = p_ned_m[2]
    d = (p_ned_m[0] - launch['north_m'], p_ned_m[1] - launch['east_m'], z)
    return ned_vec_to_scenario(d, yaw0_deg, axes)


def to_scenario(p_ned_m, launch, yaw0_deg, cfg, z_source=None):
    """ned_to_scenario with the axes / z source / FC height of the config."""
    f = cfg['frame']
    return ned_to_scenario(p_ned_m, launch, yaw0_deg, f['axes'], z_source or f['z_source'], f['fc_above_ground_m'])


def scenario_to_ned_xy(x, y, launch, yaw0_deg, axes):
    n, e, _ = scenario_vec_to_ned((x, y, 0.0), yaw0_deg, axes)
    return launch['north_m'] + n, launch['east_m'] + e


def yaw_to_scenario(yaw_W_deg, yaw0_deg, axes):
    """Long-edge direction (North-referenced, [0, 180)) -> scenario x-referenced, [0, 180)."""
    if yaw_W_deg is None or not math.isfinite(yaw_W_deg):
        return None
    return (yaw_W_deg - math.degrees(_yaw(axes, yaw0_deg))) % 180.0


def tilt_deg(normal):
    """Angle between the top-surface normal and the vertical."""
    norm = math.sqrt(sum(v * v for v in normal))
    if not norm or not math.isfinite(norm):
        return None
    return math.degrees(math.acos(min(1.0, abs(normal[2]) / norm)))


# ---- LiDAR line geometry ---------------------------------------------------------------------------
def load_lidar_geometry(perception_dir, cfg):
    """R_CL, t_CL (LiDAR -> camera) and R_BC, t_BC (camera -> body) from the perception calibration files (m)."""
    import yaml
    base = Path(perception_dir).expanduser()
    a = cfg['approach']
    cl = yaml.safe_load((base / a['calib_T_CL']).read_text())
    bc = yaml.safe_load((base / a['calib_T_BC']).read_text())
    return {'R_CL': cl['R_CL'], 't_CL': cl['t_CL_m'], 'R_BC': bc['R_BC'], 't_BC': bc['t_BC_m']}


def _matvec(R, v):
    return [sum(R[i][k] * v[k] for k in range(3)) for i in range(3)]


def lidar_foot_body(geometry, d_m):
    """Point of the LiDAR scan line on a level surface d_m below the camera, closest to the optical axis.

    Body frame (x forward, y right), metres. The scan plane passes through t_CL with normal = LiDAR z axis
    (column 3 of R_CL); on the plane z_C = d its line is n_x x + n_y y = n.t - n_z d.
    """
    if not _finite(d_m) or d_m <= 0:
        return None
    R, t = geometry['R_CL'], geometry['t_CL']
    n = [R[0][2], R[1][2], R[2][2]]
    denom = n[0] ** 2 + n[1] ** 2
    if denom < 1e-6:  # scan plane parallel to the image: no line under the camera
        return None
    k = (n[0] * t[0] + n[1] * t[1] + n[2] * t[2] - n[2] * d_m) / denom
    p_B = [a + b for a, b in zip(_matvec(geometry['R_BC'], [k * n[0], k * n[1], d_m]), geometry['t_BC'])]
    return p_B[0], p_B[1]


def camera_depth_to(target_W_mm, uav_down_m, geometry):
    """Camera height above the target (m) with the UAV level: target z - (UAV z + camera z offset)."""
    if target_W_mm is None or not _finite(target_W_mm[2], uav_down_m):
        return None
    return target_W_mm[2] / 1000.0 - (uav_down_m + (geometry['t_BC'][2] if geometry else 0.0))


# ---- flight decisions ------------------------------------------------------------------------------
def approach_setpoint(target_W_mm, launch, yaw0_deg, cfg, line_offset_body=None):
    """NED (north, east) setpoint so that the LiDAR line (or the body) is over the target. Raises if too far.

    line_offset_body: (x, y) m of the LiDAR foot point from the body origin (lidar_foot_body); the body goes to
    target - R(yaw0) offset so that the scan line crosses the target. None: body over the target.
    """
    axes = cfg['frame']['axes']
    x, y, _ = to_scenario([v / 1000.0 for v in target_W_mm], launch, yaw0_deg, cfg)
    off = cfg['approach']['body_offset_m']
    sx, sy = x + off[0], y + off[1]
    if line_offset_body is not None:
        # body offset (x forward, y right at yaw0) -> NED -> scenario axes
        a = math.radians(yaw0_deg)
        dn = math.cos(a) * line_offset_body[0] - math.sin(a) * line_offset_body[1]
        de = math.sin(a) * line_offset_body[0] + math.cos(a) * line_offset_body[1]
        ox, oy, _ = ned_vec_to_scenario((dn, de, 0.0), yaw0_deg, axes)
        sx, sy = sx - ox, sy - oy
    limit = cfg['flight']['max_target_distance_m']
    if not (math.isfinite(sx) and math.isfinite(sy)) or math.hypot(sx, sy) > limit:
        raise ValueError(f'setpoint ({sx:.2f}, {sy:.2f}) m is farther than max_target_distance_m={limit} m')
    return scenario_to_ned_xy(sx, sy, launch, yaw0_deg, axes), (sx, sy)


def correction_setpoint(current_ned, dW_mm, launch, cfg):
    """Shift the setpoint by the horizontal part of perception's dW (LiDAR line -> object centre).

    Returns (north, east) or None when the shift is below min_correction_m (nothing to do).
    """
    if dW_mm is None or len(dW_mm) < 2 or not all(math.isfinite(v) for v in dW_mm[:2]):
        return None
    dn, de = dW_mm[0] / 1000.0, dW_mm[1] / 1000.0
    size = math.hypot(dn, de)
    a = cfg['approach']
    if size < a['min_correction_m']:
        return None
    if size > a['max_correction_m']:
        dn, de = dn * a['max_correction_m'] / size, de * a['max_correction_m'] / size
    n, e = current_ned[0] + dn, current_ned[1] + de
    if math.hypot(n - launch['north_m'], e - launch['east_m']) > cfg['flight']['max_target_distance_m']:
        raise ValueError('correction would leave max_target_distance_m')
    return n, e


class SpeedWindow:
    """Mean speed of the local position over the last window_s (m/s)."""

    def __init__(self, window_s=1.0):
        self.window_s = window_s
        self.samples = []

    def add(self, t, p):
        self.samples.append((t, tuple(p)))
        while self.samples and t - self.samples[0][0] > self.window_s:
            self.samples.pop(0)

    def speed(self):
        if len(self.samples) < 2:
            return None
        (t0, p0), (t1, p1) = self.samples[0], self.samples[-1]
        if t1 - t0 < 0.5 * self.window_s:
            return None
        return math.dist(p0, p1) / (t1 - t0)


# ---- launch ArUco (evaluation only) ---------------------------------------------------------------
def _finite(*values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


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


def marker_world_sample(values):
    """MARKER line values -> ((n, e, d) m, yaw_deg) of the marker in W, or None without FC pose."""
    p, yaw = values.get('p_M_W_mm'), values.get('yaw_M_W_deg')
    if values.get('world') != 1.0 or not isinstance(p, tuple) or len(p) != 3 or not _finite(*p, yaw):
        return None
    return tuple(v / 1000.0 for v in p), yaw


def estimate_marker_frame(samples):
    """Robust marker pose in W from MARKER values (median position, circular-mean yaw) or None."""
    world = [w for w in (marker_world_sample(v) for v in samples) if w is not None]
    if not world:
        return None
    p = tuple(_median([w[0][i] for w in world]) for i in range(3))
    yaw = _circular_mean_deg([w[1] for w in world])
    spread_m = max(math.dist(p, w[0]) for w in world)
    yaw_spread = max(abs(_wrap_deg(w[1] - yaw)) for w in world)
    return {'n': len(world), 'p_W_m': p, 'yaw_deg': yaw, 'spread_m': spread_m, 'yaw_spread_deg': yaw_spread}


def ned_to_marker(p_ned_m, frame):
    """NED point (m) -> marker frame A (m): translate to the marker centre, rotate by the arrow heading."""
    d = [p_ned_m[i] - frame['p_W_m'][i] for i in range(3)]
    a = math.radians(frame['yaw_deg'])
    c, s = math.cos(a), math.sin(a)
    return (c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2])


def frame_change(before, after):
    """How far the EKF thinks the (fixed) marker moved between two estimates: EKF drift + calibration."""
    if before is None or after is None:
        return None
    return {'distance_m': math.dist(before['p_W_m'], after['p_W_m']),
            'horizontal_m': math.hypot(after['p_W_m'][0] - before['p_W_m'][0], after['p_W_m'][1] - before['p_W_m'][1]),
            'yaw_deg': _wrap_deg(after['yaw_deg'] - before['yaw_deg'])}


def first_marker_reference(samples, min_samples):
    """ArUco LAN DAU: marker trong W (EKF) tu min_samples lan thay co tu the FC dau tien. Co dinh tu do (he local de
    so sai so vat gap + moc giu vi tri). None khi chua du mau."""
    world = [v for v in samples if marker_world_sample(v) is not None][:min_samples]
    if len(world) < min_samples:
        return None
    return estimate_marker_frame(world)


def marker_gt_world(ref, cfg):
    """Thuoc (tu tam marker: x theo mui ten, y ben phai, z = -chieu cao) -> W (EKF) qua marker LAN DAU, khong qua EKF
    luc gap, khong qua vi tri dat UAV / T_BC. None khi thieu thuoc."""
    gt = cfg['marker']['ground_truth']
    if ref is None or gt['x_m'] is None or gt['y_m'] is None:
        return None
    a = math.radians(ref['yaw_deg'])
    c, s = math.cos(a), math.sin(a)
    n = ref['p_W_m'][0] + c * gt['x_m'] - s * gt['y_m']
    e = ref['p_W_m'][1] + s * gt['x_m'] + c * gt['y_m']
    d = None if gt['z_m'] is None else ref['p_W_m'][2] + gt['z_m']  # marker nam tren san: z vat = z marker + z thuoc
    return (n, e, d)


def evaluate_marker_ref(obj, ref, cfg):
    """Vat (EKF luc gap) so voi thuoc dua ve W qua ArUco LAN DAU. Sai so theo he marker (x mui ten, y phai, z xuong)."""
    rep = {'ok': False, 'reason': None, 'ref': ref, 'gt_W_m': None, 'object_marker_m': None, 'error_m': None,
           'error_2d_m': None, 'error_3d_m': None}
    if ref is None:
        rep['reason'] = 'chua thay ArUco diem cat canh du mau'
        return rep
    rep['gt_W_m'] = marker_gt_world(ref, cfg)
    if obj.get('p_W_mm') is None:
        rep['reason'] = 'khong co toa do vat'
        return rep
    o = ned_to_marker([v / 1000.0 for v in obj['p_W_mm']], ref)
    rep['object_marker_m'] = o
    gt = cfg['marker']['ground_truth']
    if rep['gt_W_m'] is None:
        rep['reason'] = 'chua nhap marker.ground_truth'
        return rep
    ex, ey = o[0] - gt['x_m'], o[1] - gt['y_m']
    ez = None if gt['z_m'] is None else o[2] - gt['z_m']
    rep.update(ok=True, error_m=(ex, ey, ez), error_2d_m=math.hypot(ex, ey),
               error_3d_m=None if ez is None else math.sqrt(ex * ex + ey * ey + ez * ez))
    return rep


def format_marker_ref_report(rep, cfg):
    lines = [' --- SAI SO THEO ArUco LAN DAU (he local: goc tam marker, x mui ten, y phai, z xuong) ---']
    ref = rep['ref']
    if ref is None:
        lines.append(f' Khong tinh: {rep["reason"]}')
        return '\n'.join(lines)
    lines.append(f' Marker lan dau  : N {ref["p_W_m"][0]:.3f}, E {ref["p_W_m"][1]:.3f}, D {ref["p_W_m"][2]:.3f} m (EKF), '
                 f'mui ten {ref["yaw_deg"]:.1f} do, {ref["n"]} mau (trai {ref["spread_m"] * 100:.1f} cm)')
    if rep['object_marker_m'] is not None:
        x, y, z = rep['object_marker_m']
        lines.append(f' Vat (he marker) : x = {x:.3f} m, y = {y:.3f} m, z = {z:.3f} m')
    if not rep['ok']:
        lines.append(f' Khong tinh sai so: {rep["reason"]}')
        return '\n'.join(lines)
    gt = cfg['marker']['ground_truth']
    ex, ey, ez = rep['error_m']
    lines += [f' Thuoc           : x = {gt["x_m"]:+.3f} m, y = {gt["y_m"]:+.3f} m, z = {_fmt(gt["z_m"])} m',
              f' Sai so          : dx = {ex * 100:+.1f} cm, dy = {ey * 100:+.1f} cm, dz = '
              + ('n/a' if ez is None else f'{ez * 100:+.1f} cm')]
    if rep['error_3d_m'] is not None:
        lines.append(f' SAI SO EUCLID 3D (ArUco lan dau): {rep["error_3d_m"] * 100:.1f} cm')
    else:
        lines.append(f' Sai so 2D (ArUco lan dau): {rep["error_2d_m"] * 100:.1f} cm')
    return '\n'.join(lines)


def hold_step(ref, current, applied, hold):
    """Giu vi tri theo marker. ref / current: marker trong W (lan dau / vai giay gan nhat). applied: (dn, de) da cong vao
    setpoint hover. EKF troi d = current - ref (marker co dinh "dich" trong EKF) -> setpoint = goc + d de UAV o yen
    tren san. Tra ve (new_applied | None, ly do). None: khong can / khong duoc chinh."""
    dn = current['p_W_m'][0] - ref['p_W_m'][0]
    de = current['p_W_m'][1] - ref['p_W_m'][1]
    drift = math.hypot(dn, de)
    if drift > hold['max_total_m']:
        return None, f'EKF troi {drift * 100:.0f} cm > max_total_m {hold["max_total_m"] * 100:.0f} cm: KHONG chinh (kiem tra)'
    rn, re_ = dn - applied[0], de - applied[1]
    rest = math.hypot(rn, re_)
    if rest < hold['deadband_m']:
        return None, f'lech {rest * 100:.1f} cm < deadband'
    k = min(1.0, hold['max_step_m'] / rest)
    return (applied[0] + k * rn, applied[1] + k * re_), f'EKF troi {drift * 100:.1f} cm, chinh {k * rest * 100:.1f} cm'


def launch_check(samples, cfg):
    """Where the UAV stands on the marker before takeoff: camera offset (m) and nose-vs-arrow yaw."""
    m = cfg['marker']
    rep = {'n': len(samples), 'ok': False, 'offset_m': None, 'offset_xy_m': None, 'height_m': None, 'yaw_rel_deg': None}
    cams = [v['p_cam_S_mm'] for v in samples if isinstance(v.get('p_cam_S_mm'), tuple) and _finite(*v['p_cam_S_mm'])]
    if not cams:
        return rep
    x, y, z = (_median([c[i] for c in cams]) / 1000.0 for i in range(3))
    rep.update(offset_xy_m=(x, y), offset_m=math.hypot(x, y), height_m=-z)
    yaws = [v['yaw_rel_deg'] for v in samples if _finite(v.get('yaw_rel_deg'))]
    if yaws:
        rep['yaw_rel_deg'] = _circular_mean_deg(yaws)
    rep['ok'] = (rep['offset_m'] <= m['launch_max_offset_m'] and rep['yaw_rel_deg'] is not None
                 and abs(rep['yaw_rel_deg']) <= m['launch_max_yaw_deg'])
    return rep


def format_launch_check(rep, cfg):
    m = cfg['marker']
    if rep['offset_m'] is None:
        return (f'ArUco diem cat canh (id {m["marker_id"]}): KHONG thay trong {m["launch_window_s"]:g} s gan nhat '
                '(dat UAV de camera nhin thay marker, kiem tra den / vat che).')
    x, y = rep['offset_xy_m']
    yaw = 'n/a (chua co tu the FC)' if rep['yaw_rel_deg'] is None else f'{rep["yaw_rel_deg"]:+.1f} do'
    verdict = 'OK' if rep['ok'] else (f'VUOT NGUONG (cho phep {m["launch_max_offset_m"] * 100:.0f} cm, '
                                     f'{m["launch_max_yaw_deg"]:g} do)')
    return (f'ArUco diem cat canh: camera lech tam marker x {x * 100:+.1f} cm, y {y * 100:+.1f} cm '
            f'(= {rep["offset_m"] * 100:.1f} cm), cao {rep["height_m"]:.3f} m; mui UAV so voi mui ten {yaw} '
            f'({rep["n"]} mau) -> {verdict}')


def _rot2(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def placement_ground_truth(check, t_BC_m, yaw0_deg, cfg):
    """Tape ground truth (marker frame) -> scenario frame, through WHERE THE UAV STOOD on the marker (not EKF).

    The scenario frame at takeoff is the nose frame (launch_heading) with its origin on the floor under the FC.
    Camera in that frame = t_BC (UAV level); marker centre = camera - R(-yaw_rel) * camera-in-marker;
    object = marker centre + R(-yaw_rel) * tape (x, y); z = tape z (floor = 0 in both).
    Returns {'marker_m', 'gt_m', 'yaw_rel_deg'} or None without a launch check / T_BC / tape values.
    """
    gt = cfg['marker']['ground_truth']
    if (check is None or check.get('offset_xy_m') is None or t_BC_m is None
            or gt['x_m'] is None or gt['y_m'] is None):
        return None
    yaw_rel = check['yaw_rel_deg'] if check.get('yaw_rel_deg') is not None else 0.0
    cx, cy = _rot2(check['offset_xy_m'], -yaw_rel)
    mx, my = t_BC_m[0] - cx, t_BC_m[1] - cy
    gx, gy = _rot2((gt['x_m'], gt['y_m']), -yaw_rel)
    marker = (mx, my, 0.0)
    obj = (mx + gx, my + gy, gt['z_m'])
    if cfg['frame']['axes'] == 'ned':  # nose frame -> NED axes (origin unchanged)
        marker = scenario_vec_to_ned(marker, yaw0_deg, 'launch_heading')
        n, e, _ = scenario_vec_to_ned((obj[0], obj[1], 0.0), yaw0_deg, 'launch_heading')
        obj = (n, e, gt['z_m'])
    return {'marker_m': marker, 'gt_m': obj, 'yaw_rel_deg': yaw_rel}


def evaluate_marker(obj, launch, yaw0_deg, check, t_BC_m, cfg):
    """Object (scenario frame, EKF) vs tape measurement converted through the UAV placement on the marker."""
    rep = {'ok': False, 'position_m': None, 'placement': None, 'error_m': None, 'error_2d_m': None,
           'error_3d_m': None, 'height_above_floor_m': None, 'reason': None}
    rep['placement'] = placement_ground_truth(check, t_BC_m, yaw0_deg, cfg)
    if obj.get('p_W_mm') is None or launch is None:
        rep['reason'] = 'khong co toa do vat'
        return rep
    p = to_scenario([v / 1000.0 for v in obj['p_W_mm']], launch, yaw0_deg, cfg)
    rep.update(ok=True, position_m=p, height_above_floor_m=-p[2])
    if rep['placement'] is None:
        rep['reason'] = ('chua nhap marker.ground_truth' if cfg['marker']['ground_truth']['x_m'] is None else
                         'khong co vi tri dat UAV tren marker (khong thay marker luc takeoff) / T_BC')
        return rep
    g = rep['placement']['gt_m']
    ex, ey = p[0] - g[0], p[1] - g[1]
    rep['error_2d_m'] = math.hypot(ex, ey)
    if g[2] is None:
        rep['error_m'] = (ex, ey, None)
    else:
        ez = p[2] - g[2]
        rep['error_m'] = (ex, ey, ez)
        rep['error_3d_m'] = math.sqrt(ex * ex + ey * ey + ez * ez)
    return rep


def format_marker_report(rep, check, ground, hover, cfg):
    gt = cfg['marker']['ground_truth']
    lines = [' --- DANH GIA THEO ArUco DIEM CAT CANH (chi danh gia; dieu khien van theo EKF) ---',
             ' Thuoc do tu tam marker -> doi sang he kich ban qua VI TRI DAT UAV (anh camera + T_BC), khong qua EKF']
    if check is not None:
        lines.append(' Luc dat UAV     : ' + format_launch_check(check, cfg))
    change = frame_change(ground, hover)
    if change is not None:
        lines.append(f' EKF TROI (marker dat -> hover): {change["distance_m"] * 100:.1f} cm '
                     f'(ngang {change["horizontal_m"] * 100:.1f} cm), yaw {change["yaw_deg"]:+.1f} do')
    elif ground is None or hover is None:
        lines.append(' EKF troi        : n/a (thieu mau marker ' + ('luc dat' if ground is None else 'khi hover') + ')')
    pl = rep['placement']
    if pl is not None:
        mx, my, _ = pl['marker_m']
        lines.append(f' Tam marker      : x = {mx:.3f} m, y = {my:.3f} m (he kich ban, tu vi tri dat; '
                     f'mui lech mui ten {pl["yaw_rel_deg"]:+.1f} do)')
    if rep['position_m'] is not None:
        x, y, z = rep['position_m']
        lines.append(f' Vat (EKF)       : x = {x:.3f} m, y = {y:.3f} m, z = {z:.3f} m (cao {-z:.3f} m so voi san)')
    if rep['error_m'] is None:
        lines.append(f' Khong tinh sai so: {rep["reason"]}')
        return '\n'.join(lines)
    g = pl['gt_m']
    ex, ey, ez = rep['error_m']
    lines += [f' Thuoc -> kich ban: x = {g[0]:.3f} m, y = {g[1]:.3f} m, z = {_fmt(g[2])} m '
              f'(thuoc: {gt["x_m"]:+.3f}, {gt["y_m"]:+.3f}, {_fmt(gt["z_m"])} tu tam marker)',
              f' Sai so          : dx = {ex * 100:+.1f} cm, dy = {ey * 100:+.1f} cm, dz = '
              + ('n/a' if ez is None else f'{ez * 100:+.1f} cm')]
    if rep['error_3d_m'] is not None:
        lines.append(f' SAI SO EUCLID 3D (ArUco): {rep["error_3d_m"] * 100:.1f} cm')
    else:
        lines.append(f' Sai so 2D (ArUco): {rep["error_2d_m"] * 100:.1f} cm  (dat marker.ground_truth.z_m de co 3D)')
    return '\n'.join(lines)


# ---- result ----------------------------------------------------------------------------------------
def _finite3(v):
    return v is not None and len(v) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)


def object_from(result_json, object_line):
    """Final object in W from the OBJECT log line (4 decimals) or result.json (0.01 rounding)."""
    out = {'p_W_mm': None, 'normal_W': None, 'yaw_W_deg': None, 'mode': None, 'd_mm': None,
           'status': None, 'source': None, 'plane_dist_mm': None, 'n_mask_pts': None, 'window_s': None,
           't_plane_s': None}
    fusion = (result_json or {}).get('fusion') or {}
    world = fusion.get('world') or {}
    lock = (result_json or {}).get('lock') or {}
    if object_line and _finite3(object_line.get('p_W_mm')):
        out.update(p_W_mm=list(object_line['p_W_mm']), normal_W=list(object_line.get('normal_W') or []) or None,
                   yaw_W_deg=object_line.get('yaw_W_deg'), mode=object_line.get('mode'),
                   d_mm=object_line.get('d_mm'), source='log OBJECT')
    elif _finite3(world.get('p_obj_W_mm')):
        out.update(p_W_mm=world['p_obj_W_mm'], normal_W=world.get('normal_W'), yaw_W_deg=world.get('yaw_W_deg'),
                   mode=fusion.get('mode'), d_mm=fusion.get('d_mm'), source='result.json fusion.world')
    elif _finite3(lock.get('p_W_mm')):
        out.update(p_W_mm=lock['p_W_mm'], mode=fusion.get('mode'), d_mm=fusion.get('d_mm'),
                   source='result.json lock.p_W_mm')
    out['status'] = fusion.get('status')
    # khoang cach camera -> mat top do bang LiDAR (diem trong mask), cua so tich luy, thoi gian ra vector huong
    line = object_line if out['source'] == 'log OBJECT' else {}
    for key, json_key in (('plane_dist_mm', 'plane_dist_mm'), ('n_mask_pts', 'n_window_points'),
                          ('window_s', 'window_used_s'), ('t_plane_s', None)):
        v = line.get(key) if line else None
        if v is None and json_key is not None:
            v = world.get(json_key)
        out[key] = v if isinstance(v, (int, float)) and math.isfinite(v) else None
    if out['normal_W'] is not None and not _finite3(out['normal_W']):
        out['normal_W'] = None
    if isinstance(out['yaw_W_deg'], (int, float)) and not math.isfinite(out['yaw_W_deg']):
        out['yaw_W_deg'] = None
    return out


def evaluate(obj, launch, yaw0_deg, cfg):
    """Scenario-frame position / yaw / normal and errors vs ground truth."""
    axes = cfg['frame']['axes']
    rep = {'ok': obj.get('p_W_mm') is not None, 'mode': obj.get('mode'), 'status': obj.get('status'),
           'source': obj.get('source'), 'd_m': None, 'position_m': None, 'height_above_floor_m': None,
           'yaw_deg': None, 'normal': None, 'tilt_deg': None, 'error_m': None, 'error_3d_m': None,
           'error_2d_m': None, 'lidar_dist_m': None, 'n_mask_pts': obj.get('n_mask_pts'),
           'window_s': obj.get('window_s'), 't_plane_s': obj.get('t_plane_s')}
    if obj.get('d_mm') is not None and math.isfinite(obj['d_mm']):
        rep['d_m'] = obj['d_mm'] / 1000.0
    if not rep['ok']:
        return rep
    # p_W = tu the camera (EKF) + khoang cach LiDAR toi mat top (chi diem trong mask, s4)
    p = to_scenario([v / 1000.0 for v in obj['p_W_mm']], launch, yaw0_deg, cfg)
    if obj.get('plane_dist_mm') is not None:
        rep['lidar_dist_m'] = obj['plane_dist_mm'] / 1000.0
    rep['position_m'] = p
    rep['height_above_floor_m'] = -p[2]
    rep['yaw_deg'] = yaw_to_scenario(obj.get('yaw_W_deg'), yaw0_deg, axes)
    if obj.get('normal_W') is not None:
        rep['normal'] = ned_vec_to_scenario(obj['normal_W'], yaw0_deg, axes)
        rep['tilt_deg'] = tilt_deg(rep['normal'])
    gt = cfg['ground_truth']
    ex, ey = p[0] - gt['x_m'], p[1] - gt['y_m']
    rep['error_2d_m'] = math.hypot(ex, ey)
    if gt['z_m'] is not None:
        ez = p[2] - gt['z_m']
        rep['error_m'] = (ex, ey, ez)
        rep['error_3d_m'] = math.sqrt(ex * ex + ey * ey + ez * ez)
    else:
        rep['error_m'] = (ex, ey, None)
    return rep


ARUCO_DISTANCES = (  # (ten, khoa thuat toan, khoa ArUco, mo ta)
    ('d', 'd_s4_mm', 'd_aruco_mm', 'doc truc quang tai tam vat'),
    ('dist', 'dist_s4_mm', 'dist_aruco_mm', 'Euclid camera -> tam vat'),
    ('plane', 'plane_s4_mm', 'plane_aruco_mm', 'vuong goc camera -> mat top'),
)


def evaluate_aruco_distance(line):
    """Distance error of the final s4 result (LiDAR points in the mask) against the ArUco on the object, from the
    ARUCO_S4 line (camera frame, mm). Evaluation only. None when perception printed no ARUCO_S4 line."""
    if not line:
        return None
    rep = {'found': line.get('found') == 1.0, 'rows': [], 'reproj_px': line.get('reproj_px'),
           'tilt_deg': line.get('tilt_deg'), 'marker_dist_mm': line.get('marker_dist_mm'),
           'center_err_mm': line.get('p_err_mm')}
    for name, k_s4, k_ar, why in ARUCO_DISTANCES:
        s4, ar = line.get(k_s4), line.get(k_ar)
        ok = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (s4, ar))
        rep['rows'].append({'name': name, 'what': why, 's4_mm': s4 if isinstance(s4, (int, float)) else None,
                            'aruco_mm': ar if ok else None, 'err_mm': (s4 - ar) if ok else None,
                            'err_pct': 100.0 * (s4 - ar) / ar if ok and ar else None})
    return rep


def format_aruco_distance(rep):
    lines = [' --- SAI SO KHOANG CACH TUYET DOI (thuat toan, buoc cuoi) vs ArUco TREN VAT ---']
    if rep is None:
        return '\n'.join(lines + [' Khong co dong ARUCO_S4 (perception khong toi buoc cuoi / aruco.check_aruco tat)'])
    if not rep['found']:
        lines.append(' KHONG thay ArUco tren vat o anh cuoi: chi co khoang cach thuat toan')
    for r in rep['rows']:
        s4 = 'n/a' if r['s4_mm'] is None or not math.isfinite(r['s4_mm']) else f"{r['s4_mm']:.1f}"
        if r['err_mm'] is None:
            lines.append(f" {r['name']:<6}: thuat toan {s4} mm ({r['what']})")
        else:
            lines.append(f" {r['name']:<6}: thuat toan {s4} mm | ArUco {r['aruco_mm']:.1f} mm | sai so "
                         f"{r['err_mm']:+.1f} mm ({r['err_pct']:+.2f} %)  [{r['what']}]")
    if rep['found']:
        lines.append(f" Lech tam vat (3D) {_fmt(rep['center_err_mm'], '.1f')} mm | ArUco: chieu lai "
                     f"{_fmt(rep['reproj_px'], '.2f')} px, nghieng {_fmt(rep['tilt_deg'], '.1f')} do, cach camera "
                     f"{_fmt(rep['marker_dist_mm'], '.0f')} mm")
    return '\n'.join(lines)


def _fmt(v, spec='.3f'):
    return 'n/a' if v is None else format(v, spec)


def format_report(rep, cfg, times):
    gt = cfg['ground_truth']
    axes = 'x = mui UAV luc cat canh, y = phai' if cfg['frame']['axes'] == 'launch_heading' else 'x = Bac, y = Dong'
    lines = ['', '=' * 72, ' KET QUA PLAN A - MAT TOP CUA VAT', '=' * 72,
             f' He toa do: goc (0,0,0) = diem cat canh (EKF optical flow); {axes}; z huong xuong']
    if not rep['ok']:
        lines.append(' KHONG co toa do vat (xem log perception).')
    else:
        x, y, z = rep['position_m']
        lines += [
            f' Tam mat top     : x = {x:.3f} m, y = {y:.3f} m, z = {z:.3f} m  (cao {-z:.3f} m so voi san)',
            f' Khoang cach LiDAR: {_fmt(rep["lidar_dist_m"])} m camera -> mat top (vuong goc), '
            f'{rep["n_mask_pts"] if rep["n_mask_pts"] is not None else "n/a"} diem LiDAR trong mask',
            f' Khoang cach d   : {_fmt(rep["d_m"])} m (camera -> mat top, doc truc quang)',
            f' Vector huong    : cua so LiDAR {_fmt(rep["window_s"], ".1f")} s'
            f' | ra PLANE sau {_fmt(rep["t_plane_s"], ".1f")} s tich luy (toi da {cfg["perception"]["max_window_s"]:g} s)',
            f' Yaw canh dai    : {_fmt(rep["yaw_deg"], ".1f")} do (so voi truc x, [0, 180))',
            ' Phap tuyen n    : ' + ('n/a' if rep['normal'] is None else
                                     '({:.4f}, {:.4f}, {:.4f})'.format(*rep['normal'])
                                     + f'  nghieng {_fmt(rep["tilt_deg"], ".1f")} do so voi phuong dung'),
            f' Che do / nguon  : {rep["mode"]} / {rep["source"]} (status {rep["status"]})',
            f' Ground truth    : x = {gt["x_m"]:.3f} m, y = {gt["y_m"]:.3f} m, z = {_fmt(gt["z_m"])} m',
        ]
        ex, ey, ez = rep['error_m']
        lines.append(f' Sai so          : dx = {ex * 100:+.1f} cm, dy = {ey * 100:+.1f} cm, dz = '
                     + ('n/a' if ez is None else f'{ez * 100:+.1f} cm'))
        if rep['error_3d_m'] is not None:
            lines.append(f' SAI SO EUCLID 3D: {rep["error_3d_m"] * 100:.1f} cm')
        else:
            lines.append(f' Sai so 2D (x, y): {rep["error_2d_m"] * 100:.1f} cm  (dat ground_truth.z_m de co 3D)')
    lines.append(f' Thoi gian       : start_perception -> toa do vat dau tien {_fmt(times.get("to_target_s"), ".2f")} s'
                 f' | -> LiDAR quet qua vat {_fmt(times.get("to_aligned_s"), ".2f")} s'
                 f' | -> ket qua cuoi {_fmt(times.get("to_result_s"), ".2f")} s')
    lines.append('=' * 72)
    return '\n'.join(lines)
