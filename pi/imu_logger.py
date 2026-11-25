"""Record Pixhawk IMU / vibration telemetry to CSV while flying (read-only for flight).

Asks PX4 for higher message rates with MAV_CMD_SET_MESSAGE_INTERVAL (telemetry only, never a flight command) and
restores PX4 defaults on stop(). Files (rx_ns = time.monotonic_ns() on the Pi = same CLOCK_MONOTONIC as perception
mono_ns, so rows line up with perception video_diag.csv):
    imu_highres.csv    HIGHRES_IMU: accel (m/s^2), gyro (rad/s). PX4 filters these (IMU_ACCEL_CUTOFF,
                       IMU_GYRO_CUTOFF) and the link limits the rate: frequencies above rate / 2 fold back (alias).
    imu_vibration.csv  VIBRATION: PX4 vibration metric per axis (m/s^2, from unfiltered accel) + clipping counters.
    imu_attitude.csv   ATTITUDE body rates (rate unchanged: perception relies on its default).
    imu_motors.csv     SERVO_OUTPUT_RAW: motor outputs (us); prop vibration frequency rises with motor speed.
    imu_state.csv      armed (HEARTBEAT) + landed_state (EXTENDED_SYS_STATE): ground off / armed / air phases.
    imu_meta.json      requested rates, PX4 IMU filter / logging parameters, sample counts.
For the true vibration spectrum (hundreds of Hz) use the Pixhawk SD log with raw IMU FIFO (SDLOG_PROFILE).
"""
import json
import logging
import threading
import time
from pathlib import Path

MSG_IDS = {'HIGHRES_IMU': 105, 'VIBRATION': 241, 'SERVO_OUTPUT_RAW': 36, 'ATTITUDE': 30,
           'HEARTBEAT': 0, 'EXTENDED_SYS_STATE': 245}
COLUMNS = {
    'HIGHRES_IMU': ('time_usec', 'xacc', 'yacc', 'zacc', 'xgyro', 'ygyro', 'zgyro'),
    'VIBRATION': ('time_usec', 'vibration_x', 'vibration_y', 'vibration_z', 'clipping_0', 'clipping_1', 'clipping_2'),
    'ATTITUDE': ('time_boot_ms', 'rollspeed', 'pitchspeed', 'yawspeed'),
    'SERVO_OUTPUT_RAW': ('time_usec', 'servo1_raw', 'servo2_raw', 'servo3_raw', 'servo4_raw', 'servo5_raw',
                         'servo6_raw', 'servo7_raw', 'servo8_raw'),
}
FILES = {'HIGHRES_IMU': 'imu_highres.csv', 'VIBRATION': 'imu_vibration.csv', 'ATTITUDE': 'imu_attitude.csv',
         'SERVO_OUTPUT_RAW': 'imu_motors.csv'}
STATE_FILE = 'imu_state.csv'
PARAMS = (('IMU_GYRO_CUTOFF', 'float'), ('IMU_ACCEL_CUTOFF', 'float'), ('IMU_GYRO_RATEMAX', 'int'),
          ('IMU_DGYRO_CUTOFF', 'float'), ('IMU_GYRO_NF0_FRQ', 'float'), ('IMU_GYRO_DNF_EN', 'int'),
          ('SDLOG_PROFILE', 'int'), ('SDLOG_MODE', 'int'))
MAV_MODE_FLAG_SAFETY_ARMED = 128


def set_message_interval(mavlink_direct, system_id, msgid, interval_us):
    """MAV_CMD_SET_MESSAGE_INTERVAL (511): interval_us > 0 sets the rate, 0 restores the PX4 default."""
    from mavsdk.plugins.mavlink_direct import MavlinkMessage
    mavlink_direct.send_message(MavlinkMessage(
        message_name='COMMAND_LONG', system_id=0, component_id=0, target_system_id=system_id, target_component_id=1,
        fields_json=json.dumps({'target_system': system_id, 'target_component': 1, 'command': 511, 'confirmation': 0,
                                'param1': float(msgid), 'param2': float(interval_us), 'param3': 0.0, 'param4': 0.0,
                                'param5': 0.0, 'param6': 0.0, 'param7': 0.0})))


class ImuLogger:
    """subscribe -> buffer rows in MAVSDK callbacks -> poll() writes them (main loop) -> stop() restores rates."""

    def __init__(self, mavlink_direct, out_dir, system_id=1, rates_hz=None, params=None,
                 set_interval=set_message_interval, clock=time.monotonic_ns, flush_period_s=1.0):
        self.direct, self.out_dir, self.system_id = mavlink_direct, Path(out_dir), system_id
        self.rates_hz = dict(rates_hz if rates_hz is not None else
                             {'HIGHRES_IMU': 200.0, 'VIBRATION': 10.0, 'SERVO_OUTPUT_RAW': 10.0})
        self.params = params  # callable name, kind -> value (OffboardControl param plugin) or None
        self.set_interval, self.clock, self.flush_period_s = set_interval, clock, flush_period_s
        self.lock = threading.Lock()
        self.rows = {name: [] for name in [*FILES, 'STATE']}
        self.counts = {name: 0 for name in self.rows}
        self.handles = []
        self.state = {'armed': -1, 'landed': -1}
        self.files = {}
        self.last_flush = 0.0
        self.started = False
        self.meta = {}

    # ---- lifecycle ---------------------------------------------------------------------------------
    def start(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for name, file in FILES.items():
            self.files[name] = open(self.out_dir / file, 'w')
            self.files[name].write('rx_ns,' + ','.join(COLUMNS[name]) + '\n')
        self.files['STATE'] = open(self.out_dir / STATE_FILE, 'w')
        self.files['STATE'].write('rx_ns,armed,landed\n')
        for name in [*FILES, 'HEARTBEAT', 'EXTENDED_SYS_STATE']:
            self.handles.append(self.direct.subscribe_message(name, self._callback))
        for name, hz in self.rates_hz.items():
            if hz and hz > 0:
                self.set_interval(self.direct, self.system_id, MSG_IDS[name], int(round(1e6 / hz)))
        self.meta = {'requested_hz': self.rates_hz, 'params': self._read_params(), 'start_ns': self.clock(),
                     'clock': 'time.monotonic_ns (= perception mono_ns)'}
        self.started = True
        logging.getLogger(__name__).info('IMU logger started: %s', self.meta)

    def _read_params(self):
        out = {}
        if self.params is None:
            return out
        for name, kind in PARAMS:
            try:
                out[name] = self.params(name, kind)
            except Exception as error:  # noqa: BLE001 - optional context only
                out[name] = f'n/a ({error})'
        return out

    def stop(self):
        if not self.started:
            return
        self.started = False
        for handle in self.handles:
            try:
                self.direct.unsubscribe_message(handle)
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).exception('IMU unsubscribe failed')
        self.handles.clear()
        for name, hz in self.rates_hz.items():
            if hz and hz > 0:
                try:
                    self.set_interval(self.direct, self.system_id, MSG_IDS[name], 0)  # PX4 default again
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).exception('restore rate %s failed', name)
        self.flush()
        with self.lock:
            counts = dict(self.counts)
        self.meta.update(stop_ns=self.clock(), samples=counts)
        for f in self.files.values():
            f.close()
        self.files.clear()
        (self.out_dir / 'imu_meta.json').write_text(json.dumps(self.meta, indent=2, default=str))

    # ---- data --------------------------------------------------------------------------------------
    def _callback(self, message, user_data=None):
        name = getattr(message, 'message_name', None)
        if getattr(message, 'system_id', None) != self.system_id or getattr(message, 'component_id', None) != 1:
            return
        try:
            fields = json.loads(message.fields_json)
        except (TypeError, ValueError):
            return
        rx = self.clock()
        if name in COLUMNS:
            row = f'{rx},' + ','.join(str(fields.get(k, '')) for k in COLUMNS[name]) + '\n'
            with self.lock:
                self.rows[name].append(row)
                self.counts[name] += 1
            return
        if name == 'HEARTBEAT':
            armed = 1 if int(fields.get('base_mode', 0)) & MAV_MODE_FLAG_SAFETY_ARMED else 0
            self._state(rx, armed=armed)
        elif name == 'EXTENDED_SYS_STATE':
            self._state(rx, landed=int(fields.get('landed_state', -1)))

    def _state(self, rx, **change):
        with self.lock:
            if all(self.state[k] == v for k, v in change.items()):
                return
            self.state.update(change)
            self.rows['STATE'].append(f'{rx},{self.state["armed"]},{self.state["landed"]}\n')
            self.counts['STATE'] += 1

    def poll(self):
        """Main loop: write buffered rows about once per flush_period_s."""
        if self.started and time.monotonic() - self.last_flush >= self.flush_period_s:
            self.flush()

    def flush(self):
        with self.lock:
            rows = {k: v for k, v in self.rows.items() if v}
            for k in rows:
                self.rows[k] = []
        for name, lines in rows.items():
            f = self.files.get(name)
            if f is not None:
                f.writelines(lines)
                f.flush()
        self.last_flush = time.monotonic()
