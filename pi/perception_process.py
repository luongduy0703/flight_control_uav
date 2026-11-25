"""Run the perception program (uavp_real_time, C++) as a SEPARATE child process.

The console never loads perception code. It only:
  * starts `uavp_real_time` when the console opens, so camera/LiDAR/model errors
    show up on the ground, before ARM;
  * reads the child's log (stderr) to know its state;
  * sends start/stop as MAVLink COMMAND_LONG to the perception component
    (route: MAVSDK -> mavlink-routerd -> uavp_fc_bridge -> perception);
  * restarts it after each scan, because one perception run = one scan.

A perception crash only ends the child; the console keeps streaming setpoints.
Perception never sends flight commands.
"""
import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Lines printed by uavp_real_time (his_perception_uav-main/apps/live_app.cpp).
READY_MARK = 'Cho lenh bat dau tu may tinh'
SCAN_MARK = 'BAT DAU: lenh'
RESULT_MARK = 'Ket qua: '
ERROR_MARK = 'LOI: '
# Machine-readable lines for planA (lock phase, live_app.cpp). Values: NED EKF, mm.
TARGET_MARK = 'TARGET p_W_mm='      # object locked: first world position
ALIGN_MARK = 'ALIGN ok='            # does the LiDAR line cross the object (e_mm <= e_max_mm)
OBJECT_MARK = 'OBJECT p_W_mm='      # final top-surface centre, normal, yaw
MARKER_MARK = 'MARKER seen='        # launch ArUco under the UAV (evaluation only, never a setpoint)
ARUCO_MARK = 'ARUCO_S4 found='     # ArUco on the object at the final s4 image: distance error (evaluation only)
CANDIDATE_MARK = 'CANDIDATE rank='  # s3_select.single_shot: other detected objects (floor point p_W_mm), next targets
RUN_DIR_MARK = 'THU MUC KET QUA '   # result directory, printed at startup
_KV = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=(\S+)')


def _numbers(text):
    values = tuple(float(v) for v in text.split(','))
    return values[0] if len(values) == 1 else values


def parse_marked(line):
    """('target' | 'align' | 'object' | 'marker' | 'aruco' | 'candidate', {key: float or tuple}) for a planA line."""
    for kind, mark in (('target', TARGET_MARK), ('align', ALIGN_MARK), ('object', OBJECT_MARK),
                       ('marker', MARKER_MARK), ('aruco', ARUCO_MARK), ('candidate', CANDIDATE_MARK)):
        index = line.find(mark)
        if index < 0:
            continue
        values = {}
        for key, raw in _KV.findall(line[index:]):
            try:
                values[key] = _numbers(raw)
            except ValueError:
                values[key] = raw
        return kind, values
    return None


@contextmanager
def all_cpus():
    """Let a child start with every CPU allowed.

    The console thread is pinned to the control cores (console.py --control-cores) and a
    child inherits that mask. Perception pins itself (cpu.cores in its config); with
    cpu.cores: [] it would otherwise stay squeezed onto the control cores.
    """
    if not hasattr(os, 'sched_getaffinity'):
        yield
        return
    saved = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
    except OSError:
        pass
    try:
        yield
    finally:
        try:
            os.sched_setaffinity(0, saved)
        except OSError:
            pass

STATES = ('STOPPED', 'STARTING', 'WAITING', 'SCANNING', 'FAILED')
MAX_FAILED_STARTS = 3


def read_perception_ids(config_path):
    """Read companion_sysid / companion_compid / command_id from the perception
    YAML so both programs use ONE source of truth (sources.mavlink in
    configs/run/real_time.yaml). Each key appears once in that file."""
    text = Path(config_path).read_text()
    ids = {}
    for key in ('companion_sysid', 'companion_compid', 'command_id'):
        match = re.search(rf'^\s*{key}:\s*([0-9]+)\b', text, re.MULTILINE)
        if match is None:
            raise ValueError(f'{config_path}: missing sources.mavlink.{key}')
        ids[key] = int(match[1])
    if not (1 <= ids['companion_sysid'] <= 255 and 1 <= ids['companion_compid'] <= 255):
        raise ValueError(f'{config_path}: companion sysid/compid must be 1..255')
    return ids


class PerceptionProcess:
    """Lifecycle of one uavp_real_time child: STOPPED -> STARTING -> WAITING ->
    SCANNING -> (exit) -> restart. Repeated startup errors -> FAILED."""

    def __init__(self, perception_dir, config='configs/run/real_time.yaml', log_dir='session-logs',
                 overrides=(), restart_delay_s=2.0, popen=subprocess.Popen, clock=time.monotonic):
        self.dir = Path(perception_dir).expanduser().resolve()
        self.binary = self.dir / 'build' / 'uavp_real_time'
        self.config = config
        self.config_path = self.dir / config
        if not self.binary.is_file():
            raise ValueError(f'Perception not built: {self.binary} (cmake --build build)')
        self.ids = read_perception_ids(self.config_path)
        # Console-launched perception: no window, and SPACE on the laptop viewer
        # cannot start a scan. Only the console (HOLD check) can.
        self.overrides = ['display.show=false', 'start.accept_viewer_key=false', *overrides]
        self.log_dir = Path(log_dir)
        self.restart_delay_s = restart_delay_s
        self.popen, self.clock = popen, clock
        self.lock = threading.Lock()
        self.child = None
        self.reader = None
        self.generation = 0
        self.auto_stopped = None  # generation already sent an automatic STOP
        self.scan_mode = None  # 'flight' or 'bench': the console state the current scan was started in
        self.state = 'STOPPED'
        self.last_error = None
        self.last_result = None
        self.last_target = None   # parse_marked values of the latest TARGET / ALIGN / OBJECT line
        self.last_align = None
        self.last_object = None
        self.last_marker = None   # latest MARKER line (launch ArUco, evaluation only)
        self.last_aruco = None
        self.last_candidate = None  # latest CANDIDATE line (single_shot: another object to fly to next)
        self.run_dir = None
        self.marked = deque(maxlen=200)  # (kind, values, monotonic time) for planA, oldest first
        self.exit_code = None
        self.failed_starts = 0
        self.restart_at = None
        self.log_path = None
        self.events = deque(maxlen=50)  # Messages for the operator, printed by the console.
        self.closed = False

    # ---- process lifecycle -------------------------------------------------------------------------
    def start(self):
        """Spawn the child if none is running."""
        with self.lock:
            if self.closed or (self.child is not None and self.child.poll() is None):
                return
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            self.log_path = self.log_dir / f'perception-{stamp}.log'
            log = open(self.log_path, 'a')
            command = [str(self.binary), '--cfg', self.config, '--set', *self.overrides]
            try:
                # stdin=DEVNULL: perception puts a TTY stdin into raw mode
                # (terminal_keys.cpp) and would steal the console's keystrokes.
                # New session: Ctrl+C in the console does not hit the child.
                with all_cpus():
                    self.child = self.popen(command, cwd=str(self.dir), stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.PIPE, text=True,
                                            bufsize=1, start_new_session=True)
            except OSError as error:
                log.close()
                self._fail_start(f'cannot run {self.binary}: {error}')
                return
            self.state = 'STARTING'
            self.exit_code = None
            self.restart_at = None
            self.last_result = None
            self.last_target = self.last_align = self.last_object = self.last_marker = self.run_dir = None
            self.last_aruco = self.last_candidate = None
            self.marked.clear()
            self.generation += 1
            self.reader = threading.Thread(target=self._read_log,
                                           args=(self.child, log, self.generation), daemon=True)
            self.reader.start()
        logging.getLogger(__name__).info('Perception started: %s (log %s)', command, self.log_path)

    def _read_log(self, child, log, generation):
        """Copy the child's log to a file and follow its state markers."""
        try:
            for line in child.stderr:
                log.write(line)
                with self.lock:
                    if generation != self.generation:
                        continue  # Output of an older run that was terminated.
                    if READY_MARK in line and self.state == 'STARTING':
                        self.state = 'WAITING'
                        self.failed_starts = 0
                        self.events.append('perception READY: waiting for "perception" command')
                    elif SCAN_MARK in line:
                        self.state = 'SCANNING'
                        self.events.append('perception SCANNING (confirmed by perception log)')
                    elif RESULT_MARK in line:
                        self.last_result = line.split(RESULT_MARK, 1)[1].strip()
                        self.events.append(f'perception DONE: result in {self.last_result} (on Pi)')
                    elif ERROR_MARK in line:
                        self.last_error = line.split(ERROR_MARK, 1)[1].strip()
                    elif RUN_DIR_MARK in line:
                        self.run_dir = line.split(RUN_DIR_MARK, 1)[1].strip()
                    else:
                        marked = parse_marked(line)
                        if marked is not None:
                            kind, values = marked
                            setattr(self, f'last_{kind}', values)
                            self.marked.append((kind, values, self.clock()))
        finally:
            log.close()

    def _fail_start(self, reason):
        self.failed_starts += 1
        self.last_error = reason
        if self.failed_starts >= MAX_FAILED_STARTS:
            self.state = 'FAILED'
            self.events.append(f'perception FAILED {self.failed_starts}x: {reason}. '
                               f'Fix it, then "perception restart". Flight control is unaffected.')
        else:
            self.state = 'STOPPED'
            self.restart_at = self.clock() + self.restart_delay_s
            self.events.append(f'perception error: {reason}; retry in {self.restart_delay_s:g} s')

    def poll(self):
        """Call often from the console loop: detect exit, schedule restart."""
        restart = False
        with self.lock:
            child, reader = self.child, self.reader
        code = None if child is None else child.poll()
        if code is not None and reader is not None:
            reader.join(1.0)  # Let the last log lines (result / error) be parsed first.
        with self.lock:
            if child is not None and child is self.child and self.state != 'STOPPED':
                if code is not None:
                    self.exit_code = code
                    was = self.state
                    self.child = None
                    if code == 0 and was in ('SCANNING', 'WAITING'):
                        # Normal end of one scan (or stop): get ready for the next one.
                        self.state = 'STOPPED'
                        self.restart_at = self.clock() + self.restart_delay_s
                        if was == 'SCANNING' and self.last_result is None:
                            self.events.append('perception ended without result (stopped)')
                    else:
                        self._fail_start(self.last_error or f'exit code {code} during {was}')
            if (self.state == 'STOPPED' and self.restart_at is not None
                    and self.clock() >= self.restart_at and not self.closed):
                restart = True
        if restart:
            self.start()

    def restart(self):
        """Operator retry after FAILED (or to reload perception config)."""
        self._terminate()
        with self.lock:
            self.failed_starts = 0
            self.last_error = None
            self.state = 'STOPPED'
            self.restart_at = None
        self.start()

    def _terminate(self, timeout=3.0):
        with self.lock:
            child, self.child = self.child, None
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout)
        except (ProcessLookupError, PermissionError):
            pass
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout)

    def close(self):
        with self.lock:
            self.closed = True
        self._terminate()
        with self.lock:
            self.state = 'STOPPED'

    def take_events(self):
        with self.lock:
            events = list(self.events)
            self.events.clear()
        return events

    def take_marked(self):
        """TARGET / ALIGN / OBJECT lines seen since the last call."""
        with self.lock:
            items = list(self.marked)
            self.marked.clear()
        return items

    def status(self):
        with self.lock:
            return {'state': self.state, 'pid': None if self.child is None else self.child.pid,
                    'last_error': self.last_error, 'last_result': self.last_result,
                    'exit_code': self.exit_code, 'failed_starts': self.failed_starts,
                    'log': None if self.log_path is None else str(self.log_path),
                    'target': f"{self.ids['companion_sysid']}/{self.ids['companion_compid']}"}


def send_perception_command(fc, ids, start, timeout_s=1.0, attempts=3, floor_down_m=None):
    """COMMAND_LONG(command_id, param1=1 start / 0 stop / 2 'scan') to the perception
    component through MAVSDK. Returns True when uavp_fc_bridge ACKs. The ACK
    only proves delivery to the Pi; the perception log confirms the scan.
    floor_down_m (start only): floor level in EKF NED z (m) -> param2, param3 = 1; perception then derives the
    camera-floor distance for detection-stage size rejection from EKF."""
    from mavsdk.plugins.mavlink_direct import MavlinkMessage
    import json
    sysid, compid, command = ids['companion_sysid'], ids['companion_compid'], ids['command_id']
    acked = []

    def on_ack(message, user_data=None):
        try:
            fields = json.loads(message.fields_json)
            if (int(fields['command']) == command and message.system_id == sysid
                    and message.component_id == compid):
                acked.append(int(fields['result']))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    fields = {'target_system': sysid, 'target_component': compid, 'command': command,
              'confirmation': 0, 'param1': 2.0 if start == 'scan' else 1.0 if start else 0.0, 'param2': 0.0 if floor_down_m is None else float(floor_down_m),
              'param3': 0.0 if floor_down_m is None else 1.0,
              'param4': 0.0, 'param5': 0.0, 'param6': 0.0, 'param7': 0.0}
    handle = fc.mavlink_direct.subscribe_message('COMMAND_ACK', on_ack)
    try:
        for attempt in range(attempts):
            fields['confirmation'] = attempt
            fc.mavlink_direct.send_message(MavlinkMessage(
                message_name='COMMAND_LONG', system_id=0, component_id=0,
                target_system_id=sysid, target_component_id=compid,
                fields_json=json.dumps(fields)))
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if acked:
                    return acked[0] == 0  # MAV_RESULT_ACCEPTED
                time.sleep(0.02)
    finally:
        fc.mavlink_direct.unsubscribe_message(handle)
    return False
