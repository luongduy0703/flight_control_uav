"""Native MAVSDK transport and telemetry; local-NED flight sequencing is in OffboardControl."""
import logging
import math
import threading
import time
from deployment import validate_connection


IDENTITY_TIMEOUT_S = 3.0  # > 2 PX4 HEARTBEAT periods (1 Hz)
PARAM_ATTEMPTS = 5        # x MAVSDK's own ~2 s retries: covers a QGC full parameter download
PARAM_RETRY_DELAY_S = 1.0


def read_param(getter, name):
    """Read one PX4 parameter, retrying only MAVSDK TIMEOUT.

    On the shared TELEM1 link, QGC downloading ~1000 parameters queues PX4's
    replies; a single read can then time out although the link is healthy.
    Any other error, or TIMEOUT after the last attempt, is raised unchanged.
    """
    for attempt in range(1, PARAM_ATTEMPTS + 1):
        try:
            return getter(name)
        except Exception as error:
            if not str(error).startswith('TIMEOUT') or attempt == PARAM_ATTEMPTS:
                raise
            logging.getLogger(__name__).warning(
                'Parameter %s timed out (attempt %d/%d; QGC parameter download?), retrying',
                name, attempt, PARAM_ATTEMPTS)
            time.sleep(PARAM_RETRY_DELAY_S)

HEALTH_FIELDS = ('is_gyrometer_calibration_ok', 'is_accelerometer_calibration_ok',
                 'is_magnetometer_calibration_ok', 'is_local_position_ok',
                 'is_global_position_ok', 'is_home_position_ok', 'is_armable')


def json_safe(value):
    """Keep diagnostics JSON readable without altering raw control telemetry."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def has_nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(has_nonfinite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_nonfinite(v) for v in value)
    return False


class Pixhawk:
    def __init__(self, connection, profile, enabled=False, system_id=1):
        validate_connection(profile, connection)
        from mavsdk import Mavsdk, Configuration, ComponentType, Autopilot, Vehicle
        from mavsdk.plugins.action import Action
        from mavsdk.plugins.telemetry import Telemetry
        from mavsdk.plugins.param import Param
        self.enabled, self.profile = enabled, profile
        self.lock = threading.Lock()
        self.cache, self.handles = {}, []
        self.phase = 'IDLE'
        self.takeoff_complete = False
        self.operation_at = 0.0
        # A console running on the PC is the GCS; serial Pi control is a companion.
        component = (ComponentType.GROUND_STATION
                     if profile == 'hitl' and connection.startswith('udpin://')
                     else ComponentType.COMPANION_COMPUTER)
        self.sdk = Mavsdk(Configuration.create_with_component_type(component))
        try:
            self.sdk.add_any_connection(connection)
            self.drone = self.sdk.first_autopilot(10.0)
            if self.drone is None:
                raise RuntimeError('No autopilot heartbeat')

            def expected_vehicle():
                return (self.drone.get_system_id() == system_id and
                        self.drone.autopilot_type() == Autopilot.PX4 and
                        self.drone.vehicle_type() == Vehicle.MULTICOPTER)
            # first_autopilot() can return on the first autopilot message (e.g. ATTITUDE
            # at 100 Hz) before the 1 Hz HEARTBEAT that sets autopilot/vehicle type;
            # until then both read GENERIC. Wait for it; the check itself is unchanged.
            deadline = time.monotonic() + IDENTITY_TIMEOUT_S
            while not expected_vehicle() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not expected_vehicle():
                raise RuntimeError(
                    f'Expected configured PX4 multicopter, got system {self.drone.get_system_id()} '
                    f'{self.drone.autopilot_type()} {self.drone.vehicle_type()}')
            self.action, self.telemetry = Action(self.drone), Telemetry(self.drone)
            self.param = Param(self.drone)
            self.hitl = read_param(self.param.get_param_int, 'SYS_HITL')
            if self.hitl != (1 if profile == 'hitl' else 0):
                raise RuntimeError(f'SYS_HITL={self.hitl} does not match {profile}')
            for name in ('armed', 'landed_state', 'flight_mode', 'health_all_ok', 'health', 'position'):
                def callback(value, user_data=None, name=name):
                    if name == 'position':
                        value = {key: getattr(value, key) for key in
                                 ('latitude_deg', 'longitude_deg', 'absolute_altitude_m', 'relative_altitude_m')}
                    elif name == 'health':
                        value = {key: getattr(value, key) for key in HEALTH_FIELDS}
                    value = getattr(value, 'name', value)
                    with self.lock:
                        self.cache[name] = (value, time.monotonic())
                self.handles.append(getattr(self.telemetry, 'subscribe_' + name)(callback))
            # PX4 applies stream rates to the whole link. Through mavlink-routerd the
            # TELEM1 link is shared with perception, which needs PX4's default rates
            # (ATTITUDE 100 Hz, LOCAL_POSITION_NED 30 Hz): never lower them there.
            self.shared_link = profile == 'hardware' and connection.startswith('udpin://')
            if not self.shared_link:
                self.telemetry.set_rate_health(2.0)
                self.telemetry.set_rate_landed_state(2.0)
        except BaseException:
            try:
                self.sdk.destroy()
            except Exception:
                logging.getLogger(__name__).exception('SDK cleanup after connection failure failed')
            raise

    def status(self):
        self.refresh()
        now = time.monotonic()
        with self.lock:
            values = {k: {'value': v, 'age_s': now-t} for k, (v, t) in self.cache.items()}
            invalid = [k for k, (v, _) in self.cache.items() if has_nonfinite(v)]
        return json_safe({'connected': self.drone.is_connected(), 'SYS_HITL': self.hitl,
                          'profile': self.profile, 'actuation_enabled': self.enabled,
                          'phase': self.phase, 'takeoff_complete': self.takeoff_complete,
                          'invalid_telemetry_fields': invalid, 'telemetry': values})

    def require(self):
        if not self.enabled:
            raise ValueError('Inspection only: use --enable-actuation to send commands')
        if not self.drone.is_connected():
            raise ValueError('PX4 disconnected')

    def state(self, key, max_age=3.0):
        with self.lock:
            value, at = self.cache.get(key, (None, 0))
        if not 0 <= time.monotonic()-at <= max_age:
            raise ValueError(f'Missing/stale {key} telemetry')
        return value

    def refresh(self):
        """Flight-state transitions are implemented by OffboardControl."""
