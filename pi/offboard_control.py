"""Direct local-NED position signals. PX4 owns estimation and flight control."""
import json
import logging
import math
import time
from dataclasses import asdict
from flight_policy import ArrivalMonitor, FlightPolicy
from validation import number
from pixhawk import Pixhawk, json_safe, read_param


class OffboardControl(Pixhawk):
    FLIGHT_PHASES = {'STARTING', 'TAKEOFF', 'FLY', 'SETTLING'}
    HOLD_PHASES = {'HOLD_REQUESTED', 'HOLD', 'FLY_READY'}
    POSITION_SOURCES = {'gps', 'flow'}
    # Optical flow gives local position only: no global position or home.
    FLOW_HEALTH = ('is_gyrometer_calibration_ok', 'is_accelerometer_calibration_ok',
                   'is_magnetometer_calibration_ok', 'is_local_position_ok', 'is_armable')
    position_source = 'gps'

    def __init__(self, connection, profile, enabled=False, system_id=1, height=2.0, policy=None,
                 position_source='gps', ekf_origin=None, max_height=100.0):
        if position_source not in self.POSITION_SOURCES:
            raise ValueError('position_source must be gps or flow')
        if ekf_origin is not None:
            if position_source != 'flow':
                raise ValueError('ekf_origin is only used with position_source=flow')
            ekf_origin = self.validate_origin(*ekf_origin)
        self.position_source = position_source
        self.ekf_origin = None
        self.max_height = number(max_height, 'max_height', 0.5, 100)
        self.height = number(height, 'height', 0.5, self.max_height)
        self.target = None
        self.launch_ned = None
        self.land_acknowledged = False
        self.arm_acknowledged = False
        self.arm_operation_at = None
        self.disarm_acknowledged = False
        self.disarm_operation_at = None
        self.motor_stop_kind = None
        self.land_request_context = None
        self.offboard = None
        self.offboard_seen = False
        self.external = False
        self.termination = False
        self.mavlink_direct = None
        self.direct_handle = None
        self.policy = policy or FlightPolicy()
        if self.policy.z_tolerance_m >= self.height:
            raise ValueError('z_tolerance_m must be less than takeoff height')
        self.arrival = ArrivalMonitor(self.policy)
        self.motion_kind = None
        self.target_started_at = None
        self.target_reached = False
        self.takeoff_complete = False
        self.hold_acknowledged = False
        self.hold_operation_at = None
        self.hold_reason = None
        self.last_error = None
        super().__init__(connection, profile, enabled, system_id)
        try:
            self._subscribe_offboard(system_id)
            if ekf_origin is not None and self.enabled:
                self.set_ekf_origin(*ekf_origin)
        except BaseException:
            try:
                self.close()
            except Exception:
                logging.getLogger(__name__).exception('SDK cleanup after subscription failure failed')
            raise

    def _subscribe_offboard(self, system_id):
        self.nav_min_ltr_alt = number(read_param(self.param.get_param_float, 'NAV_MIN_LTR_ALT'),
                                     'NAV_MIN_LTR_ALT', -1, 10000)
        from mavsdk.plugins.mavlink_direct import MavlinkDirect
        from mavsdk.plugins.offboard import Offboard, PositionNedYaw
        self.make_offboard = lambda: Offboard(self.drone)
        self.Position = PositionNedYaw
        self.mavlink_direct = MavlinkDirect(self.drone)

        def heartbeat(message, user_data=None):
            # MAVSDK maps PX4 TERMINATION to UNKNOWN flight mode. Read the raw
            # PX4 autopilot heartbeat so the operator gets an actionable state.
            if (getattr(message, 'message_name', None) != 'HEARTBEAT'
                    or getattr(message, 'system_id', None) != system_id
                    or getattr(message, 'component_id', None) != 1):
                return
            try:
                fields = json.loads(message.fields_json)
                custom_mode = int(fields['custom_mode'])
                system_status = int(fields['system_status'])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return
            main_mode = (custom_mode >> 16) & 0xff
            with self.lock:
                self.cache['px4_heartbeat'] = ({
                    'custom_mode': custom_mode,
                    'main_mode': main_mode,
                    'system_status': system_status,
                }, time.monotonic())
                # MAV_STATE_FLIGHT_TERMINATION=8 and PX4 main mode TERMINATION=10.
                if system_status == 8 or main_mode == 10:
                    self.termination = True

        self.direct_handle = self.mavlink_direct.subscribe_message('HEARTBEAT', heartbeat)

        def local(value, user_data=None):
            p = value.position
            v = value.velocity
            with self.lock:
                self.cache['local_ned'] = ({'north_m': p.north_m, 'east_m': p.east_m,
                                            'down_m': p.down_m}, time.monotonic())
                self.cache['local_velocity'] = ({
                    'north_m_s': v.north_m_s,
                    'east_m_s': v.east_m_s,
                    'down_m_s': v.down_m_s,
                }, time.monotonic())

        def attitude(value, user_data=None):
            with self.lock:
                self.cache['yaw_deg'] = (value.yaw_deg, time.monotonic())

        def mode(value, user_data=None):
            self.record_mode(getattr(value, 'name', value))

        self.handles.extend([
            self.telemetry.subscribe_position_velocity_ned(local),
            self.telemetry.subscribe_attitude_euler(attitude),
            self.telemetry.subscribe_flight_mode(mode),
        ])
        if self.shared_link:
            self.restore_default_stream_rates()
        else:
            self.telemetry.set_rate_position_velocity_ned(5.0)
            self.telemetry.set_rate_attitude_euler(5.0)

    # MAVLink ids of the streams this console and perception read on a shared link.
    SHARED_STREAMS = {'SYS_STATUS': 1, 'ATTITUDE': 30, 'LOCAL_POSITION_NED': 32,
                      'VFR_HUD': 74, 'DISTANCE_SENSOR': 132, 'EXTENDED_SYS_STATE': 245}

    def restore_default_stream_rates(self):
        """MAV_CMD_SET_MESSAGE_INTERVAL(msg, 0 = PX4 default) for each shared stream.

        Undoes any earlier lowering on this link (an older console sets 5 Hz and
        PX4 keeps it until reboot). Read-only for flight: changes telemetry rates only.
        """
        from mavsdk.plugins.mavlink_direct import MavlinkMessage
        system_id = self.drone.get_system_id()
        for name, msgid in self.SHARED_STREAMS.items():
            self.mavlink_direct.send_message(MavlinkMessage(
                message_name='COMMAND_LONG', system_id=0, component_id=0,
                target_system_id=system_id, target_component_id=1,
                fields_json=json.dumps({
                    'target_system': system_id, 'target_component': 1,
                    'command': 511, 'confirmation': 0, 'param1': float(msgid), 'param2': 0.0,
                    'param3': 0.0, 'param4': 0.0, 'param5': 0.0, 'param6': 0.0, 'param7': 0.0})))
        logging.getLogger(__name__).info('Shared link: requested PX4 default rates for %s',
                                         ', '.join(self.SHARED_STREAMS))

    @staticmethod
    def validate_origin(lat_deg, lon_deg, alt_m):
        return (number(lat_deg, 'origin latitude', -90, 90),
                number(lon_deg, 'origin longitude', -180, 180),
                number(alt_m, 'origin altitude AMSL', -500, 9000))

    def set_ekf_origin(self, lat_deg, lon_deg, alt_m, timeout=5.0):
        """Give a flow-only EKF a fixed global origin so PX4 HOLD/LAND/ARM work.

        Without GPS, PX4 has local position only and refuses HOLD ("No valid
        global position estimate"). The origin also sets magnetic declination,
        so it must be the real flying site. Lost on PX4 reboot; sent each session.
        """
        lat_deg, lon_deg, alt_m = self.validate_origin(lat_deg, lon_deg, alt_m)
        self.require()
        deadline = time.monotonic() + timeout
        while True:
            try:
                armed, landed = self.state('armed'), self.state('landed_state')
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('No ARM/landed telemetry; EKF origin was not sent') from None
                time.sleep(0.05)
        if armed is not False or landed != 'ON_GROUND':
            raise ValueError('EKF origin requires DISARM and ON_GROUND')
        from mavsdk.plugins.mavlink_direct import MavlinkMessage
        lat_e7, lon_e7 = int(round(lat_deg * 1e7)), int(round(lon_deg * 1e7))
        confirmed = []

        def echo(message, user_data=None):
            try:
                fields = json.loads(message.fields_json)
                if int(fields['latitude']) == lat_e7 and int(fields['longitude']) == lon_e7:
                    confirmed.append(True)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

        system_id = self.drone.get_system_id()
        message = MavlinkMessage(
            message_name='SET_GPS_GLOBAL_ORIGIN', system_id=0, component_id=0,
            target_system_id=system_id, target_component_id=1,
            fields_json=json.dumps({'target_system': system_id, 'latitude': lat_e7,
                                    'longitude': lon_e7, 'altitude': int(round(alt_m * 1000)),
                                    'time_usec': 0}))
        handle = self.mavlink_direct.subscribe_message('GPS_GLOBAL_ORIGIN', echo)
        try:
            deadline = time.monotonic() + timeout
            next_send = 0.0
            while not confirmed:
                now = time.monotonic()
                if now >= deadline:
                    raise RuntimeError('PX4 did not confirm EKF origin (GPS_GLOBAL_ORIGIN)')
                if now >= next_send:
                    self.mavlink_direct.send_message(message)
                    next_send = now + 1.0
                time.sleep(0.05)
        finally:
            self.mavlink_direct.unsubscribe_message(handle)
        self.ekf_origin = {'latitude_deg': lat_deg, 'longitude_deg': lon_deg, 'altitude_m': alt_m}
        logging.getLogger(__name__).info('PX4 confirmed EKF origin %s', self.ekf_origin)
        return self.ekf_origin

    def record_mode(self, name):
        with self.lock:
            if self.phase in self.FLIGHT_PHASES:
                if name == 'OFFBOARD':
                    self.offboard_seen = True
                elif self.offboard_seen:
                    self.external = True  # Latch even if modes change back before input.
            elif self.phase in {'HOLD', 'FLY_READY'} and name != 'HOLD':
                self.external = True

    def ready_for_fly(self):
        if (self.phase not in {'HOLD', 'FLY_READY'} or self.external
                or not self.takeoff_complete):
            return False
        try:
            return (self.drone.is_connected()
                    and self.state('armed') is True
                    and self.state('flight_mode') == 'HOLD'
                    and self.state('landed_state', self.policy.telemetry_max_age_s) == 'IN_AIR')
        except ValueError:
            return False

    def refresh(self):
        if self.termination:
            self.phase = 'TERMINATION'
            self.stop_stream()
            return
        if (self.phase in self.FLIGHT_PHASES | self.HOLD_PHASES
                and not self.drone.is_connected()):
            self.external = True
            self.last_error = 'Connection lost; explicit recovery required, no automatic takeover'
        if self.phase == 'ARM_REQUESTED':
            with self.lock:
                armed, at = self.cache.get('armed', (None, 0))
            now = time.monotonic()
            if (armed is True and self.arm_operation_at is not None
                    and self.arm_operation_at <= at <= now and now - at <= 3):
                self.phase = 'ARMED'
        if self.external and self.phase in self.FLIGHT_PHASES | self.HOLD_PHASES:
            self.phase = 'EXTERNAL'
            self.arrival.reset()
            self.stop_stream()  # Destroy stops sending; unlike stop(), no HOLD command.
        if self.phase in {'TAKEOFF', 'FLY', 'SETTLING'}:
            self.refresh_arrival()
        if self.phase == 'HOLD_REQUESTED':
            try:
                mode = self.state('flight_mode')
                with self.lock:
                    mode_at = self.cache['flight_mode'][1]
                if mode == 'HOLD' and mode_at >= self.hold_operation_at:
                    if self.state('armed') is True and self.state('landed_state') == 'IN_AIR':
                        self.phase = 'HOLD'
                        self.target_reached = self.hold_reason == 'TARGET_REACHED'
                        if self.target_reached and self.motion_kind == 'TAKEOFF':
                            self.takeoff_complete = True
                elif mode not in {'OFFBOARD', 'HOLD'} and mode_at >= self.hold_operation_at:
                    self.external = True
                    self.phase = 'EXTERNAL'
                    self.stop_stream()
            except ValueError:
                pass
            if self.phase == 'HOLD_REQUESTED' and time.monotonic() - self.hold_operation_at > 5:
                self.last_error = 'HOLD ACK/mode confirmation timeout; requesting LAND'
                self._recover_land()
        if self.phase == 'LAND_REQUESTED':
            with self.lock:
                mode, at = self.cache.get('flight_mode', (None, 0))
            now = time.monotonic()
            if mode == 'LAND' and self.operation_at <= at <= now and now - at <= 3:
                self.phase = 'LANDING'
        if self.phase in ({'ARMED', 'DISARM_REQUESTED',
                          'LAND_REQUESTED', 'LANDING', 'EXTERNAL', 'FAULT', 'EMERGENCY_STOP'}
                         | (self.FLIGHT_PHASES - {'STARTING'}) | self.HOLD_PHASES):
            try:
                if self.state('armed') is False and self.state('landed_state') == 'ON_GROUND':
                    with self.lock:
                        fresh_after_command = (self.cache['armed'][1] >= self.operation_at
                                               and self.cache['landed_state'][1] >= self.operation_at)
                    if fresh_after_command:
                        # LAND mode makes MAVSDK stop setpoints itself. Keep
                        # its native object alive across normal landing/reuse;
                        # destruction here must not disrupt the console.
                        if self.phase not in {'DISARM_REQUESTED',
                                              'LAND_REQUESTED', 'LANDING', 'EMERGENCY_STOP'}:
                            self.stop_stream()
                        self.phase = 'IDLE'
                        self.takeoff_complete = False
                        self.target_reached = False
                        self.arrival.reset()
            except ValueError:
                pass

    def preflight_ok(self):
        """GPS requires every MAVSDK health flag; flow requires the local subset."""
        if self.position_source == 'gps':
            return self.state('health_all_ok') is True
        health = self.state('health')
        return all(health.get(key) is True for key in self.FLOW_HEALTH)

    def preflight_failures(self):
        """Names of the MAVSDK health flags that are not True (for error messages)."""
        try:
            if self.position_source == 'gps':
                return [] if self.state('health_all_ok') is True else ['health_all_ok']
            health = self.state('health')
        except ValueError as error:
            return [f'no health telemetry ({error})']
        return [key for key in self.FLOW_HEALTH if health.get(key) is not True]

    def status(self):
        result = super().status()
        result['position_source'] = self.position_source
        result['ekf_origin'] = getattr(self, 'ekf_origin', None)
        try:
            result['preflight_ok'] = self.preflight_ok()
        except ValueError:
            result['preflight_ok'] = False
        result['can_send_target'] = self.ready_for_fly()
        result['target_reached'] = self.target_reached
        result['completion_source'] = 'Controller arrival supervision of PX4 telemetry; not a PX4 waypoint ACK'
        result['flight_policy'] = asdict(self.policy)
        result['px4_hold_min_altitude_m'] = self.nav_min_ltr_alt
        result['active_operation'] = self.motion_kind
        result['arrival_errors'] = self.arrival.errors
        result['settling_elapsed_s'] = (max(0, time.monotonic() - self.arrival.since)
                                       if self.arrival.since is not None else 0.0)
        result['hold_acknowledged'] = self.hold_acknowledged
        result['hold_reason'] = self.hold_reason
        result['last_error'] = self.last_error
        result['coordinates_enabled'] = self.phase == 'FLY_READY' and self.ready_for_fly()
        result['target_ned'] = self.target
        result['coordinate_frame'] = 'PX4 local NED; x/y absolute, height above launch'
        result['launch_ned'] = self.launch_ned
        result['horizontal_target_error_m'] = None
        result['target_error_ned_m'] = None
        try:
            p = self.state('local_ned')
            if (self.phase in self.FLIGHT_PHASES | self.HOLD_PHASES
                    and not self.external and self.target is not None):
                errors = {key: p[key] - self.target[key] for key in ('north_m', 'east_m', 'down_m')}
                error = math.hypot(errors['north_m'], errors['east_m'])
                if math.isfinite(error):
                    result['horizontal_target_error_m'] = error
                if all(math.isfinite(value) for value in errors.values()):
                    result['target_error_ned_m'] = errors
            result['position_from_launch_m'] = (
                {key: p[key] - self.launch_ned[key] for key in p}
                if self.launch_ned is not None else None)
            result['height_above_launch_m'] = (
                self.launch_ned['down_m'] - p['down_m']
                if self.launch_ned is not None else None)
        except ValueError:
            result['position_from_launch_m'] = None
            result['height_above_launch_m'] = None
        result['target_locked'] = self.phase in self.FLIGHT_PHASES | {'HOLD_REQUESTED', 'HOLD'}
        result['flight_termination'] = self.termination
        result['land_acknowledged'] = self.land_acknowledged
        result['arm_acknowledged'] = self.arm_acknowledged
        result['disarm_acknowledged'] = self.disarm_acknowledged
        result['motor_stop_kind'] = self.motor_stop_kind
        result['land_request_context'] = self.land_request_context
        with self.lock:
            armed, armed_at = self.cache.get('armed', (None, 0))
        now = time.monotonic()
        result['arm_confirmed'] = (
            armed is True and self.arm_operation_at is not None
            and self.arm_operation_at <= armed_at <= now and now - armed_at <= 3)
        result['disarm_confirmed'] = (
            self.disarm_operation_at is not None and armed is False
            and self.disarm_operation_at <= armed_at <= now and now - armed_at <= 3)
        try:
            with self.lock:
                ground_at = self.cache.get('landed_state', (None, 0))[1]
            result['ground_disarm_confirmed'] = (
                result['disarm_confirmed']
                and self.disarm_operation_at <= ground_at <= now
                and self.state('landed_state') == 'ON_GROUND')
        except ValueError:
            result['ground_disarm_confirmed'] = False
        with self.lock:
            mode, mode_at = self.cache.get('flight_mode', (None, 0))
        result['land_mode_confirmed'] = (
            self.phase in {'LAND_REQUESTED', 'LANDING'} and mode == 'LAND'
            and self.operation_at <= mode_at <= time.monotonic()
            and time.monotonic() - mode_at <= 3)
        result['hold_mode_confirmed'] = (
            self.phase in {'HOLD', 'FLY_READY'} and mode == 'HOLD'
            and self.hold_operation_at is not None and self.hold_operation_at <= mode_at
            and 0 <= time.monotonic() - mode_at <= 3)
        if self.termination:
            result['required_action'] = (
                'Secure the vehicle and confirm it is on the ground before recovering Pixhawk. '
                'Do not power-cycle in flight. After recovery, restart the console.'
                + (' Stop Gazebo only after confirming DISARM.' if self.profile == 'hitl' else ''))
        return json_safe(result)

    def _begin_motion(self, kind):
        self.motion_kind = self.phase = kind
        self.target_started_at = time.monotonic()
        self.target_reached = False
        self.hold_acknowledged = False
        self.hold_reason = None
        self.arrival.reset()

    def refresh_arrival(self):
        now = time.monotonic()
        if now - self.target_started_at >= self.policy.target_timeout_s:
            self._request_hold('TARGET_TIMEOUT')
            return
        try:
            age = self.policy.telemetry_max_age_s
            if (self.state('armed') is not True
                    or self.state('landed_state', age) != 'IN_AIR'):
                self.arrival.reset()
                self.phase = self.motion_kind
                return
            mode = self.state('flight_mode')
            if mode != 'OFFBOARD':
                self.external = True
                self.phase = 'EXTERNAL'
                self.arrival.reset()
                self.stop_stream()
                return
            p, yaw = self.state('local_ned', age), self.state('yaw_deg', age)
            with self.lock:
                sample_at = min(self.cache['local_ned'][1], self.cache['yaw_deg'][1])
            complete = self.arrival.observe(self.target, p, yaw, sample_at, now)
            self.phase = 'SETTLING' if self.arrival.since is not None else self.motion_kind
            if complete:
                self._request_hold('TARGET_REACHED')
        except (ValueError, TypeError, KeyError):
            self.arrival.reset()
            self.phase = self.motion_kind

    def _recover_land(self):
        self.phase = 'FAULT'
        try:
            self.land()
        except Exception:
            logging.getLogger(__name__).exception('Recovery LAND failed')

    def _request_hold(self, reason):
        self.hold_reason = reason
        self.hold_operation_at = self.operation_at = time.monotonic()
        self.hold_acknowledged = False
        self.phase = 'HOLD_REQUESTED'  # Set intent before callbacks change mode.
        self.arrival.reset()
        try:
            # Action.hold() selects native PX4 Hold; it can enforce a minimum
            # height. Reject an unverified climb rather than silently do it.
            if self.nav_min_ltr_alt > 0:
                altitude = self.state('position', self.policy.telemetry_max_age_s)['relative_altitude_m']
                if not math.isfinite(altitude) or altitude < self.nav_min_ltr_alt:
                    raise ValueError('NAV_MIN_LTR_ALT would force a climb; review PX4 Hold configuration')
            self.action.hold()
            self.hold_acknowledged = True
        except Exception as error:
            self.last_error = f'HOLD failed: {error}; requesting LAND'
            logging.getLogger(__name__).exception(self.last_error)
            self._recover_land()

    def require(self):
        super().require()
        if self.termination:
            raise ValueError(
                'PX4 is in FLIGHT TERMINATION; secure the vehicle on the ground before recovery')

    def stop_stream(self):
        if self.offboard is not None:
            logging.getLogger(__name__).info('Stopping Offboard stream; phase=%s',
                                            self.phase, stack_info=True)
            self.offboard.destroy()
            self.offboard = None

    def confirm_started(self, timeout=5.0):
        """Wait for post-ARM state, not old disarmed telemetry or command ACKs."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.require()
            if self.external:
                raise RuntimeError('PX4 left OFFBOARD during startup')
            with self.lock:
                armed, armed_at = self.cache.get('armed', (None, 0))
                mode, mode_at = self.cache.get('flight_mode', (None, 0))
            now = time.monotonic()
            armed_epoch = (self.arm_operation_at if self.arm_operation_at is not None
                           else self.operation_at)
            if (armed is True and mode == 'OFFBOARD'
                    and armed_epoch <= armed_at <= now
                    and self.operation_at <= mode_at <= now
                    and now - armed_at <= 3 and now - mode_at <= 3):
                self.offboard_seen = True
                return
            time.sleep(0.05)
        raise RuntimeError('No fresh ARMED + OFFBOARD confirmation within 5 s; requesting LAND')

    def _prepare_ground_arm_mode(self):
        """Do not ARM into a surviving OFFBOARD stream from the previous flight."""
        if self.state('flight_mode') == 'HOLD':
            return
        epoch = time.monotonic()
        logging.getLogger(__name__).info('Preparing ground HOLD before standalone ARM')
        self.action.hold()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self.require()
            if self.state('armed') is not False or self.state('landed_state') != 'ON_GROUND':
                raise ValueError('Vehicle state changed while preparing ground ARM')
            with self.lock:
                mode, at = self.cache.get('flight_mode', (None, 0))
            now = time.monotonic()
            if mode == 'HOLD' and epoch <= at <= now and now - at <= 3:
                return
            time.sleep(0.05)
        raise RuntimeError('No fresh ground HOLD confirmation; ARM was not sent')

    def _ground_preflight(self, timeout=3.0):
        """PX4 rates armability for its current mode. A disarmed vehicle left in
        OFFBOARD after a stream stop is never armable, so select ground HOLD
        first and re-check with health telemetry newer than that switch."""
        if self.preflight_ok():
            self._prepare_ground_arm_mode()
            return
        if self.state('flight_mode') == 'HOLD':
            raise ValueError(f'PX4 reports preflight not ready: {self.preflight_failures()}')
        epoch = time.monotonic()
        self._prepare_ground_arm_mode()
        key = 'health_all_ok' if self.position_source == 'gps' else 'health'
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                _, at = self.cache.get(key, (None, 0))
            if at >= epoch and self.preflight_ok():
                return
            time.sleep(0.05)
        raise ValueError(f'PX4 reports preflight not ready (also in ground HOLD): {self.preflight_failures()}')

    def arm(self):
        """Arm only: no Offboard plugin, position setpoint or climb command."""
        self.require()
        self.refresh()
        if (self.phase != 'IDLE' or self.state('armed') is not False
                or self.state('landed_state') != 'ON_GROUND'):
            raise ValueError('ARM requires IDLE, DISARM and ON_GROUND')
        self._ground_preflight()
        self.arm_operation_at = self.operation_at = time.monotonic()
        self.arm_acknowledged = False
        self.land_acknowledged = False
        self.phase = 'ARM_REQUESTED'
        logging.getLogger(__name__).info('Sending ARM only; no Offboard/climb requested')
        try:
            self.action.arm()
        except Exception as error:
            # An explicit PX4 rejection proves ARM did not happen; a timeout does not.
            if getattr(getattr(error, 'result', None), 'name', '').startswith('COMMAND_DENIED'):
                self.phase, self.arm_operation_at = 'IDLE', None
            raise
        self.arm_acknowledged = True
        logging.getLogger(__name__).info('ARM acknowledged; awaiting armed=true telemetry')
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self.require()
            self.refresh()
            if self.phase == 'ARMED' and self.state('landed_state') == 'ON_GROUND':
                logging.getLogger(__name__).info('ARM confirmed on ground; no climb requested')
                return
            time.sleep(0.05)
        raise RuntimeError('ARM acknowledged but no fresh armed=true + ON_GROUND '
                           'confirmation within 5 s; inspect status or request LAND')

    def check_height(self, height):
        """Height above launch (m) for takeoff / alt: 0.5..max_height and above the arrival z tolerance."""
        height = number(height, 'height', 0.5, getattr(self, 'max_height', 100.0))
        if self.policy.z_tolerance_m >= height:
            raise ValueError(f'height must exceed z tolerance {self.policy.z_tolerance_m:g} m')
        return height

    def takeoff(self, height=None):
        """TAKEOFF to `height` m above launch (default: --height of this session)."""
        if height is not None:
            height = self.check_height(height)  # reject before touching state or motors
        self.require()
        self.refresh()
        prearmed = self.phase == 'ARMED' and self.state('armed') is True
        disarmed = self.phase == 'IDLE' and self.state('armed') is False
        if not (prearmed or disarmed) or self.state('landed_state') != 'ON_GROUND':
            raise ValueError('TAKEOFF requires ON_GROUND and IDLE/disarmed or ARM confirmed in this session')
        if prearmed:
            if not self.preflight_ok():
                raise ValueError(f'PX4 reports preflight not ready: {self.preflight_failures()}')
        else:
            self._ground_preflight()
        p, yaw = self.state('local_ned'), self.state('yaw_deg')
        north = number(p['north_m'], 'north_m', -10000, 10000)
        east = number(p['east_m'], 'east_m', -10000, 10000)
        down = number(p['down_m'], 'down_m', -10000, 10000)
        yaw = number(yaw, 'yaw_deg', -180, 180)
        if height is not None:
            self.height = height
        self.launch_ned = dict(north_m=north, east_m=east, down_m=down)
        self.target = dict(north_m=north, east_m=east, down_m=down-self.height, yaw_deg=yaw)
        self.external, self.offboard_seen = False, False
        self.land_acknowledged = False
        self.operation_at = time.monotonic()
        self.phase = 'STARTING'
        arm_attempted = prearmed
        try:
            if self.offboard is None:
                self.offboard = self.make_offboard()
            self.offboard.set_position_ned(self.Position(**self.target))
            # MAVSDK repeats the same position signal at 20 Hz. PX4 requires >1 s.
            time.sleep(1.2)
            self.require()
            if self.state('armed') is not prearmed or self.state('landed_state') != 'ON_GROUND':
                raise ValueError('Vehicle state changed during Offboard preparation')
            # Disarmed heartbeats during the prestream do not describe the result
            # of ARM. Reset the epoch immediately before sending that command.
            self.operation_at = time.monotonic()
            self.offboard_seen, self.external = False, False
            arm_attempted = True
            if not prearmed:
                self.arm_operation_at = self.operation_at
                self.arm_acknowledged = False
                self.action.arm()
                self.arm_acknowledged = True
            self.offboard.start()
            self.confirm_started()
            self._begin_motion('TAKEOFF')  # ACK/mode do not indicate arrival.
        except BaseException:
            if arm_attempted:
                try:
                    # ARM ACK may be lost. Send LAND before any native teardown.
                    self.land()
                except Exception:
                    logging.getLogger(__name__).exception('Recovery LAND failed')
            else:
                self.phase = 'IDLE'
                self.stop_stream()
            raise

    def fly(self, north=None, east=None, explicit=True, down=None):
        """FLY to (north, east) at the current target altitude, or at NED down (m) when down is given."""
        self.require()
        self.refresh()
        if not self.ready_for_fly() or self.offboard is None:
            raise ValueError('FLY requires completed TAKEOFF and confirmed HOLD; targets are locked while busy')
        if north is None and east is None:
            self.phase = 'FLY_READY'  # Select input state; native PX4 remains HOLD until a target is sent.
            return True
        north = number(north, 'x/north', -10000, 10000)
        east = number(east, 'y/east', -10000, 10000)
        if not explicit and self.phase != 'FLY_READY':
            raise ValueError('Coordinates are locked in HOLD; select FLY first or use fly x y')
        down = self.target['down_m'] if down is None else number(down, 'down', -10000, 10000)
        target = dict(north_m=north, east_m=east, down_m=down, yaw_deg=self.target['yaw_deg'])
        self._start_target('FLY', target)
        return True

    def altitude(self, height):
        """From confirmed HOLD: climb / descend to `height` m above launch, keeping the current x/y target."""
        height = self.check_height(height)
        self.require()
        self.refresh()
        if self.launch_ned is None or self.target is None:
            raise ValueError('ALT requires a TAKEOFF in this session')
        down = self.launch_ned['down_m'] - height
        self.fly(self.target['north_m'], self.target['east_m'], explicit=True, down=down)
        return True

    def _start_target(self, kind, target=None):
        self.require()
        if (self.state('armed') is not True or self.state('landed_state') != 'IN_AIR'
                or self.state('flight_mode') != 'HOLD'):
            raise ValueError('Starting a target requires fresh ARMED, IN_AIR and HOLD')
        if not self.preflight_ok():
            raise ValueError('PX4 reports health not ready; inspect status before starting a new target')
        if target is not None:
            self.target = target
        self.phase = 'STARTING'
        self.external, self.offboard_seen = False, False
        try:
            self.offboard.set_position_ned(self.Position(**self.target))
            time.sleep(1.2)
            self.require()
            if (self.external or self.state('armed') is not True
                    or self.state('landed_state') != 'IN_AIR'
                    or self.state('flight_mode') != 'HOLD'):
                raise ValueError('Vehicle state changed during target preparation')
            self.operation_at = time.monotonic()
            self.offboard.start()
            self.confirm_started()
            self._begin_motion(kind)
        except BaseException:
            self._recover_land()
            raise

    def land(self):
        self.require()
        self.arrival.reset()
        self.land_request_context = {'phase_before': self.phase,
                                     'offboard_plugin_present': self.offboard is not None}
        for key in ('armed', 'flight_mode', 'landed_state'):
            try:
                self.land_request_context[key] = self.state(key)
            except ValueError:
                self.land_request_context[key] = None
        # Mark intent BEFORE the command: mode callbacks can arrive during land().
        self.phase, self.operation_at = 'LAND_REQUESTED', time.monotonic()
        self.land_acknowledged = False
        logger = logging.getLogger(__name__)
        logger.info('Sending LAND; retaining plugin/connection; context=%s', self.land_request_context)
        # MAVSDK stops setpoints on observing departure from OFFBOARD. Neither
        # destroy() (native teardown) nor stop() (requests HOLD) belongs here.
        # On timeout/rejection keep the plugin alive; do not manufacture loss of
        # Offboard or claim LAND succeeded. The operator may inspect/retry LAND.
        self.action.land()
        self.land_acknowledged = True
        self.phase = 'LANDING'
        logger.info('LAND acknowledged; awaiting fresh LAND mode and landing/disarm telemetry')

    def disarm(self):
        """The single operator DISARM command forces motors off even in flight."""
        self.require()
        self.arrival.reset()
        self.phase, self.operation_at = 'DISARM_REQUESTED', time.monotonic()
        self.disarm_operation_at = self.operation_at
        self.disarm_acknowledged = False
        self.motor_stop_kind = 'FORCED_DISARM'
        logging.getLogger(__name__).warning('Sending forced DISARM; motors may stop in flight')
        self.action.kill()  # Never block the emergency command behind native teardown.
        self.disarm_acknowledged = True
        self.phase = 'EMERGENCY_STOP'

    def close(self):
        try:
            self.stop_stream()
        finally:
            try:
                if self.mavlink_direct is not None:
                    self.mavlink_direct.destroy()
                    self.mavlink_direct = None
            finally:
                self.sdk.destroy()
