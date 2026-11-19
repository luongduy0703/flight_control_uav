#!/usr/bin/env python3
"""Console for serial Pixhawk control or local UDP through a MAVLink bridge."""
import argparse
import faulthandler
from contextlib import ExitStack
import json
import logging
import math
import os
import select
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from deployment import validate_connection
from serial_lock import serial_ownership, udp_ownership
from offboard_control import OffboardControl
from flight_policy import FlightPolicy
from perception_process import PerceptionProcess, send_perception_command

PERCEPTION_HELP = '''perception          Start one perception scan; only in confirmed HOLD after takeoff
perception stop     Stop the running scan (any phase); flight is unaffected
perception restart  Restart the perception program after FAILED or a config change
'''
PERCEPTION_BENCH_HELP = '''--perception-bench: "perception" also accepted DISARMED + ON_GROUND (props off / bench);
                    ARM, takeoff or any armed state stops a bench scan automatically
'''

HELP = '''arm         ARM only on ground; motors may spin; no Offboard or climb
takeoff     Start OFFBOARD climb; accepts separate ARM or arms from IDLE
x y         Target metres N/E; accepted only after selecting FLY
fly         Enable coordinate input from confirmed HOLD
fly x y     Explicitly leave HOLD and fly to absolute local NED x/y
land        Request PX4 LAND
disarm      Force motors off on ground OR in flight; vehicle may fall
status      Read PX4 telemetry, application phase, target confirmation and locks
help        Show commands
exit        End signal stream; does NOT send LAND
X/Y origin: PX4 estimator local origin. Nonzero x/y/z alone does not diagnose a GPS fault.
Z target: launch down_m minus --height; height stays fixed for this flight.
Targets stay locked during TAKEOFF, FLY, SETTLING and HOLD_REQUESTED.
At target: position/yaw dwell supervision by console, then native PX4 HOLD.
HOLD is indefinite. Raw coordinates stay locked until FLY is explicitly selected.
'''


def parse_command(line):
    words = shlex.split(line)
    if not words:
        return 'empty', None
    if len(words) == 1 and words[0].lower() in {
            'arm', 'takeoff', 'fly', 'land', 'disarm', 'status', 'help', 'exit', 'quit'}:
        return ('exit' if words[0].lower() == 'quit' else words[0].lower()), None
    if words[0].lower() == 'perception':
        if len(words) == 1:
            return 'perception', 'start'
        if len(words) == 2 and words[1].lower() in {'stop', 'restart'}:
            return 'perception', words[1].lower()
        raise ValueError('Use perception, perception stop or perception restart')
    if words[0].lower() == 'fly':
        words = words[1:]
    if len(words) != 2:
        raise ValueError('Use arm, takeoff, fly, fly x y, land, disarm, status, help or exit')
    try:
        xy = tuple(float(v) for v in words)
    except ValueError:
        raise ValueError('x y must be numeric metres') from None
    if not all(math.isfinite(v) and abs(v) <= 10000 for v in xy):
        raise ValueError('x y must be finite and within -10000..10000 metres')
    return 'fly', xy


def bench_ready(fc):
    """Bench scan: motors cannot spin (fresh armed=false) and the vehicle is on the ground."""
    try:
        return (fc.drone.is_connected()
                and fc.state('armed') is False
                and fc.state('landed_state') == 'ON_GROUND')
    except ValueError:
        return False


def scan_mode(fc, bench):
    """'flight' (HOLD after takeoff), 'bench' (only with --perception-bench) or None."""
    if fc.ready_for_fly():
        return 'flight'
    if bench and bench_ready(fc):
        return 'bench'
    return None


def run_perception(fc, perception, action, output=print, bench=False):
    """Perception is a separate program; this only starts/stops its scan."""
    if perception is None:
        raise ValueError('Perception is not enabled: start the console with --perception-dir DIR')
    if action == 'restart':
        perception.restart()
        output('Perception restarting; wait for "perception READY".')
        return
    if action == 'stop':
        if send_perception_command(fc, perception.ids, start=False):
            output('Perception STOP acknowledged by the Pi bridge.')
        else:
            output('No ACK for perception STOP; check router/uavp-fc-bridge. Flight is unaffected.')
        return
    state = perception.status()['state']
    if state != 'WAITING':
        raise ValueError(f'Perception is {state}, not WAITING; see status.perception')
    mode = scan_mode(fc, bench)
    if mode is None:
        raise ValueError('Perception scan needs confirmed HOLD after takeoff (armed, IN_AIR, no takeover)'
                         + (' or, for a bench test, fresh DISARMED + ON_GROUND' if bench else
                            '; for a bench test on the ground restart the console with --perception-bench'))
    if not send_perception_command(fc, perception.ids, start=True):
        raise ValueError('No ACK from perception bridge; check mavlink-routerd and uavp-fc-bridge')
    perception.scan_mode = mode
    if mode == 'bench':
        output('Perception START (BENCH, DISARMED on ground) acknowledged; watch for "perception SCANNING". '
               'ARM/takeoff stops this scan.')
    else:
        output('Perception START acknowledged by the Pi bridge; watch for "perception SCANNING". '
               'Vehicle stays in HOLD; perception never commands flight.')


def execute(fc, line, output=print, perception=None, perception_bench=False):
    command, xy = parse_command(line)
    if command == 'exit':
        output('Exit stops Offboard signals; PX4 applies its configured failsafe. Land first.')
        return False
    if command == 'empty':
        return True
    if command == 'help':
        output(HELP + perception_help(perception, perception_bench))
    elif command == 'status':
        status = fc.status()
        if perception is not None:
            status['perception'] = perception.status()
        output(json.dumps(status, indent=2, allow_nan=False))
    elif command == 'perception':
        run_perception(fc, perception, xy, output, bench=perception_bench)
    else:
        if (command in {'arm', 'takeoff'} and perception is not None
                and getattr(perception, 'scan_mode', None) == 'bench'
                and perception.status()['state'] == 'SCANNING'):
            # Stop the bench scan before motors can spin, not after the (blocking) command.
            sent = send_perception_command(fc, perception.ids, start=False)
            perception.auto_stopped = perception.generation
            output('Bench perception scan stopped before ' + command +
                   ('' if sent else ' (STOP not acknowledged)') + '.')
        if command == 'fly':
            if xy is None:
                fc.fly()
            else:
                explicit = shlex.split(line)[0].lower() == 'fly'
                fc.fly(*xy, explicit=explicit)
        else:
            getattr(fc, command)()
        if command == 'arm':
            output('PX4 confirmed armed=true on ground. No Offboard/climb requested. '
                   'Motors may spin; use status, disarm, or explicit takeoff. '
                   'PX4 may auto-disarm while waiting.')
        elif command == 'takeoff':
            output('PX4 confirmed ARMED + OFFBOARD. Observe climb; target arrival is not confirmed.')
        elif command == 'disarm':
            output('Forced DISARM acknowledged. Motors may stop in flight; vehicle may fall. '
                   'Check fresh armed=false; ACK alone does not confirm motors stopped.')
        elif command == 'land':
            output('LAND acknowledged. Connection remains open; check land_mode_confirmed, '
                   'then ON_GROUND and armed=false. Do not exit yet.')
        elif command == 'fly' and xy is None:
            output('FLY selected. PX4 remains in HOLD until you enter x y; no motion commanded yet.')
        else:
            output('Request sent. Observe PX4/Gazebo; this is not confirmation of arrival.')
    return True


def wait_initial_status(fc, timeout=3.0):
    """Let asynchronous telemetry and raw heartbeat populate before --check."""
    required = {'armed', 'landed_state', 'flight_mode', 'health_all_ok', 'health', 'local_ned'}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        fc.refresh()
        with fc.lock:
            if fc.termination or required.issubset(fc.cache):
                return
        time.sleep(0.05)


SITE_ORIGIN_FILE = Path(__file__).resolve().parents[1] / 'site_origin.json'


def load_site_origin(path):
    """Read the saved flying-site origin; None if the file does not exist.

    Only magnetic declination depends on it, which changes slowly: a point in
    the same region (~100 km) is enough, so it is written once, not per flight.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        origin = (data['latitude_deg'], data['longitude_deg'], data['altitude_m'])
        return OffboardControl.validate_origin(*origin)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Invalid site origin file {path}: {error}') from None


def poll_controller(fc, logger, include_status=False):
    """A diagnostic/refresh error must not close the session before LAND/DISARM."""
    try:
        fc.refresh()
        if include_status:
            logger.info('Status %s', json.dumps(fc.status()))
    except Exception as error:
        fc.last_error = f'Polling failed: {error}; inspect state before motion commands'
        logger.exception('Polling failed; keeping console available for LAND/DISARM')


def pin_control_cores(spec):
    """Pin this process (and every thread/child created later: MAVSDK) to the control cores.

    Perception saturates its own cores (cpu.cores: [2, 3]) while scanning; the Offboard
    setpoint stream must never wait behind it. '' disables pinning.
    """
    if not spec:
        return None
    try:
        cores = {int(c) for c in spec.split(',')}
    except ValueError:
        raise ValueError(f'--control-cores must look like 0,1 (got {spec!r})') from None
    count = os.cpu_count() or 1
    if not cores or any(not 0 <= c < count for c in cores):
        raise ValueError(f'--control-cores {spec!r}: cores must be within 0..{count - 1}')
    if len(cores) == count:
        raise ValueError('--control-cores must leave cores for perception')
    os.sched_setaffinity(0, cores)
    return sorted(cores)


def perception_help(perception, bench):
    if perception is None:
        return ''
    return PERCEPTION_HELP + (PERCEPTION_BENCH_HELP if bench else '')


def poll_perception(fc, perception, logger, output=print):
    """Follow the perception child; never let its errors reach flight control."""
    try:
        perception.poll()
        if perception.status()['state'] == 'SCANNING' and perception.auto_stopped != perception.generation:
            # A scan stays valid only in the state it was started in.
            bench = getattr(perception, 'scan_mode', None) == 'bench'
            still_ok = bench_ready(fc) if bench else fc.ready_for_fly()
            if not still_ok:
                reason = ('vehicle no longer DISARMED on ground (bench scan)' if bench
                          else 'vehicle left HOLD')  # fly/land/takeover: pose assumptions gone.
                perception.auto_stopped = perception.generation  # Once per scan.
                logger.warning('%s during perception scan; sending STOP', reason)
                sent = send_perception_command(fc, perception.ids, start=False)
                perception.events.append(f'{reason}: perception STOP ' +
                                         ('sent' if sent else 'NOT acknowledged'))
        events = perception.take_events()
    except Exception:
        logger.exception('Perception polling failed; flight control continues')
        return
    for event in events:
        logger.info('Perception event: %s', event)
        output(f'\n[{event}]\nUAV> ', end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', default='serial:///dev/ttyAMA0:921600')
    parser.add_argument('--profile', choices=['hitl', 'hardware'], default='hitl')
    parser.add_argument('--system-id', type=int, default=1)
    parser.add_argument('--height', type=float, default=2.0,
                        help='Climb in metres above the local NED position captured at takeoff')
    parser.add_argument('--xy-tolerance', type=float, default=0.5, help='Arrival horizontal tolerance, metres')
    parser.add_argument('--z-tolerance', type=float, default=0.3, help='Arrival vertical tolerance, metres')
    parser.add_argument('--yaw-tolerance', type=float, default=10.0, help='Arrival yaw tolerance, degrees')
    parser.add_argument('--settle-time', type=float, default=2.0, help='Fresh in-tolerance dwell before HOLD, seconds')
    parser.add_argument('--target-timeout', type=float, default=60.0, help='Target deadline before HOLD, seconds')
    parser.add_argument('--log-dir', default='session-logs', help='Controller diagnostic log directory')
    parser.add_argument('--position-source', choices=['gps', 'flow'], default='gps',
                        help='flow: preflight needs local position only, no GPS global/home position')
    parser.add_argument('--ekf-origin', nargs=3, type=float, metavar=('LAT', 'LON', 'ALT_M'),
                        help='flow: real site lat/lon (deg) and AMSL altitude (m) sent to PX4 as EKF '
                             'origin so HOLD/ARM work without GPS; also sets magnetic declination')
    parser.add_argument('--site-origin-file', type=Path, default=SITE_ORIGIN_FILE,
                        help='flow: JSON with the flying-site origin, used when --ekf-origin is omitted')
    parser.add_argument('--enable-actuation', action='store_true')
    parser.add_argument('--check', action='store_true', help='Read state, never send flight commands')
    parser.add_argument('--perception-dir', type=Path,
                        help='Perception project on the Pi (e.g. ~/uav_perception_pi). When given, '
                             'build/uavp_real_time is started now as a separate process and the '
                             '"perception" command becomes available')
    parser.add_argument('--perception-config', default='configs/run/real_time.yaml',
                        help='Perception config, relative to --perception-dir')
    parser.add_argument('--perception-set', nargs='+', default=[], metavar='KEY=VALUE',
                        help='Extra perception --set overrides, e.g. remote.max_fps=5')
    parser.add_argument('--perception-bench', action='store_true',
                        help='Also accept "perception" while DISARMED + ON_GROUND (props-off bench test). '
                             'On the ground the flow height is ~0: add --perception-set '
                             'distance.source=fixed distance.fixed_m=<camera-to-surface m>')
    parser.add_argument('--control-cores', default='0,1',
                        help='CPU cores for the console and MAVSDK (perception uses its own cpu.cores, '
                             'default 2,3); empty string disables pinning')
    args = parser.parse_args()
    if args.perception_bench and args.perception_dir is None:
        parser.error('--perception-bench needs --perception-dir')
    if not math.isfinite(args.height) or not 0.5 <= args.height <= 100:
        parser.error('height must be 0.5..100 m')
    if not 1 <= args.system_id <= 255:
        parser.error('system-id must be 1..255')
    if args.ekf_origin is not None:
        if args.position_source != 'flow':
            parser.error('--ekf-origin is only used with --position-source flow')
        try:
            OffboardControl.validate_origin(*args.ekf_origin)
        except ValueError as error:
            parser.error(str(error))
    elif args.position_source == 'flow' and args.enable_actuation and not args.check:
        try:
            args.ekf_origin = load_site_origin(args.site_origin_file)
        except ValueError as error:
            parser.error(str(error))
        if args.ekf_origin is None:
            parser.error(f'flow needs the flying-site origin (PX4 HOLD requires it without GPS). '
                         f'Create {args.site_origin_file} once, e.g. '
                         f'{{"latitude_deg": 21.0285, "longitude_deg": 105.8542, "altitude_m": 10}}, '
                         f'or pass --ekf-origin LAT LON ALT_M')
    try:
        policy = FlightPolicy(xy_tolerance_m=args.xy_tolerance, z_tolerance_m=args.z_tolerance,
                              yaw_tolerance_deg=args.yaw_tolerance, settle_s=args.settle_time,
                              target_timeout_s=args.target_timeout)
        if policy.z_tolerance_m >= args.height:
            raise ValueError('z-tolerance must be less than height')
    except ValueError as error:
        parser.error(str(error))
    try:
        device = validate_connection(args.profile, args.connection)
    except ValueError as error:
        parser.error(str(error))
    perception = None
    if args.perception_dir is not None and not args.check:
        if not device.startswith('udpin://'):
            # Perception reads the Pixhawk through mavlink-routerd; a console that
            # opens the UART itself would take the link away from the router.
            parser.error('--perception-dir needs --connection udpin://127.0.0.1:14540 '
                         '(mavlink-routerd owns TELEM1; see config/mavlink-router.conf)')
        try:
            perception = PerceptionProcess(args.perception_dir, args.perception_config,
                                           log_dir=args.log_dir, overrides=args.perception_set)
        except (OSError, ValueError) as error:
            parser.error(f'perception: {error}')
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / datetime.now(timezone.utc).strftime('pi-%Y%m%dT%H%M%S.%fZ.log')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s', force=True)
    logger = logging.getLogger(__name__)
    logger.info('Session started profile=%s connection=%s height=%s position_source=%s actuation=%s',
                args.profile, args.connection, args.height, args.position_source,
                args.enable_actuation and not args.check)
    for label, path in [('boot_id', '/proc/sys/kernel/random/boot_id'),
                        ('uptime', '/proc/uptime')]:
        try:
            logger.info('Host %s=%s', label, Path(path).read_text().strip())
        except OSError:
            logger.warning('Cannot read host %s', label, exc_info=True)
    print(f'Diagnostic log: {log_path}')
    try:
        pinned = pin_control_cores(args.control_cores)
    except ValueError as error:
        parser.error(str(error))
    except OSError as error:  # Pinning is a latency safeguard, not a reason to refuse flight.
        pinned = None
        logger.warning('Cannot pin console to cores %s: %s', args.control_cores, error)
        print(f'Warning: cannot pin console to cores {args.control_cores}: {error}')
    if pinned:
        logger.info('Console pinned to CPU cores %s', pinned)
        print(f'Console + MAVSDK pinned to CPU cores {pinned}')
    with ExitStack() as stack:
        fault_log = stack.enter_context(open(log_path.with_suffix('.fault.log'), 'a'))
        faulthandler.enable(file=fault_log, all_threads=True)
        stack.callback(faulthandler.disable)
        ownership = udp_ownership if device.startswith('udpin://') else serial_ownership
        stack.enter_context(ownership(device))
        fc = OffboardControl(args.connection, args.profile,
                             args.enable_actuation and not args.check, args.system_id, args.height,
                             policy=policy, position_source=args.position_source,
                             ekf_origin=args.ekf_origin)
        stack.callback(fc.close)
        if args.check:
            wait_initial_status(fc)
            execute(fc, 'status')
            return 0
        if perception is not None:
            # Registered after fc: closed first, so the child stops before the SDK.
            stack.callback(perception.close)
            perception.start()
            print(f'Perception: {perception.binary} started as a separate process '
                  f'(log {perception.status()["log"]}); wait for "perception READY" before ARM.')
            if args.perception_bench and not any(s.startswith('distance.source=fixed')
                                                 for s in args.perception_set):
                print('Warning: --perception-bench with distance.source from the flight controller; on the '
                      'ground its height is ~0 and s3 size filtering is wrong. Add --perception-set '
                      'distance.source=fixed distance.fixed_m=<m>.')
        print(f'Console -> Pixhawk: {args.connection}; climb above launch={args.height:g} m')
        print(HELP + perception_help(perception, args.perception_bench))
        last_status_log = 0.0
        while True:
            try:
                print('UAV> ', end='', flush=True)
                # Keep watching takeover/landing while the user is not typing.
                # Setpoint transmission itself remains inside MAVSDK.
                while not select.select([sys.stdin], [], [], 0.2)[0]:
                    include_status = time.monotonic() - last_status_log >= 1.0
                    poll_controller(fc, logger, include_status=include_status)
                    if perception is not None:
                        poll_perception(fc, perception, logger)
                    if include_status:
                        last_status_log = time.monotonic()
                line = input()
            except (EOFError, KeyboardInterrupt):
                logger.warning('EOF/interrupt: closing session and stopping stream', exc_info=True)
                print('\nExit: signal stream stops; no LAND command. PX4 failsafe applies.')
                break
            try:
                # Diagnostics must not refresh/destroy a plugin before an
                # operator's LAND or DISARM can be dispatched.
                logger.info('Command %r', line)
                if not execute(fc, line, perception=perception, perception_bench=args.perception_bench):
                    break
            except KeyboardInterrupt:
                logger.exception('Command interrupted')
                print('Command interrupted; inspect status before retrying.')
            except Exception as error:
                logger.exception('Command failed: %r', line)
                print(f'Error: {error}. Inspect status before retrying; no automatic retry.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        logging.getLogger(__name__).exception('Fatal console error')
        raise
