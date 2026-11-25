#!/usr/bin/env python3
"""Live drift trajectory of the UAV in PX4's local frame (read-only, runs on the laptop).

Origin (0, 0, 0) is PX4's EKF local origin: where the EKF initialised after the
Pixhawk booted. Positions come from LOCAL_POSITION_NED (~30 Hz) and are drawn as
    x = North [m], y = East [m], z = Up [m] (= -Down)
in a 3D view (top view if mplot3d is unavailable) plus x/y/z against time.
Every sample is also written to session-logs/drift-*.csv for later review.

The tool never transmits MAVLink. On the Pi, tools/run_hardware_router.sh forwards
everything to PC_IPV4:14560 (endpoint "drift"); this script listens there:

    python3 tools/drift_viewer.py                         # udpin:0.0.0.0:14560
    python3 tools/drift_viewer.py --replay session-logs/drift-20261118T101500.csv [--save drift.png]

Hover drift pipeline (tools/hover_drift.py), default target 0.8 m above launch:
the launch point is taken while PX4 reports ON_GROUND; once the vehicle holds
0.8 +- 0.15 m with speed <= 0.2 m/s for 2 s, drift is measured from that hover start
point until it leaves the height band, lands, or --hover-duration ends. Each hover
summary is printed and appended to session-logs/hover-*.json. Fly it with e.g.
    pi/console.py ... --height 0.8      (then TAKEOFF, wait, LAND)
This is PX4's own estimate: it shows position-hold error. For estimator drift,
mark launch and touchdown on the floor and compare with the printed landing offset.

Events marked on the plot and printed:
  * Pixhawk reboot (time_boot_ms goes backwards): new origin, the trail restarts.
  * Position jump > --jump-m between consecutive samples: EKF reset, e.g. after
    SET_GPS_GLOBAL_ORIGIN or optical-flow loss/recovery. Drift after it is still
    relative to the same origin, but the jump itself is not real motion.
"""
import argparse
import csv
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from hover_drift import HoverMonitor, format_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONNECTION = 'udpin:0.0.0.0:14560'
CSV_FIELDS = ['wall_time', 'time_boot_s', 'boot', 'x_north_m', 'y_east_m', 'z_up_m',
              'vx_m_s', 'vy_m_s', 'vz_up_m_s', 'event', 'landed_state', 'range_m']
MAX_PLOT_POINTS = 4000
REBOOT_BACKSTEP_S = 1.0  # time_boot_ms going back more than this = Pixhawk rebooted


def ned_to_xyz(north, east, down):
    """PX4 NED -> plot axes (North, East, Up)."""
    return north, east, -down


def decimate(values, limit=MAX_PLOT_POINTS):
    """Every k-th value (always keeping the last) so long flights still draw fast."""
    if len(values) <= limit:
        return list(values)
    step = math.ceil(len(values) / limit)
    out = list(values[::step])
    if (len(values) - 1) % step:
        out.append(values[-1])
    return out


def padded_limits(values, min_span):
    """Axis limits covering values but never narrower than min_span, so mm noise stays small."""
    lo, hi = min(values), max(values)
    half = max((hi - lo) / 2 * 1.1, min_span / 2)
    mid = (lo + hi) / 2
    return mid - half, mid + half


class Track:
    """Samples of one Pixhawk boot plus reboot / EKF-jump detection (no I/O)."""

    def __init__(self, jump_m=0.5, jump_window_s=0.5, hover=None):
        self.jump_m, self.jump_window_s = jump_m, jump_window_s
        self.hover = hover if hover is not None else HoverMonitor()
        self.lock = threading.Lock()
        self.boot = 0
        self.reset()

    def reset(self):
        self.t, self.x, self.y, self.z = [], [], [], []
        self.jumps = []  # indices of samples that arrived right after a jump
        self.max_horizontal = 0.0
        self.path_length = 0.0

    def add(self, t, x, y, z, v=(0.0, 0.0, 0.0), landed=None, range_m=None):
        """Add one sample (t = seconds since boot).

        Returns '' or '+'-joined events: reboot, jump, hover_start, hover_end, landed; or 'stale'.
        """
        with self.lock:
            event = ''
            if self.t and t < self.t[-1] - REBOOT_BACKSTEP_S:
                self.boot += 1
                self.reset()
                self.hover.reset_boot(self.boot)
                event = 'reboot'
            elif self.t and t <= self.t[-1]:
                return 'stale'  # duplicate / out-of-order packet
            elif self.t:
                step = math.dist((x, y, z), (self.x[-1], self.y[-1], self.z[-1]))
                if step > self.jump_m and t - self.t[-1] <= self.jump_window_s:
                    self.jumps.append(len(self.t))
                    event = 'jump'
                else:
                    self.path_length += step
            self.t.append(t)
            self.x.append(x)
            self.y.append(y)
            self.z.append(z)
            self.max_horizontal = max(self.max_horizontal, math.hypot(x, y))
            hover_event = self.hover.update(len(self.t) - 1, t, x, y, z, *v, landed=landed,
                                            range_m=range_m)
            return '+'.join(e for e in (event, hover_event) if e)

    def snapshot(self):
        with self.lock:
            return (list(self.t), list(self.x), list(self.y), list(self.z), list(self.jumps),
                    self.boot, self.max_horizontal, self.path_length)

    def hover_view(self):
        """(status line, segments of the current boot, last summary or None)."""
        with self.lock:
            h = self.hover
            return (h.status_line(), [(a, b) for boot, a, b in h.segments if boot == self.boot],
                    h.results[-1] if h.results else None)


def stats_text(t, x, y, z, boot, max_horizontal, path_length, jumps, age_s, rate_hz):
    if not t:
        return 'Waiting for LOCAL_POSITION_NED ...'
    horizontal = math.hypot(x[-1], y[-1])
    return (f'boot #{boot}   t_boot {t[-1]:8.1f} s   {rate_hz:4.1f} Hz   age {age_s:4.1f} s\n'
            f'x N {x[-1]:+7.2f} m   y E {y[-1]:+7.2f} m   z Up {z[-1]:+7.2f} m\n'
            f'drift from origin: horizontal {horizontal:.2f} m   3D {math.hypot(horizontal, z[-1]):.2f} m   '
            f'max horizontal {max_horizontal:.2f} m\n'
            f'path length {path_length:.2f} m   EKF jumps {len(jumps)}')


class MavlinkReader(threading.Thread):
    """Receives LOCAL_POSITION_NED, EXTENDED_SYS_STATE, DISTANCE_SENSOR; never sends anything."""

    def __init__(self, connection, track, writer, report=None):
        super().__init__(daemon=True)
        from pymavlink import mavutil
        self.conn = mavutil.mavlink_connection(connection, source_system=255, source_component=0)
        self.track, self.writer, self.report = track, writer, report
        self.landed, self.range_m = None, None
        self.last_rx = None
        self.arrivals = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            msg = self.conn.recv_match(type=['LOCAL_POSITION_NED', 'EXTENDED_SYS_STATE',
                                             'DISTANCE_SENSOR'], blocking=True, timeout=0.5)
            if msg is None or msg.get_srcComponent() != 1:  # autopilot only
                continue
            if msg.get_type() == 'EXTENDED_SYS_STATE':
                self.landed = msg.landed_state
                continue
            if msg.get_type() == 'DISTANCE_SENSOR':
                d = msg.current_distance / 100.0
                valid = msg.min_distance <= msg.current_distance <= msg.max_distance
                self.range_m = d if valid else None
                continue
            now = time.monotonic()
            self.last_rx = now
            self.arrivals = [a for a in self.arrivals if now - a < 2.0] + [now]
            x, y, z = ned_to_xyz(msg.x, msg.y, msg.z)
            vx, vy, vz = ned_to_xyz(msg.vx, msg.vy, msg.vz)
            t = msg.time_boot_ms / 1000.0
            event = self.track.add(t, x, y, z, (vx, vy, vz), self.landed, self.range_m)
            if event == 'stale':
                continue
            if event:
                print(f'{datetime.now():%H:%M:%S} {event.upper()} at t_boot {t:.1f} s '
                      f'pos ({x:+.2f}, {y:+.2f}, {z:+.2f})', flush=True)
                report_events(self.track, event, self.report)
            if self.writer:
                self.writer.writerow([datetime.now().isoformat(timespec='milliseconds'), f'{t:.3f}',
                                      self.track.boot, f'{x:.3f}', f'{y:.3f}', f'{z:.3f}',
                                      f'{vx:.3f}', f'{vy:.3f}', f'{vz:.3f}', event,
                                      '' if self.landed is None else self.landed,
                                      '' if self.range_m is None else f'{self.range_m:.3f}'])

    def rate_hz(self):
        a = self.arrivals
        return (len(a) - 1) / (a[-1] - a[0]) if len(a) > 1 and a[-1] > a[0] else 0.0


def report_events(track, event, report_path):
    """Print hover summaries / landing offsets; append them to the JSON lines report."""
    records = []
    if 'hover_end' in event:
        r = track.hover.results[-1]
        print(format_summary(r), flush=True)
        records.append({'type': 'hover', 'boot': track.boot, **r})
    if 'landed' in event or ('hover_end' in event and track.hover.state == 'GROUND'):
        if track.hover.landings:
            land = track.hover.landings[-1]
            o = land['offset_from_launch_m']
            print(f"LANDED: estimated offset from launch N {o['x_north']:+.3f} E {o['y_east']:+.3f} "
                  f"= {o['horizontal']:.3f} m (compare with the tape-measured offset)", flush=True)
            records.append({'type': 'landing', 'boot': track.boot, **land})
    if report_path and records:
        with open(report_path, 'a') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')


def optional_float(value):
    return float(value) if value not in (None, '') else None


def load_replay(path, track):
    """Replay a CSV; logs written before landed_state was recorded give no hover results."""
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            landed = optional_float(row.get('landed_state'))
            event = track.add(float(row['time_boot_s']), float(row['x_north_m']),
                              float(row['y_east_m']), float(row['z_up_m']),
                              tuple(optional_float(row.get(k)) or 0.0
                                    for k in ('vx_m_s', 'vy_m_s', 'vz_up_m_s')),
                              None if landed is None else int(landed),
                              optional_float(row.get('range_m')))
            if event and event != 'stale':
                report_events(track, event, None)


def build_figure(title):
    import matplotlib.pyplot as plt
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')
        fig = plt.figure(figsize=(13, 7.5))
        main = fig.add_subplot(1, 2, 1, projection='3d')
        three_d = True
    except Exception:
        print('mplot3d unavailable (system matplotlib conflicts with pip one): showing top view. '
              'Fix: use a venv, or sudo apt remove python3-matplotlib', file=sys.stderr)
        fig = plt.figure(figsize=(13, 7.5))
        main = fig.add_subplot(1, 2, 1)
        three_d = False
    series = [fig.add_subplot(3, 2, 2 * i + 2) for i in range(3)]
    fig.suptitle(title)
    return plt, fig, main, three_d, series


def run_plot(track, title, refresh_s, window_s, reader=None, save=None, min_span=0.5):
    plt, fig, main, three_d, series = build_figure(title)
    info = fig.text(0.01, 0.01, '', family='monospace', fontsize=9, va='bottom')
    labels = ('x North [m]', 'y East [m]', 'z Up [m]')
    colors = ('tab:blue', 'tab:orange', 'tab:green')

    def draw(_frame=None):
        t, x, y, z, jumps, boot, max_h, path = track.snapshot()
        age = time.monotonic() - reader.last_rx if reader and reader.last_rx else 0.0
        if reader:
            rate = reader.rate_hz()
        else:
            rate = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0
        hover_line, segments, last = track.hover_view()
        if last and track.hover.state != 'HOVER':
            hover_line += f"\nlast {format_summary(last)}"
        info.set_text(stats_text(t, x, y, z, boot, max_h, path, jumps, age, rate) + '\n' + hover_line)
        spans = [(a, len(t) - 1 if b is None else b) for a, b in segments if a < len(t)]
        main.cla()
        main.set_xlabel(labels[0])
        main.set_ylabel(labels[1])
        if three_d:
            main.set_zlabel(labels[2])
        main.set_title('Trajectory, origin = Pixhawk boot (EKF origin)')
        if t:
            px, py, pz = decimate(x), decimate(y), decimate(z)
            jx, jy, jz = ([x[i] for i in jumps], [y[i] for i in jumps], [z[i] for i in jumps])
            if three_d:
                main.plot(px, py, pz, lw=1, color='tab:blue')
                main.scatter([0], [0], [0], color='black', marker='+', s=80, label='origin')
                main.scatter([x[-1]], [y[-1]], [z[-1]], color='tab:red', s=30, label='now')
                if jumps:
                    main.scatter(jx, jy, jz, color='tab:purple', marker='x', s=40, label='EKF jump')
                for k, (a, b) in enumerate(spans):
                    main.plot(x[a:b + 1], y[a:b + 1], z[a:b + 1], lw=2, color='tab:red',
                              label='hover' if k == 0 else None)
                    main.scatter([x[a]], [y[a]], [z[a]], color='tab:red', marker='^', s=50)
                span = max(1.0, *(max(abs(v) for v in a) for a in (px, py, pz)))
                main.set_xlim(-span, span)
                main.set_ylim(-span, span)
                main.set_zlim(-span, span)
            else:
                # North up, East right: plot East on the horizontal axis.
                main.plot(py, px, lw=1, color='tab:blue')
                main.scatter([0], [0], color='black', marker='+', s=80, label='origin')
                main.scatter([y[-1]], [x[-1]], color='tab:red', s=30, label='now')
                if jumps:
                    main.scatter(jy, jx, color='tab:purple', marker='x', s=40, label='EKF jump')
                for k, (a, b) in enumerate(spans):
                    main.plot(y[a:b + 1], x[a:b + 1], lw=2, color='tab:red',
                              label='hover' if k == 0 else None)
                    main.scatter([y[a]], [x[a]], color='tab:red', marker='^', s=50)
                main.set_xlabel(labels[1])
                main.set_ylabel(labels[0])
                span = max(min_span, max(py) - min(py), max(px) - min(px)) * 1.1
                cy, cx = (max(py) + min(py)) / 2, (max(px) + min(px)) / 2
                cy, cx = (cy, cx) if span > min_span * 1.1 else (0.0, 0.0)
                main.set_xlim(cy - span / 2, cy + span / 2)
                main.set_ylim(cx - span / 2, cx + span / 2)
                main.set_aspect('equal', adjustable='box')
                main.grid(True)
            main.legend(loc='upper right', fontsize=8)
        start = (t[-1] - window_s) if t and window_s > 0 else -math.inf
        first = next((i for i, v in enumerate(t) if v >= start), len(t))
        tt = decimate(t[first:])
        for ax, values, label, color in zip(series, (x, y, z), labels, colors):
            ax.cla()
            shown = decimate(values[first:])
            ax.plot(tt, shown, lw=1, color=color)
            if shown:
                ax.set_ylim(*padded_limits(shown, min_span))
            for i in jumps:
                if i >= first:
                    ax.axvline(t[i], color='tab:purple', lw=0.8, ls='--')
            for a, b in spans:
                if b >= first:
                    ax.axvspan(t[max(a, first)], t[b], color='tab:red', alpha=0.12)
            ax.set_ylabel(label)
            ax.grid(True)
        series[-1].set_xlabel('time since Pixhawk boot [s]')
        fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.21, hspace=0.35)

    if reader:
        from matplotlib.animation import FuncAnimation
        fig._drift_anim = FuncAnimation(fig, draw, interval=int(refresh_s * 1000),
                                        cache_frame_data=False)
    else:
        draw()
        if save:
            fig.savefig(save, dpi=120)
            print(f'Saved {save}', flush=True)
            return
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--connection', default=DEFAULT_CONNECTION,
                        help=f'pymavlink connection to listen on (default {DEFAULT_CONNECTION})')
    parser.add_argument('--replay', type=Path, help='show a recorded drift-*.csv instead of live data')
    parser.add_argument('--save', type=Path, help='with --replay: write the plot to this PNG, no window')
    parser.add_argument('--jump-m', type=float, default=0.5,
                        help='step between consecutive samples counted as an EKF jump (default 0.5 m)')
    parser.add_argument('--window', type=float, default=120.0,
                        help='seconds shown in the x/y/z time plots, 0 = all (default 120)')
    parser.add_argument('--min-span', type=float, default=0.5,
                        help='smallest axis range in m, so sensor noise does not fill the plot (default 0.5)')
    parser.add_argument('--refresh', type=float, default=0.2, help='redraw period in s (default 0.2)')
    parser.add_argument('--hover-height', type=float, default=0.8,
                        help='hover height above launch to measure drift at (default 0.8 m)')
    parser.add_argument('--hover-tol', type=float, default=0.15,
                        help='height band that counts as hovering at target (default +-0.15 m)')
    parser.add_argument('--hover-settle', type=float, default=2.0,
                        help='seconds steady in the band before measuring starts (default 2)')
    parser.add_argument('--hover-duration', type=float, default=0.0,
                        help='stop each measurement after this many s, 0 = until it leaves (default 0)')
    parser.add_argument('--log-dir', type=Path, default=PROJECT_ROOT / 'session-logs')
    parser.add_argument('--no-csv', action='store_true', help='do not record a CSV')
    args = parser.parse_args()
    if args.min_span <= 0 or args.jump_m <= 0 or not 0.05 <= args.refresh <= 5 or args.window < 0:
        parser.error('need --jump-m > 0, --refresh 0.05..5 s, --window >= 0')
    if args.save and not args.replay:
        parser.error('--save needs --replay')

    if not 0.3 <= args.hover_height <= 100 or not 0.02 <= args.hover_tol < 0.3 \
            or not 0 <= args.hover_settle <= 30 or args.hover_duration < 0:
        parser.error('need --hover-height 0.3..100, --hover-tol 0.02..0.3, --hover-settle 0..30, '
                     '--hover-duration >= 0')
    hover = HoverMonitor(target_m=args.hover_height, tol_m=args.hover_tol, settle_s=args.hover_settle,
                         duration_s=args.hover_duration)
    track = Track(jump_m=args.jump_m, hover=hover)
    if args.replay:
        load_replay(args.replay, track)
        run_plot(track, f'Drift replay: {args.replay.name}', args.refresh, args.window, save=args.save,
                 min_span=args.min_span)
        return 0

    csv_file, writer, report = None, None, None
    if not args.no_csv:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        path = args.log_dir / datetime.now().strftime('drift-%Y%m%dT%H%M%S.csv')
        csv_file = open(path, 'w', newline='', buffering=1)
        writer = csv.writer(csv_file)
        writer.writerow(CSV_FIELDS)
        report = path.with_name(path.name.replace('drift-', 'hover-')).with_suffix('.json')
        print(f'CSV: {path}\nHover reports (JSON lines): {report}', flush=True)
    reader = MavlinkReader(args.connection, track, writer, report)
    print(f'Listening on {args.connection} (read-only). Close the window or Ctrl+C to stop.', flush=True)
    reader.start()
    try:
        run_plot(track, f'Live drift ({args.connection})', args.refresh, args.window, reader,
                 min_span=args.min_span)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop.set()
        reader.join(timeout=1.0)
        if csv_file:
            csv_file.close()
        _, _, _, _, jumps, boot, max_h, path_len = track.snapshot()
        print(f'Max horizontal drift {max_h:.2f} m, path {path_len:.2f} m, '
              f'EKF jumps {len(jumps)}, reboots {boot}', flush=True)
        for r in hover.results:
            print(format_summary(r), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
