"""planA: estimate the top surface of an object with the UAV (runs on the Pi).

    python3 planA/run_plan_a.py [--config planA/config/scenario.yaml]

SAFETY (control: in scenario.yaml): nothing moves without an operator command. arm and takeoff need "yes";
every setpoint (fly to the object, LiDAR correction, descend) is only PROPOSED and needs "go" ("skip" = no).
land always works at once. exit / Ctrl+C while armed or airborne asks again; losing the terminal (SSH drop) -> LAND.

Commands (type + Enter):
    check             preflight checklist (link, armed, landed, health, perception, IMU log)
    arm / disarm      ARM / DISARM on the ground (arm -> "yes")
    yes / no          confirm / cancel arm or takeoff
    go / skip         fly / drop the proposed setpoint
    takeoff           ARM + climb to hover_height_m; origin (0,0,0) = floor under the FC at takeoff (EKF)
    s_p               (= start_perception) after a stable hover: detect -> nearest-to-centre object -> fly over it -> LiDAR
                      alignment -> top-surface plane -> result (perception.auto_start: sent automatically after
                      hover_stabilize_s of stable hover, which is also the launch-marker / EKF-drift window)
Over the object the UAV first descends to approach.final_height_m (0.8 m, "go" / "skip"), then sends the SCAN
command (param1 = 2): perception accumulates LiDAR sweeps at that height (window 2 s -> up to max_window_s until the
normal is observable) and reports the result. Object height comes from LiDAR (floor - top). No TARGET within
perception.detect_timeout_s: perception is stopped and the UAV keeps hovering.
    status            print state
    imu               Pixhawk IMU vibration report so far (also written automatically at the end of the run)
    video             video vibration / blur report so far (also written automatically at the end of the run)
    land              LAND now (any time)
    stop              stop the perception scan (UAV keeps its HOLD)
    exit              quit (land first; leaving stops the setpoint stream, PX4 failsafe applies)

Reuses the controller (pi/offboard_control.py) and the perception child (pi/perception_process.py ->
his_perception_uav-main build/uavp_real_time). Perception never commands flight: it only reports
TARGET / ALIGN / OBJECT lines; this program decides every setpoint.
Launch ArUco (marker.enabled): perception also reports MARKER lines from takeoff to start_perception. They are
used ONLY to evaluate the result against a tape measurement from the marker; setpoints stay in EKF NED.
Log + images: planA/logs/<UTC time>/ (plan_a.log, perception-*.log, perception/ result images).
"""
import argparse
import faulthandler
import json
import logging
import math
import select
import signal
import shutil
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pi'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan_a  # noqa: E402
import imu_diag  # noqa: E402
import video_diag  # noqa: E402

# Perception run files copied next to plan_a.log (images: start / detect / aligned / final).
COPY_FILES = ('start.jpg', 'detect.jpg', 'aligned.jpg', 'frame.jpg', 'object.jpg', 'final.jpg', 'result.json',
              'run.log', 'fc_events.jsonl', 'track.csv', 'lidar_points.csv', 'config.yaml',
              'launch_marker.csv')
ABORT_PHASES = {'EXTERNAL', 'FAULT', 'LANDING', 'LAND_REQUESTED', 'TERMINATION', 'EMERGENCY_STOP', 'IDLE'}


class PlanA:
    """State machine: WAIT_TAKEOFF -> TAKEOFF -> STABILIZE -> (WAIT_START) -> SCANNING -> APPROACH -> DESCEND
    (approach.final_height_m, before scanning) -> lenh QUET -> ALIGN (LiDAR scan at that height) -> DONE.  SCANNING without TARGET for detect_timeout_s -> WAIT_RESULT -> DONE (hovering). Every step is driven by step() from the main loop (no threads of its own)."""

    def __init__(self, cfg, fc, perception, log_dir, say, send_command, clock=time.monotonic, geometry=None):
        self.cfg, self.fc, self.perception = cfg, fc, perception
        self.geometry = geometry  # T_CL + T_BC (plan_a.load_lidar_geometry): dat duong LiDAR qua tam vat
        self.clock = clock
        self.log_dir, self.say, self.send_command = log_dir, say, send_command
        self.state = 'WAIT_READY'
        self.launch = self.yaw0 = None
        self.speed = plan_a.SpeedWindow(1.0)
        self.t_state = self.clock()
        self.t0 = None
        self.times = {}
        self.setpoint = None  # current (north, east)
        self.corrections = 0
        self.align = None
        self.align_ok_seen = False
        self.object_line = None
        self.report = None
        self.pending = []        # TARGET / ALIGN / OBJECT lines not handled yet
        self.candidates = []     # single_shot: CANDIDATE lines (other objects, next targets), nearest first
        self.markers = []        # (phase, t, values) launch ArUco seen (evaluation only)
        self.last_marker = None  # (t, values)
        self.launch_marker_check = None
        self.marker_ground = self.marker_frame = None
        self.marker_report = None
        self.marker_ref = None        # ArUco LAN DAU trong W (co dinh): he local danh gia + moc giu vi tri
        self.marker_ref_report = None
        self.hold_offset = (0.0, 0.0)  # (dn, de) da cong vao setpoint hover theo marker
        self.hold_t = -math.inf        # lan chinh cuoi
        self.hold_note = None          # ly do khong chinh gan nhat (tranh in lap)
        self.hold_moves = 0
        self.no_target = False
        self.descend_sent = False
        self.scan_sent = False
        self.aruco_line = None   # ARUCO_S4: ArUco tren vat o anh cuoi (danh gia sai so khoang cach)   # lenh QUET (param1 = 2) da gui: perception bat dau tich luy o do cao nay
        self.confirm = None   # {'action': 'arm' | 'takeoff', 'until': t}: waits for "yes"
        self.proposal = None  # {'kind': 'approach' | 'correction' | 'descend', 'north', 'east', 'down', 'why'}

    # ---- helpers -----------------------------------------------------------------------------------
    def _go(self, state):
        logging.info('state %s -> %s', self.state, state)
        if self.proposal is not None:
            self.say(f'(Bo de xuat "{self.proposal["why"]}": trang thai doi sang {state})')
        self.proposal = None
        self.state, self.t_state = state, self.clock()

    def in_air_or_armed(self):
        """True when leaving the program could matter (unknown telemetry counts as True)."""
        try:
            return self.fc.state('armed') is not False or self.fc.state('landed_state') != 'ON_GROUND'
        except ValueError:
            return True

    def _ctl(self, key):
        return self.cfg['control'][key]

    def _elapsed(self):
        return self.clock() - self.t_state

    def _since_start(self):
        return None if self.t0 is None else self.clock() - self.t0

    def _scenario(self, p_ned):
        return plan_a.to_scenario(p_ned, self.launch, self.yaw0, self.cfg, 'ekf')

    def _position(self):
        try:
            p = self.fc.state('local_ned', 0.5)
            return p['north_m'], p['east_m'], p['down_m']
        except (ValueError, KeyError, TypeError):
            return None

    def _fly(self, north, east, why):
        self.fc.fly(north, east)
        self.setpoint = (north, east)
        x, y, _ = self._scenario((north, east, self.launch['down_m']))
        self.say(f'>> Setpoint {why}: x = {x:.3f} m, y = {y:.3f} m (NED {north:.3f}, {east:.3f})')
        self._go('APPROACH')

    # ---- perception lines ----------------------------------------------------------------------------
    def _drain(self):
        """Route new perception lines: MARKER -> marker log (evaluation), the rest -> pending."""
        for kind, values, t in self.perception.take_marked():
            if kind == 'marker':
                self._on_marker(values, t)
            else:
                self.pending.append((kind, values, t))

    def _on_marker(self, values, t):
        if values.get('seen') != 1.0:
            return
        self.last_marker = (t, values)
        self.markers.append((plan_a.MARKER_PHASES.get(self.state, 'other'), t, values))
        del self.markers[:-5000]
        if self.marker_ref is None:
            ref = plan_a.first_marker_reference([v for _, _, v in self.markers], self.cfg['marker']['min_samples'])
            if ref is not None:
                self.marker_ref = ref
                self.say(f'ArUco LAN DAU (co dinh, {ref["n"]} mau): N {ref["p_W_m"][0]:.3f}, E {ref["p_W_m"][1]:.3f}, '
                         f'D {ref["p_W_m"][2]:.3f} m (EKF), mui ten {ref["yaw_deg"]:.1f} do -> he local danh gia vat'
                         + (' + giu vi tri hover' if self.cfg['marker']['hold']['enabled'] else '')
                         + '. KHONG dua vao EKF Pixhawk; marker khong duoc chon lam vat gap.')

    def _marker_values(self, phases, since):
        return [v for ph, t, v in self.markers if ph in phases and t >= since]

    def _marker_frame_before_start(self):
        """Marker pose in W from the hover just before start_perception (fallback: any earlier samples)."""
        m, now = self.cfg['marker'], self.clock()
        for phases, since, label in ((('hover',), now - m['hover_window_s'], f'hover {m["hover_window_s"]:g} s cuoi'),
                                     (('hover',), -math.inf, 'ca luc hover'),
                                     (('takeoff', 'hover'), -math.inf, 'cat canh + hover'),
                                     (('ground',), -math.inf, 'luc dat duoi dat')):
            frame = plan_a.estimate_marker_frame(self._marker_values(phases, since))
            if frame is not None and frame['n'] >= m['min_samples']:
                frame['source'] = label
                return frame
        return None

    # ---- commands ----------------------------------------------------------------------------------
    def command(self, line):
        word = line.strip().lower()
        if not word:
            return True
        if word in ('exit', 'exit force'):
            if word == 'exit' and self.in_air_or_armed():
                raise ValueError('UAV dang ARM / dang bay: go "land" va doi disarm truoc. Chac chan muon thoat '
                                 '(dung stream, PX4 failsafe xu ly): go "exit force"')
            return False
        if word == 'check':
            self.say(self.checklist())
            return True
        if word in ('yes', 'no'):
            return self._answer(word == 'yes')
        if word == 'go':
            self._execute_proposal()
            return True
        if word == 'skip':
            self._skip_proposal()
            return True
        if word == 'land':
            self.confirm = None
            self.proposal = None
            self.fc.land()
            self.say('LAND da gui.')
            if self.state not in ('DONE', 'ABORTED'):
                self._go('ABORTED')
            return True
        if word == 'status':
            self.say(self.status_line())
            return True
        if word == 'stop':
            ok = self.send_command(self.fc, self.perception.ids, False)
            self.say('Perception STOP ' + ('da xac nhan' if ok else 'KHONG co ACK'))
            return True
        if word == 'arm':
            if self.state != 'WAIT_TAKEOFF':
                raise ValueError(f'arm chi duoc o WAIT_TAKEOFF (dang {self.state})')
            if self._ctl('confirm_arm'):
                self._ask('arm', 'ARM: motor + CANH QUAT SE QUAY (UAV van tren san). Dung xa UAV.')
                return True
            self._do_arm()
            return True
        if word == 'disarm':
            if self.state != 'WAIT_TAKEOFF':
                raise ValueError(f'disarm chi duoc truoc takeoff (dang {self.state}); dang bay thi go land')
            self.fc.disarm()
            self.say('DISARM da gui.')
            return True
        if word == 'takeoff':
            if self.state != 'WAIT_TAKEOFF':
                raise ValueError(f'takeoff chi duoc o WAIT_TAKEOFF (dang {self.state})')
            if self._ctl('require_arm_before_takeoff') and self.fc.phase != 'ARMED':
                raise ValueError('Chua ARM: go "arm" (+ "yes") truoc, roi moi "takeoff"')
            if self._ctl('confirm_takeoff'):
                self._ask('takeoff', f'TAKEOFF: bay len {self.cfg["flight"]["hover_height_m"]:g} m tai cho '
                                     '(Offboard), roi HOLD. Kiem tra xung quanh trong.')
                return True
            self._do_takeoff()
            return True
        if word in ('s_p', 'start_perception'):  # s_p: lenh ngan (start_perception van dung duoc)
            if self.state != 'WAIT_START':
                raise ValueError(f's_p chi duoc khi da hover on dinh (dang {self.state})')
            self._start_perception()
            return True
        raise ValueError(f'Lenh khong hop le: "{line.strip()}" (bam Ctrl+U xoa dong truoc khi go). '
                         'Lenh: check, arm, yes/no, takeoff, s_p, go/skip, status, imu, video, land, '
                         'stop, exit')

    # ---- confirmations -----------------------------------------------------------------------------
    def _ask(self, action, text):
        t = self.cfg['control']['confirm_timeout_s']
        self.confirm = {'action': action, 'until': self.clock() + t}
        self.say(self.checklist())
        self.say(f'>>> {text}\n>>> Go "yes" trong {t:g} s de thuc hien, "no" de huy. <<<')

    def _answer(self, yes):
        c, self.confirm = self.confirm, None
        if c is None:
            if self.proposal is not None:
                raise ValueError('Dang co de xuat bay: go "go" de bay hoac "skip" de bo qua')
            raise ValueError('Khong co gi dang cho xac nhan')
        if not yes:
            self.say(f'Huy {c["action"]}.')
            return True
        if self.clock() > c['until']:
            raise ValueError(f'Qua thoi gian xac nhan {c["action"]}: go lai lenh {c["action"]}')
        if self.state != 'WAIT_TAKEOFF':
            raise ValueError(f'{c["action"]} khong con hop le (dang {self.state})')
        if c['action'] == 'arm':
            self._do_arm()
        else:
            self._do_takeoff()
        return True

    def _do_arm(self):
        self.fc.arm()
        self.say('ARM xac nhan (UAV tren san, chua bay). PX4 tu DISARM sau ~10 s neu khong cat canh.\n'
                 '>>> Go "takeoff" de cat canh, "disarm" de huy. <<<')

    def _do_takeoff(self):
        if self._ctl('require_arm_before_takeoff') and self.fc.phase != 'ARMED':
            raise ValueError('UAV khong con ARM (PX4 tu disarm?): go "arm" lai')
        if self.cfg['marker']['enabled']:
            self._check_launch_marker()
        self.fc.takeoff()
        self.launch = dict(self.fc.launch_ned)
        self.yaw0 = self.fc.target['yaw_deg']
        self.say(f'TAKEOFF: goc (0,0,0) = EKF NED ({self.launch["north_m"]:.3f}, {self.launch["east_m"]:.3f}, '
                 f'{self.launch["down_m"]:.3f}) m, yaw luc cat canh {self.yaw0:.1f} do. '
                 f'Bay len (0, 0, -{self.cfg["flight"]["hover_height_m"]:g}).')
        self._go('TAKEOFF')

    def checklist(self):
        def get(key, age=3.0):
            try:
                return self.fc.state(key, age)
            except ValueError:
                return 'KHONG CO'
        try:
            health = 'DAT' if self.fc.preflight_ok() else 'CHUA DAT'
        except ValueError:
            health = 'KHONG CO'
        try:
            link = 'CO' if self.fc.drone.is_connected() else 'MAT'
        except Exception:  # noqa: BLE001
            link = '?'
        p = self._position()
        per = self.perception.status()
        lines = ['-' * 72, ' KIEM TRA', '-' * 72,
                 f' Ket noi Pixhawk : {link}   | pha dieu khien: {self.fc.phase} | kich ban: {self.state}',
                 f' Armed / landed  : {get("armed")} / {get("landed_state")} | che do bay: {get("flight_mode")}',
                 f' Suc khoe PX4    : {health}',
                 ' Vi tri EKF      : ' + ('KHONG CO' if p is None else f'N {p[0]:.3f} E {p[1]:.3f} D {p[2]:.3f} m'),
                 f' Perception      : {per["state"]}' + (f' (loi: {per["last_error"]})' if per.get('last_error') else ''),
                 f' Do cao hover    : {self.cfg["flight"]["hover_height_m"]:g} m | toi da cach goc '
                 f'{self.cfg["flight"]["max_target_distance_m"]:g} m',
                 '-' * 72]
        return '\n'.join(lines)

    # ---- proposed setpoints (need "go") ----------------------------------------------------------------
    def _propose(self, kind, north, east, why, down=None):
        if not self._ctl('confirm_moves'):
            self.proposal = {'kind': kind, 'north': north, 'east': east, 'down': down, 'why': why}
            self._execute_proposal()
            return
        self.proposal = {'kind': kind, 'north': north, 'east': east, 'down': down, 'why': why}
        p = self._position()
        dist = '' if p is None else f', cach vi tri hien tai {math.hypot(north - p[0], east - p[1]):.2f} m ngang'
        x, y, _ = self._scenario((north, east, self.launch['down_m']))
        h = '' if down is None else f', do cao {self.launch["down_m"] - down:.2f} m'
        self.say(f'>>> DE XUAT {why.upper()}: toi x = {x:.3f} m, y = {y:.3f} m{h}{dist}.\n'
                 '>>> Go "go" de bay, "skip" de bo qua (UAV giu HOLD). <<<')

    def _execute_proposal(self):
        pr = self.proposal
        if pr is None:
            raise ValueError('Khong co de xuat bay nao')
        if not self.fc.ready_for_fly():
            raise ValueError(f'UAV chua HOLD (pha {self.fc.phase}): cho roi go "go" lai')
        limit = self.cfg['flight']['max_target_distance_m']
        if math.hypot(pr['north'] - self.launch['north_m'], pr['east'] - self.launch['east_m']) > limit:
            self.proposal = None
            raise ValueError(f'Diem de xuat xa goc hon max_target_distance_m = {limit:g} m: bo')
        self.proposal = None
        if pr['kind'] == 'descend':
            self.fc.fly(pr['north'], pr['east'], down=pr['down'])
            self.descend_sent = True
            self.t_state = self.clock()
            self.say(f'>> Ha do cao: setpoint down {pr["down"]:.3f} m (NED)')
            return
        self._fly(pr['north'], pr['east'], pr['why'])

    def _skip_proposal(self):
        pr, self.proposal = self.proposal, None
        if pr is None:
            raise ValueError('Khong co de xuat bay nao')
        self.say(f'Bo qua "{pr["why"]}". UAV giu HOLD.')
        if pr['kind'] == 'descend':
            self.say('Khong ha: quet vat o do cao hien tai.')
            self._start_scan()

    def _start_perception(self):
        state = self.perception.status()['state']
        if state != 'WAITING' or not self.fc.ready_for_fly():
            raise ValueError(f'perception {state} / UAV chua HOLD: chua bat dau duoc')
        self._drain()
        self.pending.clear()  # bo dong TARGET / ALIGN / OBJECT cu (MARKER da luu)
        self.candidates = []
        if self.cfg['marker']['enabled']:
            self.marker_frame = self._marker_frame_before_start()
            if self.marker_frame is None:
                self.say('ArUco diem cat canh: KHONG du mau khi hover -> chi co sai so theo EKF.')
            else:
                f = self.marker_frame
                self.say(f'ArUco diem cat canh trong EKF ({f["source"]}, {f["n"]} mau): N {f["p_W_m"][0]:.3f}, '
                         f'E {f["p_W_m"][1]:.3f}, D {f["p_W_m"][2]:.3f} m, mui ten {f["yaw_deg"]:.1f} do '
                         f'(trai {f["spread_m"] * 100:.1f} cm). Chi de danh gia.')
                change = plan_a.frame_change(self.marker_ground, f)
                if change is not None:
                    self.say(f'EKF TROI (marker luc dat -> hover): {change["distance_m"] * 100:.1f} cm '
                             f'(ngang {change["horizontal_m"] * 100:.1f} cm), yaw {change["yaw_deg"]:+.1f} do')
        floor = self.launch['down_m'] + self.cfg['frame']['fc_above_ground_m']  # san duoi UAV luc cat canh (EKF)
        if not self.send_command(self.fc, self.perception.ids, True, floor_down_m=floor):
            raise ValueError('Khong co ACK tu uavp_fc_bridge (router / bridge?)')
        p = self._position()
        if p is not None:
            self.say(f'Muc san EKF z = {floor:.3f} m gui kem lenh: camera ~{floor - p[2]:.2f} m tren san -> buoc detect '
                     'doi bbox ra mm, loai vat qua to (s3_select.bbox_max_mm).')
        self.t0 = self.clock()
        self.say('=' * 72 + '\n PERCEPTION BAT DAU (t = 0): detect -> chon vat gan tam anh nhat\n'
                 ' Xem truc tiep tren laptop: python tools/remote_viewer.py --host <ip-pi> --port 8100\n'
                 + '=' * 72)
        self._go('SCANNING')

    def _marker_hold(self):
        """Giu UAV tren san theo marker khi hover tai cho: EKF troi -> doi setpoint hover (buoc nho, co gioi han)."""
        m = self.cfg['marker']
        h = m['hold']
        if (not m['enabled'] or not h['enabled'] or self.marker_ref is None or self.launch is None
                or self.state not in plan_a.HOLD_STATES or self.proposal is not None or self.confirm is not None):
            return
        now = self.clock()
        if now - self.hold_t < h['period_s'] or not self.fc.ready_for_fly():
            return
        if self.hold_moves > 0 and not self.fc.target_reached:  # lan chinh truoc chua toi noi
            return
        recent = [v for _, t, v in self.markers if t >= now - h['window_s']]
        cur = plan_a.estimate_marker_frame(recent)
        if cur is None or cur['n'] < h['min_samples'] or cur['spread_m'] > h['max_spread_m']:
            return  # it mau / mau tan: khong tin
        new, why = plan_a.hold_step(self.marker_ref, cur, self.hold_offset, h)
        if new is None:
            if why != self.hold_note and 'deadband' not in why:
                self.say(f'Giu vi tri theo ArUco: {why}')
            self.hold_note = why
            return
        n, e = self.launch['north_m'] + new[0], self.launch['east_m'] + new[1]
        self.hold_t = now
        try:
            self.fc.fly(n, e)
        except Exception as error:  # noqa: BLE001 - controller refuses: report, keep HOLD
            self.say(f'Giu vi tri theo ArUco: khong gui duoc setpoint ({error})')
            return
        self.hold_offset = new
        self.hold_moves += 1
        self.hold_note = why
        self.say(f'Giu vi tri theo ArUco: {why} -> setpoint hover NED ({n:.3f}, {e:.3f}) '
                 f'(lech goc {math.hypot(*new) * 100:.1f} cm)')

    def _check_launch_marker(self):
        m = self.cfg['marker']
        self._drain()
        recent = self._marker_values(('ground',), self.clock() - m['launch_window_s'])
        self.launch_marker_check = plan_a.launch_check(recent, self.cfg)
        self.marker_ground = plan_a.estimate_marker_frame(recent)
        self.say(plan_a.format_launch_check(self.launch_marker_check, self.cfg))
        if m['require_at_takeoff'] and not self.launch_marker_check['ok']:
            raise ValueError('ArUco diem cat canh chua dat (marker.require_at_takeoff: true): dat lai UAV '
                             'roi go takeoff, hoac dat require_at_takeoff: false')

    def status_line(self):
        p = self._position()
        where = ''
        if p is not None and self.launch is not None:
            x, y, z = self._scenario(p)
            where = f' | vi tri x {x:.3f} y {y:.3f} z {z:.3f} m'
        sp = self.speed.speed()
        return (f'[{self.state}] FC {self.fc.phase}{where} | toc do {sp if sp is None else round(sp, 3)} m/s | '
                f'perception {self.perception.status()["state"]}'
                + ('' if self.align is None else f' | ALIGN e_mm {self.align.get("e_mm")} ok {self.align.get("ok")}')
                + self._marker_status())

    def _marker_status(self):
        if not self.cfg['marker']['enabled']:
            return ''
        if self.last_marker is None or self.clock() - self.last_marker[0] > 1.0:
            return ' | ArUco: khong thay'
        c = self.last_marker[1].get('p_cam_S_mm')
        if not isinstance(c, tuple) or len(c) != 3:
            return ' | ArUco: thay'
        return f' | ArUco: camera x {c[0] / 1000:.3f} y {c[1] / 1000:.3f} cao {-c[2] / 1000:.3f} m (he marker)'

    # ---- perception lines --------------------------------------------------------------------------
    def _perception_lines(self):
        self._drain()
        items, self.pending = self.pending, []
        for kind, values, _ in items:
            dt = self._since_start()
            if kind == 'target' and 'to_target_s' not in self.times and dt is not None:
                self.times['to_target_s'] = dt
                p, label = self._target_point(values)
                x, y, z = plan_a.to_scenario([v / 1000 for v in p], self.launch, self.yaw0, self.cfg)
                self.say(f'[t = {dt:.2f} s] DETECT + KHOA VAT: {label} -> he the gioi x = {x:.3f} m, y = {y:.3f} m, '
                         f'z = {z:.3f} m (anh: detect.jpg)')
                self._on_target(p)
            elif kind == 'align':
                self.align = values
                if values.get('ok') == 1.0 and not self.align_ok_seen and dt is not None:
                    self.align_ok_seen = True
                    self.times['to_aligned_s'] = dt
                    self.say(f'[t = {dt:.2f} s] LiDAR QUET QUA VAT: e_px = {values.get("e_px")} px, '
                             f'e_mm = {values.get("e_mm")} <= {values.get("e_max_mm")} mm (d = {values.get("d_mm")} mm)'
                             ' -> segment + tich luy duong quet (anh: aligned.jpg)')
            elif kind == 'candidate':
                self._on_candidate(values)
            elif kind == 'aruco':
                self.aruco_line = values
            elif kind == 'object':
                self.object_line = values
                if dt is not None:
                    self.say(f'[t = {dt:.2f} s] PERCEPTION XAC DINH TAM + HUONG: mode {values.get("mode")}, '
                             f'yaw_W {values.get("yaw_W_deg")} do')

    def _on_candidate(self, values):
        """single_shot: another object seen in the s1 frame -> remember it (scenario frame) as a next target."""
        item = {'rank': values.get('rank'), 'id': values.get('id'), 'class': values.get('class'),
                'conf': values.get('conf'), 'p_W_mm': values.get('p_W_mm'), 'scenario_m': None}
        p = values.get('p_W_mm')
        if values.get('world') == 1.0 and isinstance(p, tuple) and plan_a._finite3(p) and self.launch is not None:
            item['scenario_m'] = plan_a.to_scenario([v / 1000 for v in p], self.launch, self.yaw0, self.cfg)
        self.candidates.append(item)
        where = ('khong co tu the FC' if item['scenario_m'] is None
                 else f'x = {item["scenario_m"][0]:.3f} m, y = {item["scenario_m"][1]:.3f} m (diem tren san)')
        self.say(f'Vat tiep theo #{item["rank"]}: {item["class"]} conf {item["conf"]} -> {where}')

    def _target_point(self, values):
        """Point to fly to from a TARGET line: bbox centre (approach.target: bbox) or mask centre."""
        bbox = values.get('bbox_p_W_mm')
        if self.cfg['approach']['target'] == 'bbox':
            if isinstance(bbox, tuple) and plan_a._finite3(bbox):
                return list(bbox), 'tam bbox'
            self.say('TARGET khong co bbox_p_W_mm (perception cu?): dung tam mask.')
        return list(values['p_W_mm']), 'tam mask'

    def _line_offset(self, p_W_mm):
        """Body-frame offset of the LiDAR foot point at the target depth, or None (body over the target)."""
        if not self.cfg['approach']['lidar_line'] or self.geometry is None:
            return None
        pos = self._position()
        d = plan_a.camera_depth_to(p_W_mm, None if pos is None else pos[2], self.geometry)
        if d is None or d < 0.1:
            d = self.cfg['flight']['hover_height_m']
            self.say(f'Khong tinh duoc do sau camera -> vat: dung hover_height_m = {d:g} m.')
        final = self.cfg['approach']['final_height_m']
        if final is not None and pos is not None and self.launch is not None:
            drop = (self.launch['down_m'] - final) - pos[2]  # NED: > 0 = se ha xuong truoc khi quet
            if 0.0 < drop < d - 0.1:
                self.say(f'Quet o {final:g} m: do sau camera -> vat {d:.3f} -> {d - drop:.3f} m (tinh duong LiDAR).')
                d -= drop
        off = plan_a.lidar_foot_body(self.geometry, d)
        if off is None:
            self.say('Mat quet LiDAR khong cat mat phang duoi camera: dat than UAV tren vat.')
        else:
            self.say(f'Duong LiDAR o do sau {d:.3f} m lech than UAV x {off[0] * 100:+.1f} cm, y {off[1] * 100:+.1f} cm '
                     '-> bay sao cho duong quet cat qua tam vat.')
        return off

    def _on_target(self, p_W_mm):
        if self.state != 'SCANNING':
            return
        try:
            (n, e), _ = plan_a.approach_setpoint(p_W_mm, self.launch, self.yaw0, self.cfg,
                                                 self._line_offset(p_W_mm))
        except ValueError as error:
            self.say(f'KHONG bay toi vat: {error}. UAV giu HOLD; perception van chay.')
            return
        try:
            self._propose('approach', n, e, 'bay toi vat')
        except Exception as error:  # noqa: BLE001 - controller refuses: report and keep HOLD
            logging.exception('fly failed')
            self.say(f'Loi gui setpoint: {error}')

    # ---- main step ---------------------------------------------------------------------------------
    def step(self):
        now = self.clock()
        p = self._position()
        if p is not None:
            self.speed.add(now, p)
        self._drain()
        status = self.perception.status()
        if self.state == 'WAIT_READY':
            if status['state'] == 'WAITING':
                self.say('Perception READY.\n>>> Go "check" de kiem tra, "arm" (+ "yes") roi "takeoff" (+ "yes"). '
                         'Chuong trinh KHONG tu bay. <<<')
                self._go('WAIT_TAKEOFF')
            elif status['state'] == 'FAILED':
                self.say(f'Perception FAILED: {status["last_error"]}. Sua loi roi chay lai.')
                self._go('ABORTED')
            elif self._elapsed() > self.cfg['perception']['ready_timeout_s']:
                self.say('Perception chua READY sau ready_timeout_s: xem log perception-*.log.')
                self._go('ABORTED')
            return
        if self.state in ('TAKEOFF', 'APPROACH', 'ALIGN', 'STABILIZE', 'WAIT_START', 'SCANNING', 'DESCEND') \
                and self.fc.phase in ABORT_PHASES:
            self.say(f'UAV o pha {self.fc.phase} ({self.fc.last_error}): dung kich ban.')
            self._go('ABORTED')
            return
        if self.state in ('SCANNING', 'APPROACH', 'DESCEND', 'ALIGN'):
            self._perception_lines()
            if self._finish_if_result():
                return
            limit = self.cfg['perception']['detect_timeout_s']
            if self.state == 'SCANNING' and 'to_target_s' not in self.times and self._since_start() > limit:
                self.no_target = True
                self.say(f'KHONG THAY VAT sau {limit:g} s (detect_timeout_s): gui STOP perception, UAV GIU HOVER.')
                self.send_command(self.fc, self.perception.ids, False)
                self._go('WAIT_RESULT')
                return
            if self._since_start() > self.cfg['perception']['result_timeout_s']:
                self.say('Het result_timeout_s: gui STOP perception.')
                self.send_command(self.fc, self.perception.ids, False)
                self._go('WAIT_RESULT')
                return
        self._marker_hold()
        if self.state == 'TAKEOFF':
            if self.fc.phase == 'HOLD' and self.fc.takeoff_complete:
                self.say(f'Da toi do cao hover; cho on dinh {self.cfg["flight"]["hover_stabilize_s"]:g} s ...')
                self._go('STABILIZE')
        elif self.state == 'STABILIZE':
            sp = self.speed.speed()
            if (self._elapsed() >= self.cfg['flight']['hover_stabilize_s'] and sp is not None and p is not None
                    and sp < self.cfg['flight']['hover_max_speed_m_s']):
                x, y, z = self._scenario(p)
                auto = self.cfg['perception']['auto_start']
                self.say(f'HOVER ON DINH tai x = {x:.3f}, y = {y:.3f}, z = {z:.3f} m (toc do {sp:.3f} m/s).'
                         + ('' if auto else '\n>>> Go "s_p" de bat dau perception. <<<'))
                self._go('WAIT_START')
                if auto:
                    try:
                        self._start_perception()
                    except ValueError as error:
                        self.say(f'Tu start_perception KHONG duoc: {error}\n>>> Go "s_p" thu lai. <<<')
        elif self.state == 'APPROACH':
            if self.fc.phase == 'HOLD' and self.fc.target_reached:
                self.times.setdefault('arrived_s', self._since_start())
                x, y, z = self._scenario(p) if p else (math.nan,) * 3
                if self.cfg['approach']['final_height_m'] is not None and not self.descend_sent:
                    self.say(f'Toi vi tri tren vat: x = {x:.3f}, y = {y:.3f}, z = {z:.3f} m. '
                             f'HA xuong {self.cfg["approach"]["final_height_m"]:g} m truoc khi quet ...')
                    self._go('DESCEND')
                else:
                    self.say(f'Toi vi tri tren vat: x = {x:.3f}, y = {y:.3f}, z = {z:.3f} m. Xet LiDAR quet qua vat ...')
                    self._start_scan()
        elif self.state == 'ALIGN':
            self._align_step()
        elif self.state == 'DESCEND':
            self._descend_step(p)
        elif self.state == 'WAIT_RESULT':
            self._perception_lines()
            if not self._finish_if_result() and self._elapsed() > 10:
                self.say('Perception khong tra ket qua sau STOP.')
                self._finish(None)

    def _align_step(self):
        a = self.cfg['approach']
        if self.proposal is not None:
            return  # cho nguoi dieu khien go / skip
        if self.align_ok_seen or self._elapsed() < a['wait_after_arrival_s'] + a['align_wait_s']:
            return
        if self.align is None or self.align.get('ok') == 1.0:
            return
        if self.corrections >= a['max_corrections']:
            if self.corrections == a['max_corrections']:
                self.say(f'Da chinh {self.corrections} lan, LiDAR van chua quet qua vat (e_mm {self.align.get("e_mm")}).'
                         ' Giu HOLD, cho perception (hoac go land / stop).')
                self.corrections += 1
            return
        try:
            target = plan_a.correction_setpoint(self.setpoint, self.align.get('dW_mm'), self.launch, self.cfg)
        except ValueError as error:
            self.say(f'Khong chinh: {error}')
            self.corrections = a['max_corrections']
            return
        if target is None:
            self.t_state = self.clock()  # lech nho / chua co vector: cho them
            return
        self.corrections += 1
        self.say(f'LiDAR chua quet qua vat: e_mm = {self.align.get("e_mm")} > {self.align.get("e_max_mm")} mm. '
                 f'Chinh lan {self.corrections}/{a["max_corrections"]}.')
        self._propose('correction', target[0], target[1], f'chinh lan {self.corrections}')

    def _finish_if_result(self):
        status = self.perception.status()
        if status['last_result'] is None:
            if status['state'] in ('FAILED', 'STOPPED') and status['exit_code'] is not None:
                self.say(f'Perception ket thuc khong co ket qua (ma {status["exit_code"]}): '
                         f'{status["last_error"] or "bi dung / khong chon duoc vat"}')
                self._finish(None)
                return True
            return False
        self.times['to_result_s'] = self._since_start()
        self._perception_lines()
        self._finish(status['last_result'])
        return True

    def _finish(self, result_dir):
        result_json = None
        if result_dir is not None:
            run_dir = Path(result_dir)
            if not run_dir.is_absolute():
                run_dir = self.perception.dir / run_dir
            result_json = copy_run(run_dir, self.log_dir / 'perception', self.say)
        obj = plan_a.object_from(result_json, self.object_line)
        self.report = plan_a.evaluate(obj, self.launch, self.yaw0, self.cfg)
        self.say(plan_a.format_report(self.report, self.cfg, self.times))
        self.aruco_report = plan_a.evaluate_aruco_distance(self.aruco_line)
        self.say(plan_a.format_aruco_distance(self.aruco_report) + '\n' + '=' * 72)
        summary = {'report': self.report, 'object_W': obj, 'times_s': self.times, 'launch_ned': self.launch,
                   'yaw0_deg': self.yaw0, 'corrections': self.corrections, 'last_align': self.align,
                   'perception_run': result_dir, 'no_target_timeout': self.no_target, 'config': self.cfg,
                   'aruco_object_distance': self.aruco_report, 'candidates': self.candidates}
        if self.cfg['marker']['enabled']:
            t_BC = None if self.geometry is None else self.geometry['t_BC']
            self.marker_report = plan_a.evaluate_marker(obj, self.launch, self.yaw0, self.launch_marker_check, t_BC,
                                                        self.cfg)
            self.say(plan_a.format_marker_report(self.marker_report, self.launch_marker_check, self.marker_ground,
                                                 self.marker_frame, self.cfg) + '\n' + '=' * 72)
            self.marker_ref_report = plan_a.evaluate_marker_ref(obj, self.marker_ref, self.cfg)
            self.say(plan_a.format_marker_ref_report(self.marker_ref_report, self.cfg) + '\n' + '=' * 72)
            summary['marker_first'] = {'report': self.marker_ref_report, 'hold_offset_m': self.hold_offset,
                                       'hold_moves': self.hold_moves}
            summary['marker'] = {'report': self.marker_report, 'launch_check': self.launch_marker_check,
                                 'frame_ground': self.marker_ground, 'frame_hover': self.marker_frame,
                                 'ekf_drift_ground_to_hover': plan_a.frame_change(self.marker_ground,
                                                                                  self.marker_frame),
                                 'samples': [{'phase': ph, 't': t, **v} for ph, t, v in self.markers]}
        (self.log_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
        self.say(f'Luu: {self.log_dir} (summary.json, plan_a.log, perception/*.jpg)')
        self.perception.close()  # 1 kich ban = 1 lan quet; khong khoi dong lai perception
        self._end()

    def _end(self):
        if self.cfg['flight']['auto_land']:
            self.fc.land()
            self.say('auto_land: LAND da gui.')
        else:
            self.say('>>> UAV dang HOLD. Go "land" de ha canh. <<<')
        self._go('DONE')

    def _descend_step(self, p):
        """Same (north, east), down = takeoff FC z - final_height_m (height counted like hover_height_m)."""
        h = self.cfg['approach']['final_height_m']
        if not self.descend_sent:
            if self.proposal is not None:
                return  # cho "go" / "skip"
            if not self.fc.ready_for_fly():
                if self._elapsed() > self.cfg['flight']['target_timeout_s']:
                    self.say('UAV chua HOLD de ha do cao: bo buoc ha, quet o do cao hien tai.')
                    self._start_scan()
                return
            n, e = self.setpoint if self.setpoint is not None else (p[0], p[1]) if p else (None, None)
            if n is None:
                self.say('Khong co vi tri de ha do cao: bo buoc ha, quet o do cao hien tai.')
                self._start_scan()
                return
            try:
                self._propose('descend', n, e, f'ha do cao xuong {h:g} m de quet vat', down=self.launch['down_m'] - h)
            except Exception as error:  # noqa: BLE001 - controller refuses: report and keep HOLD
                logging.exception('descend failed')
                self.say(f'Loi gui setpoint ha do cao: {error}. Quet o do cao hien tai.')
                self._start_scan()
            return
        if self.fc.phase == 'HOLD' and self.fc.target_reached:
            z = self._scenario(p)[2] if p else math.nan
            fc_h = (self.launch['down_m'] - p[2]) if p else math.nan
            self.say(f'DA HA DO CAO: Pixhawk cao {fc_h:.3f} m so voi luc cat canh (muc tieu {h:g}), '
                     f'{-z:.3f} m so voi san. Bat dau QUET vat o do cao nay.')
            self.times['descend_done_s'] = self._since_start()
            self._start_scan()
        elif self._elapsed() > self.cfg['flight']['target_timeout_s'] + 5:
            self.say('Ha do cao qua lau (chua toi dich): quet o vi tri hien tai; go "land" khi can.')
            self._start_scan()

    def _start_scan(self):
        """Lenh QUET (param1 = 2): perception bo khung chup truoc do, tich luy LiDAR o do cao nay (cua so 2 -> max_window_s
        toi khi ra vector huong), roi tra ket qua. Sau do: ALIGN (chinh de duong LiDAR cat qua vat)."""
        if not self.scan_sent:
            self.scan_sent = True
            ok = self.send_command(self.fc, self.perception.ids, 'scan')
            self.times['scan_start_s'] = self._since_start()
            p = self.cfg['perception']
            self.say(f'Lenh QUET {"da xac nhan" if ok else "KHONG co ACK (perception se ket thuc theo lock_max_duration_s)"}'
                     f': tich luy duong quet LiDAR, cua so {p["window_s"]:g} -> toi da {p["max_window_s"]:g} s '
                     'de ra vector huong.')
        self._go('ALIGN')


class VideoDiagCollector:
    """Video vibration diagnosis: remembers every perception run folder (one per perception start), then at the end
    of the run joins their video_diag.csv into <log>/video_diag.csv and writes video_diag_report.txt / .json / .png."""

    def __init__(self, perception, log_dir, say, enabled, imu=None, imu_options=None):
        self.perception, self.log_dir, self.say, self.enabled = perception, log_dir, say, enabled
        self.imu, self.imu_options = imu, imu_options or {}  # ImuLogger (pi/imu_logger.py) or None
        self.dirs = []
        self.done = False

    def poll(self):
        run_dir = self.perception.run_dir
        if run_dir and run_dir not in self.dirs:
            self.dirs.append(run_dir)

    def _paths(self):
        for d in self.dirs:
            path = Path(d)
            yield (path if path.is_absolute() else self.perception.dir / path) / 'video_diag.csv'

    def collect(self):
        """Join the csv files into the log folder; returns its path or None."""
        out = self.log_dir / 'video_diag.csv'
        header, rows = None, []
        for path in self._paths():
            try:
                lines = path.read_text().splitlines(keepends=True)
            except OSError:
                continue
            if lines:
                header = header or lines[0]
                rows += [line for line in lines[1:] if line.strip()]
        if header is None:
            return None
        out.write_text(header + ''.join(rows))
        return out

    def report(self, final=False):
        self.poll()
        csv_path = self.collect()
        if csv_path is None:
            self.say('Chan doan video: chua co video_diag.csv (perception chua chay / video_diag tat).')
            return None
        try:
            rep, text = video_diag.analyze_and_save(csv_path, self.log_dir)
        except Exception as error:  # noqa: BLE001 - diagnosis must never block landing / exit
            logging.exception('video diagnosis failed')
            self.say(f'Chan doan video loi: {error}')
            return None
        self.say(text)
        self.say(f'Luu chan doan video: {self.log_dir}/video_diag_report.txt, .json'
                 + (', video_diag.png' if rep.get('plot') else '') + ', video_diag.csv'
                 + ('' if final else ' (tam thoi; ban cuoi ghi luc thoat)'))
        return rep

    def imu_report(self, final=False):
        """IMU vibration report (uses video_diag.csv of the log folder for the sharpness correlation)."""
        if self.imu is None:
            self.say('Chan doan IMU: imu_log tat.')
            return None
        try:
            self.imu.flush()
            rep, text = imu_diag.analyze_and_save(self.log_dir, self.imu_options)
        except Exception as error:  # noqa: BLE001 - diagnosis must never block landing / exit
            logging.exception('IMU diagnosis failed')
            self.say(f'Chan doan IMU loi: {error}')
            return None
        self.say(text)
        if rep is not None:
            self.say(f'Luu chan doan IMU: {self.log_dir}/imu_diag_report.txt, .json'
                     + (', imu_diag.png' if rep.get('plot') else '') + ', imu_*.csv'
                     + ('' if final else ' (tam thoi; ban cuoi ghi luc thoat)'))
        return rep

    def finalize(self):
        """End of the run (exit, Ctrl+C, error): stop the IMU logger and perception so the csv files are complete,
        then analyse once (video first: the IMU report correlates with video_diag.csv)."""
        if self.done:
            return
        self.done = True
        try:
            if self.imu is not None:
                self.imu.stop()
        except Exception:  # noqa: BLE001
            logging.exception('IMU logger stop failed')
        try:
            self.poll()
            self.perception.close()
            if self.enabled:
                self.report(final=True)
            if self.imu is not None:
                self.imu_report(final=True)
        except Exception:  # noqa: BLE001
            logging.exception('vibration diagnosis finalize failed')


def start_imu_logger(cfg, fc, log_dir, say):
    """Pixhawk IMU / vibration logger on the console's MAVSDK link (telemetry rates only, no flight command)."""
    c = cfg['imu_log']
    if not c['enabled']:
        return None
    from imu_logger import ImuLogger
    from pixhawk import read_param

    def param(name, kind):
        return read_param(fc.param.get_param_float if kind == 'float' else fc.param.get_param_int, name)

    try:
        logger = ImuLogger(fc.mavlink_direct, log_dir, fc.drone.get_system_id(),
                           {'HIGHRES_IMU': c['highres_hz'], 'VIBRATION': c['vibration_hz'],
                            'SERVO_OUTPUT_RAW': c['motors_hz']}, params=param)
        logger.start()
    except Exception as error:  # noqa: BLE001 - logging must never stop the flight scenario
        logging.exception('IMU logger start failed')
        say(f'Canh bao: khong ghi duoc IMU: {error}')
        return None
    say(f'Ghi IMU Pixhawk: HIGHRES_IMU {c["highres_hz"]:g} Hz, VIBRATION {c["vibration_hz"]:g} Hz, motor '
        f'{c["motors_hz"]:g} Hz -> {log_dir}/imu_*.csv | PX4: {logger.meta.get("params")}')
    return logger


def copy_run(src, dst, say):
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in COPY_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
            copied.append(name)
    say(f'Chep tu {src}: {", ".join(copied) or "(khong co file)"}')
    try:
        return json.loads((src / 'result.json').read_text())
    except (OSError, ValueError) as error:
        say(f'Khong doc duoc result.json: {error}')
        return None


def make_say(logger):
    def say(text):
        print(text, flush=True)
        for line in str(text).splitlines():
            logger.info('%s', line)
    return say


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent / 'config' / 'scenario.yaml')
    args = parser.parse_args()
    cfg = plan_a.load_config(args.config)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_root = Path(cfg['logging']['dir'])
    log_dir = (log_root if log_root.is_absolute() else ROOT / log_root) / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / 'plan_a.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s', force=True)
    say = make_say(logging.getLogger('planA'))
    say(f'planA: config {args.config} | log {log_dir}')
    (log_dir / 'scenario.yaml').write_text(args.config.read_text())

    # Controller modules (pi/) are imported only here: tests of plan_a.py need no MAVSDK.
    from console import load_site_origin, pin_control_cores, poll_controller
    from deployment import validate_connection
    from flight_policy import FlightPolicy
    from offboard_control import OffboardControl
    from perception_process import PerceptionProcess, send_perception_command
    from serial_lock import serial_ownership, udp_ownership

    f = cfg['flight']
    device = validate_connection(f['profile'], f['connection'])
    if not device.startswith('udpin://'):
        raise SystemExit('flight.connection must be udpin:// (mavlink-routerd shares TELEM1 with perception)')
    ekf_origin = None
    if f['position_source'] == 'flow':
        origin_file = Path(f['site_origin_file'])
        ekf_origin = load_site_origin(origin_file if origin_file.is_absolute() else ROOT / origin_file)
        if ekf_origin is None:
            raise SystemExit(f'flow needs {origin_file} (see pi/console.py --site-origin-file)')
    policy = FlightPolicy(xy_tolerance_m=f['xy_tolerance_m'], z_tolerance_m=f['z_tolerance_m'],
                          yaw_tolerance_deg=f['yaw_tolerance_deg'], settle_s=f['settle_time_s'],
                          target_timeout_s=f['target_timeout_s'])
    geometry = None
    if cfg['approach']['lidar_line'] or cfg['marker']['enabled']:  # marker: T_BC dua thuoc ve he kich ban
        try:
            geometry = plan_a.load_lidar_geometry(cfg['perception']['dir'], cfg)
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise SystemExit(f'approach.lidar_line: khong doc duoc T_CL / T_BC trong perception.dir: {error}')
        say(f'Duong LiDAR: T_CL {cfg["approach"]["calib_T_CL"]}, T_BC {cfg["approach"]["calib_T_BC"]} '
            f'(lech than UAV tai do sau {cfg["flight"]["hover_height_m"]:g} m: '
            f'{plan_a.lidar_foot_body(geometry, cfg["flight"]["hover_height_m"])})')
    overrides = plan_a.perception_overrides(cfg)
    perception = PerceptionProcess(Path(cfg['perception']['dir']).expanduser(), cfg['perception']['config'],
                                   log_dir=log_dir, overrides=overrides)
    say('Perception --set: ' + ' '.join(overrides))
    try:
        cores = pin_control_cores(f['control_cores'])
        if cores:
            say(f'Dieu khien chay tren CPU {cores}')
    except (OSError, ValueError) as error:
        say(f'Canh bao: khong ghim duoc CPU: {error}')

    with ExitStack() as stack:
        fault_log = stack.enter_context(open(log_dir / 'fault.log', 'a'))
        faulthandler.enable(file=fault_log, all_threads=True)
        stack.callback(faulthandler.disable)
        stack.enter_context((udp_ownership if device.startswith('udpin://') else serial_ownership)(device))
        fc = OffboardControl(f['connection'], f['profile'], True, f['system_id'], f['hover_height_m'],
                             policy=policy, position_source=f['position_source'], ekf_origin=ekf_origin)
        stack.callback(fc.close)
        stack.callback(perception.close)  # registered after fc: closed first
        imu = start_imu_logger(cfg, fc, log_dir, say)
        videos = VideoDiagCollector(perception, log_dir, say, cfg['video_diag']['enabled'], imu,
                                    {'camera_f_px': cfg['imu_log']['camera_f_px'],
                                     'readout_ms': cfg['imu_log']['readout_ms']})
        stack.callback(videos.finalize)  # runs first at exit: closes perception, then writes the report
        perception.start()
        say(f'Perception dang khoi dong (log {perception.status()["log"]}) ...')
        plan = PlanA(cfg, fc, perception, log_dir, say,
                     lambda fc_, ids, start, **kw: send_perception_command(fc_, ids, start=start, **kw),
                     geometry=geometry)
        logger = logging.getLogger('planA')
        last_status = 0.0
        hangup = []  # SSH dropped: SIGHUP would kill the program mid-air -> LAND instead
        signal.signal(signal.SIGHUP, lambda *_: hangup.append(time.monotonic()))
        terminal_lost = False
        last_interrupt = -math.inf
        say('AN TOAN: chuong trinh KHONG tu arm / cat canh / bay. Moi buoc can lenh cua ban '
            '(arm + yes, takeoff + yes, go cho tung diem bay). "land" luon chay ngay.')
        while True:
            try:
                if (hangup or terminal_lost) and not getattr(plan, 'terminal_landing', False):
                    plan.terminal_landing = True
                    if plan.in_air_or_armed():
                        say('MAT TERMINAL (SSH roi?) khi UAV dang ARM / bay -> gui LAND.')
                        try:
                            plan.command('land')
                        except Exception:  # noqa: BLE001
                            logger.exception('LAND after terminal loss failed')
                if terminal_lost and not plan.in_air_or_armed():
                    say('Mat terminal, UAV da tren san va disarm: thoat.')
                    break
                ready = [] if terminal_lost else select.select([sys.stdin], [], [], 0.1)[0]
                if terminal_lost:
                    time.sleep(0.1)
                poll_controller(fc, logger)
                if plan.state not in ('DONE', 'ABORTED'):
                    perception.poll()
                for event in perception.take_events():
                    say(f'[perception] {event}')
                videos.poll()
                if imu is not None:
                    imu.poll()
                plan.step()
                if time.monotonic() - last_status >= 1.0:
                    logger.info('status %s', plan.status_line())
                    last_status = time.monotonic()
                if not ready:
                    continue
                line = sys.stdin.readline()
                if line == '':
                    terminal_lost = True  # stdin closed: no operator any more
                    continue
            except KeyboardInterrupt:
                now = time.monotonic()
                if plan.in_air_or_armed() and now - last_interrupt > 3.0:
                    last_interrupt = now
                    say('\nCtrl+C: UAV dang ARM / bay -> KHONG thoat. Go "land". '
                        'Bam Ctrl+C lan nua trong 3 s de thoat han (dung stream, PX4 failsafe xu ly).')
                    continue
                say('Thoat: dung stream setpoint, KHONG gui LAND. PX4 failsafe se xu ly.')
                break
            try:
                logger.info('command %r', line.strip())
                if line.strip().lower() == 'video':
                    videos.report()
                    continue
                if line.strip().lower() == 'imu':
                    videos.collect()  # newest video_diag.csv for the sharpness correlation
                    videos.imu_report()
                    continue
                if not plan.command(line):
                    say('Thoat. (Neu UAV con bay: PX4 failsafe.)')
                    break
            except Exception as error:  # noqa: BLE001 - keep the loop alive for land
                logger.exception('command failed')
                say(f'Loi: {error}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
