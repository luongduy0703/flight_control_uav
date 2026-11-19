#!/usr/bin/env python3
"""Run on the LAPTOP: copy every finished perception scan from the Pi into results_perception/.

    python3 tools/pull_perception_results.py                 # watch, pull each new scan (Ctrl+C stops)
    python3 tools/pull_perception_results.py --all --once    # pull every scan already on the Pi, then exit

Each scan becomes one folder named DD-MM-YYYY_HH-MM-SS_<RESULT> (Pi clock at scan start), e.g.
results_perception/18-11-2026_08-36-11_KHONG_CO_VAT/, holding the run directory from the Pi
(result.json, final.jpg, run.log, timing.csv, ...) plus perception.log (the console's log of that
perception process). RESULT is CO_VAT (selected + s4 fusion), CO_VAT_CHUA_FUSION, KHONG_CO_VAT,
LOI (perception error) or DUNG_GIUA_CHUNG (stopped / crashed before a result).

Nothing runs on the Pi except ssh/tar reads. The password is asked once; the SSH connection then stays
open 15 min after the last use (ssh ControlMaster), so reruns do not ask again (--close ends it).
The first run only remembers scans already on the Pi (pass --all to copy them too).
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

RUN_RE = re.compile(r'^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})_real_time$')
STATE_FILE = '.pulled'
KEEP_SSH = '15m'  # SSH stays open this long after the last use: rerunning --once asks no password.

# Executed on the Pi with python3 (stdlib only): lists runs with what the laptop needs to classify them.
REMOTE_LIST = r'''
import glob, json, os, sys
runs, logs = (os.path.expanduser(p) for p in sys.argv[1:3])
names = sorted(d for d in os.listdir(runs) if d.endswith('_real_time') and os.path.isdir(os.path.join(runs, d)))
log_of = {}
for path in sorted(glob.glob(os.path.join(logs, 'perception-*.log'))):
    try:
        text = open(path, errors='replace').read()
    except OSError:
        continue
    for name in names:
        if 'runs/' + name in text:
            log_of[name] = path
out = []
for name in names:
    d = os.path.join(runs, name)
    item = {'name': name, 'log': log_of.get(name), 'result': None, 'started': False, 'error': False}
    try:
        item['result'] = json.load(open(os.path.join(d, 'result.json')))
    except (OSError, ValueError):
        pass
    for path in (os.path.join(d, 'run.log'), log_of.get(name)):
        try:
            text = open(path, errors='replace').read() if path else ''
        except OSError:
            text = ''
        item['started'] |= 'BAT DAU' in text
        item['error'] |= 'LOI: ' in text
    out.append(item)
print(json.dumps(out))
'''


def folder_name(run_name, status):
    """20261118-083611_real_time -> 18-11-2026_08-36-11_<status>."""
    match = RUN_RE.match(run_name)
    if not match:
        raise ValueError(f'Unexpected perception run name: {run_name}')
    year, month, day, hour, minute, second = match.groups()
    return f'{day}-{month}-{year}_{hour}-{minute}-{second}_{status}'


def classify(item, newest):
    """Result label, or None while the run is still waiting/scanning or was never started."""
    result = item.get('result')
    if result is not None:
        if result.get('selected'):
            return 'CO_VAT' if result.get('fusion') else 'CO_VAT_CHUA_FUSION'
        return 'KHONG_CO_VAT'
    if not item.get('started') or item['name'] == newest:
        return None  # READY-only run (no scan), or the scan that is running now.
    return 'LOI' if item.get('error') else 'DUNG_GIUA_CHUNG'


class Pi:
    def __init__(self, host):
        self.host = host
        self.sock = Path(tempfile.gettempdir()) / f'uavp-pull-{os.getuid()}.sock'
        self.opts = ['-o', f'ControlPath={self.sock}', '-o', 'ConnectTimeout=8']

    def open(self):
        """Interactive once: the password prompt appears in this terminal."""
        if subprocess.run(['ssh', *self.opts, '-O', 'check', self.host],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            return
        print(f'SSH to {self.host} (password asked once for this session)...')
        subprocess.run(['ssh', *self.opts, '-o', 'ControlMaster=yes', '-o', f'ControlPersist={KEEP_SSH}',
                        '-o', 'ServerAliveInterval=10', '-fN', self.host], check=True)

    def close(self):
        subprocess.run(['ssh', *self.opts, '-O', 'exit', self.host],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run(self, args, data=None, binary=False):
        result = subprocess.run(['ssh', *self.opts, '-o', 'BatchMode=yes', self.host, *args],
                                input=data, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors='replace').strip() or f'ssh exit {result.returncode}')
        return result.stdout if binary else result.stdout.decode()


def load_state(dest):
    """{run name: 'copied' | 'skipped'}, or None before the first run."""
    path = dest / STATE_FILE
    if not path.exists():
        return None
    state = {}
    for line in path.read_text().splitlines():
        name, _, how = line.partition(' ')
        state[name] = how or 'copied'
    return state


def remember(dest, name, how='copied'):
    with open(dest / STATE_FILE, 'a') as f:
        f.write(f'{name} {how}\n')


def pull(pi, item, status, runs, dest):
    name = item['name']
    final = dest / folder_name(name, status)
    partial = dest / ('.partial-' + name)
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    data = pi.run(['tar', '-C', runs, '-cf', '-', name], binary=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():  # Only plain files/dirs inside this run.
            if not (member.isfile() or member.isdir()) or member.name.startswith(('/', '..')) or '/../' in member.name:
                raise RuntimeError(f'Refusing unexpected tar member {member.name!r}')
        tar.extractall(partial)
    for child in (partial / name).iterdir():
        child.rename(partial / child.name)
    (partial / name).rmdir()
    if item.get('log'):
        (partial / 'perception.log').write_bytes(pi.run(['cat', item['log']], binary=True))
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
    remember(dest, name)
    return final


def poll(pi, args, dest, state):
    items = json.loads(pi.run(['python3', '-', args.runs, args.logs], data=REMOTE_LIST.encode()))
    names = [i['name'] for i in items if RUN_RE.match(i['name'])]
    newest = names[-1] if names else None
    if state is None:  # First run: remember older scans without copying, unless --all.
        state = {}
        (dest / STATE_FILE).touch()
        if not args.all:
            for item in items:
                if RUN_RE.match(item['name']) and classify(item, newest) is not None:
                    remember(dest, item['name'], 'skipped')
                    state[item['name']] = 'skipped'
            print(f'First run: {len(state)} older scan(s) on the Pi skipped (use --all to copy them).')
    for item in items:
        how = state.get(item['name'])
        if how == 'copied' or (how == 'skipped' and not args.all) or not RUN_RE.match(item['name']):
            continue
        status = classify(item, newest)
        if status is None:
            continue
        final = pull(pi, item, status, args.runs, dest)
        state[item['name']] = 'copied'
        print(f'[{time.strftime("%H:%M:%S")}] {status:<20} -> {final}')
    return state


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='piros2@192.168.50.1')
    parser.add_argument('--runs', default='~/uav_perception_pi/runs', help='Perception runs dir on the Pi')
    parser.add_argument('--logs', default='~/uav/controller/session-logs', help='Console session-logs on the Pi')
    parser.add_argument('--dest', type=Path, default=root / 'results_perception')
    parser.add_argument('--interval', type=float, default=3.0, help='Seconds between checks')
    parser.add_argument('--all', action='store_true', help='Also copy scans already on the Pi')
    parser.add_argument('--once', action='store_true', help='Check once, then exit')
    parser.add_argument('--close', action='store_true',
                        help=f'Close the SSH connection on exit (default: kept {KEEP_SSH} for the next run)')
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    pi = Pi(args.host)
    state = load_state(args.dest)
    print(f'Saving perception results to {args.dest}' + ('' if args.once else ' (Ctrl+C stops)'))
    try:
        pi.open()
        while True:
            try:
                state = poll(pi, args, args.dest, state)
            except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                print(f'[{time.strftime("%H:%M:%S")}] Pi not reachable ({error}); retrying', file=sys.stderr)
                try:
                    pi.open()
                except subprocess.CalledProcessError:
                    pass
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
    finally:
        if args.close:
            pi.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
