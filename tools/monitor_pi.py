#!/usr/bin/env python3
"""Pi resource monitor to run beside the flight console (read-only, stdlib only).

Prints per-core CPU %, RAM, swap, CPU temperature/frequency and power throttling
once per interval, and appends the same values to a CSV in session-logs/ so they
can be lined up with the console log and the PX4 ULog after a flight.

    nice -n 10 python3 tools/monitor_pi.py            # 1 s interval
    python3 tools/monitor_pi.py --interval 0.5 --no-csv
"""
import argparse
import csv
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# vcgencmd get_throttled bits (Raspberry Pi firmware)
THROTTLE_NOW = {0: 'undervolt', 1: 'freq-capped', 2: 'throttled', 3: 'temp-limit'}


def read_cpu_times(text):
    """Return {'cpu': (busy, total), 'cpu0': ...} from /proc/stat text."""
    times = {}
    for line in text.splitlines():
        if not line.startswith('cpu'):
            continue
        name, *fields = line.split()
        values = [int(v) for v in fields[:8]]
        idle = values[3] + values[4]  # idle + iowait
        total = sum(values)
        times[name] = (total - idle, total)
    return times


def cpu_percent(before, after):
    """Per-name busy percentage between two read_cpu_times snapshots."""
    result = {}
    for name, (busy1, total1) in after.items():
        busy0, total0 = before.get(name, (busy1, total1))
        dt = total1 - total0
        result[name] = 0.0 if dt <= 0 else 100.0 * (busy1 - busy0) / dt
    return result


def read_meminfo(text):
    """Return MiB values for used/total RAM and swap from /proc/meminfo text."""
    kb = {}
    for line in text.splitlines():
        key, _, rest = line.partition(':')
        parts = rest.split()
        if parts:
            kb[key] = int(parts[0])
    total = kb.get('MemTotal', 0)
    available = kb.get('MemAvailable', kb.get('MemFree', 0))
    swap_total = kb.get('SwapTotal', 0)
    swap_used = swap_total - kb.get('SwapFree', 0)
    mib = lambda v: round(v / 1024, 1)
    return {'ram_used_mib': mib(total - available), 'ram_total_mib': mib(total),
            'ram_pct': round(100.0 * (total - available) / total, 1) if total else 0.0,
            'swap_used_mib': mib(swap_used), 'swap_total_mib': mib(swap_total)}


def decode_throttled(value):
    """Names of the 'happening now' bits of vcgencmd get_throttled."""
    return [name for bit, name in THROTTLE_NOW.items() if value >> bit & 1]


def read_text(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def read_throttled():
    if not shutil.which('vcgencmd'):
        return None
    try:
        out = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True,
                             text=True, timeout=1).stdout
        return int(out.strip().split('=')[1], 16)
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def sample():
    """One snapshot of everything except CPU %, which needs two /proc/stat reads."""
    mem = read_meminfo(read_text('/proc/meminfo') or '')
    temp = read_text('/sys/class/thermal/thermal_zone0/temp')
    freq = read_text('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
    load = read_text('/proc/loadavg')
    mem['temp_c'] = round(int(temp) / 1000, 1) if temp else None
    mem['freq_mhz'] = int(freq) // 1000 if freq else None
    mem['load1'] = float(load.split()[0]) if load else None
    mem['throttled'] = read_throttled()
    return mem


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--interval', type=float, default=1.0, help='seconds between samples (0.2..60)')
    parser.add_argument('--log-dir', type=Path, default=PROJECT_ROOT / 'session-logs')
    parser.add_argument('--no-csv', action='store_true', help='only print, do not write a CSV')
    args = parser.parse_args()
    if not 0.2 <= args.interval <= 60:
        parser.error('interval must be 0.2..60 s')
    stat = read_text('/proc/stat')
    if stat is None:
        parser.error('/proc/stat not readable; run this on the Linux Pi')

    writer, csv_file = None, None
    prev = read_cpu_times(stat)
    cores = sorted((n for n in prev if n != 'cpu'), key=lambda n: int(n[3:]))
    if not args.no_csv:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        path = args.log_dir / datetime.now().strftime('monitor-%Y%m%dT%H%M%S.csv')
        csv_file = open(path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['time', 'cpu_pct', *[f'{c}_pct' for c in cores], 'ram_used_mib', 'ram_total_mib',
                         'ram_pct', 'swap_used_mib', 'temp_c', 'freq_mhz', 'load1', 'throttled_hex'])
        print(f'CSV: {path}', flush=True)
    print('Ctrl+C to stop. Columns: total CPU | per core | RAM | swap | temp | freq | flags', flush=True)
    peak = {'cpu': 0.0, 'ram': 0.0, 'temp': 0.0}
    try:
        while True:
            time.sleep(args.interval)
            now = read_cpu_times(read_text('/proc/stat') or '')
            pct = cpu_percent(prev, now)
            prev = now
            s = sample()
            stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            flags = decode_throttled(s['throttled']) if s['throttled'] is not None else ['n/a']
            peak['cpu'] = max(peak['cpu'], pct.get('cpu', 0.0))
            peak['ram'] = max(peak['ram'], s['ram_pct'])
            peak['temp'] = max(peak['temp'], s['temp_c'] or 0.0)
            per_core = ' '.join(f'{pct.get(c, 0):5.1f}' for c in cores)
            print(f"{stamp[11:]} CPU {pct.get('cpu', 0):5.1f}% [{per_core}] "
                  f"RAM {s['ram_used_mib']:.0f}/{s['ram_total_mib']:.0f}MiB ({s['ram_pct']:.0f}%) "
                  f"swap {s['swap_used_mib']:.0f}MiB {s['temp_c']}°C {s['freq_mhz']}MHz "
                  f"{','.join(flags) or 'ok'}", flush=True)
            if writer:
                writer.writerow([stamp, round(pct.get('cpu', 0), 1), *[round(pct.get(c, 0), 1) for c in cores],
                                 s['ram_used_mib'], s['ram_total_mib'], s['ram_pct'], s['swap_used_mib'],
                                 s['temp_c'], s['freq_mhz'], s['load1'],
                                 None if s['throttled'] is None else hex(s['throttled'])])
                csv_file.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if csv_file:
            csv_file.close()
        print(f"\nPeak: CPU {peak['cpu']:.1f}%  RAM {peak['ram']:.1f}%  temp {peak['temp']:.1f}°C", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
