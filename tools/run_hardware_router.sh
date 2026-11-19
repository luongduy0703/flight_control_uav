#!/usr/bin/env bash
# Run the ONE MAVLink router on the Pi, in the foreground, only while you want it. Ctrl+C stops it.
# While it runs it shares the Pixhawk link with:
#   control console 127.0.0.1:14540, perception uavp_fc_bridge 127.0.0.1:14551, QGC PC_IPV4:14550.
# Nothing connects (QGC, console, perception) until you run this; nothing stays connected after Ctrl+C.
#
#   bash tools/run_hardware_router.sh PC_IPV4 [SERIAL_DEVICE]   # default /dev/ttyAMA0 (TELEM1)
#
# The first run disables the auto-start mavlink-router.service (asks for sudo once), so the router never
# runs at boot or behind your back. Re-enable it with: sudo systemctl enable --now mavlink-router
# Stop the router (Ctrl+C) only with the vehicle DISARMED and the console closed.
# The router runs on the control cores (CONTROL_CORES, default 0,1) like pi/console.py; perception
# keeps cores 2,3 (cpu.cores in its config). CONTROL_CORES= (empty) disables pinning.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
template=$here/config/mavlink-router.conf
dropin=/etc/systemd/system/mavlink-router.service.d/restart.conf

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: %s PC_IPV4 [SERIAL_DEVICE=/dev/ttyAMA0]\n' "$0" >&2
  exit 2
fi
pc_ip=$1
serial_device=${2:-/dev/ttyAMA0}
control_cores=${CONTROL_CORES-0,1}

python3 - "$pc_ip" <<'PY'
import ipaddress
import sys

address = ipaddress.IPv4Address(sys.argv[1])
if address.is_multicast or address.is_loopback or address.is_unspecified or address.is_reserved:
    raise SystemExit('PC_IPV4 must be the reachable IPv4 address of the QGC computer')
PY
case "$serial_device" in
  /dev/*) ;;
  *) printf 'Serial device must be under /dev/\n' >&2; exit 2 ;;
esac
[[ -c $serial_device ]] || { printf 'Not a character device: %s\n' "$serial_device" >&2; exit 2; }
[[ -r $serial_device && -w $serial_device ]] || { printf 'No access to %s (add your user to the dialout group)\n' "$serial_device" >&2; exit 2; }
command -v mavlink-routerd >/dev/null 2>&1 || { echo 'mavlink-routerd is missing; install MAVLink Router first' >&2; exit 2; }
[[ -f $template ]] || { echo "Missing $template" >&2; exit 2; }

# Manual mode: the system service must not run (or start at boot) next to this router.
if systemctl is-enabled --quiet mavlink-router 2>/dev/null || systemctl is-active --quiet mavlink-router 2>/dev/null; then
  if pgrep -f 'pi/console\.py' >/dev/null; then
    echo 'pi/console.py is running on the service router. LAND, check DISARMED, type exit, then rerun.' >&2
    exit 2
  fi
  echo 'Disabling the auto-start mavlink-router.service (router now runs only from this script)...'
  sudo systemctl disable --now mavlink-router
  if [[ -f $dropin ]]; then
    sudo rm -f -- "$dropin"
    sudo systemctl daemon-reload
  fi
fi
if pgrep -x mavlink-routerd >/dev/null; then
  printf 'A mavlink-routerd is already running (PID %s). One router only: stop it first (Ctrl+C in its terminal).\n' \
    "$(pgrep -x mavlink-routerd | tr '\n' ' ')" >&2
  exit 2
fi
if systemctl is-active --quiet uxrce_agent 2>/dev/null; then
  echo 'uxrce_agent is running and would steal the Pixhawk serial port; disable it first' >&2
  exit 2
fi
if command -v fuser >/dev/null 2>&1 && fuser "$serial_device" >/dev/null 2>&1; then
  printf '%s is already open by PID %s. Stop it first.\n' "$serial_device" "$(fuser "$serial_device" 2>/dev/null)" >&2
  exit 2
fi

conf=$(mktemp --suffix=.conf)
trap 'rm -f -- "$conf"' EXIT
sed -e "s|__SERIAL_DEVICE__|$serial_device|g" -e "s|__PC_IPV4__|$pc_ip|g" "$template" > "$conf"

printf 'Router: Pixhawk %s <-> control 127.0.0.1:14540 <-> perception 127.0.0.1:14551 <-> QGC %s:14550\n' \
  "$serial_device" "$pc_ip"
systemctl is-active --quiet uavp-fc-bridge 2>/dev/null && echo 'uavp-fc-bridge: active (perception receives MAVLink on 14551)'
echo "Drift viewer: on the PC run python3 tools/drift_viewer.py (UDP $pc_ip:14560, read-only)"
echo 'QGC: default UDP 14550 auto-connect. Ctrl+C stops the router (DISARMED and console closed first).'
pin=()
if [[ -n $control_cores ]]; then
  [[ $control_cores =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "CONTROL_CORES must look like 0,1" >&2; exit 2; }
  command -v taskset >/dev/null 2>&1 || { echo 'taskset missing (util-linux); set CONTROL_CORES= to skip pinning' >&2; exit 2; }
  pin=(taskset -c "$control_cores")
  echo "mavlink-routerd pinned to CPU cores $control_cores"
fi
${pin[@]+"${pin[@]}"} mavlink-routerd -c "$conf"
