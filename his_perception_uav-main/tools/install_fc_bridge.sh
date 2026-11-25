#!/bin/bash
# Cai dich vu uavp-fc-bridge (chay khi Pi khoi dong). Chay trong thu muc goc uav_perception_pi, SAU khi da build:
#   bash tools/install_fc_bridge.sh            # cai + bat + chay ngay
#   bash tools/install_fc_bridge.sh --remove   # go
set -e
cd "$(dirname "$0")/.."
DIR="$(pwd)"
UNIT=/etc/systemd/system/uavp-fc-bridge.service

if [ "$1" = "--remove" ]; then
  sudo systemctl disable --now uavp-fc-bridge || true
  sudo rm -f "$UNIT"
  sudo systemctl daemon-reload
  echo "Da go uavp-fc-bridge"
  exit 0
fi

[ -x "$DIR/build/uavp_fc_bridge" ] || { echo "Chua build: cmake --build build -j2"; exit 1; }
sed -e "s|__USER__|$USER|g" -e "s|__DIR__|$DIR|g" tools/uavp-fc-bridge.service | sudo tee "$UNIT" > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now uavp-fc-bridge
sleep 3
systemctl --no-pager status uavp-fc-bridge | head -n 12
echo
echo "Xem log: journalctl -u uavp-fc-bridge -f"
echo "Xem tu the trong RAM: ./build/uavp_fc_bridge --cfg configs/run/real_time.yaml --print"
