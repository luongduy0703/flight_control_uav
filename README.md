Terminal1: Pi
/// cd ~/uav/controller
/// bash tools/run_hardware_router.sh 192.168.xx.xxx /dev/ttyAMA0

Terminal 2: PC
Bat QGC


Terminal 3: Pi
/// cd ~/uav/controller
/// .venv/bin/python pi/console.py --profile hardware --position-source flow \
  --connection udpin://127.0.0.1:14540 --height x.x --enable-actuation


Terminal 4: Pi (theo doi % su dung)

/// cd ~/uav/controller
/// nice -n 10 python3 tools/monitor_pi.py

16:33:20.037 CPU   1.2% [  1.9   2.0   1.0   0.0] RAM 391/3790MiB (10%) swap 0MiB 35.0°C 600MHz ok

Các cột lần lượt là:

	- giờ;
	- % CPU tổng;
	- % từng nhân (4 nhân của Pi 4);
	- RAM đã dùng / tổng;
	- swap;
	- nhiệt độ CPU;
	- xung nhịp CPU;
      trạng thái nguồn và nhiệt. ok là bình thường; nếu có vấn đề sẽ hiện undervolt (thiếu điện áp), throttled (đang bị giảm xung) hoặc temp-limit (quá nhiệt).
Khi đang không điều khiển, Pi chỉ dùng khoảng 1% CPU và 391 MB RAM (10%).


Lệnh force kill mavlink router:

/// pkill mavlink-routerd -f


Test actuator: /// actuator_test set -m 1 -v 0.15 -t 2

		-m 1 : motor 1
		-p 15: muc ga 15%
		-t 2 : thoi gian quay 2s
		
		
2/1 - 1045
