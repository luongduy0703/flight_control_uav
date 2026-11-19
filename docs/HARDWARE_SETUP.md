# Chuyển từ HITL sang UAV thật: PC → Pi → Pixhawk

Sơ đồ mong muốn:

```text
QGroundControl trên PC ── UDP 14550 ──┐
                                     Pi / MAVLink Router ── USB Type-C ── Pixhawk PX4
Terminal trên PC ── SSH ── Pi/console ── UDP 127.0.0.1:14540 ──┘
```

PC dùng UDP để QGC nhận/gửi **MAVLink**, còn `arm`, `takeoff`, `fly`,
`land` được gõ trong terminal console sau khi SSH vào Pi. Console không
có API nhận các chuỗi lệnh bay từ UDP bên ngoài. Router là tiến trình duy
nhất mở USB; console dùng UDP loopback trên Pi. Không chạy console mở
`/dev/ttyACM0` đồng thời với router. [Cú pháp và hành vi endpoint của
MAVLink Router](https://github.com/mavlink-router/mavlink-router#running).

## 1. Phục hồi cấu hình PX4 của UAV thật

Tháo cánh khi chuyển cấu hình, kiểm tra motor hoặc thử ARM trên bàn. Kết
nối Pixhawk trực tiếp với QGC để sao lưu toàn bộ tham số hiện tại. Nếu
còn bản sao UAV thật trước HITL, đối chiếu/khôi phục bản đó; nếu không,
cấu hình lại từng phần trên chính UAV trong QGC. Có thể lưu/nạp tham số
tại màn hình Parameters của QGC; [PX4 hướng dẫn](https://docs.px4.io/v1.17/en/advanced_config/parameters).

[`controller/params/hardware.params`](../params/hardware.params) chỉ là
**overlay một phần**, không phải bản sao đầy đủ và chưa được xác thực bằng
chuyến bay thật. File đặt `SYS_HITL=0`, `SYS_AUTOSTART=4001` (Generic
Quad-X), xóa HIL actuator và hai breaker, đặt 6S cùng giới hạn bay chậm
dạng ứng viên. Nó không chứa ánh xạ output/motor, calibration, battery
scaling, hover/PID của UAV thật hoặc hành động khi mất Offboard. Không nạp
file này như một cấu hình bay hoàn chỉnh; kiểm tra airframe thật trước
khi dùng `SYS_AUTOSTART=4001`. File HITL ở `simulation/params/` chỉ dành
cho model X500 2,0893 kg, không chuyển các giá trị hover/PID sang UAV
thật khoảng 5–6 kg.

Trong QGC, sau khi phục hồi cấu hình thật và reboot, kiểm tra:

1. `SYS_HITL=0`; đúng loại Quad-X, kích thước, thứ tự motor, chiều quay
   và output ESC. Kiểm tra actuator từng motor **khi đã tháo cánh**.
   [Hướng dẫn actuator PX4](https://docs.px4.io/v1.17/en/config/actuators).
2. IMU, compass, GPS, radio/joystick, pin 6S và bộ đo điện áp/dòng đã
   được cấu hình/calibrate trên UAV thật. `BAT1_N_CELLS=6` không tự đặt
   đúng hệ số đo pin. Kiểm tra cảnh báo preflight của QGC.
3. Kiểm tra `COM_OF_LOSS_T`, `COM_OBL_RC_ACT`, mất datalink/RC, pin thấp,
   RTL/LAND và geofence theo sân bay thử. Các giá trị HITL không tự được
   xóa khi không có trong overlay; xem [PX4 Offboard loss](https://docs.px4.io/v1.17/en/flight_modes/offboard).
4. Xác nhận cổng USB MAVLink của Pixhawk hoạt động sau khi đổi khỏi HITL;
   `MAV_0_*` và `SER_TEL1_BAUD` trên bo dành cho TELEM1, không cấu
   hình cổng USB. Bốn log cũ từng có lỗi compass/IMU, trượt x/y và mất
   Offboard; cần ULog mới để chứng minh các lỗi đó đã hết.

## 2. Triển khai code và nối mạng

Ví dụ Pi là `192.168.50.1`. PC và Pi phải cùng mạng. Trên PC chứa
repository hiện tại, sao chép phần controller tới release trên Pi khi
Pixhawk đã DISARM và console cũ đã đóng:

```bash
ssh pi@192.168.50.1 'mkdir -p ~/uav/releases/2026-10-26/controller'
rsync -av --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='session-logs/' controller/ \
  pi@192.168.50.1:~/uav/releases/2026-10-26/controller/
```

Trên Pi, từ thư mục release:

```bash
cd ~/uav/releases/2026-10-26/controller
python3 -m venv .venv
.venv/bin/python -m pip install -r pi/requirements.txt
python3 -B -m unittest discover -s tests -p 'test_*.py'
```

Cắm USB Type-C Pixhawk vào Pi; kiểm tra tên thiết bị (thường là
`/dev/ttyACM0`, nhưng có thể khác) và quyền truy cập serial:

```bash
ls -l /dev/serial/by-id/
ls -l /dev/ttyACM*
```

MAVLink Router phải có trên Pi (`command -v mavlink-routerd`). Nếu chưa
có, cài theo [README chính thức](https://github.com/mavlink-router/mavlink-router#installation)
cho hệ điều hành Pi. Chỉ chạy **một** instance router; dừng dịch vụ
router cũ dùng cùng USB trước khi chạy lệnh bên dưới.

## 3. Router trên Pi và QGC trên PC

Trên Pi, từ thư mục controller, thay `IP_PC` bằng IPv4 hiện tại của PC
chạy QGC (`ip -4 addr` trên PC; DHCP có thể đổi IP, khi đó chạy lại):

```bash
bash tools/run_hardware_router.sh IP_PC            # mặc định /dev/ttyAMA0 (TELEM1)
```

Router chạy **thủ công, ở foreground**: script điền
[`config/mavlink-router.conf`](../config/mavlink-router.conf) vào file tạm và chạy
`mavlink-routerd` trong terminal đó. Chưa chạy script thì QGC, console và perception
đều không có MAVLink; **Ctrl+C** là ngắt hết (chỉ Ctrl+C khi DISARM và console đã đóng).
Lần chạy đầu script tắt dịch vụ tự khởi động `mavlink-router.service` (hỏi sudo một
lần) để router không tự chạy khi Pi bật. Muốn chạy nền lại như cũ:
`sudo systemctl enable --now mavlink-router`. Script từ chối nếu đã có
`mavlink-routerd` khác hoặc tiến trình khác đang mở cổng serial (chỉ một router).
PC mở QGC, dùng
UDP port 14550 và xác nhận nhìn thấy đúng Pixhawk, `DISARM`,
`ON_GROUND`, tham số/cảnh báo đúng UAV thật. QGC qua UDP là đường giám
sát và tiếp quản; terminal PC kết nối Pi qua SSH.

Trong terminal PC khác:

```bash
ssh pi@192.168.50.1
cd ~/uav/releases/2026-10-26/controller
.venv/bin/python pi/console.py --profile hardware \
  --connection udpin://127.0.0.1:14540 --check
```

`--check` chỉ đọc trạng thái rồi thoát. Cần thấy `connected=true`,
`SYS_HITL=0`, `armed=false`, `ON_GROUND`, `health_all_ok=true` và dữ liệu
telemetry còn mới. `x/y/z` local NED có thể khác 0. Nếu QGC/console
không có telemetry, kiểm tra IP PC, firewall UDP 14550, đường USB,
quyền serial và router trước khi thử ARM. Không đổi profile hoặc tắt
preflight check để đi qua lỗi kết nối.

## 4. Kiểm tra trên đất rồi mới bay

Khi tháo cánh, mở console có quyền điều khiển:

```bash
.venv/bin/python pi/console.py --profile hardware \
  --connection udpin://127.0.0.1:14540 --height 2 \
  --enable-actuation
```

Trong console, thử `status`, `arm`, `status`, `disarm`, `status`; quan sát
trên cả QGC và Pi. PX4 có thể tự DISARM nếu ARM mà không TAKEOFF trong
thời gian `COM_DISARM_PRFLT` (bốn log cũ ghi 10 s). Bài thử trên đất chưa
xác thực TAKEOFF/LAND hay nguồn/Wi-Fi Pi khi UAV bay.

Chỉ khi cấu hình thật, cảm biến, pin, actuator, failsafe và bài thử trên
đất đều đạt, người vận hành mới lắp cánh tại nơi bay phù hợp. Giữ QGC/
phương tiện tiếp quản sẵn sàng. Chuyến đầu thử `takeoff` giữ tại chỗ,
kiểm tra `phase=HOLD` và `takeoff_complete=true`, rồi `land`; xác nhận
`armed=false`, `ON_GROUND` trước `exit`. `fly x y` là đích **tuyệt đối**
theo local NED, không phải độ dời từ chỗ hiện tại. `disarm` trong code
gọi dừng motor cưỡng bức; khi đang bay có thể làm UAV rơi.

Console không phải điều khiển thời gian thực từ PC: nếu phiên SSH hoặc
Pi/router mất khi đang Offboard, PX4 xử lý theo failsafe đã cấu hình.
Sau chuyến đầu, lưu ULog và session log Pi để đối chiếu lỗi cũ.

## Bay trong nhà bằng optical flow, không GPS

Không có GPS thì PX4 chỉ có local position, nên từ chối mode HOLD
("No valid global position estimate"). Console lại ARM và giữ vị trí bằng
HOLD. Khi dùng `--position-source flow --enable-actuation`, console gửi gốc
toạ độ của nơi bay (`SET_GPS_GLOBAL_ORIGIN`) khi UAV đang DISARM và ON_GROUND,
rồi chờ PX4 trả về `GPS_GLOBAL_ORIGIN`. Gốc mất khi Pixhawk reboot nên mỗi
phiên console gửi lại.

Toạ độ lấy từ file `site_origin.json` ở thư mục controller, chỉ cần tạo **một
lần** cho khu vực bay. PX4 chỉ dùng nó để tính độ lệch từ (declination), đại
lượng thay đổi chậm: sai khoảng 100 km vẫn chấp nhận được, nhưng sai sang
vùng/quốc gia khác sẽ làm lệch hướng bay.

```bash
echo '{"latitude_deg": 21.0285, "longitude_deg": 105.8542, "altitude_m": 10}' > site_origin.json
.venv/bin/python pi/console.py --profile hardware --position-source flow \
  --connection udpin://127.0.0.1:14540 --height 1.5 --enable-actuation
```

`--ekf-origin LAT LON ALT_M` ghi đè file cho một phiên.
`--check` không gửi gốc toạ độ. Nếu Pixhawk vừa reboot thì `--check` sẽ
báo `is_armable=false` cho tới khi một phiên có `--enable-actuation` gửi gốc.

## Chạy perception từ console (lệnh `perception`)

Perception (`uavp_real_time`, C++, thư mục `his_perception_uav-main` / trên Pi
`~/uav_perception_pi`) vẫn là **chương trình riêng**. Console chỉ khởi động nó
như tiến trình con, ra lệnh bắt đầu/dừng quét và theo dõi log của nó. Perception
lỗi/crash thì chỉ nó dừng; console vẫn gửi setpoint, UAV vẫn HOLD.

### Đường MAVLink chung: một router giữ TELEM1

Chạy `bash tools/run_hardware_router.sh IP_PC` (mục 3). Router là tiến trình **duy nhất** mở
`/dev/ttyAMA0`; control nhận ở UDP 14540, `uavp-fc-bridge` ở 14551, QGC ở 14550.
USB Pixhawk không đưa vào router. Đổi dịch vụ router từ USB sang TELEM1 ảnh
hưởng tới perception của cả nhóm: thống nhất trước khi đổi.

Kiểm tra trước bay:
```bash
sudo fuser -v /dev/ttyAMA0                     # chỉ có mavlink-routerd (terminal chạy script)
systemctl is-active uavp-fc-bridge
cd ~/uav_perception_pi && ./build/uavp_fc_check --cfg configs/run/real_time.yaml --seconds 20
```

### Chạy

```bash
.venv/bin/python pi/console.py --profile hardware --position-source flow \
  --connection udpin://127.0.0.1:14540 --height 1.5 --enable-actuation \
  --perception-dir ~/uav_perception_pi
```

- Mở console là perception khởi động luôn (nạp model, camera, LiDAR, Pixhawk).
  Chờ dòng `[perception READY ...]` **trước khi ARM**. Nếu lỗi 3 lần liên tiếp:
  `FAILED` + lý do; sửa xong gõ `perception restart`. Console vẫn bay được.
- `perception`: chỉ nhận khi đang **HOLD sau takeoff** (armed, IN_AIR, không
  bị QGC tiếp quản) và perception đang `WAITING`. Console gửi `COMMAND_LONG`
  31010 tới 1/191 (lấy từ `configs/run/real_time.yaml` của perception), bridge
  trả ACK; dòng `[perception SCANNING]` đến từ log perception.
- **Thử trên bàn / tháo cánh** (`--perception-bench`): `perception` nhận thêm khi UAV
  **DISARM và ON_GROUND** (telemetry còn mới). Trên mặt đất độ cao optical flow ≈ 0 nên
  đặt khoảng cách camera → mặt phẳng bằng tay:
  ```bash
  .venv/bin/python pi/console.py --profile hardware --position-source flow \
    --connection udpin://127.0.0.1:14540 --height 1.5 --enable-actuation \
    --perception-dir ~/uav_perception_pi --perception-bench \
    --perception-set distance.source=fixed distance.fixed_m=0.7
  ```
  Gõ `arm`/`takeoff` khi đang quét bench: console gửi STOP **trước** lệnh; ARM bằng
  QGC/RC thì console tự STOP ở vòng kiểm tra kế tiếp. Khi đã bay, vẫn chỉ nhận ở HOLD.
- **Kết quả về laptop**: trên laptop, ở thư mục repo, để chạy song song
  `python3 tools/pull_perception_results.py` (hỏi mật khẩu SSH một lần, Ctrl+C dừng).
  Mỗi lần quét xong (kể cả quét hỏng/dừng giữa chừng) thành một thư mục
  `results_perception/DD-MM-YYYY_HH-MM-SS_<KET_QUA>` (giờ Pi lúc bắt đầu chạy perception):
  `CO_VAT`, `CO_VAT_CHUA_FUSION`, `KHONG_CO_VAT`, `LOI`, `DUNG_GIUA_CHUNG`; trong đó có
  `result.json`, `final.jpg`, `run.log`, `timing.csv`… và `perception.log`. Lần đầu chỉ ghi
  nhớ các lần quét cũ trên Pi; `--all --once` để chép cả chúng.
- **Chia nhân CPU** (đo 2026-11-18: perception quét dùng ~1,9 nhân, 92–99% nhân 2/3): console +
  MAVSDK (`--control-cores`, mặc định `0,1`) và `mavlink-routerd` (`CONTROL_CORES`, mặc định
  `0,1`) chạy trên nhân 0/1; perception tự ghim vào `cpu.cores: [2, 3]`. Tắt ghim:
  `--control-cores ''` / `CONTROL_CORES= bash tools/run_hardware_router.sh ...`.
- `perception stop`: dừng quét bất cứ lúc nào. Nếu UAV rời HOLD (`fly`, `land`,
  QGC tiếp quản) khi đang quét, console tự gửi STOP một lần.
- Quét xong: `[perception DONE: result in runs/...]` (thư mục trên Pi), rồi
  perception tự khởi động lại để sẵn sàng lần sau.
- `status` có thêm mục `perception` (trạng thái, lỗi gần nhất, file log).
- Viewer trên laptop chỉ để **xem**: `python3 his_perception_uav-main/tools/remote_viewer.py --host 192.168.50.1` (từ thư mục repo control trên laptop).
  SPACE trên viewer bị tắt (`start.accept_viewer_key=false`) khi perception do
  console chạy; chỉ console được ra lệnh quét. Phím `q` trên viewer vẫn làm
  perception thoát (console tự khởi động lại).
- Log perception: `session-logs/perception-*.log` cạnh log console.
- Perception **không bao giờ gửi lệnh bay**. Bay tới toạ độ vật (`p_W`) chưa
  được nối; `calib/T_BC.yaml` vẫn là giá trị tạm.

### Sự cố: console báo `ParamError: TIMEOUT ... SYS_HITL`, QGC kẹt "loading vehicle data"
PX4 nhận heartbeat từ Pi được nhưng không trả lời yêu cầu (tham số, lệnh). Kiểm tra:
1. Chỉ một router (`pgrep -a mavlink-routerd`). Nếu có hai router cùng mở `/dev/ttyAMA0`:
   DISARM, đóng console, Ctrl+C rồi chạy lại script.
2. Chiều Pi → Pixhawk: dây TX Pi (GPIO14, chân 8) → RX TELEM1 (chân 3) và GND. Nếu dữ
   liệu từ Pixhawk vẫn về đủ mà PX4 không trả lời gì, cắm lại dây rồi rút nguồn Pixhawk
   cấp lại.
3. QGC mở từ lúc lỗi có thể kẹt ở bước tải: tắt hẳn rồi mở lại QGC.

## Đo quỹ đạo trôi (drift) trực tiếp trên PC

`tools/drift_viewer.py` chạy trên laptop, vẽ trực tiếp quỹ đạo của UAV theo
x = Bắc, y = Đông, z = Lên (= −Down của PX4). Gốc (0,0,0) là gốc EKF local:
vị trí lúc EKF khởi tạo sau khi Pixhawk boot. Dữ liệu là `LOCAL_POSITION_NED`
(~30 Hz) do router trên Pi chuyển tới `PC_IP:14560` (endpoint `drift` trong
`config/mavlink-router.conf`). Tool chỉ nghe, không gửi gì cho Pixhawk.

```bash
pip install -r tools/requirements.txt          # matplotlib, pymavlink
python3 tools/drift_viewer.py                   # chạy sau khi đã bật run_hardware_router.sh
python3 tools/drift_viewer.py --replay session-logs/drift-YYYYMMDDTHHMMSS.csv --save drift.png
```

- Mỗi mẫu được ghi vào `session-logs/drift-*.csv` để xem lại sau chuyến bay.
- Pixhawk reboot (`time_boot_ms` lùi lại) nghĩa là có gốc mới, nên quỹ đạo vẽ lại từ đầu.
- Hai mẫu liên tiếp lệch nhau > `--jump-m` (mặc định 0.5 m) được đánh dấu là
  EKF reset (dấu ×, vạch tím). Ví dụ: lúc console gửi `SET_GPS_GLOBAL_ORIGIN`,
  hoặc lúc optical flow mất rồi có lại. Cú nhảy đó không phải chuyển động thật.
- Firewall trên PC phải mở UDP 14560 (giống 14550 cho QGC).
- Nếu báo "mplot3d unavailable", tool vẫn chạy nhưng hiện góc nhìn từ trên xuống.
  Lỗi này do matplotlib của hệ thống xung đột với bản cài bằng pip. Cách sửa:
  dùng venv, hoặc `sudo apt remove python3-matplotlib`.
