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
   `MAV_1_*` và `SER_TEL2_BAUD` trong overlay dành cho TELEM2, không cấu
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

Mở terminal thứ nhất trên Pi, từ thư mục release. Thay `IP_PC` bằng IPv4
thực của PC đang chạy QGC và `/dev/ttyACM0` bằng thiết bị Pixhawk thực tế:

```bash
bash tools/run_hardware_router.sh IP_PC /dev/ttyACM0
```

Script tạo cấu hình tạm, tắt TCP server mặc định của router và chỉ dùng
USB Pixhawk cùng hai endpoint UDP nêu trong sơ đồ. Giữ router chạy. Nếu
Pi đã có dịch vụ router giữ USB, dừng dịch vụ đó trước. PC mở QGC, dùng
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
