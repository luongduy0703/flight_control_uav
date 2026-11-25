# Hướng dẫn chạy `uav_perception_pi` (C++) trên Raspberry Pi qua terminal

**Máy đang dùng: Raspberry Pi 4 Model B, Ubuntu 22.04 (ROS 2 Humble), libcamera bản Raspberry Pi tự build.** Các bước cho Pi 5 / Raspberry Pi OS vẫn giữ để tham khảo.

Pipeline s1 detect → s2 segment → s3 select → s4 fusion chạy **trực tiếp trên Pi**: camera đọc bằng libcamera, LiDAR LD06 qua USB (`/dev/serial/by-id/...`), Pixhawk qua `uavp_fc_bridge`. Máy tính dùng để SSH, xem trực tiếp (`tools/remote_viewer.py`), ra lệnh bắt đầu / dừng và lấy kết quả về. Tổng quan kết nối và biến config: [README.md](README.md).

---

## 1. Chuẩn bị (một lần)

### Phần cứng
| | **Pi 4 (đang dùng)** | Pi 5 |
|---|---|---|
| Nguồn | 5 V 3 A (cấp thêm Pixhawk: dùng nguồn ≥ 3.5 A) | 5 V 5 A (USB-C PD) |
| Tản nhiệt | Tản nhiệt + quạt (2 nhân chạy NCNN liên tục: không quạt sẽ hạ xung từ 80 °C) | Active Cooler |
| Camera | Camera Module 3 Wide, cáp 15-pin | Camera Module 3 Wide, cáp 22-pin ↔ 15-pin |
| CPU | Cortex-A72: không tính fp16 (ncnn tự tính fp32) → s1 chậm hơn Pi 5 khoảng 2–3 lần | Cortex-A76, có fp16 |

LD06 cắm vào cổng USB của Pi qua bộ USB-UART CP210x.

Pixhawk nối qua TELEM2 → UART GPIO của Pi: xem **mục 1b** (nối dây, bật UART, tham số ArduPilot, dịch vụ `uavp-fc-bridge`, test cảm biến).

### Pi cần Internet để cài
Khi Pi đang phát Wi-Fi (192.168.4.1), nó không ra được Internet và `apt` báo `Temporary failure in name resolution`. Cách xử lý:
- cắm cáp LAN từ Pi vào router (Wi-Fi phát vẫn dùng được song song); hoặc
- tạm nối Wi-Fi của Pi vào mạng nhà để cài, cài xong chuyển lại.

### Cài thư viện: Pi 4 + Ubuntu 22.04 (đang dùng)
```bash
sudo apt update
sudo apt install -y cmake g++ pkg-config libopencv-dev libopencv-contrib-dev libyaml-cpp-dev libgtest-dev v4l-utils
sudo usermod -aG dialout,video $USER     # quyền /dev/ttyUSB0, /dev/ttyAMA0, camera -> đăng xuất / reboot
```
**libcamera:** Ubuntu 22.04 không có bản Raspberry Pi, phải build từ source (`github.com/raspberrypi/libcamera`, pipeline `rpi/vc4` cho Pi 4) rồi `sudo ninja -C build install && sudo ldconfig`. Thêm vào `~/.bashrc`:
```bash
export PKG_CONFIG_PATH=/usr/local/lib/aarch64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
```
Kiểm tra: `pkg-config --modversion libcamera` in số phiên bản; `cam -l` thấy `imx708`.
Kernel không nhận camera (`sudo dmesg | grep -i imx708` trống): thêm `camera_auto_detect=0` và `dtoverlay=imx708` vào `/boot/firmware/config.txt`, reboot.

### Cài thư viện: Raspberry Pi OS 64-bit (Pi 5 / Pi 4, Trixie hoặc Bookworm)
```bash
sudo apt update
sudo apt install -y cmake g++ pkg-config libopencv-dev libyaml-cpp-dev libcamera-dev libgtest-dev
sudo usermod -aG dialout $USER     # quyền đọc /dev/ttyUSB0 -> đăng xuất / reboot sau lệnh này
```

**ncnn (C++):** build từ source (khoảng 20–30 phút trên Pi 4). Pi 4 (Cortex-A72, ARMv8.0) không có fp16 arithmetic, `sdot`, `i8mm`, bf16, nên tắt các phần đó; bật OpenMP và `benchncnn` (chi tiết: [docs/PI4_NCNN.md](docs/PI4_NCNN.md)):
```bash
git clone --depth 1 --branch 20261020 https://github.com/Tencent/ncnn.git ~/ncnn && cd ~/ncnn
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_OPENMP=ON \
      -DNCNN_ARM82=OFF -DNCNN_ARM82DOT=OFF -DNCNN_ARM82FP16FML=OFF -DNCNN_ARM84BF16=OFF -DNCNN_ARM84I8MM=OFF \
      -DCMAKE_C_FLAGS="-mcpu=cortex-a72" -DCMAKE_CXX_FLAGS="-mcpu=cortex-a72" \
      -DNCNN_BUILD_BENCHMARK=ON -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$HOME/ncnn-install
cmake --build build -j2 && cmake --install build
~/ncnn/build/benchmark/benchncnn 10 2 0 -1 1      # 10 lan, 2 luong, CPU: kiem tra ncnn chay duoc
```
Pi đã có ncnn cũ (`~/ncnn-install`) thì build lại với các cờ trên rồi build lại `uav_perception_pi` (xoá `build/` trước).

### Chép thư mục từ máy tính (PowerShell, trong `object_detection\`)
```powershell
scp -r uav_perception_pi <user>@<ip-pi>:~/
```
`models/` nặng khoảng 20 MB; `parity_data/` (khoảng 70 MB, chỉ dùng cho test) không cần chép. Lần sau chỉ sửa code hoặc config thì chép riêng phần đó:
```powershell
scp -r uav_perception_pi/src uav_perception_pi/include uav_perception_pi/apps uav_perception_pi/configs uav_perception_pi/tools uav_perception_pi/tests uav_perception_pi/CMakeLists.txt <user>@<ip-pi>:~/uav_perception_pi/
```

### Build
```bash
cd ~/uav_perception_pi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dncnn_DIR=$HOME/ncnn-install/lib/cmake/ncnn
cmake --build build -j2
(cd build && ctest --output-on-failure)            # test_parity bỏ qua nếu không có parity_data/
```
**Đo tốc độ model trên Pi** (s1 cả ảnh + s2 một ROI, đúng code và config chạy thật; so fp16 storage bật / tắt):
```bash
./build/uavp_bench_model --cfg configs/run/real_time_perception.yaml --image runs/<phien>/frame.jpg --loops 30
./build/uavp_bench_model --cfg configs/run/real_time_perception.yaml --fp16 off \
    --set "cpu.cores=[0,1,2,3]" s1_detect.model.num_threads=4 s2_segment.model.num_threads=4      # 4 nhan
```
Mặc định là model **fp32** cho cả s1 và s2 (bản INT8 đã bỏ: trên Pi 4 không nhanh hơn mà kém chính xác).

CMake tự nhận Pi 4 / Pi 5 (đọc `/proc/device-tree/model`) và in `uavp: -mcpu=cortex-a72` (Pi 4) hoặc `cortex-a76` (Pi 5). Build sai CPU sẽ báo `Illegal instruction` khi chạy; ép tay: `-DUAVP_MCPU=cortex-a72`. Pi 4 RAM 2 GB bị treo khi build: `cmake --build build -j1`.

### Kiểm tra
```bash
cam -l                                # Ubuntu (libcamera tự build): thấy imx708_wide. Raspberry Pi OS: rpicam-hello --list-cameras
ls -l /dev/ttyUSB*                    # thấy /dev/ttyUSB0
cat /proc/device-tree/model; echo     # Raspberry Pi 5 / 4 Model B ...
```

---

## 1b. Pixhawk → Pi (TELEM2 → GPIO, UART) và test cảm biến

Pi chỉ ĐỌC MAVLink (không gửi lệnh cho FC). Dịch vụ **`uavp-fc-bridge`** chạy từ lúc Pi khởi động, giữ cổng UART, ghi tư thế EKF (world coordinate NED) vào RAM `/dev/shm/uavp_fc`. `uavp_real_time` và `uavp_sensor_check` đọc RAM đó (`sources.mavlink.source: shm`), nên nhiều chương trình dùng chung được, và khởi động lại perception không làm mất dữ liệu FC.

### Nối dây (3 dây, cả hai bên mức 3.3 V)

| Pixhawk TELEM2 (JST-GH 6 chân) | Pi 4 / Pi 5 (header 40 chân, cùng vị trí chân) |
|---|---|
| chân 1 `+5V` | Chỉ nối khi **Pi cấp nguồn cho Pixhawk** (bàn thử). Khi Pixhawk có power module / BEC riêng thì **bỏ dây này**: hai nguồn 5 V nối nhau sẽ chống nhau |
| chân 2 `TX` (FC gửi) | chân **10** = GPIO15 `RXD` |
| chân 3 `RX` (FC nhận) | chân **8** = GPIO14 `TXD` |
| chân 4, 5 `CTS`, `RTS` | không nối |
| chân 6 `GND` | chân **6** = GND |

TX nối RX (bắt chéo). Nối nhầm TX–TX thì không nhận được byte nào.

**Pi và Pixhawk bật nguồn cùng lúc** (Pi cấp 5 V cho Pixhawk):
- Pixhawk khởi động xong (vài giây) và phát MAVLink trước khi dịch vụ bridge trên Pi chạy (khoảng 20–30 s). Các tin phát trong khoảng đó bị bỏ, không ảnh hưởng gì: bridge đọc từ tin tiếp theo, và lúc mở cổng thì xoá rác còn trong bộ đệm.
- Pixhawk khởi động lại giữa chừng (sụt áp, reset): bridge vẫn chạy, đồng hồ FC tự đồng bộ lại ngay từ tin đầu tiên sau khi khởi động lại.
- Nguồn: Pixhawk cùng GPS, rangefinder, telemetry lấy thêm khoảng 0.3–1 A từ chân 5 V của Pi (Pi 4: adapter ≥ 3.5 A; Pi 5: 5 A). Kiểm tra sụt áp: `cat /sys/devices/platform/soc/soc:firmware/get_throttled` (Ubuntu) hoặc `vcgencmd get_throttled`, phải là `0` / `0x0`.

### Bật UART GPIO trên Pi 4 + Ubuntu 22.04 (đang dùng)
Trên Pi 4, UART tốt (PL011) mặc định dành cho Bluetooth; GPIO 14/15 chỉ có mini UART (`ttyS0`, tốc độ trôi theo xung CPU, hay lỗi ở 921600). Tắt Bluetooth để GPIO 14/15 thành PL011 = `/dev/ttyAMA0`:
```bash
sudo nano /boot/firmware/config.txt
#   thêm cuối file:
#   enable_uart=1
#   dtoverlay=disable-bt
sudo nano /boot/firmware/cmdline.txt        # XOÁ "console=serial0,115200" (nếu có), giữ nguyên phần còn lại trên 1 dòng
sudo systemctl disable --now hciuart 2>/dev/null; sudo systemctl mask serial-getty@ttyAMA0.service serial-getty@ttyS0.service
sudo reboot
ls -l /dev/ttyAMA0 /dev/serial0              # serial0 -> ttyAMA0
groups | grep dialout
```
Ubuntu không có `raspi-config`: làm tay như trên. Test nhanh: cắm Pixhawk, `./build/uavp_fc_bridge --cfg configs/run/real_time.yaml` (dừng dịch vụ trước: `sudo systemctl stop uavp-fc-bridge`) phải thấy `HEARTBEAT 1.0/s`.

### Bật UART GPIO trên Pi 5 (Raspberry Pi OS)
```bash
sudo raspi-config        # Interface Options -> Serial Port: login shell "No", serial port hardware "Yes"
sudo nano /boot/firmware/config.txt
#   thêm cuối file (Pi 5: UART0 trên GPIO 14/15 -> /dev/ttyAMA0):
#   dtparam=uart0=on
sudo reboot
ls -l /dev/ttyAMA0                     # phải có
grep -o "console=[^ ]*" /boot/firmware/cmdline.txt   # KHÔNG được có console=serial0 / ttyAMA0
groups | grep dialout                  # user trong nhóm dialout
```

### Tham số ArduPilot (Mission Planner → Config → Full Parameter List), TELEM2 = SERIAL2

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `SERIAL2_PROTOCOL` | 2 | MAVLink 2 |
| `SERIAL2_BAUD` | 921 | 921600 baud (= `sources.mavlink.baudrate`; dây dài / nhiễu: 115 và đổi config thành 115200) |
| `BRD_SER2_RTSCTS` | 0 | không dùng chân CTS / RTS |
| `SR2_EXTRA1` | 50 | ATTITUDE 50 Hz |
| `SR2_POSITION` | 20 | LOCAL_POSITION_NED 20 Hz: vị trí + **vận tốc** (cần EKF có gốc toạ độ + nguồn vị trí: GPS / optical flow / VIO) |
| `SR2_EXTRA2` | 10 | VFR_HUD (vận tốc dự phòng khi chưa có LOCAL_POSITION_NED) |
| `SR2_EXTRA3` | 10 | EXTENDED_SYS_STATE (đang bay hay chưa), OPTICAL_FLOW (**độ cao** `ground_distance`), DISTANCE_SENSOR |

HEARTBEAT (đã arm, chế độ bay) luôn được gửi, không cần tham số. Kiểm tra từng tin bằng Mission Planner → `Ctrl+F` → MAVLink Inspector, hoặc `python tools/gcs_command.py --connect <cổng> status` trên máy tính.

Ghi tham số rồi khởi động lại Pixhawk.

### Cài dịch vụ bridge (một lần, sau khi build)
```bash
cd ~/uav_perception_pi
bash tools/install_fc_bridge.sh          # cài + bật khi khởi động + chạy ngay
journalctl -u uavp-fc-bridge -f          # log mỗi 5 s: HEARTBEAT / ATTITUDE / LOCAL_POSITION_NED / rangefinder (tin/s)
./build/uavp_fc_bridge --cfg configs/run/real_time.yaml --print    # đọc RAM: roll / pitch / yaw, NED, rangefinder mỗi 1 s
```
Log đúng: `HEARTBEAT 1.0/s | ATTITUDE 50/s (...) | LOCAL_POSITION_NED 20/s | ...`. Các cảnh báo:
- `CHUA THAY PIXHAWK`: không có HEARTBEAT (dây, baud, `SERIAL2_PROTOCOL`).
- `EKF chua co vi tri`: Pixhawk chạy nhưng EKF chưa có gốc toạ độ / nguồn vị trí. Khi có, world coordinate tự xuất hiện.

Bridge tự mở lại cổng nếu lỗi, và vẫn chạy khi Pixhawk khởi động sau Pi. Bridge ghim vào nhân 1 (`fc_bridge.cores`); perception dùng nhân 2, 3. Sửa config xong: `sudo systemctl restart uavp-fc-bridge`. Gỡ: `bash tools/install_fc_bridge.sh --remove`.

Không dùng bridge: `--set sources.mavlink.source=serial` (chương trình tự mở `/dev/ttyAMA0`; phải `sudo systemctl stop uavp-fc-bridge` trước). Không có Pixhawk: `--set sources.mavlink.enabled=false`.

### Test nhanh: camera + LiDAR + Pixhawk (không chạy model)
```bash
./build/uavp_sensor_check --cfg configs/run/real_time.yaml --seconds 10
```
Mỗi giây in: `camera 30.0 khung/s (tre .. ms) | LiDAR 10.0 vong/s, ~450 diem/vong | ATTITUDE 50/s, NED 20/s, range 10/s | FC roll .. pitch .. yaw .. | NED (x, y, z) m`.

Kết quả trong `runs/<ngày-giờ>_sensor_check/`:

| File | Nội dung |
|---|---|
| `overlay.jpg` | Khung cuối + điểm LiDAR chiếu lên ảnh (đỏ gần, lam xa) + tư thế FC: kiểm `T_CL` (đường quét phải nằm đúng trên sàn / vật) |
| `lidar_world.csv` | Điểm LiDAR 1 s cuối: hệ camera và **hệ thế giới NED** (`p_W = T_WC(t) p_C`, tư thế FC lúc quét) |
| `summary.json` | Tần số từng nguồn, tư thế cuối, **điểm mặt đất dưới tâm ảnh** trong hệ camera và hệ thế giới |

Kiểm tra kết hợp world coordinate: để UAV (cầm tay) nghiêng / xoay chậm trên sàn phẳng rồi chạy lại vài lần. `ground_under_center.p_W_mm` (điểm sàn dưới tâm ảnh) dịch theo chỗ camera nhìn, nhưng toạ độ z của các điểm sàn trong `lidar_world.csv` phải gần như không đổi. Nếu z đổi nhiều khi chỉ nghiêng thì `configs/calib/T_BC.yaml` (camera → thân FC, **chưa hiệu chuẩn**) sai.

---

## 2. Hai chương trình chạy trên Pi

| | `uavp_real_time_perception` | `uavp_real_time` |
|---|---|---|
| Config | `configs/run/real_time_perception.yaml` | `configs/run/real_time.yaml` |
| Chạy | 4 stage: s1 detect → s2 segment → s3 select → s4 fusion | Cả chương trình: s1 → s3 → khoá vật (`p_W`) → bám vật → s4 |
| Pixhawk | Không dùng | Bắt buộc (qua `uavp_fc_bridge`) |
| Bắt đầu | **SPACE** trên máy tính (`tools/remote_viewer.py`) hoặc `s` trên terminal | Lệnh từ máy tính **gửi qua Pixhawk** (`tools/gcs_command.py start`) + Pi tự nhận UAV đang hover |
| D camera → sàn (s3: ước lượng W để loại vật quá to cho tay gắp) | Cố định 1 m (`distance.fixed_m`) | Độ cao optical flow lúc bắt đầu × cos(roll) cos(pitch) + `distance.camera_offset_m` |
| Dùng để | Đo thời gian chạy thực tế từng stage, phần cứng Pi (CPU, RAM, nhiệt, tần số), sai số s4 so với ArUco | Chạy thật khi bay |

Cả hai đều gửi ảnh, LiDAR, thời gian, độ trễ, phần cứng về máy tính nếu `remote.enabled: true`, và ghi `runs/<ngày-giờ>_<tên>/`. Lệnh phải chạy trong `~/uav_perception_pi`. Không chạy chương trình khác đang giữ camera / cổng LD06 cùng lúc. `Ctrl+C` lúc nào cũng dừng được.

### 2a. `uavp_real_time_perception`: đo thời gian, phần cứng, sai số (không Pixhawk)
1. Trên Pi (SSH):
   ```bash
   cd ~/uav_perception_pi
   ./build/uavp_real_time_perception --cfg configs/run/real_time_perception.yaml --set display.show=false
   ```
   Log hiện `Cho may tinh ket noi: python tools/remote_viewer.py --host <ip-pi> --port 8100`.
2. Trên máy tính (PowerShell, trong `object_detection\uav_perception_pi`):
   ```powershell
   ..\detection\.venv\Scripts\python.exe tools\remote_viewer.py --host <ip-pi>
   ```
3. Đặt vật (có ArUco nếu đo sai số), bấm **SPACE** trong cửa sổ viewer. Viewer hiện:
   - ảnh + điểm LiDAR (đỏ gần, lam xa), track (vàng DEFER, xanh OK, đỏ ABORT);
   - thời gian s1 / s2 / s3, **độ trễ** (SensorTimestamp → xong khung, đo trên Pi), median / p95 cộng dồn;
   - **phần cứng Pi** mỗi 1 s: CPU cả máy, nhân bận nhất, CPU của chương trình, RAM, nhiệt độ, tần số thấp nhất;
   - xong: `final.jpg` có 3 trục vật; terminal máy tính in **sai số s4 − ArUco** (d, dist, p_obj, độ nghiêng) và bảng thời gian (median / p95 từng bước).
   - **q** / ESC: dừng sớm.

Máy tính lưu `remote_runs/<ngày-giờ>/`: `status.jsonl` (từng khung), `hw.jsonl` (phần cứng), `result.json`, `summary.txt`, `final.jpg`. Pi lưu `runs/<ngày-giờ>_real_time_perception/` (`timing.csv` có cột `latency_ms`, `hw_usage.csv`, `result.json`...).

Không có máy tính: `--set remote.enabled=false`, bấm `s` trên terminal SSH. Số đo "sạch" (gửi ảnh tốn ~3–5 ms / ảnh trên cùng 2 nhân): `--set remote.max_fps=0` (chỉ gửi số liệu, không gửi ảnh) hoặc tắt remote.

### 2b. `uavp_real_time`: chạy thật, lệnh từ máy tính qua Pixhawk
Đường lệnh: **máy tính (GCS) → Pixhawk (telemetry / USB) → TELEM2 → `uavp_fc_bridge` trên Pi → `uavp_real_time`**.
Bridge gửi HEARTBEAT của Pi (sysid 1, compid 191) lên Pixhawk mỗi 1 s, nhờ đó Pixhawk biết chuyển lệnh có đích compid 191 sang TELEM2. Lệnh là `COMMAND_LONG` `MAV_CMD_USER_1` (31010): `param1 = 1` bắt đầu, `0` dừng; bridge trả `COMMAND_ACK`.

1. Trên Pi: bridge đang chạy (`systemctl status uavp-fc-bridge`), rồi:
   ```bash
   ./build/uavp_real_time --cfg configs/run/real_time.yaml --set display.show=false
   ```
   Log: `>>> Cho lenh bat dau tu may tinh ... + hover ... <<<`, mỗi 2 s một dòng điều kiện còn thiếu.
2. Trên máy tính, nối với Pixhawk theo một trong hai cách:
   - **Không mở Mission Planner:** cổng COM của telemetry radio / USB Pixhawk (Device Manager → Ports), ví dụ `COM5:57600` (USB: `COM5:115200`).
   - **Đang mở Mission Planner:** `Ctrl+F` → `MAVLink` → chọn `UDP Client` (hoặc `UDP Host`) `127.0.0.1:14550`, tick **Write**; rồi dùng `udp:0.0.0.0:14550`.
3. Xem trạng thái hover (tuỳ chọn), rồi gửi lệnh:
   ```powershell
   ..\detection\.venv\Scripts\python.exe tools\gcs_command.py --connect COM5:57600 status   # arm, chế độ bay, đang bay, vận tốc, độ cao flow
   ..\detection\.venv\Scripts\python.exe tools\gcs_command.py --connect COM5:57600 start    # -> "Pi tra loi COMMAND_ACK: ACCEPTED"
   ..\detection\.venv\Scripts\python.exe tools\gcs_command.py --connect COM5:57600 stop     # dừng giữa chừng
   ```
4. Pi **chỉ bắt đầu** khi có lệnh `start` (gửi sau lúc chương trình bắt đầu chờ) **và đủ điều kiện hover** (`configs/run/real_time.yaml` → `start`):

   | Điều kiện | Tin MAVLink | Mặc định |
   |---|---|---|
   | đã arm, đúng chế độ bay | HEARTBEAT | `modes: [5, 16]` (LOITER, POSHOLD) |
   | đang bay | EXTENDED_SYS_STATE | `landed_state = IN_AIR` |
   | vận tốc gần 0 | LOCAL_POSITION_NED (dự phòng VFR_HUD) | ngang ≤ 0.15 m/s, dọc ≤ 0.10 m/s |
   | độ cao ổn định 1 s | OPTICAL_FLOW `ground_distance` (dự phòng rangefinder) | dao động ≤ 5 cm trong 1 s, ≥ 0.3 m |

   Đủ thì log `BAT DAU: lenh may tinh + hover (do cao cam bien ... m -> camera - san ... m)`; khoảng cách này dùng cho s3. Sau đó: chọn vật → khoá vật (`p_W`) → bám vật → s4 cuối như mục 3.
5. Lệnh `stop` lúc đang chạy → dừng ngay, vẫn ghi kết quả.

**Bắt đầu chỉ bằng lệnh từ máy tính** (mặc định hiện tại, `start.require_hover: false`): `gcs_command.py start` (qua Pixhawk) **hoặc** SPACE trên `remote_viewer` (`start.accept_viewer_key: true`) là chạy ngay, không cần arm / bay. Khi bay thật muốn Pi tự chờ hover: `--set start.require_hover=true`.

**Pi đã chạy mavlink-routerd** (giữ UART TELEM2 ↔ GPIO, trạm mặt đất qua UDP 14550): bridge **không** mở UART mà nhận qua một đầu ra riêng của router (`sources.mavlink.link: udp`, mặc định). Thêm vào `/etc/mavlink-router/main.conf`:
```ini
[UdpEndpoint uavp]
Mode = Normal
Address = 127.0.0.1
Port = 14551
```
rồi `sudo systemctl restart mavlink-router` và cài bridge (`bash tools/install_fc_bridge.sh`). Không có mavlink-routerd: `--set sources.mavlink.link=serial` (bridge mở thẳng `/dev/ttyAMA0`). Hai chương trình **không được** mở chung một UART.

**Kiểm tra Pixhawk → Pi trước khi chạy** (không cần camera / model):
```bash
./build/uavp_fc_check --cfg configs/run/real_time.yaml --seconds 20     # trong lúc chạy: gcs_command.py start / stop
```
Mỗi giây in tần số từng loại tin, HEARTBEAT (arm, chế độ), landed, tư thế, vị trí / vận tốc NED, độ cao optical flow (`ground_distance`, quality → D camera–sàn), lệnh nhận được; cuối cùng kết luận ĐẠT / CẢNH BÁO / LỖI. Bridge không chạy: `sudo systemctl stop uavp-fc-bridge` rồi thêm `--set sources.mavlink.source=serial`.

**Log Pixhawk theo quyết định** (`runs/<phiên>/fc_events.jsonl`, chỉ `uavp_real_time`): mỗi dòng JSON = 1 sự kiện (`wait_start`, `start_gate`, `start`, `s1_detect`, `target_selected`, `target_ok`, `target_abort`, `s4_first`, `lock`, `lock_end`, `s4_final`, `stop`, `finish`) + chi tiết + trạng thái Pixhawk lúc đó (arm, chế độ, landed, vận tốc, độ cao optical flow / rangefinder, tư thế, vị trí NED, lệnh).

Viewer (`remote_viewer.py`) vẫn dùng được để xem: trước khi bắt đầu, nó hiện điều kiện nào đang thiếu (`CHO LENH + HOVER: ...`).

### Ghi đè tham số
```bash
--set aruco.check_aruco=false                                     # vật không dán ArUco
--set sources.lidar_port=/dev/ttyUSB1                             # LD06 ở cổng khác
--set s4_fusion.min_pts=30 max_frames=30 monitor.period_s=0.25
--set "cpu.cores=[0,1,2,3]" s1_detect.model.num_threads=4 s2_segment.model.num_threads=4   # dùng 4 nhân
--set lock_track.max_duration_s=5 "start.modes=[2,5,16]" start.max_speed_m_s=0.2   # uavp_real_time
--set distance.fixed_m=0.7                                        # uavp_real_time_perception: camera cách sàn 0.7 m
```
Chọn vật: chỉ xét **1 vật gần tâm ảnh**. s1 chạy lúc bắt đầu, khi vật bị loại (chuyển sang vật gần tâm tiếp theo) hoặc khi mất vật (s2 không ra mask); các khung khác chỉ chạy s2 cho vật đó. ROI của s2 lấy trên ảnh letterbox 960×544 (`s2_segment.roi_image_px`), giống lúc train.
Ý nghĩa từng khoá: [configs/README.md](configs/README.md).

### Có màn hình (HDMI / VNC)
```bash
./build/uavp_real_time_perception --cfg configs/run/real_time_perception.yaml --set remote.enabled=false   # display.show: true
```
Cửa sổ xem trước trên màn hình Pi hiện camera và đường quét LiDAR; bấm `s` (hoặc SPACE) để quét, `q` để thoát. Hiển thị tốn CPU, nên **khi đo thời gian thì dùng `display.show=false`**.

### Mất Wi-Fi giữa chừng
Chạy trong `tmux`: `tmux new -s uav`, rồi chạy lệnh. Mất kết nối thì SSH lại và `tmux attach -t uav`.

### Chạy offline (thư mục ảnh, video, một ảnh)
```bash
./build/uavp_offline --cfg configs/run/offline.yaml --source <thư mục ảnh | video>
./build/uavp_offline --cfg configs/run/offline.yaml --image <ảnh>      # thêm s1_detect.jpg, s2_segment.jpg, s3_select.jpg
```

---

## 3. Kết quả: `runs/<ngày-giờ>_real_time/`

| File | Nội dung |
|---|---|
| `run.log` | Toàn bộ log của lần chạy |
| `config.yaml` | Config đã dùng (đã gộp file stage, file model và `--set`) |
| `result.json` | `selected`; `fusion` (`p_obj_mm` = tâm mặt top, `axes` x / y / z, `tilt_deg`, `normal_observable`, `pose_compensated`, `drift_mm_s`, `d_mm`, W / L); `lock` (`p_W_mm` hệ thế giới NED, `p_W_lock_mm`, `pose_mode` world / rotation / none, số lần segment, số khung flow / NCC); `aruco`; `timing_ms`; `session`; `device` (nhân CPU, input s1...); `hw_usage` |
| `timing.csv` | Pha chọn vật, mỗi khung một dòng: `capture_ms`, `s1_ms` (trống nếu khung không chạy s1), `s2_ms`, `s3_ms`, `loop_ms` |
| `track.csv` | Pha bám, mỗi khung một dòng: chế độ tư thế, flow / NCC, số điểm flow tốt, tâm Kalman, tâm dự đoán từ FC, `p_C` từ `p_W`, có segment không, thời gian |
| `hw_usage.csv` | Mỗi `monitor.period_s` một dòng: giai đoạn, CPU cả máy, nhân bận nhất, CPU của chương trình (tối đa 400 %), RSS, RAM, nhiệt độ, tần số |
| `object.jpg` | **Ảnh cuối từ camera (1920 x 1080) đã ghép mask vật (tô màu + viền), dấu tâm vật, toạ độ tâm (px; p_obj, dist mm) và 3 trục vật** |
| `final.jpg` | Như `object.jpg` + điểm LiDAR trong mask (xanh inlier, đỏ outlier) + dòng số đo (d, W / L, sai số ArUco...) |
| `frame.jpg`, `selected_mask.png`, `lidar_points.csv` | Ảnh camera gốc của khung đó, mask (PNG, 255 = vật), điểm LiDAR trong mask (toạ độ lúc đo + đã bù chuyển động, khung, vòng quét, inlier) |

Lấy phiên về máy tính (PowerShell, trong `object_detection\uav_perception_pi`, sau khi chương trình trên Pi chạy xong):
```powershell
.\tools\fetch_run.ps1 -Pi <user>@<ip-pi>                                # phiên mới nhất -> runs_pi\<tên phiên>\
.\tools\fetch_run.ps1 -Pi <user>@<ip-pi> -Kind real_time_perception     # phiên mới nhất của uavp_real_time_perception
.\tools\fetch_run.ps1 -Pi <user>@<ip-pi> -All                           # cả thư mục runs/
```
PowerShell chặn script lạ: chạy trước `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (một lần). Không dùng script:
```powershell
ssh <user>@<ip-pi> "ls -td ~/uav_perception_pi/runs/*"                 # liệt kê phiên, mới nhất ở trên
scp -r <user>@<ip-pi>:~/uav_perception_pi/runs/<tên phiên> .\runs_pi\
```

---

## 4. So sánh Pi 4 / Pi 5 (hoặc C++ / Python cũ)

**Cách đo cho công bằng:**
- cùng `configs/`, cùng cảnh, `display.show=false`;
- governor `performance`:
  ```bash
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  ```
- chạy 1 lần làm nóng máy rồi bỏ kết quả; sau đó mỗi máy chạy 3–5 lần.

Gom các thư mục `runs/` vào một chỗ rồi chạy (chỉ cần Python 3):
```bash
python3 tools/compare_runs.py runs/<pi4_1> runs/<pi4_2> runs/<pi5_1> runs/<pi5_2> --csv so_sanh.csv
```

| Chỉ số | Ý nghĩa | Cần để ý |
|---|---|---|
| `khung_s`, `tong_ms` | Số khung s1→s3 mỗi giây, thời gian từ lúc bấm `s` tới hết s4 | Càng cao / thấp càng tốt |
| `s1_ms` … `s4_ms`, `vong_ms` | Median từng bước, cả vòng một khung | s1 chiếm phần lớn |
| `cpu_tb_%`, `tien_trinh_%` | CPU cả máy, CPU của chương trình lúc chạy model | Tối đa khoảng 400 % (4 nhân) |
| `rss_max_MB`, `nhiet_max_C` | RAM của chương trình, nhiệt độ lớn nhất | Pi 5 hạ xung từ khoảng 85 °C |
| `tan_so_min_MHz`, `throttle_cuoi` | Tần số thấp nhất, cờ `vcgencmd get_throttled` | Khác `0x0` thì từng thiếu nguồn hoặc quá nhiệt, kết quả không đại diện |

---

## 5. Lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `Khong co file model .../models/...` | Chép `models/` lên Pi; chạy lệnh trong `~/uav_perception_pi` |
| `Camera dang bi giu ...` | Tắt chương trình khác đang giữ camera (`rpicam-*`, một `uavp_*` khác: `pgrep -af uavp_`). Vẫn lỗi thì `sudo reboot` |
| `Khong mo duoc LD06 /dev/ttyUSB0` | `ls /dev/ttyUSB*`; `groups` phải có `dialout`; LD06 đã quay chưa |
| `LD06 ...: khong co vong quet ...` | LD06 chưa có nguồn 5 V, dây TX chưa nối, hoặc sai cổng |
| `net LensPosition ... != 1.25` | Camera không khoá nét đúng file K |
| `luc nhan - SensorTimestamp = ...` | Đồng hồ camera và LiDAR không khớp: gửi log để xem |
| `Camera khong ho tro 1920x1080 BGR888 ...` | libcamera đổi cấu hình; gửi log (in cấu hình libcamera đề xuất) |
| `stdin khong phai terminal ... bat dau ngay` | Đang chạy qua `ssh host "lệnh"` hoặc `nohup`: dùng SSH bình thường hoặc `ssh -t` |
| `Bo dem LiDAR khong phu het phien` | Phiên dài hơn khoảng 60 s: tăng `sources.lidar_buffer_scans` |
| `find_package(ncnn)` lỗi khi cmake | Chưa cài ncnn C++; build từ source rồi thêm `-Dncnn_DIR=...` (mục 1) |
| `MAVLink: khong co ATTITUDE sau connect_timeout_s` | Sai cổng / baud, chưa đặt `SERIAL2_PROTOCOL = 2`, `SR2_EXTRA1 = 0`, hoặc serial console của Pi còn bật |
| `KHOA vat khong co vi tri FC (chi co goc quay)` | EKF không có LOCAL_POSITION_NED và không có rangefinder: bám bằng homography quay + flow, không có `p_W` |
| `normal_observable: false` | UAV gần như không trôi vuông góc mặt quét LiDAR trong lúc bám: chỉ đo được nghiêng dọc mặt quét; tăng `lock_track.max_duration_s` |
| `cpu.cores: nhan ... khong ton tai` | Máy ít nhân hơn (PC / WSL): `--set "cpu.cores=[]"` |
| `throttle_cuoi` khác `0x0` | Nguồn yếu (bit 0/16) hoặc nóng (bit 2/3/18/19): đổi nguồn, thêm quạt rồi đo lại |
