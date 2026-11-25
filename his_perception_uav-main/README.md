# uav_perception_pi

Phần mềm nhận thức (perception) bằng C++ chạy **trực tiếp trên Raspberry Pi 4 / Pi 5** cho UAV gắp vật. Camera nhìn thẳng xuống, UAV hover. Mỗi phiên chương trình chọn **một vật gần tâm ảnh**, đo tâm và hướng mặt trên của vật bằng camera + LiDAR LD06, khoá vật trong hệ toạ độ thế giới (EKF của Pixhawk qua MAVLink) rồi bám vật. Được điều khiển và theo dõi từ máy tính.

- Hướng dẫn cài đặt và chạy từng bước trên Pi: **[GUIDE.md](GUIDE.md)**
- Ý nghĩa từng khoá config: [configs/README.md](configs/README.md) · Model: [models/README.md](models/README.md)
- Ghi chú phần cứng Pi 4 và NCNN: [docs/PI4_NCNN.md](docs/PI4_NCNN.md) · Quy tắc viết code: [CPP_RULES.md](CPP_RULES.md)

## Pipeline

| Bước | Việc | Model | Code | Config |
|---|---|---|---|---|
| **s1 detect** | Ảnh 1920 × 1080 → letterbox 960 × 544 → box vật. Chỉ chạy lúc bắt đầu, khi vật bị loại hoặc khi mất vật | YOLO26n, NCNN fp32 | [src/stages/s1_detect/](src/stages/s1_detect/) | [stages/s1_detect.yaml](configs/stages/s1_detect.yaml), [models/s1_detect_ncnn.yaml](configs/models/s1_detect_ncnn.yaml) |
| **s2 segment** | ROI quanh box (cắt trên ảnh 960 × 544, như lúc train) → mask mặt trên, tâm, yaw. 1 vật / khung | YOLO26n-seg, NCNN fp32, 320 | [src/stages/s2_segment/](src/stages/s2_segment/) | [stages/s2_segment.yaml](configs/stages/s2_segment.yaml), [models/s2_segment_ncnn.yaml](configs/models/s2_segment_ncnn.yaml) |
| **s3 select** | Chỉ xét **1 vật gần tâm**; đo bề rộng W từ mask; DEFER / OK / ABORT; ABORT → vật gần tâm tiếp theo | — | [src/stages/s3_select/](src/stages/s3_select/), [src/pipeline/](src/pipeline/) | [stages/s3_select.yaml](configs/stages/s3_select.yaml) |
| **khoá + bám** (`uavp_real_time`) | p_W = T_WC · p_C; mỗi khung: dự đoán ROI từ tư thế Pixhawk, optical flow / NCC, Kalman; s2 theo nhịp LiDAR | — | [src/tracking/](src/tracking/) | [stages/lock_track.yaml](configs/stages/lock_track.yaml), [calib/T_BC.yaml](configs/calib/T_BC.yaml) |
| **s4 fusion** | LiDAR quanh các khung có mask → bù chuyển động → RANSAC mặt phẳng → tâm (x, y, z = d) + 3 trục, W, L | — | [src/stages/s4_fusion/](src/stages/s4_fusion/) | [stages/s4_fusion.yaml](configs/stages/s4_fusion.yaml), [calib/T_CL.yaml](configs/calib/T_CL.yaml) |

## Chạy ở đâu: Pi hay máy tính

### Trên Raspberry Pi (build bằng CMake → `build/`)

| Chương trình | Nguồn | Việc | Cần Pixhawk |
|---|---|---|---|
| `uavp_real_time` | [apps/real_time.cpp](apps/real_time.cpp) | **Chương trình chính**: chờ lệnh từ máy tính → s1–s3 → khoá → bám → s4; ghi log Pixhawk theo từng quyết định | Có |
| `uavp_real_time_perception` | [apps/real_time_perception.cpp](apps/real_time_perception.cpp) | Chỉ 4 stage (không khoá / bám), D cố định; đo thời gian, phần cứng, sai số so với ArUco | Không |
| `uavp_fc_bridge` | [apps/fc_bridge.cpp](apps/fc_bridge.cpp) | **Dịch vụ nền** (systemd): Pixhawk → RAM `/dev/shm/uavp_fc`; gửi HEARTBEAT, trả `COMMAND_ACK` | Có |
| `uavp_fc_check` | [apps/fc_check.cpp](apps/fc_check.cpp) | Kiểm tra Pixhawk → Pi: tần số từng loại tin, tư thế, độ cao optical flow, lệnh nhận được → ĐẠT / LỖI | Có |
| `uavp_sensor_check` | [apps/sensor_check.cpp](apps/sensor_check.cpp) | Kiểm tra nhanh camera + LiDAR + Pixhawk (không model) | Tuỳ chọn |
| `uavp_bench_model` | [apps/bench_model.cpp](apps/bench_model.cpp) | Đo thời gian s1, s2 trên Pi (so fp16 storage bật / tắt) | Không |
| `uavp_offline` | [apps/offline.cpp](apps/offline.cpp) | Chạy pipeline trên thư mục ảnh / video (Pi hoặc Linux PC) | Không |
| `tools/install_fc_bridge.sh` + `tools/uavp-fc-bridge.service` | | Cài `uavp_fc_bridge` thành dịch vụ chạy từ lúc Pi khởi động | |

### Trên máy tính (Windows / Linux, Python 3)

| File | Việc | Cần |
|---|---|---|
| [tools/remote_viewer.py](tools/remote_viewer.py) | Xem trực tiếp ảnh camera + đường LiDAR + mask + thời gian + CPU Pi; **SPACE** = bắt đầu, **q** = dừng | `pip install opencv-python numpy` |
| [tools/gcs_command.py](tools/gcs_command.py) | Gửi lệnh `start` / `stop` tới Pi **qua Pixhawk**; `status`: xem arm, chế độ bay, độ cao | Python 3 (+ `pyserial` khi dùng cổng COM) |
| [tools/fetch_run.ps1](tools/fetch_run.ps1) | PowerShell: tải phiên chạy từ Pi về `runs_pi/` | `ssh`, `scp` |
| [tools/compare_runs.py](tools/compare_runs.py) | So sánh thời gian / phần cứng giữa nhiều phiên (`result.json`) | Python 3 |
| [tools/quantize_int8.sh](tools/quantize_int8.sh) | (Tuỳ chọn, Linux / WSL) Lượng tử model NCNN sang INT8. Mặc định **không dùng**, xem docs/PI4_NCNN.md | Công cụ ncnn |
| [tools/dump_parity.py](tools/dump_parity.py) | (Phát triển) Sinh dữ liệu so khớp với bản Python cũ cho `tests/test_parity.cpp` | Dự án Python cũ |

## Các kết nối

```
                                   ┌────────────────────────── Raspberry Pi ───────────────────────────┐
 Pixhawk TELEM2 ──UART 921600──▶ GPIO 14/15 (/dev/ttyAMA0)                                           │
   (3 dây: TX, RX, GND)          │   ├─(a) mavlink-routerd ──UDP 14550──▶ trạm mặt đất (điều khiển bay) │
                                 │   │                     └─UDP 127.0.0.1:14551─┐                      │
                                 │   └─(b) không có router: bridge mở thẳng UART─┤                      │
                                 │                                    uavp_fc_bridge (dịch vụ)         │
                                 │                                    └─▶ /dev/shm/uavp_fc (RAM)        │
                                 │                                          └─▶ uavp_real_time, fc_check│
 LD06 ──USB (CP2102)──▶ /dev/serial/by-id/...  ──▶ uavp_real_time                                   │
 Camera Module 3 ──CSI (libcamera)──────────────▶ uavp_real_time                                   │
                                   └───────────────────────────────────────────────────────────────────┘
 Máy tính ──SSH (22)──────────────▶ terminal Pi, scp, fetch_run.ps1
          ──TCP 8100 (Wi-Fi/LAN)──▶ remote_viewer.py  (Pi → máy tính: ảnh, LiDAR, trạng thái; máy tính → Pi: SPACE / q)
          ──MAVLink (COM / UDP)───▶ gcs_command.py → Pixhawk / mavlink-routerd → Pi  (lệnh start / stop)
```

| Kết nối | Phương thức | Ghi chú |
|---|---|---|
| **Pixhawk → Pi** | UART TELEM2 ↔ GPIO 14/15 (`/dev/ttyAMA0`, 921600, MAVLink 2) | Pi 4: cần `dtoverlay=disable-bt`. Pixhawk: `SERIAL2_PROTOCOL = 2`, `SERIAL2_BAUD = 921`, `SR2_*` (GUIDE.md mục 1b) |
| ↳ (a) **có mavlink-routerd** (khuyên dùng khi còn điều khiển bay qua Pi) | Router giữ UART; bridge nhận qua UDP `127.0.0.1:14551` (`sources.mavlink.link: udp`) | Thêm endpoint `[UdpEndpoint uavp]` (mục cấu hình bên dưới). **Không** để 2 chương trình mở chung 1 UART, **không** dùng chung cổng 14550 của trạm mặt đất |
| ↳ (b) **không có router** | Bridge mở thẳng `/dev/ttyAMA0` (`sources.mavlink.link: serial`) | |
| **Bridge → perception** | Shared memory `/dev/shm/uavp_fc` (vòng đệm seqlock) | Nhiều chương trình đọc cùng lúc; bridge tự tạo lại file nếu bị xoá |
| **Máy tính → Pi: lệnh bắt đầu / dừng** | (1) **SPACE / q trên `remote_viewer.py`** (TCP 8100) hoặc (2) **`gcs_command.py start / stop`**: `COMMAND_LONG` `MAV_CMD_USER_1` (31010) tới sysid 1 / compid 191, đi qua Pixhawk hoặc router | (2) Pi trả `COMMAND_ACK`. Kết nối của `gcs_command.py`: `COM5:57600` (USB / radio Pixhawk), `udp:0.0.0.0:<port>` (nghe), `udpout:<ip>:<port>` (gửi tới) |
| **Máy tính ↔ Pi: xem trực tiếp** | TCP 8100: Pi gửi JPEG + điểm LiDAR + JSON trạng thái; máy tính gửi phím | `remote.*` trong config |
| **Máy tính ↔ Pi: terminal, file** | SSH / scp | `fetch_run.ps1` lấy kết quả |
| **LD06 → Pi** | USB–UART CP2102, 230400 8N1 | Dùng đường dẫn `/dev/serial/by-id/...` (tên `ttyUSBx` đổi theo thứ tự cắm) |
| **Camera → Pi** | CSI, libcamera, 1920 × 1080 (sensor mode 2304 × 1296), lật 180° | `calib/K_1920x1080.yaml` |

## Biến cấu hình khi chạy với Pixhawk

Tất cả nằm trong [configs/run/real_time.yaml](configs/run/real_time.yaml) (code không có giá trị mặc định: thiếu khoá → báo lỗi). Ghi đè tạm: `--set khoa.con=gia_tri`.

**Nhận MAVLink (`sources.mavlink`)**

| Khoá | Mặc định | Ý nghĩa |
|---|---|---|
| `enabled` | `true` | `uavp_real_time` cần Pixhawk |
| `source` | `shm` | Perception đọc: `shm` = RAM do bridge ghi · `serial` = tự mở `port` (bridge phải tắt) |
| `link` | `udp` | Bridge nhận từ: `udp` = mavlink-routerd · `serial` = mở thẳng `port` |
| `udp_address`, `udp_port` | `127.0.0.1`, `14551` | Địa chỉ bridge nghe = endpoint `uavp` của mavlink-routerd |
| `port`, `baudrate` | `/dev/ttyAMA0`, `921600` | UART TELEM2 (chỉ khi `link: serial` / `source: serial`) |
| `shm_name`, `shm_poll_ms` | `/uavp_fc`, `5` | File RAM; chu kỳ đọc |
| `companion_sysid`, `companion_compid` | `1`, `191` | Danh tính của Pi (= `SYSID_THISMAV`; `MAV_COMP_ID_ONBOARD_COMPUTER`). Lệnh từ máy tính gửi tới đây |
| `command_id` | `31010` | `MAV_CMD_USER_1`: `param1 = 1` bắt đầu, `0` dừng |
| `max_age_s`, `latency_ms`, `buffer_samples` | `0.2`, `0`, `1024` | Mẫu tư thế quá cũ coi như không có; trễ bù thêm; số mẫu ATTITUDE giữ lại |
| `use_local_position`, `use_rangefinder` | `true`, `true` | Dùng vị trí NED (p_W) / rangefinder hướng xuống |

**Dịch vụ bridge (`fc_bridge`)**: `cores` (`[1]`), `shm_capacity` (`4096`), `status_period_s` (`5`, log + kiểm tra file RAM), `reopen_wait_s` (`2`), `heartbeat_period_s` (`1`).

**Điều kiện bắt đầu (`start`)**

| Khoá | Mặc định | Ý nghĩa |
|---|---|---|
| `require_hover` | `false` | `false`: **chỉ cần lệnh** từ máy tính (test trên bàn). `true`: lệnh **và** Pi tự nhận đang hover |
| `accept_viewer_key` | `true` | SPACE trên `remote_viewer.py` cũng là lệnh bắt đầu |
| `modes` | `[5, 16]` | Chế độ bay ArduCopter được phép (5 LOITER, 16 POSHOLD) |
| `max_speed_m_s`, `max_vspeed_m_s` | `0.15`, `0.10` | Đứng yên: vận tốc ngang / dọc |
| `max_height_change_m`, `hover_window_s`, `min_height_m` | `0.05`, `1.0`, `0.3` | Độ cao ổn định trong cửa sổ |
| `max_msg_age_s`, `status_period_s` | `1.5`, `2.0` | Tin quá cũ; chu kỳ log điều kiện còn thiếu |

**Khoảng cách camera → sàn cho s3 (`distance`)**: `source` (`fc`: độ cao optical flow / rangefinder của Pixhawk × cos(roll) cos(pitch) + `camera_offset_m` · `fixed`: `fixed_m`), `fixed_m` (`1.0`, dùng khi FC không có độ cao), `camera_offset_m` (`-0.05`: cảm biến ToF cao hơn camera 5 cm).

**Hiệu chuẩn liên quan**: [calib/T_BC.yaml](configs/calib/T_BC.yaml) camera → thân Pixhawk (dự đoán ROI, p_W; **hiện là giá trị tạm**), [calib/T_CL.yaml](configs/calib/T_CL.yaml) LiDAR → camera, [calib/K_1920x1080.yaml](configs/calib/K_1920x1080.yaml) camera.

**mavlink-routerd** (`/etc/mavlink-router/main.conf`, khi dùng `link: udp`):
```ini
[UdpEndpoint uavp]          # cho uavp_fc_bridge
Mode = Normal
Address = 127.0.0.1
Port = 14551

[UdpEndpoint gcs_cmd]       # (tuỳ chọn) cho gcs_command.py trên máy tính: udpout:<ip-pi>:14552
Mode = Server
Address = 0.0.0.0
Port = 14552
```

**ArduPilot** (TELEM2 = SERIAL2): `SERIAL2_PROTOCOL = 2`, `SERIAL2_BAUD = 921`, `SR2_EXTRA1 = 20` (ATTITUDE), `SR2_POSITION = 10` (LOCAL_POSITION_NED), `SR2_EXTRA2 = 5` (VFR_HUD), `SR2_EXTRA3 = 10` (OPTICAL_FLOW, DISTANCE_SENSOR), `SR2_EXT_STAT = 2` (EXTENDED_SYS_STATE).

## Chạy nhanh

**Build trên Pi** (chi tiết cài thư viện: GUIDE.md):
```bash
cd ~/uav_perception_pi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dncnn_DIR=$HOME/ncnn-install/lib/cmake/ncnn
cmake --build build -j2 && (cd build && ctest --output-on-failure)
```

**Pixhawk → Pi** (một lần cài, mỗi lần kiểm tra):
```bash
bash tools/install_fc_bridge.sh                                     # cài dịch vụ (sau khi build)
sudo systemctl restart uavp-fc-bridge                               # sau mỗi lần build lại
journalctl -u uavp-fc-bridge -n 20 --no-pager                       # "Nhan MAVLink tu mavlink-routerd ..."
./build/uavp_fc_check --cfg configs/run/real_time.yaml --seconds 20 # ĐẠT / CẢNH BÁO / LỖI
```

**Chạy chương trình chính** — trên Pi:
```bash
./build/uavp_real_time --cfg configs/run/real_time.yaml
```
Trên máy tính:
```bash
python tools/remote_viewer.py --host <ip-pi>                        # xem; SPACE = bắt đầu, q = dừng
python tools/gcs_command.py --connect COM5:57600 start              # hoặc lệnh qua Pixhawk (status / stop)
```
```powershell
.\tools\fetch_run.ps1 -Pi <user>@<ip-pi> -Kind real_time            # lấy kết quả về runs_pi\
```

**Chỉ đo perception** (không Pixhawk, D cố định): `./build/uavp_real_time_perception --cfg configs/run/real_time_perception.yaml --set distance.fixed_m=<D>`, rồi SPACE trên viewer.

## Kết quả mỗi phiên (`runs/<ngày-giờ>_<chương trình>/` trên Pi)

| File | Nội dung |
|---|---|
| `result.json` | Vật được chọn, s4 (tâm, 3 trục, W, L, độ nghiêng), khoá (`p_W_mm`), ArUco và sai số, thời gian từng bước, thiết bị, phần cứng |
| `fc_events.jsonl` | (`uavp_real_time`) Mỗi dòng = 1 quyết định (`start_gate`, `s1_detect`, `target_*`, `lock`, `s4_final`, `stop`, ...) + trạng thái Pixhawk lúc đó |
| `timing.csv`, `track.csv`, `hw_usage.csv` | Thời gian từng khung (chọn vật), từng khung bám, CPU / RAM / nhiệt |
| `frame.jpg`, `final.jpg`, `object.jpg`, `selected_mask.png`, `lidar_points.csv` | Ảnh gốc, ảnh có mask + tâm + LiDAR + 3 trục, mask, điểm LiDAR |
| `run.log`, `config.yaml` | Log, config đã gộp (sau `--set`) |

## Cấu trúc thư mục

```
uav_perception_pi/
├── apps/       chương trình (bảng "Chạy ở đâu"); live_app.cpp: phần chung của real_time và real_time_perception
├── configs/    run/ (real_time, real_time_perception, offline) · stages/ (s1–s4, lock_track) · models/ · calib/
├── include/uavp/, src/
│   ├── stages/     s1_detect · s2_segment · s3_select · s4_fusion
│   ├── tracking/   world_pose (FC → T_WC, p_W), roi_flow (LK + NCC), roi_kalman, locked_tracker
│   ├── pipeline/   session (chọn 1 vật), start_gate (điều kiện bắt đầu)
│   ├── model/      ncnn_model (pool allocator, tiền xử lý ncnn)
│   ├── sources/    pi_camera (libcamera), ld06, mavlink (giải mã, tư thế FC), fc_shm (RAM bridge ↔ perception), serial_port
│   ├── support/    remote_view (TCP 8100), hw_monitor, aruco_check, draw, cpu_affinity, terminal_keys
│   └── common/     config, log, run_files, report, boxes
├── models/     model NCNN (chỉ source.yaml + README trong repo)
├── tests/      GTest: từng stage, tracking, MAVLink, điều kiện bắt đầu, config, parity
├── tools/      chạy trên máy tính / script cài dịch vụ (bảng "Chạy ở đâu")
├── docs/       PI4_NCNN.md
└── GUIDE.md, CPP_RULES.md, CMakeLists.txt
```

## Test

```bash
cmake -S . -B build -DUAVP_WITH_NCNN=OFF -DUAVP_WITH_LIBCAMERA=OFF   # PC Linux / WSL: chỉ logic, không model / camera
cmake --build build -j && (cd build && ctest --output-on-failure)
```
`test_parity` tự bỏ qua khi không có `parity_data/`.

## Quy ước

- Mọi tham số nằm trong `configs/` (không có giá trị mặc định ẩn trong code); tên biến ghi rõ đơn vị (`_px`, `_mm`, `_m`, `_ns`).
- Ảnh trong pipeline là RGB uint8; toạ độ ảnh theo OpenCV; hệ thế giới = NED của EKF Pixhawk (mm), thân = FRD; mọi luồng dùng chung đồng hồ `CLOCK_MONOTONIC`.
- File hiệu chuẩn có `version` và `date`; hiệu chuẩn lại thì sao lưu bản cũ, ghi bộ số mới, tăng `version`.
