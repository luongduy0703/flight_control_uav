# planA: ước lượng mặt top của vật

Chạy **trên Pi**. Chương trình nối phần điều khiển (`pi/offboard_control.py`) với phần perception
(`his_perception_uav-main`, chạy `build/uavp_real_time` như một tiến trình con).

| File | Vai trò |
|---|---|
| `config/scenario.yaml` | Tham số kịch bản: độ cao hover, dung sai, `e_max_mm`, ground truth (x = 0.635, y = 0.255 m), hệ trục |
| `run_plan_a.py` | Chương trình chính (máy trạng thái), nhận lệnh từ terminal |
| `plan_a.py` | Logic thuần: đổi hệ tọa độ, setpoint, chỉnh căn LiDAR, sai số, báo cáo (test: `tests/test_plan_a.py`) |
| `imu_diag.py` | Phân tích rung từ IMU Pixhawk cuối mỗi lần chạy (ghi bằng `pi/imu_logger.py`) |
| `video_diag.py` | Chẩn đoán rung / mờ video cuối mỗi lần chạy (chạy lại được trên laptop) |
| `make_launch_marker.py` | Tạo tờ A4 in ArUco điểm cất cánh (`markers/`), dùng để đánh giá |

Phần perception sửa cho kịch bản này (nằm trong `his_perception_uav-main`):
- `src/stages/s4_fusion/lidar_align.cpp`: e_px (tâm vật → đường LiDAR trên ảnh), e_mm = e_px·d/f, vector chỉnh vị trí.
- `apps/live_app.cpp`: in `TARGET` khi khóa vật; cổng `lock_track.align_gate` (chỉ segment và tích lũy khi e_mm ≤ e_max)
  và dòng `ALIGN`; `OBJECT` (tâm, pháp tuyến, yaw thế giới); ảnh `start.jpg`, `detect.jpg`, `aligned.jpg`.
- `world_plane.cpp`: `fill_world` → `yaw_W_deg`, `axis_x_W` trong result.json.
- `src/support/launch_marker.cpp`: bám ArUco điểm cất cánh trong vòng chờ (READY → cất cánh → hover), in dòng
  `MARKER` và ghi `launch_marker.csv`. Bật bằng `--set launch_marker.enabled=true` (planA tự thêm khi `marker.enabled`).

## Chạy

```bash
# Pi, terminal 1: router (giữ TELEM1)
cd ~/uav/controller
bash tools/run_hardware_router.sh 192.168.50.143 /dev/ttyAMA0
# Pi, terminal 2: planA (python3 hệ thống có yaml; mavsdk lấy từ .venv)
cd ~/uav/controller
PYTHONPATH=$(ls -d .venv/lib/python3*/site-packages) python3 planA/run_plan_a.py   # hoặc thêm --config <file khác>
# Laptop: xem trực tiếp camera + detect + LiDAR
cd ~/Downloads/control
/usr/bin/python3 his_perception_uav-main/tools/remote_viewer.py --host 192.168.50.1 --port 8100
```
Thư mục trên Pi: `~/uav/controller` (điều khiển + planA), `~/uavp_planA` (perception, `perception.dir` trong `scenario.yaml`).
Nhật ký và lệnh đầy đủ: [README_2026-11-21.md](README_2026-11-21.md).

Sau `start_perception`, khi perception khóa vật (dòng `TARGET`), planA **đề xuất** điểm bay (gõ `go` mới bay) ở độ cao hover sao cho **đường quét
LiDAR cắt qua tâm bbox** của vật đã chọn (`approach.target: bbox`, `approach.lidar_line: true`). Độ lệch của đường quét
so với thân UAV tính từ `T_CL` + `T_BC` (v1) của perception, ở độ sâu camera → mặt vật (khoảng +10.5 cm theo trục x
thân, −2.9 cm theo y: camera ở trước Pixhawk 13 cm, đường quét sau camera ~2.4 cm). Sau đó bước ALIGN vẫn chỉnh tối đa 3 lần như cũ. Mọi setpoint vẫn theo EKF.

### Quy trình an toàn (chương trình KHÔNG tự arm / cất cánh / bay)

Mở console chỉ khởi động perception và chờ. Mỗi bước nguy hiểm phải gõ lệnh **và** xác nhận:

| Bước | Gõ | Chương trình làm |
|---|---|---|
| 0 | `check` | In bảng kiểm tra: kết nối, armed/landed, chế độ bay, sức khỏe PX4, vị trí EKF, perception |
| 1 | `arm` → `yes` | In bảng kiểm tra, hỏi lại; `yes` trong 8 s mới ARM (motor quay, UAV vẫn trên sàn). `no` = hủy. `disarm` để tắt |
| 2 | `takeoff` → `yes` | Chỉ nhận khi **đã ARM** (PX4 tự disarm sau ~10 s → phải `arm` lại). Bay lên 1.0 m tại chỗ rồi HOLD |
| 3 | `s_p` (hoặc `start_perception`) | Sau khi hover ổn định (`perception.auto_start: false`) |
| 4 | `go` / `skip` | Mỗi lần bay tới vật, chỉnh LiDAR, hạ độ cao: in **ĐỀ XUẤT** (tọa độ, khoảng cách) và chờ. `go` = bay, `skip` = bỏ, UAV giữ HOLD. `go` chỉ nhận khi UAV đang HOLD |
| 5 | `land` | Hạ cánh (gõ được bất cứ lúc nào, hủy mọi xác nhận / đề xuất đang chờ) |
| 6 | `exit` | Chỉ thoát khi UAV đã ở dưới đất và disarm (`exit force` để thoát cưỡng bức) |

Bảo vệ thêm: mất terminal / SSH (SIGHUP, stdin đóng) khi đang ARM hoặc bay → tự gửi `land`, chờ chạm đất rồi mới thoát.
Ctrl+C khi đang bay phải bấm 2 lần trong 3 s. Tắt từng cổng trong `scenario.yaml` khối `control:` (không khuyến khích).

Kịch bản bay (`scenario.yaml`):

1. `takeoff` → hover cao 1.0 m (`flight.hover_height_m`, tính từ Pixhawk lúc cất cánh).
2. Hover ổn định 5 s (`hover_stabilize_s`): perception đo ArUco cất cánh → **EKF trôi** (marker lúc đặt → lúc hover).
3. Gõ `start_perception`. Không có `TARGET` sau `perception.detect_timeout_s` → dừng perception, UAV **giữ hover**, in báo cáo.
4. Có vật → đề xuất bay để đường quét LiDAR cắt tâm bbox (`go`) → ALIGN (đề xuất chỉnh, `go`) → s4 → in tâm + chiều cao vật.
5. Đề xuất hạ tại chỗ xuống `approach.final_height_m` = 0.8 m (`go`, hoặc `skip` = giữ 1.0 m) → HOLD, chờ `land`.
Lúc nào cũng gõ được `land`, `stop` (dừng perception), `status`, `check`.

## Kết quả

`planA/logs/<giờ UTC>/`:
- `plan_a.log`: toàn bộ màn hình và trạng thái mỗi giây.
- `perception-*.log`: log perception.
- `perception/`: `start.jpg`, `detect.jpg`, `aligned.jpg`, `final.jpg`, `object.jpg`, `result.json`, `run.log`.
- `summary.json`: vị trí, yaw, pháp tuyến, sai số, các mốc thời gian.

Gốc hệ kịch bản = **mặt sàn ngay dưới Pixhawk lúc cất cánh** (giống `handoff_uav`): z = z_EKF − (z Pixhawk lúc cất cánh
+ `frame.fc_above_ground_m` = 0.25). Sàn có z = 0, mặt top cao h có z = −h. Đặt `ground_truth.z_m` (= −chiều cao mặt top
so với sàn) thì có sai số Euclid 3D; để `null` thì chỉ có sai số 2D. Cất cánh từ giá / hộp thì đổi `fc_above_ground_m`.

## ArUco điểm cất cánh: chỉ để đánh giá

Marker này chỉ dùng để so tọa độ vật (do perception ước lượng trong hệ EKF) với vị trí đo bằng thước. Nó **không đi vào
vòng điều khiển**: takeoff, bay tới vật và chỉnh LiDAR vẫn dùng EKF NED như cũ.

1. `python3 planA/make_launch_marker.py` → in `markers/launch_dict_4x4_50_id1_100mm_A4.pdf` ở chế độ **100 %**.
   Đo cạnh viền đen thật bằng thước rồi ghi vào `marker.side_mm`. Cắt theo đường xám để giữ viền trắng 20 mm
   (`--border-mm`, tối thiểu 1 ô = cạnh / 6), nhất là khi sàn tối màu.
2. Dán marker phẳng trên sàn, kẻ trục x, y theo bốn vạch tâm. Đo tâm mặt trên của vật bằng thước theo hệ marker
   (x theo mũi tên, y bên phải, z = −chiều cao) rồi ghi vào `marker.ground_truth`.
3. Đặt UAV sao cho **camera** nằm trên tâm marker, mũi UAV theo +x. Khi gõ `takeoff`, chương trình in vị trí camera
   so với tâm marker và độ lệch mũi. Nếu `require_at_takeoff: true` mà lệch quá `launch_max_offset_m` /
   `launch_max_yaw_deg`, hoặc không thấy marker, lệnh takeoff bị từ chối. `status` hiện vị trí camera theo marker.
4. Khi cất cánh và hover, perception đo marker 5 lần/giây (`MARKER` → tư thế marker trong EKF). Lúc gõ `start_perception`,
   planA lấy trung vị của 3 s hover cuối làm hệ marker trong EKF, rồi dừng đo marker.
5. Kết quả có thêm phần **DANH GIA THEO ArUco**. Marker **không phải gốc hệ world**:
   - Thước (đo từ tâm marker) được đổi sang hệ kịch bản qua **vị trí đặt UAV**: camera so với marker (ảnh lúc takeoff)
     + `T_BC` + độ lệch mũi. Không đi qua EKF.
   - Vật (perception, EKF) − thước đã đổi = sai số vật; kèm chiều cao vật so với sàn.
   - "EKF trôi" = vị trí marker trong EKF lúc hover − lúc đặt UAV.
   `summary.json → marker` lưu mọi mẫu đo; `perception/launch_marker.csv` lưu từng lần đo.

## Chẩn đoán rung / mờ video (`video_diag`)

**Mặc định TẮT từ 2026-11-22** (đã đủ số liệu, xem `runs_pi/planA_20261122T*`). Bật lại: `enabled: true` trong `scenario.yaml`.

Perception đo **mọi khung ảnh ở mọi pha**: chờ trên sàn, arm, cất cánh, hover, quét. Kết quả ghi vào
`video_diag.csv` (code ở `his_perception_uav-main/src/support/video_diag.cpp`; bật bằng `video_diag.enabled` trong
`scenario.yaml`).

Mỗi khung đo:
- khoảng cách khung (rớt khung), phơi sáng, gain, lens;
- độ nét: phương sai Laplacian và gradient x/y ở ô giữa ảnh;
- ảnh dịch giữa 2 khung, và độ lệch dải trên − dải dưới ảnh (rung trong lúc đọc rolling shutter);
- tốc độ quay camera lấy từ ATTITUDE của FC, độ mờ dự đoán = |ω|·f·t_phơi_sáng, độ dịch ảnh dự đoán từ góc quay;
- trạng thái FC (armed, landed).

**Cuối lần chạy** (`exit`, Ctrl+C, lỗi) planA dừng perception rồi ghi vào thư mục log:
`video_diag.csv`, `video_diag_report.txt` / `.json`, `video_diag.png`. Báo cáo cũng in ra màn hình.
Gõ `video` để xem báo cáo tạm giữa chừng. Phân tích lại trên laptop:
```bash
/usr/bin/python3 planA/video_diag.py runs_pi/planA_<giờ>
```

Báo cáo so sánh các pha để tách nguồn rung:

| So sánh | Kết luận |
|---|---|
| Arm trên sàn (motor quay, chưa bay) nét kém hẳn lúc tắt motor | Rung cơ khí từ motor / cánh quạt truyền tới giá camera |
| Dải trên và dải dưới ảnh dịch khác nhau khi bay | Rung tần số cao trong lúc đọc cảm biến ("jello") |
| Dịch ảnh khớp với góc quay FC (R² cao), độ mờ dự đoán ≥ 1.5 px | Lắc thân UAV (PID tư thế, gió) |
| Dịch ảnh còn lại sau khi trừ phần quay | Trôi vị trí |
| Rớt khung, phơi sáng dài, gain / độ sáng khác nhau, lens đổi | Lỗi camera / ánh sáng (hoặc so sánh bị lệch) |

**Để có đủ đoạn so sánh:** sau `Perception READY` để UAV tắt motor khoảng 5 s. Sau đó gõ `arm` và chờ khoảng 5 s
(PX4 tự disarm sau 10 s nếu không cất cánh), rồi mới `takeoff`.

## Log IMU Pixhawk: tần số rung (`imu_log`)

**Mặc định TẮT từ 2026-11-22** (đã đủ số liệu, xem `runs_pi/planA_20261122T*`). Bật lại: `enabled: true` trong `scenario.yaml`.

`pi/imu_logger.py` ghi telemetry IMU từ lúc kết nối Pixhawk tới lúc thoát. Nó chỉ xin PX4 tăng tốc độ gửi tin
(MAV_CMD_SET_MESSAGE_INTERVAL), không gửi lệnh bay, và trả lại tốc độ mặc định khi thoát.

| File | Nội dung |
|---|---|
| `imu_highres.csv` | HIGHRES_IMU 200 Hz: gyro (rad/s), accel (m/s²) |
| `imu_vibration.csv` | VIBRATION: chỉ số rung PX4 (từ accel chưa lọc) + số lần clipping |
| `imu_motors.csv` | SERVO_OUTPUT_RAW: mức motor (µs) |
| `imu_attitude.csv` | tốc độ quay từ ATTITUDE (giữ tốc độ mặc định vì perception dùng) |
| `imu_state.csv`, `imu_meta.json` | armed / landed (chia pha); tham số lọc IMU và log của PX4 |

Cột `rx_ns` cùng đồng hồ với `video_diag.csv`, nên ghép được từng giây với độ nét ảnh.

**Cuối lần chạy** ghi `imu_diag_report.txt` / `.json` / `imu_diag.png`. Gõ `imu` để xem báo cáo tạm. Phân tích lại:
`/usr/bin/python3 planA/imu_diag.py runs_pi/planA_<giờ>`.

Báo cáo theo pha (tắt motor / arm trên sàn / bay) gồm:
- chỉ số VIBRATION (dưới 30 m/s² là tốt, trên 60 là xấu) và clipping;
- tần số rung chính từ phổ gyro, kèm biên độ;
- rung đó làm ảnh mờ bao nhiêu px trong thời gian phơi sáng thật, và lệch rolling shutter bao nhiêu;
- rung tăng bao nhiêu lần khi motor quay;
- tương quan rung với độ nét video theo từng giây.

**Giới hạn:** HIGHRES_IMU đã qua bộ lọc của PX4 (`IMU_GYRO_CUTOFF`, `IMU_ACCEL_CUTOFF`) và chỉ lấy mẫu 200 Hz.
Rung cánh quạt trên 100 Hz sẽ bị gập về tần số thấp hơn (alias) hoặc bị lọc mất. Chỉ số VIBRATION vẫn bắt được
những rung này, nhưng không cho biết tần số. Muốn có phổ thật thì dùng log thẻ SD của Pixhawk với raw IMU FIFO
(`SDLOG_PROFILE`).

Khi đẩy code lên Pi, chép thêm `pi/imu_logger.py` vào `~/uav/controller/pi/`.

## Thay doi 2026-11-22 (ha 0.8 m truoc khi quet, cua so thich ung, do cao vat tu LiDAR)

- Thu tu moi: bay toi vat (`go`) -> **ha xuong `approach.final_height_m` = 0.8 m** (`go` / `skip`) -> gui lenh
  **QUET** (COMMAND_LONG param1 = 2) -> ALIGN + tich luy -> ket qua -> HOLD cho `land`. Khong con buoc ha sau ket qua.
- Perception (`lock_track.wait_scan_command=true`) van bam vat khi bay / ha nhung chi tich luy va ket thuc sau lenh
  QUET; bo cac khung co mask chup o 1.0 m.
- s4 world: cua so 2 s, mo rong tung 1 s toi `perception.max_window_s` (10 s) cho toi khi ra vector huong (PLANE).
  Log: `VECTOR HUONG: PLANE sau X s tich luy (cua so Y s)`; bao cao planA: dong "Vector huong". Het 10 s -> HEIGHT_ONLY.
- Hai buoc do khoang cach:
  1. Buoc detection: planA gui muc san EKF kem lenh bat dau -> perception tinh D = do cao camera tren san (EKF) ->
     canh ngan bbox x D / f > `s3_select.bbox_max_mm` (200 mm) -> bo detection ngay (qua to), khong segment.
  2. Sau segment (co mask): LiDAR tich luy, CHI diem trong mask -> mat top -> khoang cach chinh xac camera -> mat top
     (`plane_dist_mm`); toa do vat = tu the camera (EKF) + khoang cach LiDAR. Bao cao: dong "Khoang cach LiDAR".
- Perception tu choi mo cong LD06 neu chuong trinh khac dang giu cong (vd `ldlidar_ros2_node`): hai chuong trinh cung
  doc mot tty se chia nhau byte -> ~28% goi loi CRC (phien 090646Z / 091222Z).
