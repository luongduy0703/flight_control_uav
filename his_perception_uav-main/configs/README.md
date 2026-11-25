# configs/

Mọi tham số nằm ở đây; code không có giá trị mặc định, thiếu khoá thì báo lỗi lúc khởi động. Đường dẫn trong các file tính từ thư mục gốc `uav_perception_pi/`.

```
configs/
├── run/                    # cấu hình CHẠY: file stage, calib, nơi ghi kết quả, số nhân CPU + phần cứng
│   ├── real_time.yaml      #   uavp_real_time: cpu.cores, camera, LD06, MAVLink (Pixhawk), cửa sổ / phím, ArUco
│   └── offline.yaml        #   uavp_offline: thư mục ảnh / video / một ảnh
├── stages/                 # tham số TỪNG STAGE (thuật toán, ngưỡng)
│   ├── s1_detect.yaml      #   cả ảnh (full) / tile, conf, NMS  ── model_cfg → models/s1_detect_ncnn.yaml
│   ├── s2_segment.yaml     #   ROI quanh bbox, ngưỡng, chọn instance  ── model_cfg → models/s2_segment_ncnn.yaml
│   ├── s3_select.yaml      #   chọn nhóm, n_min, W tối đa + tracking (s1 mỗi N khung, vận tốc ROI)
│   ├── lock_track.yaml     #   khoá + bám vật sau s3: tư thế FC, optical flow / NCC, Kalman, segment theo LiDAR
│   └── s4_fusion.yaml      #   LiDAR quanh khung có mask, RANSAC mặt phẳng top (bù chuyển động), lọc tầm / góc
├── models/                 # MODEL NCNN: file, lớp, PIXEL INPUT, tiền xử lý, số luồng
│   ├── s1_detect_ncnn.yaml #   YOLO26n, input 960 x 544 (cả ảnh)
│   └── s2_segment_ncnn.yaml#   YOLO26n-seg, input 320 x 320
└── calib/                  # hiệu chuẩn (một bản chính, có version / date; sao lưu bản cũ vào _backup/)
    ├── K_1920x1080.yaml    #   K + méo + sensor_mode (camera mở đúng như lúc hiệu chuẩn)
    ├── T_CL.yaml           #   LiDAR → camera, quy ước góc LD06, hiệu chỉnh tầm
    └── T_BC.yaml           #   camera → thân FC (Pixhawk). CHƯA HIỆU CHUẨN (giá trị danh nghĩa)
```

## Cách nạp

1. `run/*.yaml` liệt kê các file stage trong mục `stages`.
2. Mỗi file stage được nạp vào mục cùng tên (`s1_detect`, ..., `lock_track`).
3. Stage có model (s1, s2) trỏ tới file model bằng `model_cfg`. File model được nạp vào mục `model` của stage đó.
4. Ghi đè lúc chạy bằng `--set <mục>.<khoá>=<giá trị>`:
   ```bash
   --set max_frames=30 s3_select.n_min=3
   --set s1_detect.mode=tile "s1_detect.model.input_px=[640,640]"     # quay về 8 tile 640 (model tile cũ)
   --set "cpu.cores=[0,1,2,3]" s1_detect.model.num_threads=4            # dùng 4 nhân
   --set sources.mavlink.enabled=false lock_track.max_duration_s=5
   ```
5. Toàn bộ config đã gộp (sau `--set`) được ghi ra `runs/<ngày-giờ>_*/config.yaml`.

## Pixel đầu vào model (`models/*.yaml`)

| | s1 detect | s2 segment |
|---|---|---|
| `input_px` | **[960, 544]** (chữ nhật, bội số của 32) | **320** (vuông) |
| Ảnh đưa vào | Cả ảnh 1920 x 1080 (`mode: full`), 1 lần chạy model / khung | ROI vuông quanh bbox, cạnh = `roi_scale` x cạnh dài bbox |
| Cách đưa về `input_px` | Letterbox: thu 0.5 lần → 960 x 540, đệm `pad_value` (114) 2 px trên / dưới | Resize thẳng về 320 x 320 |
| `interpolation` | `linear` | `linear` |
| Chuẩn hoá (cố định trong code) | RGB uint8 / 255 → float32 CHW | RGB uint8 / 255 → float32 CHW |

`input_px` phải bằng kích thước lúc train / export. Model s1 hiện tại (`models/det_yolo26n_960x544_s0_ncnn_fp32`) được train trên ảnh đã letterbox sẵn 960 × 544 và export `[960, 544]` fp32 (2026-11-16; test mAP50 96.7, recall 97.0; nguồn trong `source.yaml` của model). Model tile 640 cũ (`det_yolo26n_v3ft_s2_ncnn_fp16`) vẫn còn để quay lại chế độ 8 tile.

## Hai nhân CPU (`cpu.cores`)

Cả chương trình (luồng chính s1–s4, NCNN, camera, LD06, MAVLink, log, theo dõi phần cứng) được ghim vào `cpu.cores` (mặc định `[2, 3]`) ngay đầu `main()`, trước khi tạo luồng nào. NCNN chạy 2 luồng (`num_threads: 2`) trên đúng 2 nhân đó; các luồng I/O chủ yếu ngủ (poll / chờ) và chen vào giữa các lần chạy model. Nhân 0, 1 để cho hệ điều hành và chương trình khác (FC bridge, ROS...).
