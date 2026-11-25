# models/

Model NCNN mà s1 detect và s2 segment nạp, theo khoá `dir` trong `configs/models/s1_detect_ncnn.yaml` và `configs/models/s2_segment_ncnn.yaml` (kèm pixel input, lớp, tiền xử lý). Mỗi thư mục chứa `model.ncnn.param`, `model.ncnn.bin` và `source.yaml` (nguồn gốc, số đo). **Chỉ `source.yaml` được commit**; file `.param` / `.bin` không có trong repo, chép riêng vào đây (ví dụ `scp -r models/<thư mục> <user>@<ip-pi>:~/uav_perception_pi/models/`).

| Thư mục | Model | Dùng khi |
|---|---|---|
| `det_yolo26n_960x544_s0_ncnn_fp32` | **s1 (mặc định)**: YOLO26n detect, train trên ảnh 960 × 544 (0.7 m + 1.3 m), input [960, 544], fp32 | Luôn |
| `seg_yolo26n_roi960_s1_ncnn_fp32` | **s2 (mặc định)**: YOLO26n-seg mặt trên, ROI cắt trên ảnh 960 × 544 (0.7 m), input 320, fp32 | Luôn (`s2_segment.roi_image_px: [960, 544]`) |
| `det_yolo26n_v3ft_s2_ncnn_fp16` | s1 cũ: chế độ 8 tile 640 | `tests/test_parity.cpp`; quay lại chế độ tile |
| `seg_yolo26n_roi07_ft1109_s1_ncnn_fp16` | s2 cũ: ROI cắt trên ảnh 1920 × 1080 | Chế độ cũ: `--set "s2_segment.roi_image_px=[0, 0]" s2_segment.model.dir=models/seg_yolo26n_roi07_ft1109_s1_ncnn_fp16` |

Cập nhật model: export từ phần train (dự án train riêng, ultralytics) rồi chép vào đây.
- **detect:** `YOLO(best.pt).export(format="ncnn", imgsz=[544, 960], nms=None)` (fp32), kiểm tra lại mAP so với PyTorch.
- **segment:** `YOLO(best.pt).export(format="ncnn", imgsz=320, half=False, nms=None)`; output phải là `out0 (38, 2100)`, `out1 (32, 80, 80)`.
- INT8 (không dùng mặc định: Pi 4 không nhanh hơn mà kém chính xác): `tools/quantize_int8.sh`, xem `docs/PI4_NCNN.md`.
