# Pi 4: phần cứng và hướng viết lại NCNN

Ghi ngày 2026-11-16. Dùng khi viết lại `src/model/ncnn_model.cpp`, export model, và chọn cấu hình chạy real-time.

## 1. Phần cứng (đo trên máy thật)

| Mục | Giá trị | Hệ quả |
|---|---|---|
| Máy | Raspberry Pi 4 Model B Rev 1.4, Ubuntu 22.04.5, kernel 5.15.0-1110-raspi | |
| CPU | 4 × Cortex-A72 r0p3, 1 cluster, 600–1800 MHz | 4 nhân giống nhau: không có nhân "lớn/nhỏ", ghim nhân nào cũng như nhau |
| Tập lệnh | ARMv8.0-A, `Features: fp asimd evtstrm crc32 cpuid` | Chỉ có NEON fp32 / int8 cơ bản |
| fp16 | **không có `fphp` / `asimdhp`** | Không tính được fp16. `use_fp16_arithmetic` vô tác dụng; fp16 chỉ dùng để **lưu** trọng số / activation (đổi qua lại bằng `vcvt`) |
| int8 dot product | **không có `asimddp`** (`sdot` / `udot`), không có `i8mm` | ncnn int8 chạy bằng `smull` / `smlal`: vẫn nhanh hơn fp32 nhưng không bằng CPU có `sdot` (Pi 5) |
| bf16 | không có | |
| L1d | 32 KiB / nhân (2-way, line 64 B) | Tile tính toán nhỏ |
| L1i | 48 KiB / nhân | |
| L2 | **1 MiB, chung cho 4 nhân** (16-way) | Activation ảnh lớn không vừa cache: một feature map 16 kênh × 480 × 272 × fp32 = 8.4 MB. Model bị giới hạn bởi băng thông RAM → giảm độ phân giải input có lợi gần tỉ lệ số pixel |
| RAM | 3.8 GB (LPDDR4) | |
| Đo được (phiên 2026-11-14, 2 luồng, fp16 storage) | s1 960×544: 711 ms; s2 320×320: ~195 ms / vật | Tỉ lệ s1/s2 khớp với PC (3.65 so với 3.81): không có lỗi bất thường, chỉ là CPU chậm |

## 2. Viết lại `NcnnModel` (không đổi model)

| Việc | Lý do |
|---|---|
| `use_fp16_arithmetic = false` luôn; tách `use_fp16_storage` thành tham số riêng và **đo cả hai** (bật / tắt) bằng `benchncnn` | A72 không tính fp16. fp16 storage giảm băng thông RAM (có lợi vì L2 nhỏ) nhưng tốn thêm lệnh đổi kiểu; chưa đo nên chưa biết bên nào nhanh hơn |
| Gán `opt.blob_allocator` và `opt.workspace_allocator` = `ncnn::UnlockedPoolAllocator` / `ncnn::PoolAllocator`, tạo một lần cho mỗi model | Hiện mỗi lần chạy, mỗi lớp `malloc` / `free` buffer mới |
| Giữ `opt.lightmode = true`, `use_packing_layout = true`, `use_winograd_convolution = true`, `use_sgemm_convolution = true` (mặc định ncnn) | Winograd 3×3 có lợi trên A72 fp32 |
| Đổi tiền xử lý sang `ncnn::Mat::from_pixels_resize(..., PIXEL_RGB, ...)` + `copy_make_border` (pad 114) + `substract_mean_normalize(0, 1/255)` | Resize + đổi kiểu bằng NEON của ncnn; bỏ buffer CHW float trung gian và một lần `memcpy` (~6 MB với 960×544) |
| Build ncnn trên Pi: `-DCMAKE_BUILD_TYPE=Release -DNCNN_OPENMP=ON -DNCNN_BUILD_BENCHMARK=ON -DNCNN_ARM82=OFF -DNCNN_ARM82DOT=OFF -DNCNN_ARM84BF16=OFF -DNCNN_ARM84I8MM=OFF`, cờ `-mcpu=cortex-a72` | Bỏ code cho tập lệnh Pi 4 không có; `benchncnn` để đo trực tiếp file `.param` |
| Đo trên Pi: `benchncnn` với `loop=20 num_threads=2/4 powersave=0 gpu=-1 cooling=1` cho từng model, từng cấu hình | Quyết định bằng số đo trên Pi, không suy từ PC |

## 3. Đề xuất tăng tốc (ưu tiên từ trên xuống)

Các số "ước lượng" tính theo tỉ lệ pixel / số luồng, **chưa đo trên Pi**.

| # | Phương án | Ước lượng | Đổi lại |
|---|---|---|---|
| 1 | **s1 input 640×384** (1920×1080 thu 1/3 → 640×360, đệm 12 px) thay 960×544 | s1 ~0.47× số pixel → ~330 ms (2 luồng) | Vật ở 1.3 m còn ~17–27 px: phải train lại ở 640 và đo recall 1.3 m trên test |
| 2 | **INT8** cho s1 và s2 (`ncnn2table` với ~100 ảnh hiệu chuẩn đúng kích thước chạy, rồi `ncnn2int8`) | thường 1.3–1.8× trên A72 (không có `sdot`) | Phải kiểm tra lại mAP như bước `verify_export` (giới hạn tụt 1 điểm); lớp đầu / lớp detect có thể phải giữ fp32 |
| 3 | **s2 nhỏ hơn và sắc hơn**: cắt ROI từ ảnh **1920×1080** (camera đã có sẵn, cắt không tốn gì) và cho s2 input **224 hoặc 256** thay 320 | s2 0.5–0.65× | ROI ở 1920 có cạnh ~228 px: thu về 256 vẫn giữ chi tiết, trong khi ROI cắt từ 960 (~114 px) bị phóng lên 320 → mờ và tốn gấp ~8 lần pixel. Nắp chai (đang kém nhất ở 960) có lợi nhất. Phải train lại s2 theo cách cắt này |
| 4 | **Chạy s1 ít hơn**: chỉ lúc bắt đầu và khi mất vật; giữa chừng bám bằng ROI dự đoán (s3) / optical flow (đã có) | bỏ phần lớn số lần s1 | Cần ngưỡng "mất vật" chắc chắn (không có mask N khung liên tiếp) |
| 5 | **s2 chỉ cho vật gần tâm nhất** trong pha chọn vật, vật khác chỉ khi vật đầu bị loại | s2 từ N vật xuống 1 vật / khung | Chọn chậm hơn nếu vật đầu bị loại |
| 6 | **3–4 luồng cho NCNN** khi chạy model (luồng LiDAR, camera rất nhẹ) thay 2 nhân cố định | ~1.4–1.7× | Cần quạt: 4 nhân liên tục dễ chạm 80 °C và hạ xung; theo dõi `get_throttled` |
| 7 | **Camera ra 2 luồng** (libcamera: `main` 1920×1080 cho s2, `lores` 960×540 hoặc 640×360 cho s1) | bỏ resize CPU (~5–10 ms) | ISP Pi 4 thu nhỏ miễn phí; luồng `lores` là YUV420 → đổi sang RGB (rẻ) |
| 8 | Một model seg duy nhất trên cả ảnh 640×384 (thay s1 + s2) | bỏ hẳn s2 | **Không khuyến nghị**: proto 160×96 → mỗi ô ~12 px ảnh gốc, biên mặt trên hộp quá thô |

Thứ tự thử: (6) và §2 trước vì không cần train lại → (1) + (3) train lại → (2) INT8 sau cùng, khi model đã chốt.

## 4. Đề xuất tăng độ chính xác

1. **Dữ liệu từ chính Pi 4** (libcamera tự build trên Ubuntu có thể cho màu / phơi sáng khác camera cũ), ở **~1.0 m** (độ cao bay thật; hiện chỉ có 0.7 m và 1.3 m). Cho `real_time_perception` lưu khung, chạy model hiện tại để gán nhãn nháp, sửa trong Labelme.
2. **Nhãn segment cho 1.3 m** và thêm nắp chai: test hiện chỉ có 7 nắp chai, IoU 0.60 ở ROI 960.
3. **Train đúng cách chạy**: ảnh / ROI cắt đúng như trên Pi (letterbox, nguồn ảnh của ROI, kích thước input). Mọi thay đổi ở Pi phải đi kèm thay đổi tương ứng lúc tạo dataset.
4. **Quyết định bằng tập test** (val đã dùng để chọn seed / epoch); báo cáo 3 seed.
5. **Kiểm tra lượng tử hoá**: mỗi model INT8 / fp16 chạy lại bộ đánh giá như `verify_export` trước khi chép lên Pi.

## 5. Đã làm (2026-11-16)

- `NcnnModel` viết lại: fp16 arithmetic luôn tắt, `use_fp16_storage` là khoá riêng (mặc định tắt), pool allocator cho blob + workspace, tiền xử lý bằng `from_pixels(_resize)` + `substract_mean_normalize` + `copy_make_border` của ncnn (`IInfer::run_rgb`; model giả trong test vẫn đi đường cũ). Đo: `uavp_bench_model` (GUIDE.md mục Build).
- s2 cắt ROI trên ảnh letterbox 960×544 (`s2_segment.roi_image_px`): chỉ vùng ROI của ảnh gốc được thu 0.5 lần, pixel trùng tuyệt đối với "letterbox cả ảnh rồi cắt" (test `LetterboxRoiMatchesResizeThenCrop`); mask / tâm trả về toạ độ ảnh gốc.
- Chọn vật: chỉ 1 vật gần tâm (`kTargets = 1`), bỏ `n_targets`; s1 chỉ chạy lúc bắt đầu / vật bị loại / mất vật, bỏ `detect_every_n_frames`.
- **Đã bỏ INT8 (2026-11-16)**: trên Pi 4 s1 INT8 744 ms so với fp16 711 ms (không nhanh hơn) mà nắp chai kém hẳn → dùng fp32 cho cả hai stage. Cách đã thử: INT8 bằng `tools/quantize_int8.sh` (ncnn2table KL + ncnn2int8), **trộn**: backbone + neck INT8, head (s1) / head + proto (s2) fp32.

| Model | Vị trí | Test (PC, python-ncnn) |
|---|---|---|
| s1 INT8 trộn (đã bỏ, không trong repo) | `models/det_yolo26n_960x544_s0_ncnn_int8` | mAP50 95.7, mAP50-95 77.3, recall 97.0 |
| s1 fp32 (**mặc định**) | `models/det_yolo26n_960x544_s0_ncnn_fp32` | mAP50 96.7, mAP50-95 82.0, recall 97.0 (= PyTorch) |
| s2 INT8 trộn (đã bỏ, không trong repo) | `models/seg_yolo26n_roi960_s1_ncnn_int8` | có mask 91.4 %, IoU mặt hộp 0.917, **nắp chai 0.51** |
| s2 fp32 (**mặc định**) | `models/seg_yolo26n_roi960_s1_ncnn_fp32` | có mask 97.1 %, IoU mặt hộp 0.925, nắp chai 0.77 |
| s1 / s2 cũ | `models/det_yolo26n_v3ft_s2_ncnn_fp16` (tile 640), `models/seg_yolo26n_roi07_ft1109_s1_ncnn_fp16` | s2 cũ cần `roi_image_px=[0, 0]` |

INT8 toàn bộ (cả head) kém hơn bản trộn ở s1 (recall test 93.0); ở s2 nắp chai kém như nhau → sai số nằm ở backbone với vật nhỏ. Còn phải đo trên Pi: INT8 có nhanh hơn fp32 đủ để đổi lấy độ chính xác không (A72 không có `sdot`). Trên WSL x86: s1 30 ms (INT8) / 44 ms (fp32), s2 8 / 10 ms.
