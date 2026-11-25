# Quy tắc viết code C++ (`uav_perception_pi`)

Áp dụng cho **toàn bộ code C++ của hệ thống**. Đọc file này trước mỗi lần viết hoặc sửa code C++, cho cả người và AI. Các quy tắc chung của repo (config trong YAML, đơn vị trong tên biến, ảnh RGB, calib một bản chính...) trong mục "Quy ước" của [README.md](README.md) vẫn áp dụng.

---

## 0. Nguyên tắc ưu tiên

Code được chia hai loại:
- **chỗ nóng**: chạy mỗi khung, mỗi scan hoặc mỗi phần tử (ô dự đoán của model, điểm LiDAR, pixel mask, gói LD06);
- **chỗ lạnh**: chạy một lần lúc khởi động hoặc khi có sự kiện (đọc config, nạp model, ghi kết quả, log).

Thứ tự áp dụng quy tắc trong tài liệu này:
1. **Đúng trước, nhanh sau.** Bản C++ phải khớp bản Python (bbox ≤ 1 px, mask IoU ≥ 0.99, W ≤ 1 mm) trước khi tối ưu. Kiểm bằng `tests/test_parity.cpp` với dữ liệu do `tools/dump_parity.py` sinh từ bản Python.
2. **Đo trước khi sửa.** Chỉ tối ưu bước chiếm nhiều thời gian nhất theo bench (`timing.csv`, `hw_usage.csv`, `uavp_bench`), không theo cảm tính.
3. **Chỗ nóng tuân thủ nghiêm, chỗ lạnh ưu tiên dễ đọc.**
4. **Trình biên dịch đã làm nhiều việc.** Với `-O3 -mcpu=<CPU>` (CMake tự chọn: `cortex-a72` cho Pi 4, `cortex-a76` cho Pi 5), GCC tự unroll, tự inline, tự vector hóa NEON. Quy tắc thủ công chỉ dùng khi đo thấy có lợi.

---

## 1. Chọn đúng cấu trúc dữ liệu

Ưu tiên **mảng liên tục**: `std::array` (kích thước biết lúc biên dịch), `std::vector` đã `reserve` (kích thước biết lúc khởi động). Duyệt mảng liên tục tận dụng cache và prefetch. Tránh `std::list`, `std::map`, `std::unordered_map` trong chỗ nóng: mỗi phần tử là một lần cấp phát và một lần nhảy con trỏ. Hàng đợi giữa các luồng dùng ring buffer có dung lượng cố định (lũy thừa 2).

## 2. Kiểu biến tiết kiệm byte

Dùng kiểu nhỏ cho dữ liệu **được lưu nhiều bản** (mảng, struct, buffer); dùng `int` và `float` cho **biến tính toán cục bộ**. Biến cục bộ nằm trong thanh ghi 64-bit, nên khai báo `uint8_t` cho biến đếm vòng lặp không tiết kiệm gì mà còn thêm lệnh cắt bit. Tiết kiệm byte có ý nghĩa khi dữ liệu lớn: struct nhỏ hơn thì nhiều phần tử nằm trong một dòng cache 64 byte hơn.

| Đại lượng | Phạm vi | Kiểu lưu trữ | Byte | Ghi chú |
|---|---|---|---|---|
| Thời gian | ns từ lúc khởi động | `int64_t` | 8 | Không rút gọn: `int32_t` tràn sau 2.1 s |
| Frame ID | < 4 tỷ khung | `uint32_t` | 4 | 30 fps → 4.5 năm mới tràn |
| Tọa độ pixel (lưu trữ) | 0–1920 | `int16_t` | 2 | Khi tính toán thì chuyển sang `int` hoặc `float` |
| Tọa độ bbox sau decode | có phần thập phân | `float` | 4 | Không dùng `double` |
| Confidence | 0–1 | `float` (hoặc `uint8_t` × 1/255 khi lưu log) | 4 / 1 | |
| Class ID | 2 lớp | `uint8_t` (chỉ số trong `classes` của config) | 1 | |
| Trạng thái, lý do ABORT | < 16 giá trị | `enum class : uint8_t` | 1 | |
| Khoảng cách LiDAR | 0–12 000 mm | `uint16_t` (mm) | 2 | Độ phân giải LD06 cũng là mm |
| Điểm 3D, pháp tuyến | mm / m | `float` | 4 mỗi thành phần | `double` chỉ cho ma trận hiệu chỉnh, phép chiếu OpenCV và EKF |
| Mask | 0/1 | `uint8_t` (`CV_8U`) | 1 / pixel | Không dùng mask float |
| Số track, số lần đo | ≤ 255 | `uint8_t` trong struct | 1 | |
| Biến đếm vòng lặp, chỉ số | | `int` hoặc `std::size_t` | | Không dùng `uint8_t` |

### Sắp xếp thành viên struct để giảm padding

Trình biên dịch căn mỗi trường theo kích thước của nó và chèn byte trống (padding). **Xếp thành viên từ lớn đến nhỏ.**

```cpp
// XẤU: 32 byte (9 byte padding)
struct DetBad {
    uint8_t  cls;      // 1 + 3 padding
    float    x1, y1;   // 8
    uint8_t  status;   // 1 + 3 padding
    float    x2, y2;   // 8
    float    conf;     // 4
    uint8_t  n_obs;    // 1 + 3 padding
};

// TỐT: 24 byte, cùng thông tin
struct Detection {
    float  x1, y1, x2, y2;   // 16 (pixel, ảnh gốc)
    float  conf;             // 4
    Cls    cls;              // 1  (enum class : uint8_t)
    Status status;           // 1
    uint8_t n_obs;           // 1
    uint8_t pad_{};          // 1  (ghi rõ để không bị bất ngờ)
};
static_assert(sizeof(Detection) == 24, "Detection layout changed");
```
Một dòng cache 64 byte chứa 2 `DetBad` hoặc gần 3 `Detection`.

### Quy tắc kiểu dữ liệu
- Dùng kiểu có kích thước cố định trong `<cstdint>` (`int16_t`, `uint32_t`) cho dữ liệu lưu trữ và truyền đi. Không dùng `long` (kích thước khác nhau giữa nền tảng).
- Mọi enum là `enum class X : uint8_t`. Có phần tử cuối `kCount` để làm kích thước bảng tra.
- Thêm `static_assert(sizeof(T) == N)` cho mọi struct nằm trong mảng lớn hoặc ghi ra log nhị phân.
- Không dùng bitfield (`uint8_t a : 3;`) cho dữ liệu truy cập thường xuyên: đọc/ghi cần thêm lệnh dịch bit và không an toàn giữa các luồng. Chỉ cân nhắc cho gói tin gửi đi.
- Không dùng `#pragma pack(1)` cho struct tính toán: truy cập không căn chỉnh chậm hơn. Chỉ dùng cho định dạng gói LD06 khi parse.
- Cẩn thận khi tính toán với kiểu nhỏ: `int16_t a, b; a * b` được nâng lên `int`, nhưng gán lại vào `int16_t` có thể tràn. Chuyển sang `int` hoặc `float` trước khi tính, rút gọn khi lưu.
- Bật `-Wconversion` để trình biên dịch báo các chỗ chuyển kiểu ngầm làm mất dữ liệu (đã bật trong `CMakeLists.txt`).

---

## 3. Tránh sao chép dữ liệu

Một khung 1920×1080 RGB là 6.2 MB. Copy nó một lần tốn khoảng 1–2 ms trên Pi 5 và đẩy dữ liệu khác ra khỏi cache. **Mỗi khung ảnh chỉ nằm ở một chỗ trong bộ nhớ, mọi bước đều nhìn vào đó** (ROI, tile là `cv::Mat` view, không clone).

### Cách truyền tham số

| Loại tham số | Cách truyền | Ví dụ |
|---|---|---|
| Kiểu nhỏ (≤ 16 byte): `int`, `float`, enum, `cv::Point2f` | Truyền giá trị | `bool in_center(float u, float v)` |
| Đối tượng lớn, chỉ đọc | `const T&` | `void update(const Detection& d)` |
| Đối tượng lớn, hàm sửa nó | `T&` | `void decode(const Tensor& out, std::vector<Candidate>& cands)` |
| Dãy phần tử, chỉ đọc | `std::span<const T>` (C++20) | `float median(std::span<const float> v)` |
| Tham số có thể không có | `const T*` (có thể `nullptr`) | `const Track* best = nullptr;` |
| Chuyển quyền sở hữu | Truyền giá trị rồi `std::move`, hoặc `std::unique_ptr<T>` | `void set_model(std::unique_ptr<NcnnModel> m)` |

### Output của hàm
- Trả về đối tượng mới: trả bằng giá trị (`return result;`). Trình biên dịch áp dụng RVO, không copy. Không viết `return std::move(result);` vì nó chặn RVO.
- **Trong vòng lặp chính: dùng tham số output `T&` ghi vào buffer đã có.** Trả về `std::vector` mới mỗi khung là một lần cấp phát mỗi khung.

### Trao ảnh giữa các luồng không copy
Luồng camera ghi vào một trong các buffer đã cấp phát sẵn (`FramePool`), rồi đổi **chỉ số** với luồng perception. Chỉ có chỉ số được trao đổi, không có pixel nào bị copy thêm.

---

## 4. Constructor, destructor và giải phóng bộ nhớ

Mọi tài nguyên đều do một đối tượng sở hữu (**RAII**). Constructor lấy tài nguyên, destructor trả lại. Tài nguyên gồm: bộ nhớ, file log, cổng serial LD06, camera, socket/ROS publisher, luồng, mutex, model. Khi đối tượng ra khỏi phạm vi hoặc bị hủy, tài nguyên tự được giải phóng, kể cả khi hàm thoát sớm hay có lỗi. Không có hàm `close()` hay `free()` nào phải nhớ gọi.

- Không dùng `new` và `delete` trực tiếp. Tạo bằng `std::make_unique<T>(...)`, `std::make_shared<T>(...)`.
- Ưu tiên `unique_ptr`. `shared_ptr` tốn thêm bộ đếm nguyên tử (atomic) mỗi lần copy, và làm khó biết khi nào bộ nhớ được giải phóng.
- Truyền `shared_ptr` vào hàm bằng `const T&` nếu hàm chỉ dùng đối tượng, không cần giữ lại. Tránh copy `shared_ptr` trong vòng lặp nóng.
- Không tạo vòng `shared_ptr` (A giữ B, B giữ A): cả hai không bao giờ được giải phóng.

### Rule of zero và rule of five
- **Rule of zero (mặc định):** nếu mọi thành viên đã tự quản lý tài nguyên (`std::vector`, `cv::Mat`, `unique_ptr`, `std::array`), không viết destructor, copy hay move. Trình biên dịch sinh đúng và nhanh nhất.
- **Rule of five:** chỉ lớp bọc trực tiếp một tài nguyên hệ thống (file descriptor, handle C) mới viết destructor. Khi đó phải xử lý đủ 5 hàm: destructor, copy constructor, copy assignment, move constructor, move assignment. Thường là cấm copy và cho phép move.
- **Lớp nặng thì cấm copy:** model, camera, cổng serial, buffer lớn, luồng. Viết `T(const T&) = delete; T& operator=(const T&) = delete;`. Một lần copy vô tình sẽ bị trình biên dịch báo lỗi thay vì âm thầm tốn RAM.
- Move constructor và move assignment phải có `noexcept`. Nếu không, `std::vector` sẽ copy thay vì move khi cần mở rộng.
- Destructor không bao giờ ném exception (mặc định là `noexcept`).
- **Không dùng biến toàn cục hoặc `static` có constructor phức tạp:** thứ tự khởi tạo giữa các file không xác định. Tạo mọi thứ trong `main()` theo thứ tự rõ ràng. (Bảng `constexpr` tính lúc biên dịch không bị cấm.)

---

## 5. Bộ nhớ và cache

Pi 5 có cache L1 64 KB và L2 512 KB cho mỗi core, L3 2 MB dùng chung. Đọc từ RAM chậm hơn đọc từ L1 khoảng 50–100 lần. Code nhanh là code đọc dữ liệu liên tục, theo thứ tự, và **không cấp phát khi đang chạy**.

- **Cấp phát trước:** buffer ảnh, input model, vector kết quả được tạo và `reserve` lúc khởi động; mỗi khung chỉ `clear()` và ghi lại.
- **Tách dữ liệu nóng và lạnh:** trường dùng mỗi khung (vị trí, trạng thái, bộ đếm) nằm trong struct chính. Trường hiếm dùng (lịch sử để debug, chuỗi mô tả) tách ra struct khác hoặc chỉ có trong bản debug.
- **Duyệt theo thứ tự bộ nhớ:** ảnh (`cv::Mat`) lưu theo hàng. Vòng ngoài là hàng, vòng trong là cột. Lấy con trỏ đầu hàng một lần bằng `mat.ptr<uint8_t>(r)`, không dùng `mat.at<>(r, c)` trong vòng lặp vì nó tính lại địa chỉ và kiểm tra. Mảng 2 chiều tự viết cũng lưu theo hàng, duyệt theo hàng.
- **Chỉ duyệt vùng cần thiết:** mask và điểm LiDAR chỉ trong bbox của ROI, không trên cả ảnh.
- **Tránh false sharing giữa các luồng:** hai biến do hai luồng khác nhau ghi thường xuyên, nếu nằm chung một dòng cache 64 byte, sẽ làm hai core liên tục giành dòng cache đó. Đặt chúng cách nhau bằng `alignas(64)`.

---

## 6. Vòng lặp và phép toán

Việc gì không đổi giữa các lần lặp thì làm **một lần bên ngoài**; việc gì có thể bỏ qua sớm thì **bỏ qua trước khi làm việc đắt**. Hai quy tắc này mang lại nhiều lợi hơn mọi mẹo vi mô cộng lại.

### Phép toán rẻ thay cho phép toán đắt

| Thay | Bằng |
|---|---|
| Chia cho hằng số trong vòng lặp | Nhân với nghịch đảo tính sẵn |
| So khoảng cách `sqrt(dx²+dy²) < r` | So bình phương `dx*dx + dy*dy < r*r` |
| `sigmoid(x) > t` cho mọi phần tử | `x > logit(t)` với `logit(t) = ln(t/(1-t))` tính sẵn |
| `pow(x, 2)` | `x * x` |
| `% N` với N là lũy thừa 2 | `& (N - 1)` |
| Đổi đơn vị lặp lại (mm ↔ m, độ ↔ rad) | Giữ một đơn vị nội bộ, đổi ở biên vào/ra |
| `std::sort` toàn bộ để lấy median | `std::nth_element` |
| Tính hàm lượng giác mỗi điểm | Bảng tra theo chỉ số góc |
| `uint8_t → float` chia 255 cho từng pixel | Bảng tra 256 giá trị tính sẵn |

Ví dụ lọc bằng logit, dùng chung cho mọi chỗ có ngưỡng xác suất **trên logit thô** (output YOLO26n hiện tại đã qua sigmoid trong model, nên so trực tiếp với ngưỡng conf):
```cpp
const float logit_thr = std::log(cfg.conf_thr / (1.f - cfg.conf_thr));   // một lần
for (int i = 0; i < n; ++i) {
    if (score[i] <= logit_thr) continue;                     // bỏ qua sớm, không tính exp
    const float conf = 1.f / (1.f + std::exp(-score[i]));    // chỉ vài chục phần tử còn lại
    ...
}
```

### Viết vòng lặp để trình biên dịch tối ưu được
- Dùng range-based for khi duyệt cả container: `for (const auto& d : dets)`. Dễ đọc, không sai biên, tốc độ bằng vòng lặp chỉ số.
- Dùng vòng lặp chỉ số khi cần chỉ số, duyệt hai mảng song song, hoặc chỉ duyệt một đoạn.
- **Thân vòng lặp nóng: không gọi hàm ảo, không cấp phát, không khóa mutex, không ghi log.**
- Đánh dấu con trỏ không chồng lấn bằng `__restrict` khi một hàm đọc mảng này và ghi mảng kia. Trình biên dịch khi đó mới dám vector hóa.
- Loop unrolling: với `-O3`, GCC tự unroll và tự vector hóa NEON. Không tự viết unroll bằng tay. Nếu đo thấy một vòng lặp cụ thể chưa được unroll, dùng `#pragma GCC unroll 4` ngay trước nó. Kiểm tra bằng `-fopt-info-vec-missed`.
- Thoát sớm: dùng `continue` để bỏ phần tử không đạt và `break` khi đã có kết quả (ví dụ RANSAC đã đủ inlier).

### Toán tử gán kết hợp
- Viết `sum += x;`, `w_mm *= scale;` thay cho `sum = sum + x;`. Với kiểu số, trình biên dịch sinh cùng mã máy, nên lợi ích chính là dễ đọc.
- Với đối tượng (`cv::Mat`, ma trận, vector Eigen), `a += b` cập nhật tại chỗ, còn `a = a + b` tạo đối tượng tạm. Ở đây quy tắc này thật sự tiết kiệm bộ nhớ và thời gian.
- Dùng `++i` thay cho `i++` với iterator.

### Độ chính xác và an toàn số học
- Không so sánh float bằng `==`. Dùng ngưỡng: `std::abs(a - b) < 1e-6f`.
- Kiểm tra mẫu số trước khi chia (ví dụ tia song song với mặt phẳng: `|n·r| < 1e-6`).
- Không dùng `-ffast-math` cho toàn bộ dự án: nó bỏ kiểm tra NaN, làm hỏng các phép kiểm tra `std::isnan` trong fusion. Nếu cần, chỉ bật cho từng file đã đo và kiểm tra.
- **Làm tròn giống Python:** `round()` của Python làm tròn nửa về số chẵn. Chỗ nào bản Python dùng `round()` / `np.round` thì C++ dùng `uavp::round_half_even()` (`std::nearbyint`), không dùng `std::round`.

---

## 7. Rẽ nhánh: if, switch, lookup table, ternary

Rẽ nhánh chỉ đáng tối ưu khi nằm trong vòng lặp qua nhiều phần tử (ô của model, điểm LiDAR, pixel mask, packet LD06). Ở logic chọn vật hay máy trạng thái chạy vài lần mỗi khung, ưu tiên viết rõ ràng.

### Thứ tự điều kiện trong if
`&&` và `||` được tính từ trái sang phải và dừng ngay khi biết kết quả (short-circuit). Vì vậy:
1. Nếu xác suất đúng/sai của các điều kiện tương đương nhau: điều kiện rẻ trước, đắt sau.
2. Với `&&`: đặt trước điều kiện hay sai nhất (loại được nhiều nhất).
3. Với `||`: đặt trước điều kiện hay đúng nhất.
4. Khi một điều kiện vừa rẻ vừa loại được nhiều, nó luôn đứng đầu.
5. Điều kiện bảo vệ (kiểm tra `nullptr`, kiểm tra biên mảng) luôn đứng trước điều kiện dùng nó, bất kể chi phí.

### switch thay cho chuỗi if-else theo cùng một biến
Khi so một biến enum hoặc số nguyên với nhiều giá trị, dùng `switch`. Trình biên dịch có thể biến nó thành bảng nhảy, và với `-Wswitch` sẽ báo khi quên xử lý một giá trị enum. **Không viết `default:` cho enum** để cảnh báo này còn hoạt động.
```cpp
switch (stage) {                       // máy trạng thái của hệ thống
    case Stage::kSearch:   run_search(); break;
    case Stage::kSelect:   run_select(); break;
    case Stage::kApproach: run_track();  break;
    case Stage::kFuse:     run_fuse();   break;
    case Stage::kDone:     break;
}
```

### Lookup table thay cho switch khi chỉ trả về một giá trị
Khi mỗi nhánh chỉ trả về một hằng số (tên, màu, ngưỡng, hệ số), dùng bảng `constexpr` tra theo chỉ số enum. Gọn hơn, không có rẽ nhánh, và thêm một lớp mới chỉ là thêm một dòng.

### Ternary operator
Dùng khi chọn một giá trị dựa trên một điều kiện, và cả hai nhánh đều rẻ. Trình biên dịch thường sinh lệnh chọn không rẽ nhánh (`csel`, `fcsel` trên ARM), nhưng một `if` đơn giản cũng được tối ưu như vậy, nên lợi ích chính là cho phép khai báo `const`.

---

## 8. Hàm: virtual, inline, constexpr, template và khởi tạo {}

### Hàm ảo chỉ ở ranh giới module, không ở vòng lặp nóng
Gọi hàm ảo tốn thêm một lần đọc bảng vtable và chặn trình biên dịch inline. Gọi một lần mỗi khung thì không đáng kể (vài ns). Gọi một lần mỗi điểm LiDAR hay mỗi ô thì đáng kể.

| Tình huống | Cách làm |
|---|---|
| Chọn nguồn dữ liệu lúc chạy: camera thật / replay / Gazebo; LD06 thật / replay; model NCNN / kết quả ghi sẵn | Interface có hàm ảo (`IFrameSource`, `IInfer`). Gọi một lần mỗi khung (mỗi tile): chấp nhận được |
| Chọn thuật toán lúc biên dịch (tile hay cả ảnh, RANSAC hay median) | Template hoặc `if constexpr`, không dùng virtual |
| Chọn thuật toán theo config, ít lựa chọn | `enum` + `switch` một lần ngoài vòng lặp |
| Hàm gọi cho từng phần tử (chiếu điểm, IoU, sigmoid) | Hàm thường, khai báo trong header để inline được |

Quy tắc khi dùng virtual:
- Lớp có hàm ảo phải có destructor ảo (`virtual ~IFrameSource() = default;`). Nếu không, hủy qua con trỏ lớp cha sẽ không gọi destructor lớp con và gây rò tài nguyên.
- Lớp con ghi `override` cho mọi hàm ghi đè, và `final` cho lớp không kế thừa thêm. `final` cho phép trình biên dịch bỏ vtable khi biết kiểu cụ thể.
- Không gọi hàm ảo trong constructor hoặc destructor.
- Các bước s1–s4 không cần chung một lớp cha ảo. Pipeline gọi trực tiếp từng bước theo thứ tự cố định.

### inline
- Trong C++ hiện đại, `inline` chủ yếu cho phép định nghĩa hàm trong header mà không lỗi trùng định nghĩa. Việc có chèn thân hàm vào chỗ gọi hay không do trình biên dịch quyết định.
- Hàm nhỏ (1–5 dòng) gọi trong vòng lặp nóng: định nghĩa trong header (hàm thành viên viết trong thân lớp đã tự là inline). Ví dụ: `sigmoid`, `iou`, `sq_dist`, `in_center`.
- Hàm lớn hoặc gọi ít: để trong file `.cpp`. Inline hàm lớn làm mã phình to và giảm hiệu quả cache lệnh.
- Bật LTO (`CMAKE_INTERPROCEDURAL_OPTIMIZATION ON`, đã bật cho bản Release) để trình biên dịch inline được cả giữa các file `.cpp`.
- Không dùng `__attribute__((always_inline))` trừ khi đã đo và thấy trình biên dịch không inline một hàm nóng.

### constexpr và tính toán lúc biên dịch
- Mọi hằng số cố định dùng `constexpr`, không dùng `#define`: có kiểu, có phạm vi, debug được.
- Giá trị phụ thuộc phần cứng hoặc cần chỉnh khi thử (ngưỡng, bán kính, số khung) để trong YAML, không `constexpr`.
- Bảng tính được từ hằng số (lưới stride 8/16/32, bảng CRC8 của LD06, bảng thông tin lớp, bảng uint8 → float) tính lúc biên dịch bằng hàm `constexpr`.
- Bảng phụ thuộc dữ liệu lúc chạy (sin/cos theo góc thực của scan, bảng undistort theo K) tính một lần trong constructor.
- Dùng `static_assert` để kiểm tra giả định lúc biên dịch: kích thước struct, dung lượng ring buffer là lũy thừa 2.

```cpp
constexpr std::array<uint8_t, 256> make_crc8_table(uint8_t poly) {
    std::array<uint8_t, 256> t{};
    for (int i = 0; i < 256; ++i) {
        auto c = static_cast<uint8_t>(i);
        for (int b = 0; b < 8; ++b) c = (c & 0x80) ? uint8_t((c << 1) ^ poly) : uint8_t(c << 1);
        t[static_cast<std::size_t>(i)] = c;
    }
    return t;
}
inline constexpr auto kCrc8 = make_crc8_table(0x4D);   // tính lúc biên dịch, dùng khi parse LD06

constexpr bool is_pow2(unsigned x) { return x && !(x & (x - 1)); }
static_assert(is_pow2(kCap), "ring buffer cần là lũy thừa 2");   // đặt trong thân lớp ring buffer
```

### Template
- Dùng template khi cùng một thuật toán chạy cho nhiều kiểu (median cho `float` và `uint16_t`, ring buffer cho scan và IMU).
- Không dùng template metaprogramming phức tạp chỉ để tính hằng số: `constexpr` làm được việc đó và dễ đọc hơn nhiều.
- Template làm tăng thời gian biên dịch và kích thước mã. Giữ ở mức cần thiết.

### Khởi tạo bằng {}
- Khởi tạo mọi biến khi khai báo. Biến số không khởi tạo mang giá trị rác và gây lỗi khó tái hiện.
- Dùng `{}` để khởi tạo: `float w_mm{0.f};`, `Detection d{};` (mọi trường về 0), `std::array<float, 5> hist{};`.
- `{}` cấm chuyển kiểu thu hẹp: `int16_t u{1920.7f};` bị báo lỗi biên dịch, còn `int16_t u = 1920.7f;` âm thầm cắt.
- Khởi tạo container bằng danh sách: `std::vector<float> w{62.f, 64.f, 63.f};`.
- Cẩn thận với vector: `std::vector<int> v{5};` là một phần tử có giá trị 5, còn `std::vector<int> v(5);` là năm phần tử bằng 0. Muốn cấp trước dung lượng thì dùng `v.reserve(5)`.
- Thành viên lớp có giá trị mặc định ngay tại chỗ khai báo: `uint8_t n_obs{0};`, `int fd_{-1};`.

### Quy tắc chung cho hàm
- Đánh dấu `const` cho hàm thành viên không sửa đối tượng, và `noexcept` cho hàm không ném lỗi (hàm tính toán, move, destructor).
- Dùng `[[nodiscard]]` cho hàm trả về kết quả hoặc trạng thái lỗi, để trình biên dịch báo khi quên kiểm tra: `[[nodiscard]] bool fuse(..., TargetOutput& out);`.
- Một hàm làm một việc, dưới khoảng 50 dòng. Hàm nóng càng ngắn càng dễ được inline và tối ưu.
- Lỗi trong vòng lặp nóng trả về bằng `bool`, `std::optional<T>` hoặc enum trạng thái. **Exception chỉ dùng lúc khởi động** (config sai, thiếu model, không mở được camera / LiDAR).

---

## 9. I/O, log và thuật toán STL

Hệ thống không đọc dữ liệu từ bàn phím (trừ phím `s` / `q` lúc chờ), nên `std::cin` hầu như không dùng. I/O thật sự là: đọc cổng serial LD06, nhận ảnh camera, gửi kết quả cho FC và tay gắp, ghi log. Quy tắc chính: **luồng tính toán không bao giờ chờ I/O.**

- Ghi log qua luồng riêng. Luồng perception chỉ đẩy một bản ghi nhỏ (định dạng sẵn vào slot cố định) vào ring buffer. Luồng log định dạng và ghi ra file (`uavp::log`).
- Không dùng `std::endl` trong log: nó xả buffer (flush) mỗi dòng. Dùng `'\n'` và để hệ thống xả theo khối, hoặc gọi `flush()` mỗi 1 s.
- Nếu dùng iostream, gọi `std::ios::sync_with_stdio(false);` và `std::cin.tie(nullptr);` một lần đầu `main()`. Sau đó không trộn `printf` với `std::cout`.
- Định dạng số nhanh hơn với `std::to_chars` (C++17) hoặc `snprintf` vào buffer có sẵn, thay cho `std::stringstream` (chậm, cấp phát).
- Log số lượng lớn (điểm LiDAR, bbox từng khung): ghi nhị phân (struct có `static_assert(sizeof)`), chuyển sang CSV bằng script Python khi phân tích. (Giai đoạn đầu: CSV ghi một lần cuối phiên, ở chỗ lạnh.)
- Có mức log (ERROR, WARN, INFO, DEBUG). Khi bay chỉ bật tới INFO. Kiểm tra mức log trước khi định dạng chuỗi, để log bị tắt không tốn gì.
- Đọc serial LD06 theo khối (đọc nhiều byte một lần vào buffer có sẵn), không đọc từng byte. Dùng `poll()` có timeout để luồng dừng được khi tắt.
- Gửi kết quả cho FC và tay gắp bằng struct cố định kích thước, không dùng chuỗi JSON trong vòng lặp nóng.
- Đọc config, file hiệu chỉnh, model chỉ một lần lúc khởi động.
- Thuật toán STL: ưu tiên `std::nth_element`, `std::partial_sort`, `std::stable_sort` (khi cần thứ tự giống `np.argsort(kind="stable")`), `std::max_element` thay cho vòng lặp tự viết, nhưng gọi trên buffer có sẵn.

---

## 10. Đo đạc, profiling và checklist review

Mọi tối ưu phải có **số đo trước và sau trên Pi 5, cùng bộ dữ liệu replay**. Nếu không nhanh hơn rõ ràng (> 5 % hoặc p95 giảm), giữ bản dễ đọc hơn.

- Đo bằng `timing.csv` / `result.json` của `uavp_real_time` (median, p95 từng bước), `hw_usage.csv` (CPU, nhiệt độ, tần số), `uavp_offline` trên cùng thư mục ảnh. So nhiều lần chạy bằng `python -m uavp.compare_runs` (file kết quả cùng định dạng với bản Python).
- Profiling: `perf record -g ./uavp_offline ...`, `perf report`; xem vòng lặp chưa vector hóa: build với `-fopt-info-vec-missed`.
- Đo trên Pi với governor `performance`, `display.show=false`, đã làm nóng máy (bỏ lần chạy đầu).

### Checklist review (mỗi thay đổi)
- [ ] Build không cảnh báo với `-Wall -Wextra -Wconversion -Wshadow` (bản Release).
- [ ] `ctest` qua, gồm `test_parity` (bbox ≤ 1 px, mask IoU ≥ 0.99, W ≤ 1 mm so với bản Python).
- [ ] Chỗ nóng: không cấp phát, không mutex, không log, không hàm ảo trong vòng lặp; buffer cấp phát lúc khởi động.
- [ ] Struct trong mảng lớn: thành viên xếp lớn → nhỏ, có `static_assert(sizeof)`.
- [ ] Enum là `enum class : uint8_t`, `switch` không có `default:`.
- [ ] Tài nguyên có chủ (RAII), lớp nặng cấm copy, move có `noexcept`, không `new` / `delete`.
- [ ] Hằng số cố định là `constexpr`; ngưỡng / tham số chỉnh được nằm trong YAML.
- [ ] Làm tròn giống Python (`round_half_even`) ở mọi chỗ bản Python dùng `round()`.
- [ ] Có số đo trước / sau trên Pi 5 cho mọi thay đổi nhằm tăng tốc.
