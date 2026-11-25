// Gioi han ca chuong trinh vao mot so nhan CPU (configs/run/*.yaml: cpu.cores, vd [2, 3]).
// Linux: luong tao SAU khi goi ham ke thua tap nhan cua luong goi -> goi o dau main(), TRUOC khi tao logger, LD06,
// camera (libcamera tao luong rieng), theo doi phan cung va NCNN (luong OpenMP).
// Cach chia 2 nhan: luong chinh (s1 -> s4) + 1 luong OpenMP cua NCNN dung ca 2 nhan khi chay model (num_threads = 2);
// cac luong I/O (LD06, camera, log, theo doi phan cung) phan lon thoi gian ngu (poll / cho), chen vao giua cac lan
// chay model. Nhan con lai (0, 1) danh cho he dieu hanh va chuong trinh khac (FC, ROS...).
#pragma once

#include <span>
#include <string>

namespace uavp {

// Ghim luong goi (va moi luong tao sau do) vao `cores`. Rong: khong lam gi. Nhan khong ton tai -> exception.
// Tra ve mo ta tap nhan (de log).
std::string pin_process_to_cores(std::span<const int> cores);

}  // namespace uavp
