// Hai chuong trinh thoi gian thuc tren Pi dung chung apps/live_app.cpp:
//   uavp_real_time_perception (apps/real_time_perception.cpp): chi 4 stage s1 -> s4, bat dau bang SPACE tren may tinh
//     (tools/remote_viewer.py) / s tren terminal, khong Pixhawk -> do thoi gian chay, phan cung, sai so so voi ArUco.
//   uavp_real_time (apps/real_time.cpp): ca chuong trinh (s1 -> s4 + khoa + bam) ket hop Pixhawk: bat dau bang lenh
//     tu may tinh gui qua Pixhawk (tools/gcs_command.py) khi dang hover, D camera - san tu do cao optical flow.
#pragma once

#include <cstdint>

enum class AppMode : uint8_t { kPerception, kFull };

int run_live(int argc, char** argv, AppMode mode);
