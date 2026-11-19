// uavp_real_time: CA chuong trinh ket hop Pixhawk (qua uavp_fc_bridge):
//   cho lenh bat dau tu may tinh (tools/gcs_command.py start, gui qua Pixhawk) + Pi tu nhan dang hover
//   -> s1 -> s3 (D camera - san tu do cao optical flow) -> khoa vat (p_W he the gioi) -> bam (tu the FC, optical
//   flow, Kalman, segment theo LiDAR) -> s4 cuoi (tam + 3 truc mat top). Lenh dung (gcs_command.py stop) -> dung.
//   ./build/uavp_real_time --cfg configs/run/real_time.yaml --set display.show=false
#include "live_app.hpp"

int main(int argc, char** argv) { return run_live(argc, argv, AppMode::kFull); }
