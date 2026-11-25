// uavp_real_time_perception: chi 4 stage (s1 detect -> s2 segment -> s3 select -> s4 fusion), KHONG Pixhawk.
// Muc dich: do thoi gian chay thuc te tung stage, phan cung Pi (CPU, RAM, nhiet, tan so), sai so s4 so voi ArUco.
// May tinh xem truc tiep + bam SPACE de bat dau (tools/remote_viewer.py, qua Wi-Fi / LAN). D camera - san = 1 m.
//   ./build/uavp_real_time_perception --cfg configs/run/real_time_perception.yaml --set display.show=false
#include "live_app.hpp"

int main(int argc, char** argv) { return run_live(argc, argv, AppMode::kPerception); }
