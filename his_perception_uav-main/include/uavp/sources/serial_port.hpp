// Cong serial raw 8N1 (LD06 qua CP210x, Pixhawk qua UART / USB). Doc bang poll() co timeout.
#pragma once

#include <string>
#include <vector>

namespace uavp {

// Mo cong khong chan, raw 8N1 (chi doc, hoac doc + ghi neu writable). Loi -> exception (kem goi y). Tra ve file
// descriptor (nguoi goi dong). Cong DOC QUYEN: neu tien trinh khac dang mo cung thiet bi -> exception (hai chuong
// trinh cung doc mot tty se chia nhau byte -> goi bi cat, bao loi CRC: 2026-11-22 ldlidar_ros2_node giu ttyUSB0 ->
// mat ~28% goi LD06); sau khi mo dat TIOCEXCL de chuong trinh khac khong mo chen vao duoc.
[[nodiscard]] int open_serial_fd(const std::string& port, int baudrate, const std::string& hint, bool writable = false);

// Cac tien trinh KHAC dang mo cung thiet bi voi `port` (quet /proc/*/fd, so sanh st_rdev): "pid lenh".
[[nodiscard]] std::vector<std::string> port_holders(const std::string& port);

}  // namespace uavp
