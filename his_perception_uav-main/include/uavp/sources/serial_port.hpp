// Cong serial raw 8N1 (LD06 qua CP210x, Pixhawk qua UART / USB). Doc bang poll() co timeout.
#pragma once

#include <string>

namespace uavp {

// Mo cong khong chan, raw 8N1 (chi doc, hoac doc + ghi neu writable). Loi -> exception (kem goi y). Tra ve file
// descriptor (nguoi goi dong).
[[nodiscard]] int open_serial_fd(const std::string& port, int baudrate, const std::string& hint, bool writable = false);

}  // namespace uavp
