// Doc phim tu terminal (SSH) khong can Enter: che do cbreak cua termios (Linux). RAII: huy doi tuong -> tra
// terminal ve che do cu, ke ca khi loi. stdin khong phai terminal -> enabled() = false.
#pragma once

#include <optional>
#include <termios.h>

namespace uavp {

class TerminalKeys {
public:
    TerminalKeys();
    ~TerminalKeys();
    TerminalKeys(const TerminalKeys&) = delete;
    TerminalKeys& operator=(const TerminalKeys&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    // Mot phim (chu thuong) neu co trong timeout_s, khong thi nullopt.
    [[nodiscard]] std::optional<char> read_key(double timeout_s) const;

private:
    termios saved_{};
    bool enabled_{false};
};

}  // namespace uavp
