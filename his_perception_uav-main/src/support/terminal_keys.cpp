#include "uavp/support/terminal_keys.hpp"

#include <cctype>

#include <poll.h>
#include <unistd.h>

namespace uavp {

TerminalKeys::TerminalKeys() {
    enabled_ = isatty(STDIN_FILENO) != 0;
    if (!enabled_ || tcgetattr(STDIN_FILENO, &saved_) != 0) {
        enabled_ = false;
        return;
    }
    termios raw = saved_;
    raw.c_lflag &= ~static_cast<tcflag_t>(ICANON | ECHO);  // cbreak: tung phim, khong hien; Ctrl+C van dung
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
}

TerminalKeys::~TerminalKeys() {
    if (enabled_) tcsetattr(STDIN_FILENO, TCSADRAIN, &saved_);
}

std::optional<char> TerminalKeys::read_key(double timeout_s) const {
    if (!enabled_) return std::nullopt;
    pollfd pfd{STDIN_FILENO, POLLIN, 0};
    if (::poll(&pfd, 1, static_cast<int>(timeout_s * 1000.0)) <= 0) return std::nullopt;
    char c = 0;
    if (::read(STDIN_FILENO, &c, 1) != 1) return std::nullopt;
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

}  // namespace uavp
