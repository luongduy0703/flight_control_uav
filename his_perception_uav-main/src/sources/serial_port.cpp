// Mo cong serial (cho lanh: mot lan luc khoi dong).
#include "uavp/sources/serial_port.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

namespace uavp {
namespace {

speed_t baud_constant(int baudrate) {
    switch (baudrate) {
        case 57600: return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
        case 460800: return B460800;
        case 500000: return B500000;
        case 921600: return B921600;
        default: throw std::invalid_argument("Baud khong ho tro: " + std::to_string(baudrate));
    }
}

}  // namespace

int open_serial_fd(const std::string& port, int baudrate, const std::string& hint, bool writable) {
    const int fd = ::open(port.c_str(), (writable ? O_RDWR : O_RDONLY) | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) throw std::runtime_error("Khong mo duoc " + port + " (" + std::strerror(errno) + "): " + hint);
    termios tio{};
    tcgetattr(fd, &tio);
    cfmakeraw(&tio);  // 8 bit, khong parity; KHONG tu xoa 2 stop bit / RTS-CTS con lai tu lan cau hinh truoc
    cfsetispeed(&tio, baud_constant(baudrate));
    cfsetospeed(&tio, baud_constant(baudrate));
    tio.c_cflag |= CLOCAL | CREAD;
    tio.c_cflag &= ~static_cast<tcflag_t>(CSTOPB | CRTSCTS);  // 8N1, khong flow control (chi noi TX / RX / GND)
    if (tcsetattr(fd, TCSANOW, &tio) != 0) {
        ::close(fd);
        throw std::runtime_error("Khong cau hinh duoc cong " + port);
    }
    tcflush(fd, TCIFLUSH);  // bo byte da nam trong bo dem tu luc khoi dong (rac / sai baud truoc khi cau hinh)
    return fd;
}

}  // namespace uavp
