// Mo cong serial (cho lanh: mot lan luc khoi dong).
#include "uavp/sources/serial_port.hpp"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
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

std::vector<std::string> port_holders(const std::string& port) {
    namespace fs = std::filesystem;
    std::vector<std::string> out;
    struct stat target {};
    if (::stat(port.c_str(), &target) != 0 || !S_ISCHR(target.st_mode)) return out;
    const std::string self = std::to_string(::getpid());
    std::error_code ec;
    for (const auto& proc : fs::directory_iterator("/proc", ec)) {
        const std::string pid = proc.path().filename().string();
        if (pid.empty() || pid.find_first_not_of("0123456789") != std::string::npos || pid == self) continue;
        std::error_code ec2;
        for (const auto& fd : fs::directory_iterator(proc.path() / "fd", ec2)) {  // tien trinh cua user khac: bo qua
            struct stat st {};
            if (::stat(fd.path().c_str(), &st) != 0 || !S_ISCHR(st.st_mode) || st.st_rdev != target.st_rdev) continue;
            std::ifstream cmd(proc.path() / "cmdline");
            std::string args;
            std::getline(cmd, args, '\0');
            out.push_back(pid + " " + args);
            break;
        }
    }
    return out;
}

int open_serial_fd(const std::string& port, int baudrate, const std::string& hint, bool writable) {
    const int fd = ::open(port.c_str(), (writable ? O_RDWR : O_RDONLY) | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) throw std::runtime_error("Khong mo duoc " + port + " (" + std::strerror(errno) + "): " + hint);
    if (const auto others = port_holders(port); !others.empty()) {
        ::close(fd);
        std::string list;
        for (const auto& o : others) list += "\n    PID " + o;
        throw std::runtime_error("Cong " + port + " dang bi tien trinh khac mo (hai chuong trinh cung doc se chia nhau "
                                 "byte -> mat goi / loi CRC). Dung tien trinh do truoc (vd: kill <PID>):" + list);
    }
    ::ioctl(fd, TIOCEXCL);  // doc quyen: open() khac (khong phai root) se bi EBUSY
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
