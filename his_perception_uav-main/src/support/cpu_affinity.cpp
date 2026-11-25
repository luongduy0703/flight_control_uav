// Ghim chuong trinh vao mot so nhan CPU (cho lanh: mot lan luc khoi dong).
#include "uavp/support/cpu_affinity.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <sched.h>
#include <unistd.h>

namespace uavp {

std::string pin_process_to_cores(std::span<const int> cores) {
    if (cores.empty()) return "moi nhan (khong gioi han)";
    const long online = ::sysconf(_SC_NPROCESSORS_ONLN);
    cpu_set_t set;
    CPU_ZERO(&set);
    std::string text;
    for (const int c : cores) {
        if (c < 0 || c >= online || c >= CPU_SETSIZE)
            throw std::runtime_error("cpu.cores: nhan " + std::to_string(c) + " khong ton tai (may co " +
                                     std::to_string(online) + " nhan)");
        CPU_SET(static_cast<std::size_t>(c), &set);
        text += (text.empty() ? "" : ", ") + std::to_string(c);
    }
    if (::sched_setaffinity(0, sizeof(set), &set) != 0)  // 0 = luong goi; luong tao sau ke thua
        throw std::runtime_error(std::string("sched_setaffinity loi: ") + std::strerror(errno));
    return "nhan [" + text + "]";
}

}  // namespace uavp
