// Vung nho chung MAVLink (bridge -> perception), xem fc_shm.hpp.
#include "uavp/sources/fc_shm.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <new>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "uavp/common/common.hpp"

namespace uavp {
namespace {

constexpr uint64_t kBacklog = 256;  // lan doc dau: lay toi da 256 tin gan nhat (~2 s) de co tu the ngay

std::size_t total_size(std::size_t capacity) { return sizeof(FcShmHeader) + capacity * sizeof(FcShmEntry); }

std::string shm_error(const std::string& what, const std::string& name) {
    return what + " " + name + " (" + std::strerror(errno) + ")";
}

}  // namespace

FcShmWriter::FcShmWriter(const std::string& name, std::size_t capacity) {
    if (capacity == 0 || (capacity & (capacity - 1)) != 0) throw std::invalid_argument("shm capacity phai la luy thua 2");
    const int fd = ::shm_open(name.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0) throw std::runtime_error(shm_error("Khong tao duoc shared memory", name));
    size_ = total_size(capacity);
    if (::ftruncate(fd, static_cast<off_t>(size_)) != 0) {
        ::close(fd);
        throw std::runtime_error(shm_error("Khong dat kich thuoc shared memory", name));
    }
    mem_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mem_ == MAP_FAILED) throw std::runtime_error(shm_error("Khong mmap duoc", name));
    header_ = static_cast<FcShmHeader*>(mem_);
    header_->magic.store(0, std::memory_order_relaxed);  // nguoi doc thay "chua san sang" trong luc khoi tao
    header_ = new (mem_) FcShmHeader{};
    entries_ = reinterpret_cast<FcShmEntry*>(static_cast<char*>(mem_) + sizeof(FcShmHeader));
    for (std::size_t i = 0; i < capacity; ++i) new (&entries_[i]) FcShmEntry{};
    header_->version = kFcShmVersion;
    header_->capacity = static_cast<uint32_t>(capacity);
    header_->entry_bytes = sizeof(FcShmEntry);
    header_->writer_start_ns = mono_ns();
    header_->writer_pid = static_cast<int32_t>(::getpid());
    mask_ = capacity - 1;
    header_->magic.store(kFcShmMagic, std::memory_order_release);
}

FcShmWriter::~FcShmWriter() {
    if (mem_ != nullptr && mem_ != MAP_FAILED) ::munmap(mem_, size_);  // giu /dev/shm: bridge chay lai dung tiep
}

void FcShmWriter::push(const MavMessage& m, int64_t t_rx_ns) noexcept {
    const uint64_t i = header_->write_index.load(std::memory_order_relaxed);
    FcShmEntry& e = entries_[i & mask_];
    e.seq.store(2 * i + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    e.t_rx_ns = t_rx_ns;
    e.msgid = m.msgid;
    e.len = static_cast<uint8_t>(std::min<std::size_t>(m.len, kFcShmPayload));
    e.sysid = m.sysid;
    e.compid = m.compid;
    std::copy_n(m.payload.begin(), kFcShmPayload, e.payload.begin());
    e.seq.store(2 * i + 2, std::memory_order_release);
    header_->write_index.store(i + 1, std::memory_order_release);
    header_->last_rx_ns.store(t_rx_ns, std::memory_order_relaxed);
    switch (m.msgid) {
        case kMsgHeartbeat: header_->n_heartbeat.fetch_add(1, std::memory_order_relaxed); break;
        case kMsgAttitude: header_->n_attitude.fetch_add(1, std::memory_order_relaxed); break;
        case kMsgLocalPositionNed: header_->n_position.fetch_add(1, std::memory_order_relaxed); break;
        case kMsgDistanceSensor:
        case kMsgRangefinder: header_->n_range.fetch_add(1, std::memory_order_relaxed); break;
        default: break;
    }
}

void FcShmWriter::add_io(std::size_t bytes, std::size_t crc_errors) noexcept {
    header_->bytes.fetch_add(bytes, std::memory_order_relaxed);
    header_->crc_errors.fetch_add(crc_errors, std::memory_order_relaxed);
}

FcShmReader::FcShmReader(const std::string& name) {
    const int fd = ::shm_open(name.c_str(), O_RDONLY, 0);
    if (fd < 0)
        throw std::runtime_error(shm_error("Khong mo duoc shared memory", name) +
                                 ": uavp_fc_bridge chua chay? (systemctl status uavp-fc-bridge)");
    struct stat st {};
    ::fstat(fd, &st);
    size_ = static_cast<std::size_t>(st.st_size);
    if (size_ < sizeof(FcShmHeader)) {
        ::close(fd);
        throw std::runtime_error("Shared memory " + name + " qua nho: bridge chua khoi tao xong");
    }
    mem_ = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mem_ == MAP_FAILED) throw std::runtime_error(shm_error("Khong mmap duoc", name));
    header_ = static_cast<const FcShmHeader*>(mem_);
    entries_ = reinterpret_cast<const FcShmEntry*>(static_cast<const char*>(mem_) + sizeof(FcShmHeader));
}

FcShmReader::~FcShmReader() {
    if (mem_ != nullptr && mem_ != MAP_FAILED) ::munmap(const_cast<void*>(mem_), size_);
}

bool FcShmReader::poll(std::vector<MavMessage>& out, std::vector<int64_t>& t_rx_ns, std::size_t& lost) {
    out.clear();
    t_rx_ns.clear();
    lost = 0;
    if (header_->magic.load(std::memory_order_acquire) != kFcShmMagic || header_->version != kFcShmVersion) return false;
    const uint64_t cap = header_->capacity;
    if (sizeof(FcShmHeader) + cap * sizeof(FcShmEntry) > size_) return false;  // bridge doi kich thuoc: mo lai
    const uint64_t w = header_->write_index.load(std::memory_order_acquire);
    if (header_->writer_start_ns != writer_start_ns_) {  // lan dau / bridge khoi dong lai
        writer_start_ns_ = header_->writer_start_ns;
        read_index_ = w > kBacklog ? w - kBacklog : 0;
    }
    if (w - read_index_ > cap) {
        lost = static_cast<std::size_t>(w - read_index_ - cap);
        read_index_ = w - cap;
    }
    for (uint64_t i = read_index_; i < w; ++i) {
        const FcShmEntry& e = entries_[i & (cap - 1)];
        const uint64_t s1 = e.seq.load(std::memory_order_acquire);
        if (s1 != 2 * i + 2) continue;  // dang ghi / da bi ghi de
        MavMessage m;
        m.msgid = e.msgid;
        m.len = e.len;
        m.sysid = e.sysid;
        m.compid = e.compid;
        std::copy_n(e.payload.begin(), kFcShmPayload, m.payload.begin());
        const int64_t t = e.t_rx_ns;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (e.seq.load(std::memory_order_relaxed) != s1) continue;
        out.push_back(m);
        t_rx_ns.push_back(t);
    }
    read_index_ = w;
    return true;
}

}  // namespace uavp
