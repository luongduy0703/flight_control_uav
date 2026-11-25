// Vung nho chung (POSIX shared memory, RAM: /dev/shm/<ten>) giua uavp_fc_bridge (ghi) va chuong trinh perception
// (doc). Bridge doc MAVLink tu Pixhawk (TELEM2 -> UART GPIO cua Pi) va ghi moi tin can dung (HEARTBEAT, ATTITUDE,
// LOCAL_POSITION_NED, DISTANCE_SENSOR, RANGEFINDER, EXTENDED_SYS_STATE, OPTICAL_FLOW, VFR_HUD, COMMAND_LONG) kem thoi diem nhan (CLOCK_MONOTONIC: chung moi tien trinh tren
// Pi) vao ring buffer. Mot nguoi ghi, nhieu nguoi doc, khong khoa: moi o co so thu tu (seqlock).
//   seq = 2 i + 1: dang ghi o thu i;  seq = 2 i + 2: o thu i da ghi xong.
// Nguoi doc chep o roi kiem lai seq: doi -> o vua bi ghi de (bo qua).
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "uavp/sources/mavlink.hpp"

namespace uavp {

inline constexpr uint32_t kFcShmMagic = 0x55465348;  // "UFSH"
inline constexpr uint32_t kFcShmVersion = 2;  // 2: them sysid, compid ben gui
inline constexpr std::size_t kFcShmPayload = 48;     // >= payload lon nhat dung (DISTANCE_SENSOR 39 byte)

struct FcShmEntry {
    std::atomic<uint64_t> seq{0};
    int64_t t_rx_ns{0};
    uint32_t msgid{0};
    uint8_t len{0};
    uint8_t sysid{0}, compid{0};
    uint8_t pad_{0};
    std::array<uint8_t, kFcShmPayload> payload{};
};
static_assert(sizeof(FcShmEntry) == 72, "FcShmEntry layout changed");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "can atomic khong khoa de dung giua cac tien trinh");

struct FcShmHeader {
    std::atomic<uint32_t> magic{0};  // ghi cuoi cung khi khoi tao xong
    uint32_t version{0};
    uint32_t capacity{0};            // so o, luy thua 2
    uint32_t entry_bytes{0};
    int64_t writer_start_ns{0};      // doi = bridge khoi dong lai
    int32_t writer_pid{0};
    uint32_t pad_{0};
    std::atomic<uint64_t> write_index{0};
    std::atomic<int64_t> last_rx_ns{0};
    std::atomic<uint64_t> bytes{0}, crc_errors{0};
    std::atomic<uint64_t> n_heartbeat{0}, n_attitude{0}, n_position{0}, n_range{0};
};

// Bridge: tao / khoi tao lai vung nho (giu file /dev/shm de nguoi doc dang mo khong loi), ghi tin.
class FcShmWriter {
public:
    FcShmWriter(const std::string& name, std::size_t capacity);
    ~FcShmWriter();
    FcShmWriter(const FcShmWriter&) = delete;
    FcShmWriter& operator=(const FcShmWriter&) = delete;

    void push(const MavMessage& m, int64_t t_rx_ns) noexcept;
    void add_io(std::size_t bytes, std::size_t crc_errors) noexcept;
    [[nodiscard]] const FcShmHeader& header() const noexcept { return *header_; }

private:
    FcShmHeader* header_{nullptr};
    FcShmEntry* entries_{nullptr};
    void* mem_{nullptr};
    std::size_t size_{0};
    std::size_t mask_{0};
};

// Nguoi doc: mo vung nho co san (bridge chua chay -> exception kem goi y).
class FcShmReader {
public:
    explicit FcShmReader(const std::string& name);
    ~FcShmReader();
    FcShmReader(const FcShmReader&) = delete;
    FcShmReader& operator=(const FcShmReader&) = delete;

    // Them cac tin moi tu lan doc truoc vao out / t_rx (xoa truoc). Lan dau: lay toi da `backlog` tin gan nhat.
    // Tra ve so tin bi mat (doc cham hon ring). false neu bridge chua khoi tao xong vung nho.
    bool poll(std::vector<MavMessage>& out, std::vector<int64_t>& t_rx_ns, std::size_t& lost);
    [[nodiscard]] const FcShmHeader& header() const noexcept { return *header_; }

private:
    const FcShmHeader* header_{nullptr};
    const FcShmEntry* entries_{nullptr};
    const void* mem_{nullptr};
    std::size_t size_{0};
    uint64_t read_index_{0};
    int64_t writer_start_ns_{-1};
};

}  // namespace uavp
