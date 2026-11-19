// Doc / ghi tensor float32 ra file nhi phan (du lieu parity do tools/dump_parity.py sinh tu ban Python):
//   int32 ndim, int32 shape[ndim], float32 data (little-endian, C order).
// ReplayInfer: IInfer tra lai output da ghi san theo thu tu (test logic C++ ma khong can NCNN).
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "uavp/common/common.hpp"

namespace uavp {

void read_tensor(const std::filesystem::path& path, Tensor& out);
void write_tensor(const std::filesystem::path& path, const Tensor& t);

class ReplayInfer final : public IInfer {
public:
    // Moi lan run() tra ve nhom file tiep theo trong `calls` (moi phan tu: cac file output out0, out1...).
    explicit ReplayInfer(std::vector<std::vector<std::filesystem::path>> calls) : calls_(std::move(calls)) {}
    void run(const InputCHW& input, std::vector<Tensor>& outputs) override;
    [[nodiscard]] const InputCHW& first_input() const noexcept { return first_input_; }
    [[nodiscard]] std::size_t calls_done() const noexcept { return next_; }

private:
    std::vector<std::vector<std::filesystem::path>> calls_;
    InputCHW first_input_;  // input lan goi dau (so tien xu ly)
    std::size_t next_{0};
};

}  // namespace uavp
