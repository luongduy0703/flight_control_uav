#include "uavp/common/tensor_io.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace uavp {

void read_tensor(const std::filesystem::path& path, Tensor& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Khong doc duoc tensor " + path.string());
    int32_t ndim = 0;
    f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    if (ndim < 1 || ndim > 3) throw std::runtime_error("Tensor ndim khong ho tro: " + path.string());
    out.shape = {1, 1, 1};
    std::size_t count = 1;
    for (int i = 0; i < ndim; ++i) {
        int32_t d = 0;
        f.read(reinterpret_cast<char*>(&d), sizeof(d));
        out.shape[static_cast<std::size_t>(i)] = d;
        count *= static_cast<std::size_t>(d);
    }
    out.dims = ndim;
    out.data.resize(count);
    f.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) throw std::runtime_error("Tensor thieu du lieu: " + path.string());
}

void write_tensor(const std::filesystem::path& path, const Tensor& t) {
    std::ofstream f(path, std::ios::binary);
    const int32_t ndim = t.dims;
    f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (int i = 0; i < t.dims; ++i) {
        const int32_t d = t.shape[static_cast<std::size_t>(i)];
        f.write(reinterpret_cast<const char*>(&d), sizeof(d));
    }
    f.write(reinterpret_cast<const char*>(t.data.data()), static_cast<std::streamsize>(t.data.size() * sizeof(float)));
}

void ReplayInfer::run(const InputCHW& input, std::vector<Tensor>& outputs) {
    if (next_ >= calls_.size()) throw std::runtime_error("ReplayInfer: het output ghi san");
    if (next_ == 0) first_input_ = input;
    const auto& files = calls_[next_++];
    outputs.resize(files.size());
    for (std::size_t i = 0; i < files.size(); ++i) read_tensor(files[i], outputs[i]);
}

}  // namespace uavp
