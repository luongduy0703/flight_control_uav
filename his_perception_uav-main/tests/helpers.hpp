// Ham dung chung cho test (giong cac helper trong tests/*.py cua ban Python).
#pragma once

#include <filesystem>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "uavp/common/common.hpp"
#include "uavp/common/config.hpp"
#include "uavp/stages/s4_fusion.hpp"
#include "uavp/stages/s2_segment.hpp"

namespace uavp::test {

inline const std::filesystem::path kRoot = UAVP_ROOT;

// Camera that (K + meo) cua du an.
inline CameraModel camera() { return load_camera(kRoot / "configs/calib/K_1920x1080.yaml"); }

// Tensor 2 chieu tu cac hang.
inline Tensor tensor2(const std::vector<std::vector<float>>& rows) {
    Tensor t;
    t.dims = 2;
    t.shape = {static_cast<int>(rows.size()), static_cast<int>(rows.at(0).size()), 1};
    for (const auto& r : rows) t.data.insert(t.data.end(), r.begin(), r.end());
    return t;
}

inline Tensor tensor3(int c, int h, int w, float value) {
    Tensor t;
    t.dims = 3;
    t.shape = {c, h, w};
    t.data.assign(static_cast<std::size_t>(c * h * w), value);
    return t;
}

// Model gia: luon tra ve cung output, dem so lan goi va luu shape input.
class FakeInfer final : public IInfer {
public:
    explicit FakeInfer(std::vector<Tensor> outs) : outs_(std::move(outs)) {}
    void run(const InputCHW& input, std::vector<Tensor>& outputs) override {
        shapes.push_back({input.c, input.h, input.w});
        outputs = outs_;
    }
    std::vector<std::array<int, 3>> shapes;

private:
    std::vector<Tensor> outs_;
};

}  // namespace uavp::test
