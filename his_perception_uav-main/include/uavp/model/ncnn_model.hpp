// Model NCNN (.param + .bin, fp32 / fp16 / INT8) -> IInfer: input (3, H, W) float32 -> output sap theo ten (out0, ...).
// Nap mot lan luc khoi dong; loi file -> exception. Pool allocator dung lai bo nho giua cac lan chay; phep tinh fp16
// tat (Pi 4 / A72 khong co); fp16 storage theo config (do bat / tat bang uavp_bench_model).
#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "uavp/common/common.hpp"
#include "uavp/common/config.hpp"

namespace uavp {

class NcnnModel final : public IInfer {
public:
    explicit NcnnModel(const ModelConfig& cfg);
    ~NcnnModel() override;
    NcnnModel(const NcnnModel&) = delete;
    NcnnModel& operator=(const NcnnModel&) = delete;

    void run(const InputCHW& input, std::vector<Tensor>& outputs) override;
    // Anh uint8 RGB -> from_pixels(_resize) + chuan hoa + copy_make_border cua ncnn (NEON) -> model.
    bool run_rgb(const cv::Mat& src_rgb, const ImagePrep& prep, std::vector<Tensor>& outputs) override;

private:
    struct Impl;  // giu header ncnn ngoai giao dien
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string ncnn_version();

// Luong OpenMP cua NCNN chi chay tren `cores` (them lop bao ve: binh thuong chung da ke thua tap nhan cua tien trinh,
// nhung NCNN co the tu dat lai affinity). Rong: khong lam gi. Goi sau khi nap model, truoc lan chay dau.
void pin_ncnn_threads(std::span<const int> cores);

}  // namespace uavp
