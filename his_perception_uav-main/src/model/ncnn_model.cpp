#include "uavp/model/ncnn_model.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include <opencv2/core.hpp>

#include "allocator.h"  // ncnn::PoolAllocator, UnlockedPoolAllocator
#include "cpu.h"        // ncnn::CpuSet, set_cpu_thread_affinity
#include "net.h"        // ncnn (include/ncnn tu target ncnn)

namespace uavp {

namespace {

constexpr float kInv255 = 1.f / 255.f;
constexpr float kNorm[3] = {kInv255, kInv255, kInv255};

// ncnn::Mat -> Tensor (dung lai bo nho cua Tensor). Moi kenh mot lan chep (cstep cua ncnn co the > h * w).
void to_tensor(const ncnn::Mat& out, Tensor& t) {
    const bool three = out.dims == 3;
    t.dims = three ? 3 : 2;
    t.shape = three ? std::array<int, 3>{out.c, out.h, out.w} : std::array<int, 3>{out.h, out.w, 1};
    const std::size_t ch_size = static_cast<std::size_t>(out.h) * static_cast<std::size_t>(out.w);
    t.data.resize(ch_size * static_cast<std::size_t>(three ? out.c : 1));
    if (!three) {
        std::memcpy(t.data.data(), out.data, ch_size * sizeof(float));
        return;
    }
    for (int c = 0; c < out.c; ++c)
        std::memcpy(t.data.data() + static_cast<std::size_t>(c) * ch_size, out.channel(c), ch_size * sizeof(float));
}

}  // namespace

// Bo nho: 2 pool cap phat mot lan roi dung lai (truoc day moi lop moi lan chay deu malloc / free buffer moi).
// blob: tensor giua cac lop, chi mot luong goi -> ban khong khoa; workspace: buffer tam ben trong lop (OpenMP) -> co khoa.
struct NcnnModel::Impl {
    ncnn::UnlockedPoolAllocator blob_pool;
    ncnn::PoolAllocator workspace_pool;
    ncnn::Net net;
    ncnn::Option prep_opt;  // tien xu ly (copy_make_border): cung pool, 1 luong
    ncnn::Mat input;        // run(): cap phat mot lan theo kich thuoc input dau tien
    ncnn::Mat padded;       // run_rgb(): anh da dem
    std::string input_name;
    std::vector<std::string> output_names;

    void extract(const ncnn::Mat& in, std::vector<Tensor>& outputs) {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_blob_allocator(&blob_pool);
        ex.set_workspace_allocator(&workspace_pool);
        ex.input(input_name.c_str(), in);
        outputs.resize(output_names.size());
        for (std::size_t i = 0; i < output_names.size(); ++i) {
            ncnn::Mat out;
            if (ex.extract(output_names[i].c_str(), out) != 0)
                throw std::runtime_error("ncnn extract " + output_names[i] + " loi");
            to_tensor(out, outputs[i]);
        }
    }
};

NcnnModel::NcnnModel(const ModelConfig& cfg) : impl_(std::make_unique<Impl>()) {
    ncnn::Option& opt = impl_->net.opt;
    opt.num_threads = cfg.num_threads;
    opt.use_vulkan_compute = cfg.use_vulkan;
    opt.lightmode = true;                 // giai phong tensor trung gian ngay khi lop sau da dung xong
    opt.use_packing_layout = true;        // NEON 4 kenh / lan
    opt.use_winograd_convolution = true;  // conv 3x3 fp32 nhanh hon tren A72
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;        // model INT8 (ncnn2int8): lop da luong tu chay int8, con lai fp32
    opt.use_fp16_packed = cfg.use_fp16_storage;
    opt.use_fp16_storage = cfg.use_fp16_storage;  // chi LUU fp16 (giam bang thong RAM); tinh van fp32
    opt.use_fp16_arithmetic = false;              // Pi 4 / A72 khong co asimdhp
    opt.use_bf16_storage = false;
    impl_->prep_opt.num_threads = 1;
    impl_->prep_opt.blob_allocator = &impl_->blob_pool;
    impl_->prep_opt.workspace_allocator = &impl_->workspace_pool;
    for (const auto& [file, is_param] : {std::pair{cfg.param_file, true}, std::pair{cfg.bin_file, false}}) {
        const std::filesystem::path path = cfg.dir / file;
        if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error("Khong co file model " + std::filesystem::absolute(path).string() +
                                     " (chep thu muc models/ len Pi; chay trong thu muc goc uav_perception_pi/)");
        const int rc = is_param ? impl_->net.load_param(path.c_str()) : impl_->net.load_model(path.c_str());
        if (rc != 0) throw std::runtime_error("Khong doc duoc " + path.string() + " (file hong / chep thieu?)");
    }
    impl_->input_name = impl_->net.input_names().at(0);
    for (const char* name : impl_->net.output_names()) impl_->output_names.emplace_back(name);
    std::sort(impl_->output_names.begin(), impl_->output_names.end());
}

NcnnModel::~NcnnModel() {
    impl_->input.release();  // tra bo nho ve pool truoc khi pool bi huy
    impl_->padded.release();
    impl_->net.clear();
}

void NcnnModel::run(const InputCHW& input, std::vector<Tensor>& outputs) {
    ncnn::Mat& in = impl_->input;
    if (in.w != input.w || in.h != input.h || in.c != input.c) in.create(input.w, input.h, input.c);
    const std::size_t plane = static_cast<std::size_t>(input.h) * static_cast<std::size_t>(input.w);
    for (int k = 0; k < input.c; ++k)
        std::memcpy(in.channel(k), input.data.data() + static_cast<std::size_t>(k) * plane, plane * sizeof(float));
    impl_->extract(in, outputs);
}

bool NcnnModel::run_rgb(const cv::Mat& src_rgb, const ImagePrep& p, std::vector<Tensor>& outputs) {
    if (src_rgb.type() != CV_8UC3) return false;
    Impl& im = *impl_;
    const auto stride = static_cast<int>(src_rgb.step[0]);
    const bool same = p.resized_w_px == src_rgb.cols && p.resized_h_px == src_rgb.rows;
    // uint8 HWC RGB -> float CHW (+ resize bilinear NEON), mot buoc, khong buffer trung gian
    ncnn::Mat resized = same ? ncnn::Mat::from_pixels(src_rgb.data, ncnn::Mat::PIXEL_RGB, src_rgb.cols, src_rgb.rows,
                                                      stride, &im.blob_pool)
                             : ncnn::Mat::from_pixels_resize(src_rgb.data, ncnn::Mat::PIXEL_RGB, src_rgb.cols,
                                                             src_rgb.rows, stride, p.resized_w_px, p.resized_h_px,
                                                             &im.blob_pool);
    resized.substract_mean_normalize(nullptr, kNorm);  // [0, 255] -> [0, 1]
    const int bottom = p.out_h_px - p.resized_h_px - p.pad_y_px;
    const int right = p.out_w_px - p.resized_w_px - p.pad_x_px;
    if (p.pad_x_px == 0 && p.pad_y_px == 0 && bottom == 0 && right == 0) {
        im.extract(resized, outputs);
        return true;
    }
    ncnn::copy_make_border(resized, im.padded, p.pad_y_px, bottom, p.pad_x_px, right, ncnn::BORDER_CONSTANT,
                           static_cast<float>(p.pad_value) * kInv255, im.prep_opt);
    im.extract(im.padded, outputs);
    return true;
}

void pin_ncnn_threads(std::span<const int> cores) {
    if (cores.empty()) return;
    ncnn::CpuSet set;
    for (const int c : cores) set.enable(c);
    if (ncnn::set_cpu_thread_affinity(set) != 0) throw std::runtime_error("ncnn::set_cpu_thread_affinity loi");
}

std::string ncnn_version() {
#ifdef NCNN_VERSION_STRING
    return NCNN_VERSION_STRING;
#else
    return "?";
#endif
}

}  // namespace uavp
