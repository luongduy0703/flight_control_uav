#include "uavp/sources/pi_camera.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <sys/mman.h>

#include <libcamera/libcamera.h>

#include "uavp/common/common.hpp"
#include "uavp/common/log.hpp"

namespace uavp {

namespace lc = libcamera;

struct PiCamera::Impl {
    struct Mapped {
        const lc::FrameBuffer* buffer{nullptr};
        void* base{nullptr};
        std::size_t length{0};
        const uint8_t* data{nullptr};
    };

    std::unique_ptr<lc::CameraManager> manager;
    std::shared_ptr<lc::Camera> camera;
    std::unique_ptr<lc::CameraConfiguration> config;
    std::unique_ptr<lc::FrameBufferAllocator> allocator;
    std::vector<std::unique_ptr<lc::Request>> requests;
    std::vector<Mapped> mapped;
    lc::Stream* stream{nullptr};
    std::mutex mutex;
    std::condition_variable cv;
    lc::Request* latest{nullptr};  // request moi nhat chua lay (chua tra lai camera)
    uint64_t seq{0};
    uint64_t taken{0};
    int width{0}, height{0}, stride{0};
    bool acquired{false}, started{false};

    void configure(const CaptureConfig& cap, int buffer_count, int bit_depth);
    void map_buffers();
    void start(const CaptureConfig& cap);
    void on_complete(lc::Request* req);
    const uint8_t* data_of(const lc::FrameBuffer* fb) const;
};

void PiCamera::Impl::configure(const CaptureConfig& cap, int buffer_count, int bit_depth) {
    config = camera->generateConfiguration({lc::StreamRole::VideoRecording});
    lc::StreamConfiguration& sc = config->at(0);
    sc.pixelFormat = lc::formats::BGR888;  // byte [R, G, B]
    sc.size = lc::Size(static_cast<unsigned>(cap.width_px), static_cast<unsigned>(cap.height_px));
    sc.bufferCount = static_cast<unsigned>(buffer_count);
    if (cap.rotate_deg != 0 && cap.rotate_deg != 180) throw std::invalid_argument("rotate_deg chi ho tro 0 / 180");
    config->orientation = cap.rotate_deg == 180 ? lc::Orientation::Rotate180 : lc::Orientation::Rotate0;
    config->sensorConfig = lc::SensorConfiguration();  // che do cam bien cua file K (du goc nhin)
    config->sensorConfig->outputSize = lc::Size(static_cast<unsigned>(cap.sensor_width_px),
                                                static_cast<unsigned>(cap.sensor_height_px));
    config->sensorConfig->bitDepth = static_cast<unsigned>(bit_depth);
    if (config->validate() == lc::CameraConfiguration::Invalid) throw std::runtime_error("Cau hinh camera khong hop le");
    if (sc.size.width != static_cast<unsigned>(cap.width_px) || sc.size.height != static_cast<unsigned>(cap.height_px) ||
        sc.pixelFormat != lc::formats::BGR888)
        throw std::runtime_error("Camera khong ho tro " + std::to_string(cap.width_px) + "x" +
                                 std::to_string(cap.height_px) + " BGR888 (libcamera doi thanh " + sc.toString() + ")");
    if (camera->configure(config.get()) < 0) throw std::runtime_error("camera->configure that bai");
    stream = sc.stream();
    width = cap.width_px;
    height = cap.height_px;
    stride = static_cast<int>(sc.stride);
}

void PiCamera::Impl::map_buffers() {
    allocator = std::make_unique<lc::FrameBufferAllocator>(camera);
    if (allocator->allocate(stream) < 0) throw std::runtime_error("Khong cap phat duoc bo dem camera");
    for (const auto& buf : allocator->buffers(stream)) {
        const lc::FrameBuffer::Plane& plane = buf->planes()[0];  // RGB: mot plane
        const std::size_t length = plane.offset + plane.length;
        void* base = mmap(nullptr, length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
        if (base == MAP_FAILED) throw std::runtime_error("mmap bo dem camera that bai");
        mapped.push_back({buf.get(), base, length, static_cast<const uint8_t*>(base) + plane.offset});
        std::unique_ptr<lc::Request> req = camera->createRequest();
        if (!req || req->addBuffer(stream, buf.get()) < 0) throw std::runtime_error("Khong tao duoc request camera");
        requests.push_back(std::move(req));
    }
}

void PiCamera::Impl::start(const CaptureConfig& cap) {
    lc::ControlList ctrls(lc::controls::controls);
    const auto frame_us = static_cast<int64_t>(std::llround(1e6 / cap.fps));  // fps co dinh
    // Kieu gia tri lay tu chinh control: libcamera cu = libcamera::Span, ban moi (Raspberry Pi 0.8) = std::span.
    using DurationLimits = std::remove_cvref_t<decltype(lc::controls::FrameDurationLimits)>::type;
    const std::array<int64_t, 2> limits{frame_us, frame_us};
    ctrls.set(lc::controls::FrameDurationLimits, DurationLimits(limits));
    if (camera->controls().count(&lc::controls::AfMode) != 0U) {  // net KHOA o lens_position cua file K
        ctrls.set(lc::controls::AfMode, lc::controls::AfModeManual);
        ctrls.set(lc::controls::LensPosition, static_cast<float>(cap.lens_position));
    }
    camera->requestCompleted.connect(this, &Impl::on_complete);
    if (camera->start(&ctrls) < 0) throw std::runtime_error("camera->start that bai");
    started = true;
    for (auto& r : requests) camera->queueRequest(r.get());
}

void PiCamera::Impl::on_complete(lc::Request* req) {
    if (req->status() == lc::Request::RequestCancelled) return;
    lc::Request* old = nullptr;
    {
        std::lock_guard lock(mutex);
        old = latest;
        latest = req;
        ++seq;
    }
    cv.notify_one();
    if (old != nullptr) {  // khung cu khong ai lay: tra lai camera ngay
        old->reuse(lc::Request::ReuseBuffers);
        camera->queueRequest(old);
    }
}

const uint8_t* PiCamera::Impl::data_of(const lc::FrameBuffer* fb) const {
    for (const auto& m : mapped)
        if (m.buffer == fb) return m.data;
    return nullptr;
}

PiCamera::PiCamera(const CaptureConfig& capture, int buffer_count, int sensor_bit_depth) : impl_(std::make_unique<Impl>()) {
    Impl& im = *impl_;
    im.manager = std::make_unique<lc::CameraManager>();
    if (im.manager->start() < 0) throw std::runtime_error("Khong khoi dong duoc libcamera");
    if (im.manager->cameras().empty()) throw std::runtime_error("Khong thay camera: rpicam-hello --list-cameras");
    im.camera = im.manager->cameras().front();
    if (im.camera->acquire() < 0)
        throw std::runtime_error("Camera dang bi giu (rpicam-* / uavp_* khac: pgrep -af uavp_): tat roi chay lai");
    im.acquired = true;
    im.configure(capture, buffer_count, sensor_bit_depth);
    im.map_buffers();
    im.start(capture);
    UAVP_LOG_INFO("uavp.sources", "Camera: %s, %dx%d, che do cam bien %dx%d, lat %d do, %.0f fps, LensPosition %.3f",
                  im.camera->id().c_str(), capture.width_px, capture.height_px, capture.sensor_width_px,
                  capture.sensor_height_px, capture.rotate_deg, capture.fps, capture.lens_position);
}

PiCamera::~PiCamera() {
    Impl& im = *impl_;
    if (im.started) {
        im.camera->stop();
        im.camera->requestCompleted.disconnect(&im);
    }
    im.requests.clear();
    for (const auto& m : im.mapped) munmap(m.base, m.length);
    if (im.allocator && im.stream != nullptr) im.allocator->free(im.stream);
    im.allocator.reset();
    if (im.acquired) im.camera->release();
    im.camera.reset();
    if (im.manager) im.manager->stop();
}

bool PiCamera::next_frame(cv::Mat& image_rgb, FrameMeta& meta, double timeout_s) {
    Impl& im = *impl_;
    lc::Request* req = nullptr;
    {
        std::unique_lock lock(im.mutex);
        const auto timeout = std::chrono::nanoseconds(static_cast<int64_t>(timeout_s * 1e9));
        if (!im.cv.wait_for(lock, timeout, [&] { return im.latest != nullptr && im.seq > im.taken; })) return false;
        req = im.latest;
        im.latest = nullptr;
        im.taken = im.seq;
    }
    const lc::FrameBuffer* fb = req->findBuffer(im.stream);
    const uint8_t* src = im.data_of(fb);
    image_rgb.create(im.height, im.width, CV_8UC3);
    const auto row_bytes = static_cast<std::size_t>(im.width) * 3;
    for (int r = 0; r < im.height; ++r)  // mot lan chep: DMA -> slot FramePool (stride co the > width * 3)
        std::memcpy(image_rgb.ptr<uint8_t>(r), src + static_cast<std::size_t>(r) * static_cast<std::size_t>(im.stride), row_bytes);
    const lc::ControlList& md = req->metadata();
    const auto ts = md.get(lc::controls::SensorTimestamp);
    meta.t_ns = ts ? *ts - boottime_minus_monotonic_ns() : mono_ns();  // CLOCK_BOOTTIME -> CLOCK_MONOTONIC
    const auto lens = md.get(lc::controls::LensPosition);
    const auto exposure = md.get(lc::controls::ExposureTime);
    const auto gain = md.get(lc::controls::AnalogueGain);
    meta.lens_position = lens ? static_cast<double>(*lens) : -1.0;
    meta.exposure_us = exposure ? static_cast<double>(*exposure) : -1.0;
    meta.analogue_gain = gain ? static_cast<double>(*gain) : -1.0;
    req->reuse(lc::Request::ReuseBuffers);
    im.camera->queueRequest(req);
    return true;  // thieu SensorTimestamp: check_camera bao lech dong ho o khung dau
}

}  // namespace uavp
