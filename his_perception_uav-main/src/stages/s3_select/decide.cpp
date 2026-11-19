#include "uavp/stages/s3_select/decide.hpp"

#include <span>

namespace uavp {

// --- decide ---------------------------------------------------------------------------------------------------
double median_W_mm(const Track& track) {
    std::vector<double> v;
    for (const auto& o : track.observations) v.push_back(o.W_mm);
    return median_inplace(std::span<double>(v));
}

double median_L_mm(const Track& track) {
    std::vector<double> v;
    for (const auto& o : track.observations) v.push_back(o.L_mm);
    return median_inplace(std::span<double>(v));
}

double mean_conf(const Track& track) {
    if (track.observations.empty()) return kNaN;
    double sum = 0.0;
    for (const auto& o : track.observations) sum += o.conf;
    return sum / static_cast<double>(track.observations.size());
}

void update_state(Track& track, Observation obs, const DecideConfig& cfg) {
    if (track.state != State::kDefer) return;
    if (obs.area_px < cfg.mask_min_px) {  // khong co mask / mask qua nho: giu DEFER, khong luu so do
        if (++track.mask_small_streak >= cfg.mask_small_max_frames) {
            track.state = State::kAbort;
            track.reason = Reason::kMaskSmall;
        }
        return;
    }
    track.mask_small_streak = 0;
    track.observations.push_back(std::move(obs));
    if (track.observations.size() < static_cast<std::size_t>(cfg.n_min)) return;
    if (median_W_mm(track) > cfg.w_max_mm) {
        track.state = State::kAbort;
        track.reason = Reason::kTooBig;
    } else if (mean_conf(track) < cfg.conf_min) {
        track.state = State::kAbort;
        track.reason = Reason::kLowConf;
    } else {
        track.state = State::kOk;
    }
}

}  // namespace uavp
