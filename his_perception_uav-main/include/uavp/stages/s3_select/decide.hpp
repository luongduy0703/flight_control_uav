// s3 select / decide: DEFER -> OK / ABORT (n_min, w_max_mm, conf_min, mask_small). Giong decide.py.
#pragma once

#include <optional>
#include <span>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

// --- decide ---------------------------------------------------------------------------------------------------
[[nodiscard]] double median_W_mm(const Track& track);
[[nodiscard]] double median_L_mm(const Track& track);
[[nodiscard]] double mean_conf(const Track& track);
void update_state(Track& track, Observation obs, const DecideConfig& cfg);

}  // namespace uavp
