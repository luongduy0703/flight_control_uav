#include "uavp/common/boxes.hpp"

#include <numeric>

namespace uavp {

void nms_per_class(std::span<const BoxD> boxes, std::span<const double> scores, std::span<const uint8_t> class_ids,
                   double thr, Overlap metric, std::vector<int>& keep, std::vector<int>& order,
                   std::vector<uint8_t>& removed) {
    const auto n = static_cast<int>(boxes.size());
    keep.clear();
    order.resize(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    // np.argsort(-scores, kind="stable"): score giam dan, bang nhau giu thu tu chi so
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[static_cast<std::size_t>(a)] > scores[static_cast<std::size_t>(b)];
    });
    removed.assign(boxes.size(), 0);
    for (int i = 0; i < n; ++i) {
        const auto best = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
        if (removed[best] != 0) continue;
        keep.push_back(static_cast<int>(best));
        for (int j = i + 1; j < n; ++j) {
            const auto other = static_cast<std::size_t>(order[static_cast<std::size_t>(j)]);
            if (removed[other] != 0 || class_ids[other] != class_ids[best]) continue;
            if (overlap(boxes[best], boxes[other], metric) > thr) removed[other] = 1;
        }
    }
}

}  // namespace uavp
