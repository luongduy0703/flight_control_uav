#include <gtest/gtest.h>

#include "uavp/common/boxes.hpp"
#include "uavp/common/common.hpp"

using namespace uavp;

TEST(Boxes, IouIosIoa) {
    const BoxD a{0, 0, 10, 10}, b{5, 0, 15, 10}, small{2, 2, 4, 4};
    EXPECT_NEAR(overlap(a, b, Overlap::kIoU), 50.0 / 150.0, 1e-12);
    EXPECT_NEAR(overlap(a, small, Overlap::kIoS), 1.0, 1e-12);
    EXPECT_NEAR(ioa(small, a), 1.0, 1e-12);
    EXPECT_NEAR(ioa({8, 0, 12, 10}, a), 0.5, 1e-12);
}

TEST(Boxes, NmsPerClassKeepsOtherClassAndStableOrder) {
    const std::vector<BoxD> boxes{{0, 0, 10, 10}, {1, 0, 11, 10}, {0, 0, 10, 10}, {50, 50, 60, 60}};
    const std::vector<double> scores{0.8, 0.9, 0.7, 0.9};
    const std::vector<uint8_t> cls{0, 0, 1, 0};
    std::vector<int> keep, order;
    std::vector<uint8_t> removed;
    nms_per_class(boxes, scores, cls, 0.5, Overlap::kIoU, keep, order, removed);
    // score bang nhau (0.9): giu thu tu chi so; box 0 bi box 1 bo; box 2 khac class -> giu
    EXPECT_EQ(keep, (std::vector<int>{1, 3, 2}));
}

TEST(Common, MedianAndPercentileLikeNumpy) {
    std::vector<double> even{4, 1, 3, 2};
    EXPECT_DOUBLE_EQ(median_inplace(std::span<double>(even)), 2.5);
    std::vector<double> odd{5, 1, 3};
    EXPECT_DOUBLE_EQ(median_inplace(std::span<double>(odd)), 3.0);
    std::vector<double> with_nan{1, kNaN, 3};
    EXPECT_TRUE(std::isnan(median_inplace(std::span<double>(with_nan))));
    std::vector<double> p{1, 2, 3, 4, 5};
    EXPECT_DOUBLE_EQ(percentile_inplace(std::span<double>(p), 95.0), 4.8);
    EXPECT_DOUBLE_EQ(py_mod(-10.0, 360.0), 350.0);
}
