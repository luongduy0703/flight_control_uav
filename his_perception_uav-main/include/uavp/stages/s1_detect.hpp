// Stage 1 - detect (YOLO26n NCNN, ca anh 960 x 544). Config: configs/stages/s1_detect.yaml (full / tile, nguong) +
// configs/models/s1_detect_ncnn.yaml (model, input_px [w, h], tien xu ly).
#pragma once

#include "uavp/stages/s1_detect/detector.hpp"
#include "uavp/stages/s1_detect/letterbox.hpp"
#include "uavp/stages/s1_detect/tiling.hpp"
