// Mot phien chon vat: CHI MOT vat dang xet = vat gan tam anh nhat (khong xep hang nhieu vat). Moi khung:
// (s1 neu can) -> s2 segment tren ROI du doan cua vat -> s3 do, quyet dinh: OK -> ket thuc; ABORT -> s1 chon vat gan tam
// tiep theo (chua xet); het vat -> ket thuc.
// s1 chi chay khi bat dau / chua co vat / vat bi loai, hoac khi mat vat (s2 khong ra mask o khung truoc). Cac khung
// khac chi chay s2 (1 vat / khung).
#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/stages/s1_detect.hpp"
#include "uavp/stages/s2_segment.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

inline constexpr int kTargets = 1;  // so vat xet cung luc: 1 vat gan tam

struct Session {
    std::vector<Track> tracks;          // moi track da xet (ca vat cu)
    std::vector<int> group;             // chi so (trong tracks) cua vat dang xet (0 hoac kTargets phan tu)
    std::vector<Detection> detections;  // detection khung hien tai (buffer dung lai; rong neu khung nay khong chay s1)
    bool need_detect{true};             // khung sau phai chay s1 (dau phien / can chon nhom moi)
    bool reselect{false};               // nhom truoc ABORT het o khung khong chay s1: s1 ke tiep chon nhom moi
    bool done{false};
};

struct StepTimes {
    double detect_ms{0.0};
    double segment_ms{0.0};
    double select_ms{0.0};
    bool detected{false};  // khung nay co chay s1
};

// Mot khung (anh RGB chup luc t_ns, dong ho chung): sua tai cho `s`. Tra ve thoi gian tung buoc.
StepTimes session_step(Session& s, uint32_t frame_idx, int64_t t_ns, const cv::Mat& image_rgb, const Config& cfg,
                       Detector& detector, Segmenter& segmenter);

// s2 tren ROI du doan cua mot track (khong qua s1) + cap nhat van toc. true neu co mask (ghi vao `out`).
[[nodiscard]] bool track_segment(Track& t, uint32_t frame_idx, const cv::Mat& image_rgb, const TrackConfig& cfg,
                                 Segmenter& segmenter, SegObject& out);

// Log 1 dong thoi gian + 1 dong moi track cua nhom (giong log_frame cua ban Python).
void log_frame(uint32_t frame_idx, std::size_t n_detections, const Session& s, const StepTimes& t,
               const Config& cfg);

}  // namespace uavp
