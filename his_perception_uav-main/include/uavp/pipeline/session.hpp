// s3_select.single_shot = true: MOT khung duy nhat. s1 (960 x 544) -> bo detection conf < conf_min (va vung loai / qua
// to nhu duoi) -> xet tu vat gan tam anh nhat: s2 tren bbox, mask >= mask_min_px -> OK (chon, sang khoa + bam nhu cu);
// mask nho -> ABORT mask_small, xet vat ke tiep. Moi vat KHAC (ke ca vat mask nho) -> next_targets (p_W tren san).
//
// single_shot = false (cu): Mot phien chon vat: CHI MOT vat dang xet = vat gan tam anh nhat (khong xep hang nhieu vat). Moi khung:
// (s1 neu can) -> s2 segment tren ROI du doan cua vat -> s3 do, quyet dinh: OK -> ket thuc; ABORT -> s1 chon vat gan tam
// tiep theo (chua xet); het vat -> ket thuc.
// s1 chi chay khi bat dau / chua co vat / vat bi loai, hoac khi mat vat (s2 khong ra mask o khung truoc). Cac khung
// khac chi chay s2 (1 vat / khung).
#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "uavp/common/config.hpp"
#include "uavp/stages/s1_detect.hpp"
#include "uavp/stages/s2_segment.hpp"
#include "uavp/stages/s3_select/tracker.hpp"
#include "uavp/common/types.hpp"

namespace uavp {

inline constexpr int kTargets = 1;  // so vat xet cung luc: 1 vat gan tam

// Vung anh KHONG chon vat (vd to giay ArUco diem cat canh): detection co bbox chua `center_px` hoac tam bbox cach
// `center_px` < radius_px bi bo khi chon vat; track dang xet nam trong vung -> ABORT launch_marker.
struct ExcludeZone {
    cv::Point2d center_px;
    double radius_px{0.0};
};
[[nodiscard]] bool in_zone(const Detection& d, const ExcludeZone& z) noexcept;
[[nodiscard]] bool excluded(const Detection& d, const std::vector<ExcludeZone>& zones) noexcept;

struct Session {
    std::vector<Track> tracks;          // moi track da xet (ca vat cu)
    std::vector<int> group;             // chi so (trong tracks) cua vat dang xet (0 hoac kTargets phan tu)
    std::vector<Detection> detections;  // detection khung hien tai (buffer dung lai; rong neu khung nay khong chay s1)
    bool need_detect{true};             // khung sau phai chay s1 (dau phien / can chon nhom moi)
    bool reselect{false};               // nhom truoc ABORT het o khung khong chay s1: s1 ke tiep chon nhom moi
    bool done{false};
    std::vector<ExcludeZone> exclude;   // dat lai moi khung truoc session_step (rong: khong loai)
    int excluded_detections{0};         // so detection da bo vi nam trong vung loai (ca phien)
    int too_big_detections{0};          // so detection da bo vi bbox qua to (s3_select.bbox_max_mm, ca phien)
    // Che do the gioi (s3_select.world_track): app dat `pose` = tu the camera LUC CHUP truoc moi session_step.
    // Mat san chot o khung dau co tu the: z = z camera + D cos(truc quang) (co vi tri) hoac D (chi goc quay, t = 0).
    std::optional<CamPose> pose;
    double floor_z_W_mm{kNaN};
    bool world_position{false};  // mat san chot khi co vi tri (khung khong cung kieu -> bo che do the gioi khung do)
    // s3_select.single_shot: chi so (trong tracks) cac vat KHAC vat duoc chon, gan tam truoc. Co tu the FC -> track.p_W_mm
    // = tam bbox chieu xuong mat san (NED EKF, mm): vi tri bay toi tiep theo.
    std::vector<int> next_targets;
};

// FloorView cua khung hien tai (nullopt: khong co tu the / tat world_track). Chot mat san o lan goi dau co tu the.
[[nodiscard]] std::optional<FloorView> floor_view(Session& s, const Config& cfg);

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
                                 Segmenter& segmenter, SegObject& out, const FloorView* view = nullptr);

// Log 1 dong thoi gian + 1 dong moi track cua nhom (giong log_frame cua ban Python).
void log_frame(uint32_t frame_idx, std::size_t n_detections, const Session& s, const StepTimes& t,
               const Config& cfg);

// Canh ngan bbox doi ra mm tai khoang cach D_m (m): px x D / fx. nan neu D khong hop le.
[[nodiscard]] double bbox_short_mm(const Detection& d, double D_m, const CameraModel& camera) noexcept;

}  // namespace uavp
