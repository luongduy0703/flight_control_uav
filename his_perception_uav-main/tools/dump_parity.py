"""(Phat trien, can du an Python cu uav_perception - KHONG co trong repo nay.)
Sinh du lieu parity cho ban C++ (CPP_RULES.md muc 0.1). Chuan so sanh la ban Python o ../uav_perception
(cung code s1 - s3, cung model): chay voi model NCNN THAT tren vai anh, ghi lai:
  - anh (PNG, khong mat du lieu: C++ va Python giai ma JPEG co the lech),
  - moi lan goi model: output (out0, out1, ...) -> C++ phat lai bang ReplayInfer, so logic giai ma / NMS / mask,
  - input tile dau va ROI dau (so tien xu ly),
  - ket qua Python: detection, mask / tam / yaw / W cua tung vat trong nhom dau,
  - thu muc model da dung (test C++ kiem tra configs/models/*.yaml tro cung model).
C++ doc lai trong tests/test_parity.cpp.

    cd ../uav_perception          # ban Python (PYTHONPATH=src), model trong uav_perception/models/
    PYTHONPATH=src ../detection/.venv/Scripts/python.exe ../uav_perception_pi/tools/dump_parity.py         --cfg configs/select_simple.yaml --images "../uav_perception_pi/runs/*/frame.jpg" "runs/*/frame.jpg"         --out ../uav_perception_pi/parity_data [--max-images 6]

Dinh dang tensor: int32 ndim, int32 shape[ndim], float32 data (C order, little-endian).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from uavp.config import load_config
from uavp.ncnn_model import NcnnModel
from uavp.run_files import read_image_rgb
from uavp.stages.s1_detect.detector import Detector
from uavp.stages.s2_segment.segmenter import Segmenter
from uavp.stages.s3_select.measure import measure
from uavp.stages.s3_select.tracker import select_group

LOGGER = logging.getLogger(__name__)


def write_tensor(path: Path, a: np.ndarray) -> None:
    """int32 ndim + int32 shape + float32 data."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    path.write_bytes(struct.pack(f"<i{a.ndim}i", a.ndim, *a.shape) + a.tobytes())


class Recorder:
    """infer_fn ghi output moi lan goi (va input lan dau) ra file, roi tra output cua model that."""

    def __init__(self, model: NcnnModel, out_dir: Path) -> None:
        self.model, self.out_dir = model, out_dir
        self.prefix = ""
        self.calls: list[list[str]] = []
        self.first_input: str | None = None

    def start(self, prefix: str) -> None:
        self.prefix, self.calls, self.first_input = prefix, [], None

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        outs = self.model(x)
        idx = len(self.calls)
        if idx == 0:
            self.first_input = f"{self.prefix}_in.bin"
            write_tensor(self.out_dir / self.first_input, x)
        files = []
        for k, o in enumerate(outs):
            name = f"{self.prefix}_{idx:02d}_out{k}.bin"
            write_tensor(self.out_dir / name, o)
            files.append(name)
        self.calls.append(files)
        return outs


def dump_image(i: int, path: Path, cfg: Any, det: Detector, seg: Segmenter, rec_d: Recorder, rec_s: Recorder,
               out: Path) -> dict[str, Any]:
    image = read_image_rgb(path)
    png = f"img_{i}.png"
    cv2.imwrite(str(out / png), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # OpenCV ghi BGR
    rec_d.start(f"det_{i}")
    dets = det(image)
    entry: dict[str, Any] = {"source": path.as_posix(), "image": png, "det_calls": rec_d.calls, "det_input": rec_d.first_input,
                             "detections": [[*map(float, d.xyxy_px), d.conf, d.class_id] for d in dets], "segs": []}
    for j, track in enumerate(select_group([], dets, cfg.camera.center_px, cfg.n_targets, cfg.match_radius_px, 0)):
        rec_s.start(f"seg_{i}_{j}")
        obj = seg(image, track.detection.xyxy_px)
        s: dict[str, Any] = {"det": [*map(float, track.detection.xyxy_px), track.detection.conf,
                                     track.detection.class_id], "calls": rec_s.calls, "input": rec_s.first_input,
                             "found": obj is not None}
        if obj is not None:
            mask_name = f"seg_{i}_{j}_mask.png"
            cv2.imwrite(str(out / mask_name), obj.mask.mask.astype(np.uint8) * 255)
            o = measure(obj, track.detection.conf, 0, cfg.camera, cfg.D_m)
            s |= {"x0": obj.mask.x0_px, "y0": obj.mask.y0_px, "mask": mask_name, "center": obj.center_px.tolist(),
                  "yaw": obj.yaw_deg, "area": obj.area_px, "conf": obj.mask.conf, "W_mm": o.W_mm, "L_mm": o.L_mm}
        entry["segs"].append(s)
    LOGGER.info("%s: %d detection, %d vat segment", path, len(dets), len(entry["segs"]))
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--images", nargs="+", required=True, help="duong dan / mau glob anh")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.cfg, [])
    args.out.mkdir(parents=True, exist_ok=True)
    rec_d = Recorder(NcnnModel(cfg.detector.model), args.out)
    rec_s = Recorder(NcnnModel(cfg.segmenter.model), args.out)
    det, seg = Detector(cfg.detector, rec_d), Segmenter(cfg.segmenter, rec_s)
    paths = sorted({Path(p) for pattern in args.images for p in glob.glob(pattern)})[: args.max_images]
    images = [dump_image(i, p, cfg, det, seg, rec_d, rec_s, args.out) for i, p in enumerate(paths)]
    manifest = {"cfg": args.cfg.as_posix(), "D_m": cfg.D_m, "images": images,
                "models": {"s1": cfg.detector.model.dir.name, "s2": cfg.segmenter.model.dir.name}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    LOGGER.info("Da ghi %d anh vao %s", len(images), args.out)


if __name__ == "__main__":
    main()
