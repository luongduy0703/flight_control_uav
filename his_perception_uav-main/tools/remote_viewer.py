"""Xem + dieu khien uavp_real_time_perception / uavp_real_time (chay tren Pi, remote.enabled: true) tu MAY TINH.

    python tools/remote_viewer.py --host <ip-pi>                 # cong 8100 (run/real_time.yaml: remote.port)
    (can Python 3 + opencv-python + numpy: pip install opencv-python numpy)

Cua so hien anh camera thu nho + diem LiDAR (do gan, lam xa) + track / ROI + thoi gian tung buoc, do tre
(SensorTimestamp -> xong buoc, do tren Pi), phan cung Pi (CPU, RAM, nhiet, tan so). Phim (cua so dang chon):
    SPACE   uavp_real_time_perception bat dau quet vat (s1 -> s4). uavp_real_time bat dau bang lenh qua Pixhawk
            (tools/gcs_command.py start) khi dang hover; viewer hien dieu kien con thieu
    q / ESC dung som / thoat
Luu vao remote_runs/<ngay-gio>/: status.jsonl (moi khung), hw.jsonl (phan cung moi 1 s), result.json (ket qua cuoi tu Pi), final.jpg, summary.txt.
Giao thuc: xem include/uavp/support/remote_view.hpp.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger("remote_viewer")
HEADER = struct.Struct("<BI")  # loai, do dai
WINDOW = "uavp remote (SPACE: bat dau, q: thoat)"
FONT = cv2.FONT_HERSHEY_SIMPLEX
RECONNECT_S = 1.0
STATE_COLORS_BGR = {"defer": (0, 215, 255), "ok": (0, 200, 0), "abort": (30, 30, 230)}
LIDAR_GAP_PX = 25.0  # 2 diem LiDAR lien tiep cach xa hon -> khong noi (mep vat / mat tia)
LIDAR_COLOR_MIN_SPAN_MM = 50.0  # thang mau LiDAR toi thieu 5 cm
MASK_ALPHA = 0.35  # do dam mask to mau tren anh
CENTER_BAND_PX = 12  # diem LiDAR trong +-12 px quanh cot giua anh -> khoang cach tai tam anh


@dataclass
class Shared:
    """Du lieu moi nhat tu Pi (luong nhan ghi, luong hien thi doc)."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    frame_bgr: np.ndarray | None = None
    lidar: np.ndarray | None = None  # (n, 3): u, v (px anh thu nho), z_mm
    status: dict | None = None
    result: dict | None = None
    final_bgr: np.ndarray | None = None
    statuses: list[dict] = field(default_factory=list)
    hw: list[dict] = field(default_factory=list)
    connected: bool = False
    closed: bool = False


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Pi ngat ket noi")
        buf.extend(chunk)
    return bytes(buf)


def handle(kind: str, payload: bytes, sh: Shared) -> None:
    if kind == "J":
        img = cv2.imdecode(np.frombuffer(payload[8:], np.uint8), cv2.IMREAD_COLOR)  # JPEG -> BGR (OpenCV)
        with sh.lock:
            sh.frame_bgr = img
            if sh.status is not None and sh.status.get("phase") == "final":
                sh.final_bgr = img
    elif kind == "L":
        n = struct.unpack_from("<I", payload, 8)[0]
        pts = np.frombuffer(payload, np.float32, count=3 * n, offset=12).reshape(n, 3)
        with sh.lock:
            sh.lidar = pts
    elif kind == "S":
        st = json.loads(payload.decode("utf-8"))
        with sh.lock:
            if st.get("phase") == "hw":  # phan cung: rieng, khong thay trang thai khung
                sh.hw.append(st)
                return
            sh.status = st
            if st.get("phase") in ("select", "lock"):
                sh.statuses.append(st)
    elif kind == "R":
        with sh.lock:
            sh.result = json.loads(payload.decode("utf-8"))


def receiver(sock: socket.socket, sh: Shared) -> None:
    try:
        while True:
            t, n = HEADER.unpack(recv_exact(sock, HEADER.size))
            handle(chr(t), recv_exact(sock, n), sh)
    except (ConnectionError, OSError) as e:
        LOGGER.warning("%s", e)
    finally:
        with sh.lock:
            sh.connected = False


def connect(host: str, port: int) -> socket.socket:
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=3.0)
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            LOGGER.info("Da ket noi %s:%d", host, port)
            return sock
        except OSError as e:
            LOGGER.info("Chua ket noi duoc %s:%d (%s) - uavp_real_time da chay tren Pi? thu lai...", host, port, e)
            time.sleep(RECONNECT_S)


def stats(values: list[float]) -> str:
    v = [x for x in values if x is not None]
    if not v:
        return "-"
    return f"median {np.median(v):.0f} | p95 {np.percentile(v, 95):.0f} ms (n {len(v)})"


def hw_line(hw: list[dict]) -> str:
    if not hw:
        return "phan cung Pi: chua co"
    h = hw[-1]
    return (f"Pi: CPU {h['cpu_total_pct']:.0f}% (nhan max {h['cpu_core_max_pct']:.0f}%, chuong trinh {h['proc_cpu_pct']:.0f}%) "
            f"| RAM {h['proc_rss_mb']:.0f} MB | {h['temp_c']:.1f} C | {h['freq_mhz_min']:.0f} MHz")


def text_lines(st: dict | None, statuses: list[dict], hw: list[dict]) -> list[str]:
    if st is None:
        return ["cho du lieu...", hw_line(hw)]
    phase = st.get("phase")
    if phase == "wait":
        return [f"CHO BAM SPACE | LiDAR loi CRC {st.get('lidar_crc')}", hw_line(hw)]
    if phase == "wait_fc":
        hover = "can hover" if st.get("hover_required", True) else "khong can hover (chi de xem)"
        lines = ["CHO LENH: " + (st.get("reason") or "du dieu kien"),
                 f"lenh {st['commanded']} | {hover}: arm {st['armed']} | che do {st['mode_ok']} | dang bay {st['in_air']} | "
                 f"dung yen {st['still']} | do cao on dinh {st['height_stable']} ({st['height_m']} m)"]
        fc = st.get("fc")
        if fc:  # gia tri that tu Pixhawk: thay du lieu co cap nhat khong
            def f(v, fmt):
                return "-" if v is None else format(v, fmt)
            lines.append(f"FC: HB tuoi {f(fc['hb_age_s'], '.1f')} s | mode {fc['mode']} | landed {fc['landed']} | "
                         f"ATT {fc['n_att']} mau, tuoi {f(fc['att_age_s'], '.2f')} s, roll {f(fc['roll_deg'], '.1f')} "
                         f"pitch {f(fc['pitch_deg'], '.1f')} | v {f(fc['v_h'], '.2f')} m/s | do cao {fc['height_src']} "
                         f"{fc['height_samples']} mau | flow {f(fc['flow_m'], '.3f')} m q{fc['flow_q']}")
        return lines + [hw_line(hw)]
    sel = [s for s in statuses if s["phase"] == "select"]
    lck = [s for s in statuses if s["phase"] == "lock"]
    lines: list[str] = []
    if phase == "select":
        s1 = "-" if st["s1_ms"] is None else f"{st['s1_ms']:.0f}"
        lines.append(f"CHON VAT khung {st['frame']} | s1 {s1} s2 {st['s2_ms']:.0f} s3 {st['s3_ms']:.1f} | "
                     f"vong {st['loop_ms']:.0f} | tre {st['latency_ms']:.0f} ms | {st['n_det']} vat")
    elif phase == "lock":
        lines.append(f"BAM khung {st['frame']} | tu the {st['pose_mode']} | {st['method']} ({st['n_good']} diem) | "
                     f"buoc {st['step_ms']:.1f} s2 {st['s2_ms']:.0f} | tre {st['latency_ms']:.0f} ms | segment {st['segments']}")
    elif phase == "final":
        lines.append("XONG - ket qua trong result.json (xem terminal)")
    lines.append("tre chon vat: " + stats([s["latency_ms"] for s in sel]))
    lines.append("s1: " + stats([s["s1_ms"] for s in sel]))
    if lck:
        lines.append("tre bam: " + stats([s["latency_ms"] for s in lck]))
    lines.append(hw_line(hw))
    return lines


def lidar_color(z_mm: float, lo_mm: float, hi_mm: float) -> tuple[int, int, int]:
    t = float(np.clip((z_mm - lo_mm) / (hi_mm - lo_mm), 0.0, 1.0))
    return int(255 * t), 80, int(255 * (1 - t))  # BGR: gan nhat do, xa nhat lam (vat cao hon san -> do hon)


def draw_lidar(img: np.ndarray, lidar: np.ndarray) -> float | None:
    """Duong quet LiDAR (noi diem lien tiep theo goc) + cham; tra khoang cach median tai cot giua anh (mm).
    Mau theo chinh vong quet nay (p5..p95, toi thieu 50 mm): vat nho cao hon san vai cm van khac mau ro."""
    if len(lidar) == 0:
        return None
    lo, hi = np.percentile(lidar[:, 2], [5, 95])
    hi = max(hi, lo + LIDAR_COLOR_MIN_SPAN_MM)
    for i in range(len(lidar)):
        u, v, z = lidar[i]
        if i > 0:
            pu, pv, _ = lidar[i - 1]
            if (u - pu) ** 2 + (v - pv) ** 2 <= LIDAR_GAP_PX**2:
                cv2.line(img, (int(pu), int(pv)), (int(u), int(v)), lidar_color(z, lo, hi), 2, cv2.LINE_AA)
        cv2.circle(img, (int(u), int(v)), 2, lidar_color(z, lo, hi), -1)
    mid = img.shape[1] / 2
    near = lidar[np.abs(lidar[:, 0] - mid) <= CENTER_BAND_PX] if len(lidar) else lidar
    return float(np.median(near[:, 2])) if len(near) else None


def put_text(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Chu tren nen toi (doc duoc tren moi nen anh)."""
    (w, h), base = cv2.getTextSize(text, FONT, 0.55, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, 0.55, color, 1, cv2.LINE_AA)


def draw_center(img: np.ndarray) -> None:
    """Vach tam anh (truc quang ~ giua anh) de can vat."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 40, cy), (cx + 40, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 40), (cx, cy + 40), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 60, (255, 255, 255), 1, cv2.LINE_AA)


def draw_mask(img: np.ndarray, obj: dict, k: float, color: tuple[int, int, int], fresh: bool) -> None:
    """Mask (duong vien tu Pi, toa do anh goc) to mau ban trong suot + vien + tam. Mask cu (khung truoc): chi vien."""
    poly = obj.get("mask")
    if poly:
        pts = (np.asarray(poly, np.float32) * k).astype(np.int32)
        if fresh:
            layer = img.copy()
            cv2.fillPoly(layer, [pts], color)
            cv2.addWeighted(layer, MASK_ALPHA, img, 1.0 - MASK_ALPHA, 0.0, dst=img)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    center = obj.get("center_px")
    if center and center[0] is not None:
        cu, cv_ = int(center[0] * k), int(center[1] * k)
        cv2.drawMarker(img, (cu, cv_), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
        put_text(img, f"({center[0]:.0f}, {center[1]:.0f})", (cu + 10, cv_ + 18), (255, 0, 255))


def draw(sh: Shared, full_width_px: int) -> np.ndarray | None:
    with sh.lock:
        img = None if sh.frame_bgr is None else sh.frame_bgr.copy()
        lidar, st, statuses, connected, hw = sh.lidar, sh.status, list(sh.statuses), sh.connected, list(sh.hw[-1:])
    if img is None:
        return None
    k = img.shape[1] / full_width_px  # toa do box tu Pi: anh goc
    center_z = None
    if lidar is not None and (st is None or st.get("phase") != "final"):
        center_z = draw_lidar(img, lidar)
    if st is None or st.get("phase") in ("wait", "wait_fc"):
        draw_center(img)
        msg = "CAN VAT: dat vat duoi tam anh, tren duong LiDAR | SPACE: chay | q: thoat"
        if center_z is not None:
            msg += f" | LiDAR tai tam anh: {center_z:.0f} mm"
        h = img.shape[0]
        put_text(img, msg, (8, h - 14), (0, 255, 255))
    if st is not None and st.get("phase") == "select":
        for tr in st["tracks"]:
            x1, y1, x2, y2 = (int(c * k) for c in tr["box"])
            color = STATE_COLORS_BGR.get(tr["state"], (255, 255, 255))
            draw_mask(img, tr, k, color, fresh=tr.get("fresh", True))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            put_text(img, f"#{tr['id']} {tr['class']} {tr['state']} n{tr['n_obs']} W{tr['W_mm'] or 0:.0f}",
                     (x1, max(y1 - 6, 14)), color)
    if st is not None and st.get("phase") == "lock":
        x1, y1, x2, y2 = (int(c * k) for c in st["roi"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        draw_mask(img, st.get("object") or {}, k, (255, 0, 255), fresh=bool(st.get("segmented")))
        cu, cv_ = st["center_px"]
        cv2.drawMarker(img, (int(cu * k), int(cv_ * k)), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        pu, pv = st["pose_pred_px"]
        if pu is not None:
            cv2.circle(img, (int(pu * k), int(pv * k)), 6, (0, 255, 255), 2)  # du doan tu tu the FC
    lines = text_lines(st, statuses, hw) + ([] if connected else ["MAT KET NOI - dang thu lai"])
    for i, line in enumerate(lines):
        y = 22 + 22 * i
        put_text(img, line, (8, y), (255, 255, 255))
    return img


def summarize(res: dict) -> str:
    """Tom tat ket qua cuoi: sai so so voi ArUco, do tre, thoi gian tung buoc."""
    out = []
    fu = res.get("fusion") or {}
    if fu:
        out.append(f"s4 {fu.get('status')}: p_obj {fu.get('p_obj_mm')} mm | d {fu.get('d_mm')} | nghieng {fu.get('tilt_deg')} do "
                   f"(do duoc 2 huong: {fu.get('normal_observable')}) | W {fu.get('W_mm')} L {fu.get('L_mm')}")
    lk = res.get("lock") or {}
    if lk:
        out.append(f"p_W {lk.get('p_W_mm')} mm | tu the {lk.get('pose_mode')} | segment {lk.get('segments')} | "
                   f"flow/NCC {lk.get('methods')}")
    ar = res.get("aruco")
    if ar and ar.get("found"):
        e = ar["error_s4_minus_aruco"]
        out.append(f"SAI SO s4 - ArUco: d {e['d_mm']} mm ({e['d_pct']} %) | dist {e['dist_mm']} mm | p_obj {e['p_obj_mm']} mm "
                   f"| nghieng ArUco {ar.get('tilt_deg')} do")
    elif ar is not None:
        out.append("ArUco: khong thay marker")
    for name, t in (res.get("timing_ms") or {}).items():
        out.append(f"  {name:16s} median {t['median_ms']:8.1f} ms | p95 {t['p95_ms']:8.1f} ms | n {t['n']}")
    return "\n".join(out)


def save(sh: Shared, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with sh.lock:
        statuses, result, final, hw = list(sh.statuses), sh.result, sh.final_bgr, list(sh.hw)
    (out_dir / "status.jsonl").write_text("\n".join(json.dumps(s) for s in statuses), encoding="utf-8")
    (out_dir / "hw.jsonl").write_text("\n".join(json.dumps(h) for h in hw), encoding="utf-8")
    if result is not None:
        (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "summary.txt").write_text(summarize(result), encoding="utf-8")
    if final is not None:
        cv2.imwrite(str(out_dir / "final.jpg"), final)  # anh BGR (OpenCV)
    LOGGER.info("Da luu %s", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Xem + dieu khien uavp_real_time tu may tinh")
    ap.add_argument("--host", required=True, help="IP cua Pi")
    ap.add_argument("--port", type=int, default=8100, help="= remote.port trong configs/run/real_time.yaml")
    ap.add_argument("--full-width-px", type=int, default=1920, help="chieu rong anh goc camera (toa do box tu Pi)")
    ap.add_argument("--out", default="remote_runs", help="thu muc luu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    sh = Shared()
    sock = connect(args.host, args.port)
    sh.connected = True
    threading.Thread(target=receiver, args=(sock, sh), daemon=True).start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    reported = False
    while True:
        img = draw(sh, args.full_width_px)
        if img is not None:
            cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        with sh.lock:
            result, connected = sh.result, sh.connected
        if key == ord(" "):
            sock.sendall(b"s")
            LOGGER.info("SPACE: Pi bat dau quet vat")
        elif key in (ord("q"), 27):
            try:
                sock.sendall(b"q")
            except OSError:
                pass
            break
        if result is not None and not reported:
            LOGGER.info("KET QUA:\n%s", summarize(result))
            save(sh, out_dir)
            reported = True
        if not connected and not reported:
            sock = connect(args.host, args.port)  # Pi chay lai: noi lai
            with sh.lock:
                sh.connected = True
            threading.Thread(target=receiver, args=(sock, sh), daemon=True).start()
    if not reported:
        save(sh, out_dir)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
