#!/usr/bin/env python3
"""Tạo tờ A4 in ArUco điểm cất cánh cho planA (chạy trên laptop hoặc Pi).

    python3 planA/make_launch_marker.py                       # DICT_4X4_50 id 1, cạnh viền đen 100 mm
    python3 planA/make_launch_marker.py --side-mm 120 --id 2

Kết quả: planA/markers/launch_<dict>_id<id>_<side>mm_A4.pdf (+ .png 300 dpi). In bằng "Actual size / 100 %"
(không "Fit to page"), rồi đo lại cạnh viền đen bằng thước và ghi vào marker.side_mm (planA/config/scenario.yaml).

Tờ in định nghĩa hệ A dùng để ĐÁNH GIÁ, không dùng để điều khiển:
  gốc = tâm marker trên sàn, x = mũi tên "+x" (cạnh trên marker), y = mũi tên "+y" (bên phải), z = hướng xuống.
Bốn vạch giữa bốn cạnh nằm trên hai đường tâm, dùng để kẻ trục trên sàn và đo vật bằng thước.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 300
A4_MM = (210.0, 297.0)
BORDER_MM = 20.0       # viền trắng quanh marker, có đường cắt xám (detector cần >= 1 ô = cạnh / 6)
ARROW_GAP_MM = 6.0     # mũi tên / vạch tâm bắt đầu cách đường cắt
CUT_GRAY = 150         # màu đường cắt + vạch tâm trong viền trắng (nhạt, không làm nhiễu detector)
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
FONT_BOLD = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
DICTIONARIES = ('DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_5X5_50', 'DICT_5X5_100', 'DICT_6X6_50',
                'DICT_6X6_250', 'DICT_ARUCO_ORIGINAL')  # giống src/support/aruco_check.cpp


def px(mm):
    return int(round(mm * DPI / 25.4))


def marker_image(dictionary, marker_id, side_px):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    if hasattr(cv2.aruco, 'generateImageMarker'):
        return cv2.aruco.generateImageMarker(d, marker_id, side_px, borderBits=1)
    return cv2.aruco.drawMarker(d, marker_id, side_px, borderBits=1)


def font(size_mm, bold=False):
    path = FONT_BOLD if bold else FONT
    if path.is_file():
        return ImageFont.truetype(str(path), px(size_mm))
    return ImageFont.load_default()


def arrow(draw, start, end, width, head):
    draw.line([start, end], fill=0, width=width)
    (x0, y0), (x1, y1) = start, end
    v = np.array([x1 - x0, y1 - y0], float)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    tip = np.array([x1, y1], float)
    base = tip - v * head
    draw.polygon([tuple(tip), tuple(base + n * head * 0.5), tuple(base - n * head * 0.5)], fill=0)


def make_sheet(dictionary, marker_id, side_mm, border_mm=BORDER_MM):
    w, h = px(A4_MM[0]), px(A4_MM[1])
    page = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(page)
    if border_mm < side_mm / 6.0:
        raise ValueError(f'viền trắng {border_mm:g} mm < 1 ô marker ({side_mm / 6.0:.1f} mm): detector khó tìm')
    side, b = px(side_mm), px(border_mm)
    cx, cy = w // 2, px(20.0 + border_mm + ARROW_GAP_MM + 30.0) + side // 2  # chừa chỗ mũi tên +x phía trên
    x0, y0 = cx - side // 2, cy - side // 2
    page.paste(Image.fromarray(marker_image(dictionary, marker_id, side)), (x0, y0))

    # Viền trắng: đường cắt xám mảnh quanh vùng trắng + vạch tâm và nhãn nhỏ ở mép ngoài (giữ trục sau khi cắt).
    ox0, oy0, ox1, oy1 = x0 - b, y0 - b, x0 + side + b, y0 + side + b
    draw.rectangle([ox0, oy0, ox1, oy1], outline=CUT_GRAY, width=px(0.3))
    notch = px(4)
    for start, end in (((cx, oy0), (cx, oy0 + notch)), ((cx, oy1), (cx, oy1 - notch)),
                       ((ox0, cy), (ox0 + notch, cy)), ((ox1, cy), (ox1 - notch, cy))):
        draw.line([start, end], fill=CUT_GRAY, width=px(0.5))
    draw.text((cx + px(2), oy0 + px(1.5)), '+x', font=font(3.5, True), fill=CUT_GRAY)
    draw.text((ox1 - px(7), cy + px(1.5)), '+y', font=font(3.5, True), fill=CUT_GRAY)

    gap, line_w = b + px(ARROW_GAP_MM), px(0.6)
    # Mũi tên +x (lên, cạnh trên) và +y (sang phải), nằm trên hai đường tâm.
    arrow(draw, (cx, y0 - gap), (cx, y0 - gap - px(28)), px(1.2), px(7))
    draw.text((cx + px(4), y0 - gap - px(27)), '+x  (mũi UAV)', font=font(5, True), fill=0)
    arrow(draw, (x0 + side + gap, cy), (x0 + side + gap + px(15), cy), px(1.2), px(6))
    draw.text((x0 + side + gap + px(2), cy + px(3)), '+y', font=font(5, True), fill=0)
    # Vạch tâm phía trái và phía dưới (cùng đường tâm với hai mũi tên).
    draw.line([(x0 - gap, cy), (x0 - gap - px(15), cy)], fill=0, width=line_w)
    draw.line([(cx, y0 + side + gap), (cx, y0 + side + gap + px(15))], fill=0, width=line_w)

    # Thước kiểm tra tỉ lệ in: 100 mm, vạch mỗi 10 mm.
    sx, sy = px(30), y0 + side + gap + px(26)
    draw.line([(sx, sy), (sx + px(100), sy)], fill=0, width=px(0.5))
    for k in range(11):
        tick = px(4 if k % 5 == 0 else 2.5)
        draw.line([(sx + px(10 * k), sy), (sx + px(10 * k), sy - tick)], fill=0, width=px(0.3))
    draw.text((sx + px(103), sy - px(4)), '100 mm: đo lại, phải đúng 100 mm', font=font(3.2), fill=0)

    lines = [
        (f'ArUco {dictionary}  id {marker_id}  -  cạnh viền đen {side_mm:g} mm', True),
        ('Điểm cất cánh planA. CHỈ ĐỂ ĐÁNH GIÁ (so với thước); điều khiển vẫn theo EKF.', False),
        ('Hệ A: gốc = tâm marker, x = mũi tên +x, y = +y (bên phải), z = hướng xuống (sàn = 0).', False),
        ('1. In "Actual size / 100 %". Đo cạnh viền đen -> marker.side_mm (planA/config/scenario.yaml).', False),
        (f'2. Cắt theo đường xám (giữ viền trắng {border_mm:g} mm). Dán phẳng, không bóng; kẻ trục theo 4 vạch tâm.', False),
        ('3. Đặt UAV sao cho CAMERA ở trên tâm marker, mũi UAV theo +x.', False),
        ('4. Đo tâm mặt trên của vật bằng thước theo x, y; z = -(chiều cao) -> marker.ground_truth.', False),
        ('Marker trên vật là id khác (aruco.marker_id = 0); không dán 2 marker cùng id.', False),
    ]
    ty = sy + px(10)
    for text, bold in lines:
        draw.text((px(15), ty), text, font=font(3.6 if bold else 3.0, bold), fill=0)
        ty += px(6.0 if bold else 5.0)
    if ty > h - px(5):
        raise ValueError('marker + viền trắng quá lớn cho A4: giảm --side-mm hoặc --border-mm')
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dict', default='DICT_4X4_50', choices=DICTIONARIES)
    parser.add_argument('--id', type=int, default=1, help='KHÁC id marker trên vật (0)')
    parser.add_argument('--side-mm', type=float, default=100.0, help='cạnh viền đen (40..120 mm cho A4)')
    parser.add_argument('--border-mm', type=float, default=BORDER_MM, help='viền trắng ngoài (>= cạnh / 6)')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'markers')
    args = parser.parse_args()
    if not 40.0 <= args.side_mm <= 120.0:
        parser.error('--side-mm phải trong 40..120 mm (vừa A4 với vùng trắng + chú thích)')
    if args.id < 0:
        parser.error('--id phải >= 0')
    try:
        page = make_sheet(args.dict, args.id, args.side_mm, args.border_mm)
    except ValueError as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f'launch_{args.dict.lower()}_id{args.id}_{args.side_mm:g}mm_A4'
    page.save(stem.with_suffix('.png'), dpi=(DPI, DPI))
    page.convert('RGB').save(stem.with_suffix('.pdf'), resolution=DPI)
    print(f'{stem}.pdf (+ .png, {DPI} dpi). In 100 %, đo cạnh viền đen -> marker.side_mm.')


if __name__ == '__main__':
    main()
