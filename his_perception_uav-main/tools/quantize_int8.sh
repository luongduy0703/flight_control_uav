#!/bin/bash
# Luong tu model NCNN fp32 -> INT8 (ncnn PTQ): ncnnoptimize -> ncnn2table (hieu chuan KL tren anh train) -> ncnn2int8.
# Chay tren Linux / WSL co cong cu ncnn (ban prebuilt Ubuntu: github.com/Tencent/ncnn/releases, thu muc bin/).
#   tools/quantize_int8.sh <ncnn_bin_dir> <thu muc fp32 (model.ncnn.param/bin)> <danh sach anh hieu chuan .txt> \
#                          <w> <h> <thu muc ra> [regex lop giu fp32]
# Anh hieu chuan: dung kich thuoc chay that (s1: anh 960 x 544; s2: crop ROI 320 x 320), RGB / 255 nhu luc chay.
# Regex lop giu fp32 (tron INT8 / fp32): ten lop trong .param, vd head detect 'conv_(7[6-9]|8[0-9]|9[0-3])|convdw_21[2-7]'.
# Lop khop bi bo khoi bang thang do -> ncnn2int8 khong luong tu (head / proto nhay voi sai so INT8).
# Ket qua: <thu muc ra>/model.ncnn.param, model.ncnn.bin (INT8), model.table (thang do), quantize.log.
set -euo pipefail
BIN=$1; SRC=$2; LIST=$3; W=$4; H=$5; OUT=$6; KEEP=${7:-}
mkdir -p "$OUT"
LOG="$OUT/quantize.log"
: > "$LOG"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
"$BIN/ncnnoptimize" "$SRC/model.ncnn.param" "$SRC/model.ncnn.bin" "$TMP/opt.param" "$TMP/opt.bin" 0 >> "$LOG" 2>&1
"$BIN/ncnn2table" "$TMP/opt.param" "$TMP/opt.bin" "$LIST" "$OUT/model.table" \
    mean=[0,0,0] norm=[0.003921569,0.003921569,0.003921569] shape=[$W,$H,3] pixel=RGB thread=8 method=kl >> "$LOG" 2>&1
if [ -n "$KEEP" ]; then  # bang: "<lop>_param_0 <thang do trong so...>" va "<lop> <thang do input>"
    grep -vE "^(${KEEP})(_param_0)? " "$OUT/model.table" > "$TMP/table" || true
    echo "giu fp32: $(( $(wc -l < "$OUT/model.table") - $(wc -l < "$TMP/table") )) dong bang (regex: $KEEP)" | tee -a "$LOG"
    cp "$TMP/table" "$OUT/model.table"
fi
"$BIN/ncnn2int8" "$TMP/opt.param" "$TMP/opt.bin" "$OUT/model.ncnn.param" "$OUT/model.ncnn.bin" "$OUT/model.table" \
    >> "$LOG" 2>&1
echo "INT8: $OUT ($(du -h "$OUT/model.ncnn.bin" | cut -f1), fp32 $(du -h "$SRC/model.ncnn.bin" | cut -f1))"
echo "so lop da luong tu INT8: $(grep -c '^quantize_' "$LOG" || true)"
if grep -qiE "read image .* failed|imread .* failed|no image" "$LOG"; then echo "CANH BAO: co anh hieu chuan doc loi (xem $LOG)"; fi
