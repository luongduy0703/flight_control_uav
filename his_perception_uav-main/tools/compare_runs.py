"""So sanh thoi gian chay + phan cung giua nhieu lan chay uavp_real_time (vd Pi 4 va Pi 5, hoac ban C++ va ban
Python cu: cung dinh dang result.json).

    python tools/compare_runs.py runs/<lan_1> runs/<lan_2> ... [--csv so_sanh.csv]   (chi can Python 3, khong can thu vien)

Doc result.json cua tung thu muc (muc device, session, timing_ms, hw_usage) -> bang (log ra man hinh) va CSV (neu
co --csv). Phan cung lay o phase s1_s3 (luc chay model).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
# (ten cot, duong dan trong result.json)
COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("thiet_bi", ("device", "model")),
    ("khung", ("session", "frames")),
    ("khung_s", ("session", "fps_s1_s3")),
    ("tong_ms", ("session", "total_ms")),
    ("anh_ms", ("timing_ms", "capture", "median_ms")),
    ("s1_ms", ("timing_ms", "s1_detect", "median_ms")),
    ("s1_p95_ms", ("timing_ms", "s1_detect", "p95_ms")),
    ("s2_ms", ("timing_ms", "s2_segment", "median_ms")),
    ("s3_ms", ("timing_ms", "s3_select", "median_ms")),
    ("s4_ms", ("timing_ms", "s4_fusion", "median_ms")),
    ("vong_ms", ("timing_ms", "loop", "median_ms")),
    ("cpu_tb_%", ("hw_usage", "phases", "s1_s3", "cpu_total_pct_mean")),
    ("cpu_max_%", ("hw_usage", "phases", "s1_s3", "cpu_total_pct_max")),
    ("tien_trinh_%", ("hw_usage", "phases", "s1_s3", "proc_cpu_pct_mean")),
    ("rss_max_MB", ("hw_usage", "phases", "s1_s3", "proc_rss_mb_max")),
    ("ram_max_MB", ("hw_usage", "phases", "s1_s3", "mem_used_mb_max")),
    ("nhiet_max_C", ("hw_usage", "phases", "s1_s3", "temp_c_max")),
    ("tan_so_min_MHz", ("hw_usage", "phases", "s1_s3", "freq_mhz_min")),
    ("throttle_cuoi", ("hw_usage", "throttled_end", "hex")),
)


def lookup(data: Any, path: tuple[str, ...]) -> Any:
    """data[path[0]][path[1]]...; thieu khoa -> None. Ham thuan."""
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def run_row(result: dict[str, Any]) -> list[str]:
    """Mot dong bang tu result.json. Ham thuan."""
    return ["-" if (v := lookup(result, path)) is None else str(v) for _, path in COLUMNS]


def format_table(names: list[str], rows: list[list[str]]) -> str:
    """Bang chu: moi COT la mot lan chay (de doc khi co nhieu chi so). Ham thuan."""
    header = ["chi_so", *names]
    lines = [[col, *(row[i] for row in rows)] for i, (col, _) in enumerate(COLUMNS)]
    widths = [max(len(r[j]) for r in [header, *lines]) for j in range(len(header))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in [header, *lines])


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="thu muc runs/<ngay-gio>_real_time")
    parser.add_argument("--csv", type=Path, default=None, help="ghi bang ra CSV (moi dong mot lan chay)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    names, rows = [], []
    for run_dir in args.runs:
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        names.append(run_dir.name)
        rows.append(run_row(result))
    LOGGER.info("%s", format_table(names, rows))
    if args.csv is not None:
        lines = [",".join(["lan_chay", *(c for c, _ in COLUMNS)])]
        lines += [",".join([n, *(f'"{v}"' if "," in v else v for v in r)]) for n, r in zip(names, rows, strict=True)]
        args.csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        LOGGER.info("Da ghi %s", args.csv)


if __name__ == "__main__":
    main()
