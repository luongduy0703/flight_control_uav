"""May tinh (GCS) -> Pixhawk -> Pi: lenh bat dau / dung uavp_real_time, va xem trang thai hover.

    python tools/gcs_command.py --connect COM5:57600 start     # telemetry radio / USB cua Pixhawk (dong Mission Planner)
    python tools/gcs_command.py --connect udp:0.0.0.0:14550 start   # qua Mission Planner (MAVLink Mirror, xem GUIDE)
    python tools/gcs_command.py --connect COM5:57600 stop
    python tools/gcs_command.py --connect COM5:57600 status    # arm, che do bay, dang bay, van toc, do cao optical flow

Lenh: COMMAND_LONG (MAV_CMD_USER_1 = 31010, param1 = 1 bat dau / 0 dung) gui toi (sysid 1, compid 191) = Pi.
Pixhawk chuyen lenh toi cong TELEM2 vi uavp_fc_bridge tren Pi gui HEARTBEAT voi compid 191 tu cong do.
Cho COMMAND_ACK tu Pi (bridge tra loi); khong co -> thu lai. Pi chi BAT DAU khi them dieu kien hover dat
(configs/run/real_time.yaml: start). Chi can Python 3 + pyserial (khi dung cong COM). Tu dong goi MAVLink 2.
"""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import time
from dataclasses import dataclass

LOGGER = logging.getLogger("gcs_command")
CRC_EXTRA = {0: 50, 30: 39, 32: 185, 74: 20, 76: 152, 77: 143, 100: 175, 132: 85, 245: 130}
MSG_HEARTBEAT, MSG_COMMAND_LONG, MSG_COMMAND_ACK = 0, 76, 77
MAV_TYPE_GCS, MAV_AUTOPILOT_INVALID = 6, 8
MAV_RESULT = {0: "ACCEPTED", 1: "TEMPORARILY_REJECTED", 2: "DENIED", 3: "UNSUPPORTED", 4: "FAILED"}
LANDED = {0: "khong ro", 1: "duoi dat", 2: "DANG BAY", 3: "cat canh", 4: "ha canh"}
COPTER_MODES = {0: "STABILIZE", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED", 5: "LOITER", 6: "RTL", 9: "LAND", 16: "POSHOLD"}


def crc_x25(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        t = (b ^ (crc & 0xFF)) & 0xFF
        t = (t ^ (t << 4)) & 0xFF
        crc = ((crc >> 8) ^ (t << 8) ^ (t << 3) ^ (t >> 4)) & 0xFFFF
    return crc


def pack(msgid: int, payload: bytes, seq: int, sysid: int, compid: int) -> bytes:
    hdr = bytes([0xFD, len(payload), 0, 0, seq & 0xFF, sysid, compid, msgid & 0xFF, (msgid >> 8) & 0xFF, msgid >> 16])
    crc = crc_x25(bytes([CRC_EXTRA[msgid]]), crc_x25(hdr[1:] + payload))
    return hdr + payload + struct.pack("<H", crc)


@dataclass
class Msg:
    msgid: int
    sysid: int
    compid: int
    payload: bytes


def parse(buf: bytearray) -> list[Msg]:
    """Tach goi MAVLink 1 / 2 dung CRC (tin da biet) tu dau buf; xoa phan da dung."""
    out: list[Msg] = []
    i = 0
    while i < len(buf):
        if buf[i] not in (0xFD, 0xFE):
            i += 1
            continue
        v2 = buf[i] == 0xFD
        hlen = 10 if v2 else 6
        if len(buf) - i < hlen:
            break
        n = buf[i + 1]
        total = hlen + n + 2 + (13 if v2 and buf[i + 2] & 1 else 0)
        if len(buf) - i < total:
            break
        msgid = (buf[i + 7] | buf[i + 8] << 8 | buf[i + 9] << 16) if v2 else buf[i + 5]
        if msgid in CRC_EXTRA:
            crc = crc_x25(bytes([CRC_EXTRA[msgid]]), crc_x25(bytes(buf[i + 1 : i + hlen + n])))
            if crc != struct.unpack_from("<H", buf, i + hlen + n)[0]:
                i += 1
                continue
            payload = bytes(buf[i + hlen : i + hlen + n]) + bytes(64)  # MAVLink 2 cat byte 0 cuoi: them lai
            out.append(Msg(msgid, buf[i + (5 if v2 else 3)], buf[i + (6 if v2 else 4)], payload))
        i += total
    del buf[:i]
    return out


class Link:
    """Cong serial (COMx:baud) hoac UDP (udp:ip:port nghe, tra loi ve noi gui; udpout:ip:port gui toi)."""

    def __init__(self, spec: str) -> None:
        self.ser = None
        self.sock: socket.socket | None = None
        self.peer: tuple[str, int] | None = None
        if spec.startswith(("udp:", "udpout:")):
            kind, host, port = spec.split(":")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(0.05)
            if kind == "udp":
                self.sock.bind((host, int(port)))
            else:
                self.peer = (host, int(port))
        else:
            import serial  # pip install pyserial

            port, baud = spec.rsplit(":", 1)
            self.ser = serial.Serial(port, int(baud), timeout=0.05)

    def read(self) -> bytes:
        if self.ser is not None:
            return self.ser.read(4096)
        assert self.sock is not None
        try:
            data, addr = self.sock.recvfrom(65535)
        except TimeoutError:
            return b""
        self.peer = self.peer or addr
        return data

    def write(self, data: bytes) -> None:
        if self.ser is not None:
            self.ser.write(data)
        elif self.sock is not None and self.peer is not None:
            self.sock.sendto(data, self.peer)


class Gcs:
    def __init__(self, link: Link, sysid: int, compid: int) -> None:
        self.link, self.sysid, self.compid, self.seq = link, sysid, compid, 0
        self.buf = bytearray()

    def send(self, msgid: int, payload: bytes) -> None:
        self.link.write(pack(msgid, payload, self.seq, self.sysid, self.compid))
        self.seq = (self.seq + 1) & 0xFF

    def heartbeat(self) -> None:
        self.send(MSG_HEARTBEAT, struct.pack("<IBBBBB", 0, MAV_TYPE_GCS, MAV_AUTOPILOT_INVALID, 0, 4, 3))

    def poll(self, seconds: float) -> list[Msg]:
        out: list[Msg] = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.buf.extend(self.link.read())
            out.extend(parse(self.buf))
        return out


def send_command(gcs: Gcs, start: bool, args: argparse.Namespace) -> bool:
    payload = struct.pack("<7fHBBB", 1.0 if start else 0.0, 0, 0, 0, 0, 0, 0, args.command_id, args.target_sysid,
                          args.target_compid, 0)
    for attempt in range(1, args.retries + 1):
        gcs.heartbeat()  # Pixhawk biet duong ve GCS (de chuyen COMMAND_ACK)
        gcs.send(MSG_COMMAND_LONG, payload)
        LOGGER.info("Gui %s (lan %d) toi sysid %d compid %d", "BAT DAU" if start else "DUNG", attempt, args.target_sysid,
                    args.target_compid)
        end = time.monotonic() + args.ack_timeout_s
        while time.monotonic() < end:  # doc tung 0.05 s: co ACK la tra loi ngay
            for m in gcs.poll(0.05):
                if m.msgid == MSG_COMMAND_ACK and m.compid == args.target_compid:
                    cmd, result = struct.unpack_from("<HB", m.payload, 0)
                    if cmd == args.command_id:
                        LOGGER.info("Pi tra loi COMMAND_ACK: %s", MAV_RESULT.get(result, result))
                        return result == 0
    LOGGER.error("Khong co COMMAND_ACK tu Pi: uavp_fc_bridge chay chua (journalctl -u uavp-fc-bridge)? Pixhawk co thay "
                 "HEARTBEAT compid %d tren TELEM2 khong (Mission Planner: MAVLink Inspector)?", args.target_compid)
    return False


def show_status(gcs: Gcs, seconds: float) -> None:
    """In trang thai de biet vi sao Pi chua bat dau (dieu kien hover)."""
    st: dict[str, str] = {}
    end = time.monotonic() + seconds
    last = 0.0
    while time.monotonic() < end:
        gcs.heartbeat()
        for m in gcs.poll(0.5):
            p = m.payload
            if m.msgid == MSG_HEARTBEAT and p[4] != MAV_TYPE_GCS and p[5] != MAV_AUTOPILOT_INVALID:
                mode = struct.unpack_from("<I", p, 0)[0]
                st["FC"] = f"{'DA ARM' if p[6] & 128 else 'chua arm'}, {COPTER_MODES.get(mode, mode)}"
            elif m.msgid == MSG_HEARTBEAT and m.compid == 191:
                st["Pi"] = "co HEARTBEAT (bridge chay)"
            elif m.msgid == 245:
                st["bay"] = LANDED.get(p[1], p[1])
            elif m.msgid == 32:
                vx, vy, vz = struct.unpack_from("<3f", p, 16)
                st["van toc"] = f"ngang {(vx * vx + vy * vy) ** 0.5:.2f} m/s, doc {vz:.2f} m/s"
            elif m.msgid == 100:
                st["do cao flow"] = f"{struct.unpack_from('<f', p, 16)[0]:.3f} m (quality {p[25]})"
        if time.monotonic() - last >= 1.0:
            last = time.monotonic()
            LOGGER.info(" | ".join(f"{k}: {v}" for k, v in st.items()) or "chua nhan duoc tin nao")


def main() -> None:
    ap = argparse.ArgumentParser(description="Lenh bat dau / dung perception tren Pi qua Pixhawk")
    ap.add_argument("action", choices=["start", "stop", "status"])
    ap.add_argument("--connect", required=True, help="COM5:57600 | udp:0.0.0.0:14550 | udpout:127.0.0.1:14550")
    ap.add_argument("--target-sysid", type=int, default=1, help="= sources.mavlink.companion_sysid")
    ap.add_argument("--target-compid", type=int, default=191, help="= sources.mavlink.companion_compid")
    ap.add_argument("--command-id", type=int, default=31010, help="= sources.mavlink.command_id (MAV_CMD_USER_1)")
    ap.add_argument("--gcs-sysid", type=int, default=255)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--ack-timeout-s", type=float, default=1.5)
    ap.add_argument("--status-s", type=float, default=30.0, help="thoi gian in trang thai (status)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    gcs = Gcs(Link(args.connect), args.gcs_sysid, 190)
    if args.action == "status":
        show_status(gcs, args.status_s)
        return
    ok = send_command(gcs, args.action == "start", args)
    if ok and args.action == "start":
        LOGGER.info("Pi da nhan lenh. Pi bat dau khi UAV dang hover (xem: gcs_command.py status, log Pi)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
