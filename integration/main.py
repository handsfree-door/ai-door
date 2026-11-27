# CSV를 실제 시간처럼 흘려보내면서 판단(controller) → 명령 → 문(가짜 문 또는 진짜 보드) 까지 한 번에 돌려보는 코드

import time
import argparse
from pathlib import Path

import pandas as pd

from controller import DoorController
from fake_door import FakeDoor

TICK_SEC = 0.1     # 0.1초마다 한 번씩 판단
TAIL_SEC = 5.0     # CSV가 끝난 뒤에도 5초 더 돌려서 문이 닫히는 것까지 확인


def thin_out(df, fps):
    # 파이 속도를 흉내 냄: 1/fps초 간격의 프레임만 남김 (모든 사람이 같은 프레임에 찍히므로 frame 기준)
    gap = 1.0 / fps
    keep_frames = []
    last = -999.0
    for frame_id, t in df.groupby("frame_id")["t"].first().sort_values().items():
        if t - last >= gap - 1e-6:
            keep_frames.append(frame_id)
            last = t
    return df[df["frame_id"].isin(keep_frames)]


def load_csv(path):
    df = pd.read_csv(path)
    df = df[df["class"] == "person"].copy()
    df["x"] = df["center_x"] / df["frame_w"]
    df["h"] = df["height"] / df["frame_h"]
    df["t"] = df["elapsed_sec"] - df["elapsed_sec"].min()  # 0초부터 시작
    return df.sort_values("t")


def run(path, fast=False, fps=None, port=None):
    df = load_csv(path)
    if fps:
        df = thin_out(df, fps)
        print(f"파이 속도 흉내: {fps}fps로 솎아서 실행")
    if port:
        from serial_door import SerialDoor  # 진짜 보드가 있을 때만 불러옴
        door = SerialDoor(port)
    else:
        door = FakeDoor()
    ctrl = DoorController(door)

    end_time = df["t"].max() + TAIL_SEC
    rows = df.to_dict("records")
    i = 0
    now = 0.0
    while now <= end_time:
        # 지금 시각까지 들어온 CSV 줄을 모아서 판단에 넘김
        people = []
        while i < len(rows) and rows[i]["t"] <= now:
            r = rows[i]
            people.append((int(r["track_id"]), r["t"], r["x"], r["h"]))
            i += 1
        ctrl.step(now, people)

        if not fast:
            time.sleep(TICK_SEC)
        now = round(now + TICK_SEC, 2)
    if hasattr(door, "close"):
        door.close()
    print("\n파일:", Path(path).name)
    ctrl.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="흘려보낼 추적 CSV 경로")
    parser.add_argument("--fast", action="store_true", help="기다리지 않고 빠르게 실행")
    parser.add_argument("--fps", type=float, help="파이 속도 흉내 (예: 3이면 1초에 3장만 처리)")
    parser.add_argument("--port", help="진짜 보드 포트 (예: COM3). 없으면 가짜 문으로 실행")
    args = parser.parse_args()
    if args.port and args.fast:
        parser.error("진짜 보드로 할 때는 --fast를 빼 주세요 (실제 시간대로 보내야 함)")
    run(args.csv, fast=args.fast, fps=args.fps, port=args.port)