# 채원 님 CSV를 실제 시간처럼 흘려보내면서
# 판단(rule_baseline) → 명령 → 가짜 문 까지 한 번에 돌려보는 코드
#
# 실행 예시 (ai-door 폴더에서)
#   python integration/main.py data/vision/raw/20261114_세미나실/hesitate_back/20261114_154615_tracking.csv
#   python integration/main.py data/vision/raw/20261114_세미나실/hesitate_enter/20261114_154537_tracking.csv --fast

import sys
import time
import argparse
from collections import deque
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "intent"))  # intent 폴더 코드 불러오기

from rule_baseline import decide, WINDOW_SEC   # noqa: E402
from door_state import decide_while_open       # noqa: E402
from fake_door import FakeDoor                 # noqa: E402

TICK_SEC = 0.1     # 0.1초마다 한 번씩 판단
HOLD_MS = 2000     # 사람이 보이는 동안 2초씩 열림 유지를 연장
REFRESH_SEC = 0.5  # HOLD 연장 명령을 보내는 간격
LOST_SEC = 1.0     # 문을 열어준 사람이 1초 동안 안 보이면 지나간 걸로 봄
TAIL_SEC = 5.0     # CSV가 끝난 뒤에도 5초 더 돌려서 문이 닫히는 것까지 확인
TAILGATE_SEC = 2.0 # 앞사람 때문에 문이 열린 뒤 이 시간 안에 다음 사람이 붙으면 → 강한 경고

def load_csv(path):
    df = pd.read_csv(path)
    df = df[df["class"] == "person"].copy()
    df["x"] = df["center_x"] / df["frame_w"]
    df["h"] = df["height"] / df["frame_h"]
    df["t"] = df["elapsed_sec"] - df["elapsed_sec"].min()  # 0초부터 시작
    return df.sort_values("t")


def log(now, msg):
    print(f"[{now:6.2f}s] {msg}")


def run(path, fast=False):
    df = load_csv(path)
    door = FakeDoor()

    history = {}     # track_id → 최근 기록 (t, x, h)
    opened_for = {}  # 문을 열어준 track_id → 마지막으로 보인 시각
    crossed = set()  # 가로지른다고 판단한 track_id (로그 한 번만 찍으려고)
    staying = set()  # 문 앞에 서 있다고 판단한 track_id (로그 한 번만 찍으려고)
    last_hold = -999.0
    open_since = None  # 지금 열림이 시작된 시각 (따라옴 간격 계산용)
    events = []      # (시각, 명령) 기록

    def send(now, cmd):
        if not cmd.startswith("HOLD"):  # HOLD 연장은 자주 보내서 출력 생략
            log(now, f"→ 명령: {cmd}")
        door.send(cmd)
        events.append((now, cmd))

    end_time = df["t"].max() + TAIL_SEC
    rows = df.to_dict("records")
    i = 0
    now = 0.0

    while now <= end_time:
        door.update(now)

        # 1) 지금 시각까지 들어온 CSV 줄 읽기
        seen = set()
        while i < len(rows) and rows[i]["t"] <= now:
            r = rows[i]
            tid = int(r["track_id"])
            history.setdefault(tid, deque()).append((r["t"], r["x"], r["h"]))
            seen.add(tid)
            i += 1

        # 2) 사람마다 최근 1초 움직임으로 판단
        for tid in seen:
            hist = history[tid]
            while hist and hist[0][0] < now - WINDOW_SEC:
                hist.popleft()
            if hist[-1][0] - hist[0][0] < WINDOW_SEC * 0.8:
                continue  # 아직 1초치 기록이 안 쌓임

            window = pd.DataFrame(list(hist), columns=["t", "x", "h"])

            # 문을 열어준 사람이면: 서 있는지 / 떠나는지 판단
            if tid in opened_for:
                motion = decide_while_open(window)
                if motion == "LEAVING":
                    log(now, f"판단: ID {tid} 문에서 멀어지는 중 → 이 사람 몫의 열림 해제")
                    del opened_for[tid]
                    staying.discard(tid)
                    if not opened_for:  # 기다리는 다른 사람이 없으면 바로 닫기
                        send(now, "CLOSE")
                    continue
                if motion == "STAY" and tid not in staying:
                    log(now, f"판단: ID {tid} 문 앞에 서 있음 → 열어 둠")
                    staying.add(tid)
                elif motion != "STAY":
                    staying.discard(tid)

            decision = decide(window)

            # 멀어지는 중이면 박스가 커져도 다시 열지 않음 (문 덜컹거림 방지)
            if decision == "OPEN" and decide_while_open(window) == "LEAVING":
                decision = "WAIT"

            if decision == "OPEN" and tid not in opened_for:
                others = [o for o, last in opened_for.items() if now - last <= LOST_SEC]
                if others:
                    # 앞사람 때문에 이미 열린 문으로 다른 사람이 따라옴
                    gap = now - open_since
                    level = 2 if gap < TAILGATE_SEC else 1
                    log(now, f"판단: ID {tid}가 ID {others[0]} 뒤를 따라옴 (간격 {gap:.1f}초) → 경고 {level}단계")
                    send(now, f"ALERT {level}")
                else:
                    log(now, f"판단: ID {tid} 다가오는 중 → 문 열기")
                    open_since = now
                if door.target == 0:
                    send(now, "OPEN")
                send(now, f"HOLD {HOLD_MS}")
                last_hold = now
            elif decision == "CROSS" and tid not in crossed and tid not in opened_for:
                log(now, f"판단: ID {tid} 가로지르는 중 → 안 엶")
                crossed.add(tid)

            if decision == "OPEN" or tid in opened_for:
                opened_for[tid] = now  # 마지막으로 보인 시각 갱신

        # 3) 문을 열어준 사람이 아직 보이면 열림 유지 연장
        active = [tid for tid, last in opened_for.items() if now - last <= LOST_SEC]
        if active and now - last_hold >= REFRESH_SEC:
            send(now, f"HOLD {HOLD_MS}")
            last_hold = now

        # 4) 문을 열어준 사람이 모두 사라지면 닫기
        if opened_for and not active:
            log(now, "판단: 문을 연 사람이 지나감 → 문 닫기")
            send(now, "CLOSE")
            opened_for.clear()

        if not fast:
            time.sleep(TICK_SEC)
        now = round(now + TICK_SEC, 2)

    print("\n=== 요약 ===")
    print("파일:", Path(path).name)
    if not events:
        print("문이 한 번도 열리지 않았어요.")
    for t, cmd in events:
        if not cmd.startswith("HOLD"):
            print(f"{t:6.2f}s  {cmd}")
    print("마지막 상태:", door.read_state())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="흘려보낼 추적 CSV 경로")
    parser.add_argument("--fast", action="store_true", help="기다리지 않고 빠르게 실행")
    args = parser.parse_args()
    run(args.csv, fast=args.fast)