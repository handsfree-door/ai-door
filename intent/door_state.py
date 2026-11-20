import sys
import pandas as pd
from load_tracks import load_all
from rule_baseline import decide, get_window, NEAR_H

DOOR_SIDE = "right"   # 카메라 화면에서 문이 있는 쪽 ("right" 또는 "left")
LEAVE_DH = 0.10       # 1초 동안 박스 높이가 0.10 이상 줄면 → 뒤로 물러남
AWAY_DX = 0.15        # 1초 동안 문 반대쪽으로 x가 0.15 이상 가면 → 옆으로 빠짐
STAY_H = 0.9          # 이 높이 이상에서
STAY_MOVE = 0.05      # x, h 변화가 둘 다 이보다 작으면 → 문 앞에 서 있음
LOST_SEC = 1.0        # 사람이 화면에서 사라지고 이만큼 지나면 닫음
REOPEN_BLOCK_SEC = 2.0  # 멀어져서 닫은 뒤 이 시간 동안은 같은 사람 때문에 다시 열지 않음


def decide_while_open(window):
    # 문이 열려 있는 동안 사람의 움직임 판단
    dx = window["x"].iloc[-1] - window["x"].iloc[0]
    dh = window["h"].iloc[-1] - window["h"].iloc[0]
    h_now = window["h"].iloc[-1]
    away = -dx if DOOR_SIDE == "right" else dx   # +면 문에서 멀어지는 방향

    if dh <= -LEAVE_DH:
        return "LEAVING"   # 뒤로 물러남
    if h_now >= NEAR_H and away >= AWAY_DX:
        return "LEAVING"   # 옆으로 빠짐
    if h_now >= STAY_H and abs(dx) < STAY_MOVE and abs(dh) < STAY_MOVE:
        return "STAY"      # 문 앞에 서 있음
    return "MOVING"


def simulate(traj):
    # 한 사람 궤적으로 문을 열고 닫아 봄
    # 반환: 명령 목록 [(시각, 명령, 이유)], 문 앞에 서 있던 시간(초)
    traj = traj.sort_values("t")
    events = []
    is_open = False
    blocked_until = -1.0  # 이 시각 전까지는 다시 열지 않음
    stay_sec = 0.0
    prev = None
    for now in traj["t"]:
        window = get_window(traj, now)
        if window is None:
            prev = now
            continue
        if not is_open:
            # 멀어지는 중이면 박스가 커져도 다시 열지 않음
            if (decide(window) == "OPEN" and decide_while_open(window) != "LEAVING"
                    and now >= blocked_until):
                events.append((now, "OPEN", "다가옴"))
                is_open = True
        else:
            state = decide_while_open(window)
            if state == "LEAVING":
                events.append((now, "CLOSE", "LEAVING"))
                is_open = False
                blocked_until = now + REOPEN_BLOCK_SEC
            elif state == "STAY":
                stay_sec += now - prev
        prev = now
    if is_open:
        events.append((traj["t"].max() + LOST_SEC, "CLOSE", "사라짐"))
    return events, stay_sec


if __name__ == "__main__":
    # 기본은 세미나실만 (집은 카메라 구도가 달라 문 방향 규칙이 안 맞음)
    sessions = sys.argv[1:] or ["20261114_세미나실"]
    data = load_all(sessions)

    rows = []
    for traj_id, traj in data.groupby("traj_id"):
        scenario = traj["scenario"].iloc[0]
        events, stay_sec = simulate(traj)
        opens = [t for t, cmd, _ in events if cmd == "OPEN"]
        leaving = [t for t, _, why in events if why == "LEAVING"]
        old_close = traj["t"].max() + LOST_SEC   # 예전 방식: 사라지고 1초 뒤 닫음

        if not opens:
            verdict = "-"
        elif len(opens) > 1:
            verdict = "다시 열림!"   # 닫았다가 또 열면 문이 덜컹거림
        elif scenario in ("hesitate_back", "approach_leave"):
            verdict = "OK" if leaving else "안 닫힘"
        else:
            verdict = "도중에 닫힘!" if leaving else "OK"

        rows.append({
            "궤적": traj_id.split("/", 1)[1],
            "시나리오": scenario,
            "열림": round(opens[0], 1) if opens else None,
            "STAY(초)": round(stay_sec, 1),
            "LEAVING닫힘": round(leaving[0], 1) if leaving else None,
            "예전방식닫힘": round(old_close, 1),
            "앞당김(초)": round(old_close - leaving[0], 1) if leaving else None,
            "판정": verdict,
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    print("\n=== 판정 요약 ===")
    print(result.groupby(["시나리오", "판정"]).size().to_string())