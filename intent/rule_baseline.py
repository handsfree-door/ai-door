import sys
import pandas as pd
from load_tracks import load_all, FULL_HEIGHT

# 지난번(11/10 데이터) 그래프 보고 정한 기준값 그대로 사용
WINDOW_SEC = 1.0    # 최근 몇 초를 보고 판단할지
CROSS_DX = 0.3      # 1초 동안 x가 화면 폭의 30% 이상 움직이면 → 가로지름
APPROACH_DH = 0.08  # 1초 동안 박스 높이가 화면의 8% 이상 커지면 → 다가옴
NEAR_H = 0.6        # 박스 높이가 화면의 60% 이상일 때만 문을 엶

# 채점 기준
SHOULD_OPEN = {"entry", "diagonal", "hesitate_enter", "tailgate"}  # 열려야 정답
SHOULD_NOT_OPEN = {"lateral"}                                      # 안 열려야 정답
# approach_leave, hesitate_back 은 다가오는 구간이 있어서 열리는 게 정상
# → 나중에 "얼마나 빨리 닫나"로 따로 평가


def decide(window):
    dx = abs(window["x"].iloc[-1] - window["x"].iloc[0])
    dh = window["h"].iloc[-1] - window["h"].iloc[0]
    h_now = window["h"].iloc[-1]

    if dx >= CROSS_DX:
        return "CROSS"   # 가로지르는 중 → 안 엶
    if dh >= APPROACH_DH and h_now >= NEAR_H:
        return "OPEN"    # 다가오는 중 → 엶
    return "WAIT"


def run_trajectory(traj):
    # 한 사람의 궤적을 처음부터 한 줄씩 흘려보내면서, 처음 OPEN이 나온 시각을 찾음
    traj = traj.sort_values("t")
    for now in traj["t"]:
        window = traj[(traj["t"] > now - WINDOW_SEC) & (traj["t"] <= now)]
        if window["t"].iloc[-1] - window["t"].iloc[0] < WINDOW_SEC * 0.8:
            continue  # 아직 1초치 기록이 안 쌓였으면 판단 보류
        if decide(window) == "OPEN":
            return now
    return None


def evaluate(data):
    rows = []
    for traj_id, traj in data.groupby("traj_id"):
        open_time = run_trajectory(traj)

        # 바운딩박스 높이가 화면의 95%에 도달한 시점 (임시 도착 기준, 카메라 위치에 따라 달라질 수 있음)
        full_times = traj.loc[traj["h"] >= FULL_HEIGHT, "t"]
        arrive_time = full_times.min() if len(full_times) else None

        margin = None
        if open_time is not None and arrive_time is not None:
            margin = round(arrive_time - open_time, 2)

        scenario = traj["scenario"].iloc[0]
        opened = open_time is not None
        if scenario in SHOULD_OPEN:
            correct = opened
        elif scenario in SHOULD_NOT_OPEN:
            correct = not opened
        else:
            correct = None  # 채점 안 함

        rows.append({
            "traj_id": traj_id,
            "session": traj["session"].iloc[0],
            "scenario": scenario,
            "test_only": traj["test_only"].iloc[0],
            "opened": opened,
            "correct": correct,
            "open_time": open_time,
            "margin_sec": margin,  # 도착 몇 초 전에 열렸는지 (+면 미리 열림)
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 특정 촬영날만 보고 싶으면: python intent/rule_baseline.py 20261114_세미나실
    sessions = sys.argv[1:] or None
    result = evaluate(load_all(sessions))

    print("=== 촬영날 · 시나리오별 문 열림 ===")
    summary = result.groupby(["session", "scenario"]).agg(
        열린_수=("opened", "sum"), 전체=("opened", "count"))
    summary["열림_비율"] = (summary["열린_수"] / summary["전체"]).round(2)
    print(summary.to_string())

    graded = result[result["correct"].notna()]
    print("\n=== 정확도 (열려야 할 때 열림 + 안 열려야 할 때 안 열림) ===")
    acc = graded.groupby("session")["correct"].agg(맞음="sum", 전체="count")
    acc["정확도"] = (acc["맞음"] / acc["전체"]).round(3)
    print(acc.to_string())

    print("\n=== 도착 몇 초 전에 열렸나 (열려야 하는 시나리오, 평균) ===")
    opened_ok = result[result["scenario"].isin(SHOULD_OPEN) & result["margin_sec"].notna()]
    print(opened_ok.groupby(["session", "scenario"])["margin_sec"].agg(["mean", "min", "count"]).round(2).to_string())

    print("\n=== 틀린 궤적 ===")
    wrong = graded[graded["correct"] == False]  # noqa: E712
    print(wrong[["traj_id", "opened", "open_time"]].to_string(index=False) if len(wrong) else "없음")