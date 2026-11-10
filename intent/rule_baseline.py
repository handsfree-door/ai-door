import pandas as pd
from load_tracks import load_all, FULL_HEIGHT

# 그래프 보고 정한 첫 기준값 (결과 보고 조정)
WINDOW_SEC = 1.0    # 최근 몇 초를 보고 판단할지
CROSS_DX = 0.3      # 1초 동안 x가 화면 폭의 30% 이상 움직이면 → 가로지름
APPROACH_DH = 0.08  # 1초 동안 박스 높이가 화면의 8% 이상 커지면 → 다가옴
NEAR_H = 0.6        # 박스 높이가 화면의 60% 이상일 때만 문을 엶


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


if __name__ == "__main__":
    data = load_all()
    rows = []
    for traj_id, traj in data.groupby("traj_id"):
        open_time = run_trajectory(traj)

        # 바운딩박스 높이가 화면의 95%에 도달한 시점 (임시 도착 기준, 카메라 위치에 따라 달라질 수 있음)
        full_times = traj.loc[traj["h"] >= FULL_HEIGHT, "t"]
        arrive_time = full_times.min() if len(full_times) else None

        margin = None
        if open_time is not None and arrive_time is not None:
            margin = round(arrive_time - open_time, 2)

        rows.append({
            "traj_id": traj_id,
            "scenario": traj["scenario"].iloc[0],
            "opened": open_time is not None,
            "open_time": open_time,
            "margin_sec": margin,  # 도착 몇 초 전에 열렸는지 (+면 미리 열림)
        })

    result = pd.DataFrame(rows)

    print("=== 클래스별 문 열림 비율 ===")
    summary = result.groupby("scenario")["opened"].agg(열린_수="sum", 전체="count")
    summary["비율"] = (summary["열린_수"] / summary["전체"]).round(2)
    print(summary)

    print("\n=== 정면 진입: 도착 몇 초 전에 열렸나 ===")
    entry = result[result["scenario"] == "entry"]
    print(entry[["traj_id", "open_time", "margin_sec"]].to_string(index=False))

    print("\n=== 좌우 이동인데 문이 열린 궤적 (잘못 열림) ===")
    wrong = result[(result["scenario"] == "lateral") & result["opened"]]
    print(wrong[["traj_id", "open_time"]].to_string(index=False) if len(wrong) else "없음")