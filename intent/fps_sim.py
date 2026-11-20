# 파이가 느릴 때(FPS가 낮을 때)도 판단이 제대로 되는지 미리 확인하는 코드

import pandas as pd
from load_tracks import load_all
from rule_baseline import evaluate, SHOULD_OPEN
from door_state import simulate

FPS_LIST = [20, 10, 5, 3, 2, 1.5, 1]  # 1.5 = 지금 파이 속도
SESSION = "20261114_세미나실"


def thin_out(data, fps):
    # 궤적마다 1/fps초 간격으로 한 줄씩만 남김 (파이가 그 속도로만 처리한다고 가정)
    gap = 1.0 / fps
    keep = []
    for _, traj in data.groupby("traj_id"):
        last = -999.0
        for idx, t in traj.sort_values("t")["t"].items():
            if t - last >= gap - 1e-6:
                keep.append(idx)
                last = t
    return data.loc[keep]


def door_check(data):
    # door_state.py와 같은 채점: 도중에 닫힘 / 안 닫힘 / 다시 열림이 있으면 실패
    bad = []
    for traj_id, traj in data.groupby("traj_id"):
        scenario = traj["scenario"].iloc[0]
        events, _ = simulate(traj)
        opens = [t for t, cmd, _ in events if cmd == "OPEN"]
        leaving = [t for t, _, why in events if why == "LEAVING"]
        if not opens:
            continue
        if len(opens) > 1:
            bad.append((traj_id, "다시 열림"))
        elif scenario in ("hesitate_back", "approach_leave") and not leaving:
            bad.append((traj_id, "안 닫힘"))
        elif scenario not in ("hesitate_back", "approach_leave") and leaving:
            bad.append((traj_id, "도중에 닫힘"))
    return bad


if __name__ == "__main__":
    full = load_all([SESSION])
    rows = []
    problems = {}
    for fps in FPS_LIST:
        data = thin_out(full, fps)
        result = evaluate(data)
        graded = result[result["correct"].notna()]
        margins = result[result["scenario"].isin(SHOULD_OPEN)]["margin_sec"].dropna()
        bad = door_check(data)
        problems[fps] = bad
        rows.append({
            "FPS": fps,
            "문 열기 판단": f"{int(graded['correct'].sum())}/{len(graded)}",
            "도착 몇 초 전 열림(평균)": round(margins.mean(), 2) if len(margins) else None,
            "가장 늦게 열린 것(초 전)": round(margins.min(), 2) if len(margins) else None,
            # 파이는 사진 한 장 처리에 1/fps초가 걸려서 판단이 그만큼 늦게 나옴
            "처리 지연 빼면(초 전)": round(margins.min() - 1.0 / fps, 2) if len(margins) else None,
            "열린 뒤 판단 실패": len(bad),
        })

    print(f"=== {SESSION}: FPS별 결과 ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n※ 처리 지연: 파이가 사진 한 장을 처리하는 동안(1/fps초) 판단이 늦어지는 것까지 뺀 값")
    print("※ 이 값이 작을수록 문이 다 열리기 전에 사람이 도착할 수 있음 (가짜 문 기준 다 열리는 데 약 1초)")

    for fps, bad in problems.items():
        if bad:
            print(f"\n[{fps}fps] 열린 뒤 판단 실패")
            for traj_id, why in bad:
                print(f"  {why}: {traj_id.split('/', 1)[1]}")