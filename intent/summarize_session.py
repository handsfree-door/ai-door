# 촬영한 CSV를 폴더째 읽어서, 파일마다 "어떻게 움직였는지" 요약해 주는 코드
# 데이터 목록 시트에 메모(속도, 멈춘 시간, 따라온 간격, 방향)를 채울 때 참고용

import sys
from pathlib import Path
import pandas as pd

MIN_ROWS = 10      # 이보다 짧은 ID 조각은 무시
ARRIVE_H = 0.95    # 박스 높이가 화면의 95% 이상 = 문 앞 도착
STILL_SEC = 0.5    # 이 시간 동안
STILL_MOVE = 0.03  # x, 박스 높이 변화가 이보다 작으면 '멈춤'


def load(path):
    df = pd.read_csv(path)
    df = df[df["class"] == "person"].copy()
    df["x"] = df["center_x"] / df["frame_w"]
    df["h"] = df["height"] / df["frame_h"]
    df["t"] = df["elapsed_sec"] - df["elapsed_sec"].min()
    sizes = df.groupby("track_id")["t"].transform("size")
    return df[sizes >= MIN_ROWS].sort_values("t")


def arrive_time(tr):
    hit = tr.loc[tr["h"] >= ARRIVE_H, "t"]
    return round(hit.min(), 1) if len(hit) else None


def stops(tr):
    # 0.5초 구간 안에서 x, h가 거의 안 변한 프레임을 '멈춤'으로 보고, 이어진 멈춤 구간을 찾음
    tr = tr.set_index("t")
    still = []
    for t in tr.index:
        w = tr.loc[t - STILL_SEC:t]
        ok = (w.index[-1] - w.index[0] >= STILL_SEC * 0.8
              and w["x"].max() - w["x"].min() < STILL_MOVE
              and w["h"].max() - w["h"].min() < STILL_MOVE)
        still.append(ok)
    tr["still"] = still

    runs, start = [], None
    times = list(tr.index)
    for i, t in enumerate(times):
        if tr["still"].iloc[i] and start is None:
            start = t
        last = i == len(times) - 1
        if start is not None and (not tr["still"].iloc[i] or last):
            end = t
            if end - start >= 0.5:
                runs.append((round(max(0.0, start - STILL_SEC), 1), round(end - start + STILL_SEC, 1),
                             round(tr.loc[start, "h"], 2)))
            start = None
    return runs


def last_move(tr):
    end = tr[tr["t"] >= tr["t"].max() - 1.0]
    dx = end["x"].iloc[-1] - end["x"].iloc[0]
    dh = end["h"].iloc[-1] - end["h"].iloc[0]
    if dh < -0.05:
        return "멀어짐(뒤로)"
    if dx > 0.05:
        return "오른쪽으로"
    if dx < -0.05:
        return "왼쪽으로"
    return "거의 그대로"


def summarize(path, scenario):
    df = load(path)
    if df.empty:
        return {"시나리오": scenario, "파일": path.name, "메모": "사람 기록 없음"}

    tracks = {tid: tr for tid, tr in df.groupby("track_id")}
    main_id = max(tracks, key=lambda k: len(tracks[k]))
    main = tracks[main_id]

    row = {
        "시나리오": scenario,
        "파일": path.name,
        "ID수": len(tracks),
        "시작높이": round(main["h"].iloc[0], 2),
        "도착(초)": arrive_time(main) if arrive_time(main) is not None else "-",
        "마지막1초": last_move(main),
        "x시작→끝": f"{main['x'].iloc[0]:.2f}→{main['x'].iloc[-1]:.2f}",
        "평균높이": round(main["h"].mean(), 2),
    }

    # 멈춘 구간: (시작 초, 멈춘 시간, 그때 박스 높이)
    runs = stops(main)
    row["멈춤(시작,길이,높이)"] = "; ".join(f"{s}s,{d}s,h{h}" for s, d, h in runs) or "-"

    # 2명 이상이면 도착 시간 차이 = 따라온 간격
    arrives = sorted(a for a in (arrive_time(tr) for tr in tracks.values()) if a is not None)
    row["도착간격(초)"] = round(arrives[1] - arrives[0], 1) if len(arrives) >= 2 else "-"
    return row


if __name__ == "__main__":
    session = Path(sys.argv[1])
    rows = []
    for folder in sorted(p for p in session.iterdir() if p.is_dir()):
        for path in sorted(folder.rglob("*.csv")):
            rows.append(summarize(path, folder.name))

    result = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 40)
    print(result.to_string(index=False))

    out = session / "summary.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n저장:", out)