from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "vision" / "raw"
# 폴더 구조: data/vision/raw/<촬영 날짜_장소>/<시나리오>/*.csv
#   예) data/vision/raw/20261114_세미나실/entry/20261114_150221_tracking.csv

# 시나리오 폴더 이름 → 정답 라벨
LABELS = {
    "entry": "통과",            # 정면 진입
    "diagonal": "통과",         # 비스듬히 진입
    "hesitate_enter": "통과",   # 문 앞에서 멈칫 후 들어감
    "approach_leave": "지나감",  # 다가오다 빠짐
    "hesitate_back": "지나감",   # 문 앞에서 멈칫 후 돌아감
    "lateral": "지나감",         # 좌우 이동
    "tailgate": "따라옴",        # 2명, 뒤에서 따라 들어옴
}

# 분석에서 아예 빼는 파일 (데이터 목록 시트에 "분석 제외"로 적은 것)
EXCLUDE_FILES = {
    "20261114_151049_tracking.csv",
    "20261114_151121_tracking.csv",
}

# 채점에만 쓰고 기준값 정할 때는 안 쓰는 것 (촬영날, 시나리오)
TEST_ONLY = {
    ("20261114_세미나실", "diagonal"),
}

SKIP_START_SEC = 0.0  # 앞부분 버리기 (자동 녹화로 바뀌어서 지금은 0)
SKIP_END_SEC = 0.0    # 뒷부분 버리기
MIN_ROWS = 10         # 이보다 짧은 궤적 조각은 버림
FULL_HEIGHT = 0.95    # 박스 높이가 화면의 95% 이상이면 '꽉 참'으로 표시


def load_file(path, session, scenario):
    df = pd.read_csv(path)
    df = df[df["class"] == "person"].copy()

    # 화면 크기로 나눠서 0~1 값으로 맞춤
    df["x"] = df["center_x"] / df["frame_w"]
    df["h"] = df["height"] / df["frame_h"]
    df["foot_y"] = (df["center_y"] + df["height"] / 2) / df["frame_h"]
    df["full"] = df["h"] >= FULL_HEIGHT

    df["file"] = path.name
    df["session"] = session
    df["scenario"] = scenario
    df["label"] = LABELS[scenario]
    df["test_only"] = (session, scenario) in TEST_ONLY
    return df


def trim_file(df):
    # 파일 앞뒤의 의도하지 않은 움직임 잘라내기
    start = df["elapsed_sec"].min() + SKIP_START_SEC
    end = df["elapsed_sec"].max() - SKIP_END_SEC
    return df[(df["elapsed_sec"] >= start) & (df["elapsed_sec"] <= end)]


def load_all(sessions=None):
    # sessions: 읽을 촬영 폴더 이름 목록. None이면 전부 읽음
    frames = []
    for session_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        if sessions and session_dir.name not in sessions:
            continue
        for scenario in LABELS:
            for path in sorted((session_dir / scenario).glob("*.csv")):
                if path.name in EXCLUDE_FILES:
                    continue
                frames.append(trim_file(load_file(path, session_dir.name, scenario)))

    data = pd.concat(frames, ignore_index=True)

    # 촬영날/시나리오/파일명#track_id = 궤적 하나
    data["traj_id"] = (
        data["session"] + "/" +
        data["scenario"] + "/" +
        data["file"] + "#" +
        data["track_id"].astype(str)
    )

    # 너무 짧은 조각 버리기
    sizes = data.groupby("traj_id")["frame_id"].transform("size")
    data = data[sizes >= MIN_ROWS].copy()

    # 궤적마다 시작 시각을 0초로 맞춤
    data["t"] = data["elapsed_sec"] - data.groupby("traj_id")["elapsed_sec"].transform("min")
    return data


if __name__ == "__main__":
    data = load_all()
    counts = data.groupby(["session", "scenario"])["traj_id"].nunique()
    print("촬영날 · 시나리오별 궤적 수")
    print(counts.to_string())
    print("\n전체 궤적 수:", data["traj_id"].nunique())