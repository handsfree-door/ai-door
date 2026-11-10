from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "vision" / "raw"

# 폴더 이름 → 정답 라벨
LABELS = {
    "entry": "통과",
    "approach_leave": "지나감",
    "lateral": "지나감",
}

SKIP_START_SEC = 0.0  # 앞부분 버리기: 시작 위치로 걸어가는 구간
SKIP_END_SEC = 0.0    # 뒷부분 버리기: 프로그램 끄러 카메라로 다가오는 구간
MIN_ROWS = 10         # 이보다 짧은 궤적 조각은 버림
FULL_HEIGHT = 0.95    # 박스 높이가 화면의 95% 이상이면 '꽉 참'으로 표시


def load_file(path, scenario):
    df = pd.read_csv(path)
    df = df[df["class"] == "person"].copy()

    # 화면 크기로 나눠서 0~1 값으로 맞춤
    df["x"] = df["center_x"] / df["frame_w"]
    df["h"] = df["height"] / df["frame_h"]
    df["foot_y"] = (df["center_y"] + df["height"] / 2) / df["frame_h"]
    df["full"] = df["h"] >= FULL_HEIGHT

    df["file"] = path.name
    df["scenario"] = scenario
    df["label"] = LABELS[scenario]
    return df


def trim_file(df):
    # 파일 앞뒤의 의도하지 않은 움직임 잘라내기
    start = df["elapsed_sec"].min() + SKIP_START_SEC
    end = df["elapsed_sec"].max() - SKIP_END_SEC
    return df[(df["elapsed_sec"] >= start) & (df["elapsed_sec"] <= end)]


def load_all():
    frames = []
    for scenario in LABELS:
        for path in sorted((RAW_DIR / scenario).glob("*.csv")):
            frames.append(trim_file(load_file(path, scenario)))

    data = pd.concat(frames, ignore_index=True)

    # 파일명 + track_id = 궤적 하나
    data["traj_id"] = data["file"] + "#" + data["track_id"].astype(str)

    # 너무 짧은 조각 버리기
    sizes = data.groupby("traj_id")["frame_id"].transform("size")
    data = data[sizes >= MIN_ROWS].copy()

    # 궤적마다 시작 시각을 0초로 맞춤
    data["t"] = data["elapsed_sec"] - data.groupby("traj_id")["elapsed_sec"].transform("min")
    return data


if __name__ == "__main__":
    data = load_all()
    counts = data.groupby("scenario")["traj_id"].nunique()
    print("클래스별 궤적 수")
    print(counts)
    print("전체 궤적 수:", data["traj_id"].nunique())