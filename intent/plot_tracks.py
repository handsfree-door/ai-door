import matplotlib.pyplot as plt
from load_tracks import load_all, PROJECT_ROOT, FULL_HEIGHT

plt.rcParams["font.family"] = "Malgun Gothic"  # 윈도우 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

SCENARIOS = ["entry", "approach_leave", "lateral"]
TITLES = {
    "entry": "1. 정면 진입",
    "approach_leave": "2. 접근 후 이탈",
    "lateral": "3. 좌우 이동",
}

data = load_all()
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")

for col, scenario in enumerate(SCENARIOS):
    part = data[data["scenario"] == scenario]
    for _, traj in part.groupby("traj_id"):
        axes[0, col].plot(traj["t"], traj["x"], alpha=0.6)
        axes[1, col].plot(traj["t"], traj["h"], alpha=0.6)
    axes[0, col].set_title(f"{TITLES[scenario]} (궤적 {part['traj_id'].nunique()}개)")
    axes[1, col].axhline(FULL_HEIGHT, color="gray", linestyle="--", linewidth=1)
    axes[1, col].set_xlabel("시간 (초)")

axes[0, 0].set_ylabel("x 위치 (0=왼쪽, 1=오른쪽)")
axes[1, 0].set_ylabel("박스 높이 (클수록 가까움)")

out_dir = PROJECT_ROOT / "intent" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out_dir / "tracks_by_class.png", dpi=120)
plt.show()