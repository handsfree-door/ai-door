import cv2
import csv
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# yolo11n.pt 모델 가중치 파일은 저장소 용량 문제로 업로드하지 않았습니다.
# 최초 실행 시 자동 다운됩니다. (인터넷 연결 필요)
model = YOLO("yolo11n.pt")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CSV 저장 경로 git폴더에 맞춰서 수정했습니다. : ai-door/data/vision/
#실행 시 기존 data가 지워질 수도 있으므로 생성 후 파일 변경 추천드립니다.
DATA_DIR = PROJECT_ROOT / "data" / "vision"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "tracking_results.csv"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("웹캠 연결 X")

start_time = time.perf_counter()
frame_id = 0

with open(CSV_PATH, "w",
          newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "timestamp",
        "elapsed_sec",
        "frame_id",
        "track_id",
        "class",
        "center_x",
        "center_y",
        "width",
        "height",
        "confidence"
    ])

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False
        )

        result = results[0]
        annotated = result.plot()

        if (result.boxes is not None
                and result.boxes.id is not None):

            boxes = result.boxes

            for i, track_id in enumerate(
                    boxes.id.int().cpu().tolist()):

                x, y, w, h = boxes.xywh[i].tolist()
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                timestamp = datetime.now().isoformat(
                    timespec="milliseconds"
                )

                elapsed = time.perf_counter() - start_time

                writer.writerow([
                    timestamp,
                    round(elapsed, 3),
                    frame_id,
                    track_id,
                    model.names[cls_id],
                    round(x, 2),
                    round(y, 2),
                    round(w, 2),
                    round(h, 2),
                    round(confidence, 3)
                ])

        cv2.imshow("YOLO11 + ByteTrack", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

# CSV 저장 위치 확인 및 파일 미생성 이슈 점검용
print("저장 완료!!!")
print("CSV 저장 위치:", CSV_PATH.resolve())
print("파일 존재 여부:", CSV_PATH.exists())

#나중에 추가할 코드 : 데이터 학습용으로 사용할 때를 위해 사람 외 객체 인식 제거 
#라즈베리파이용 코드는 따로 파일 만들 예정