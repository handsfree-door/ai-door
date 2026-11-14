import cv2
import csv
import time
import argparse
import platform
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# 실행 옵션
# 예) python vision/webcam_tracking.py --camera 1 --delay 5 --duration 10
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0, help="카메라 번호 (0: 노트북 내장, 1: 외장 웹캠)")
parser.add_argument("--delay", type=float, default=5, help="녹화 시작 전 대기 시간(초)")
parser.add_argument("--duration", type=float, default=10, help="녹화 시간(초), 지나면 자동 종료")
args = parser.parse_args()

# YOLO11 모델 가중치 파일은 저장소 용량 문제로 업로드하지 않았습니다.
# 최초 실행 시 자동 다운
model = YOLO("yolo11n.pt")

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CSV 저장 경로: ai-door/data/vision/
DATA_DIR = PROJECT_ROOT / "data" / "vision"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 파일명 영어 및 CSV파일 이름 자동 생성 (번거롭게 지울 필요 없이 실행하면 자동 생성됨)
CSV_PATH = DATA_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_tracking.csv"

# 운영체제에 따른 다른 설정 Windows: DirectShow / Linux 및 Raspberry Pi: 기본 백엔드
if platform.system() == "Windows":
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(args.camera)

if not cap.isOpened():
    cap.release()
    raise RuntimeError(f"웹캠 연결 X (카메라 번호 {args.camera})")

# 첫 실행이 3~4초 느려서 녹화 앞부분이 빠지는 문제 방지 → 미리 한 번 돌려둠
import numpy as np
print("모델 준비 중...")
model.track(np.zeros((480, 640, 3), dtype=np.uint8),
            persist=False, tracker="bytetrack.yaml", classes=[0], verbose=False)
start_time = time.perf_counter()
frame_id = 0
saved_count = 0

try:
    # 만약 동일한 파일명이 존재하면 덮어쓰지 않고 오류 발생
    with open(CSV_PATH, "x", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)
        # frame_w,frame_h 컬럼 추가
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
            "confidence",
            "frame_w",
            "frame_h"
        ])

        while True:
            ret, frame = cap.read()

            if not ret:
                print("웹캠 프레임을 읽을 수 없습니다.")
                break

            now = time.perf_counter() - start_time

            # 1) 녹화 전 대기: 화면에 카운트다운만 보여주고 기록은 안 함
            if now < args.delay:
                cv2.putText(frame, f"START IN {args.delay - now:.0f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow("YOLO11 + ByteTrack", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # 2) 녹화 시간이 지나면 자동 종료 (카메라로 다가가서 끌 필요 없음)
            if now > args.delay + args.duration:
                print("녹화 시간 종료 → 자동 종료")
                break

            frame_id += 1

            # 현재 프레임 해상도 기록 추가한 부분
            frame_h, frame_w = frame.shape[:2]

            # YOLO11 + ByteTrack
            # COCO class 0 = person (사람만 탐지 및 추적으로 수정된 코드)
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.25,
                classes=[0],
                verbose=False
            )

            result = results[0]
            annotated = result.plot()

            # 시간은 프레임당 한 번만 기록하도록 수정
            timestamp = datetime.now().isoformat(
                timespec="milliseconds"
            )
            elapsed = round(now, 3)

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes

                track_ids = boxes.id.int().cpu().tolist()
                xywh_list = boxes.xywh.cpu().tolist()
                cls_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for track_id, (x, y, w, h), cls_id, confidence in zip(
                    track_ids,
                    xywh_list,
                    cls_ids,
                    confidences
                ):
                    writer.writerow([
                        timestamp,
                        elapsed,
                        frame_id,
                        track_id,
                        model.names[cls_id],
                        round(x, 2),
                        round(y, 2),
                        round(w, 2),
                        round(h, 2),
                        round(confidence, 3),
                        frame_w,
                        frame_h
                    ])

                    saved_count += 1

            # 녹화 중 남은 시간 표시
            remain = args.delay + args.duration - now
            cv2.putText(annotated, f"REC {remain:.0f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("YOLO11 + ByteTrack", annotated)

            # q 입력 시 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("\nCtrl+C 입력으로 추적 종료")

finally:
    # Ctrl+C 입력시 정상 정리할 수 있는 코드
    cap.release()
    cv2.destroyAllWindows()

    print("처리한 프레임 수:", frame_id)
    print("CSV에 기록한 객체 수:", saved_count)
    print("CSV 저장 위치:", CSV_PATH.resolve())
    print("파일 존재 여부:", CSV_PATH.exists())