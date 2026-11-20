# 웹캠 실시간: 카메라 → YOLO + ByteTrack → 판단(controller) → 문(가짜 문 또는 진짜 보드)

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from controller import DoorController
from fake_door import FakeDoor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "data" / "vision" / "raw" / "_live"  # --save 하면 여기에 CSV 저장 (git에 안 올라감)
CSV_COLUMNS = ["timestamp", "elapsed_sec", "frame_id", "track_id", "class",
               "center_x", "center_y", "width", "height", "confidence", "frame_w", "frame_h"]


def open_camera(source):
    # 숫자면 카메라 번호, 아니면 동영상 파일 경로
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {source}를 열 수 없어요. 번호를 바꿔보거나 다른 프로그램이 쓰고 있는지 확인하세요.")
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0", help="카메라 번호 (노트북 내장 0, USB 웹캠 1) 또는 동영상 파일")
    parser.add_argument("--model", default="yolo11n.pt", help="모델 (파이에서는 yolo11n_320_ncnn_model 추천)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 입력 크기 (NCNN 모델은 변환할 때 크기와 같게)")
    parser.add_argument("--port", help="진짜 보드 포트 (예: COM3, /dev/ttyACM0). 없으면 가짜 문")
    parser.add_argument("--show", action="store_true", help="카메라 화면 띄우기 (SSH로 파이 쓸 때는 빼기)")
    parser.add_argument("--save", action="store_true", help="추적 결과를 CSV로 저장 (나중에 분석용)")
    parser.add_argument("--duration", type=float, help="몇 초 뒤 자동 종료 (없으면 Ctrl+C로 종료)")
    args = parser.parse_args()

    cap = open_camera(args.camera)
    model = YOLO(args.model, task="detect") if args.model.endswith(".onnx") else YOLO(args.model)

    if args.port:
        from serial_door import SerialDoor  # 진짜 보드가 있을 때만 불러옴
        door = SerialDoor(args.port)
    else:
        door = FakeDoor()
    ctrl = DoorController(door)

    writer = None
    if args.save:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = SAVE_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_live.csv"
        save_file = open(save_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(save_file)
        writer.writerow(CSV_COLUMNS)
        print("저장:", save_path)

    def track(frame):
        return model.track(frame, persist=True, tracker="bytetrack.yaml",
                           classes=[0], imgsz=args.imgsz, verbose=False)[0]

    print("모델 준비 중...")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("카메라에서 화면을 못 읽었어요.")
    track(frame)  # 첫 실행은 느려서 미리 한 번 돌려둠

    print("시작! (끝내려면 Ctrl + C)")
    start = time.time()
    frame_id = 0
    fps_count, fps_start = 0, start
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 읽기 끝")
                break
            now = time.time() - start
            frame_id += 1
            result = track(frame)
            frame_h, frame_w = frame.shape[:2]

            # 이번 프레임에 보인 사람들 → (ID, 시각, x, h)
            people = []
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for (cx, cy, bw, bh), tid, conf in zip(boxes.xywh.tolist(), boxes.id.int().tolist(),
                                                       boxes.conf.tolist()):
                    people.append((tid, now, cx / frame_w, bh / frame_h))
                    if writer:
                        writer.writerow([datetime.now().isoformat(timespec="milliseconds"), round(now, 3),
                                         frame_id, tid, "person", round(cx, 2), round(cy, 2),
                                         round(bw, 2), round(bh, 2), round(conf, 3), frame_w, frame_h])

            ctrl.step(now, people)

            # 5초마다 속도 출력
            fps_count += 1
            if time.time() - fps_start >= 5:
                print(f"[{now:6.2f}s] 속도 {fps_count / (time.time() - fps_start):.1f} fps, 보이는 사람 {len(people)}명")
                fps_count, fps_start = 0, time.time()

            if args.show:
                view = result.plot()
                state = "OPEN" if door.target > 0 else "CLOSED"
                cv2.putText(view, f"DOOR: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if state == "OPEN" else (0, 0, 255), 2)
                cv2.imshow("live", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.duration and now >= args.duration:
                break
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        if door.target > 0:
            ctrl.send(time.time() - start, "CLOSE")  # 끝낼 때 문이 열려 있으면 닫기
        cap.release()
        if writer:
            save_file.close()
        if args.show:
            cv2.destroyAllWindows()
        ctrl.summary()


if __name__ == "__main__":
    main()