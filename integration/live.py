# 웹캠 실시간: 카메라 → YOLO + ByteTrack → (물품 탐지) → 판단(controller) → 문(가짜 문 또는 진짜 보드)
#
# 실행 예시 (ai-door 폴더에서)
#   노트북, 가짜 문:     python integration/live.py --show
#   물품 탐지까지:       python integration/live.py --show --items
#   노트북 + 보드:       python integration/live.py --show --items --port COM3
#   파이 + 보드:         python integration/live.py --model yolo11n_320_ncnn_model --imgsz 320 --port /dev/ttyACM0
# 끝낼 때: Ctrl + C (화면을 띄웠으면 화면에서 q)

import argparse
import csv
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from controller import DoorController, ITEM_LABELS
from fake_door import FakeDoor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "vision"))  # 채원 님 object_association.py 불러오기
SAVE_DIR = PROJECT_ROOT / "data" / "vision" / "raw" / "_live"  # --save 하면 여기에 CSV 저장
CSV_COLUMNS = ["timestamp", "elapsed_sec", "frame_id", "track_id", "class",
               "center_x", "center_y", "width", "height", "confidence", "frame_w", "frame_h", "items"]


class ItemDetector:
    # 물품 모델(유모차·휠체어·캐리어·자전거 / 택배상자)을 돌려서 사람마다 확인된 물품을 붙여줌
    # 연결 방식은 채원 님 vision/object_association.py 그대로 사용

    def __init__(self, model_paths, imgsz, conf, every, window, hits):
        from object_association import Detection, match_objects
        self.Detection, self.match_objects = Detection, match_objects
        self.models = []
        for path in model_paths:
            model = YOLO(path, task="detect")
            labels = {}
            for idx, name in model.names.items():
                label = str(name).split("-", 1)[-1].lower()  # "3-stroller" → "stroller"
                if label in ITEM_LABELS:
                    labels[idx] = label
            print(f"물품 모델: {path} → {sorted(labels.values())}")
            self.models.append((model, labels, Path(path).stem))
        self.imgsz, self.conf, self.every, self.hits = imgsz, conf, every, hits
        self.history = defaultdict(lambda: deque(maxlen=window))  # (사람 ID, 물품) → 최근 연결 여부
        self.count = 0

    def run(self, frame, people_boxes):
        # people_boxes: [{"track_id": 1, "bbox_xyxy": [x1, y1, x2, y2]}, ...]
        # 반환: {사람 ID: {확인된 물품 이름들}}  (이번 프레임에 물품 모델을 안 돌렸으면 빈 dict)
        self.count += 1
        if (self.count - 1) % self.every != 0:
            return {}
        found = []
        for model, labels, name in self.models:
            result = model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
            if result.boxes is None:
                continue
            for box in result.boxes:
                idx = int(box.cls.item())
                if idx in labels:
                    found.append(self.Detection(labels[idx], float(box.conf.item()),
                                                tuple(map(float, box.xyxy[0].tolist())), name))
        matched, _ = self.match_objects(people_boxes, found)
        confirmed = {}
        for person in people_boxes:
            tid = person["track_id"]
            seen = {o["label"] for o in matched.get(tid, [])}
            for label in ITEM_LABELS:
                self.history[(tid, label)].append(label in seen)
            items = {label for label in seen if sum(self.history[(tid, label)]) >= self.hits}
            if items:
                confirmed[tid] = items
        return confirmed


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
    parser.add_argument("--items", action="store_true", help="물품 탐지 켜기 (유모차·휠체어·캐리어·자전거·택배상자)")
    parser.add_argument("--item-models", nargs="+",
                        default=["models/stroller_best.pt", "models/cardboard_best.pt"],
                        help="물품 모델 파일들 (파이에서는 NCNN 변환한 폴더)")
    parser.add_argument("--item-imgsz", type=int, default=480, help="물품 모델 입력 크기")
    parser.add_argument("--item-conf", type=float, default=0.5, help="물품 탐지 최소 신뢰도")
    parser.add_argument("--object-every", type=int, default=3, help="물품 모델을 N프레임마다 한 번 돌림 (속도 절약)")
    parser.add_argument("--confirm-hits", type=int, default=2, help="최근 --window번 중 몇 번 연결돼야 확인으로 볼지")
    parser.add_argument("--window", type=int, default=4)
    args = parser.parse_args()

    cap = open_camera(args.camera)
    model = YOLO(args.model, task="detect") if args.model.endswith(".onnx") else YOLO(args.model)
    items = None
    if args.items:
        paths = [str(p if Path(p).is_absolute() else PROJECT_ROOT / p) for p in args.item_models]
        items = ItemDetector(paths, args.item_imgsz, args.item_conf, args.object_every,
                             args.window, args.confirm_hits)

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

            # 이번 프레임에 보인 사람들
            found = []  # (ID, cx, cy, bw, bh, conf)
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for (cx, cy, bw, bh), tid, conf in zip(boxes.xywh.tolist(), boxes.id.int().tolist(),
                                                       boxes.conf.tolist()):
                    found.append((tid, cx, cy, bw, bh, conf))

            # 물품 탐지: 사람마다 확인된 물품 이름들
            person_items = {}
            if items and found:
                boxes_xyxy = [{"track_id": tid, "bbox_xyxy": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]}
                              for tid, cx, cy, bw, bh, _ in found]
                person_items = items.run(frame, boxes_xyxy)

            # 판단으로 넘기기 → (ID, 시각, x, h, 물품)
            people = []
            for tid, cx, cy, bw, bh, conf in found:
                carried = person_items.get(tid, set())
                people.append((tid, now, cx / frame_w, bh / frame_h, carried))
                if writer:
                    writer.writerow([datetime.now().isoformat(timespec="milliseconds"), round(now, 3),
                                     frame_id, tid, "person", round(cx, 2), round(cy, 2),
                                     round(bw, 2), round(bh, 2), round(conf, 3), frame_w, frame_h,
                                     "|".join(sorted(carried))])

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
                # 물품 동반으로 기억 중인 사람은 박스 아래에 물품 이름 표시
                for tid, cx, cy, bw, bh, _ in found:
                    carried = ctrl.items_of(tid, now)
                    if carried:
                        cv2.putText(view, "+" + ",".join(sorted(carried)), (int(cx - bw / 2), int(cy + bh / 2) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
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
        if hasattr(door, "close"):
            door.close()
        cap.release()
        if writer:
            save_file.close()
        if args.show:
            cv2.destroyAllWindows()
        ctrl.summary()


if __name__ == "__main__":
    main()