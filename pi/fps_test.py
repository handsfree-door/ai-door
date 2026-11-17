# 라즈베리파이에서 YOLO + ByteTrack이 초당 몇 프레임(FPS) 측정 코드
# 화면 없이 SSH로 실행해도 됨

import argparse
import time

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0, help="카메라 번호 (파이에 웹캠 하나면 0)")
parser.add_argument("--model", default="yolo11n.pt", help="모델 파일 (.pt 또는 .onnx)")
parser.add_argument("--imgsz", type=int, default=640, help="YOLO 입력 크기 (작을수록 빠름)")
parser.add_argument("--frames", type=int, default=100, help="몇 프레임 동안 잴지")
args = parser.parse_args()

cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit(f"카메라 {args.camera}번을 열 수 없어요. 웹캠 연결을 확인하세요.")

# ONNX 모델은 무슨 작업인지 알려줘야 함
model = YOLO(args.model, task="detect") if args.model.endswith(".onnx") else YOLO(args.model)

def track(frame):
    return model.track(frame, persist=True, tracker="bytetrack.yaml",
                       classes=[0], imgsz=args.imgsz, verbose=False)[0]

print("모델 준비 중...")
ok, frame = cap.read()
if not ok:
    raise SystemExit("카메라에서 화면을 못 읽었어요.")
track(frame)  # 첫 실행은 느려서 측정에서 뺌

print(f"측정 시작: {args.model}, 입력 {args.imgsz}, {args.frames}프레임")
count = 0
with_person = 0
start = time.time()
while count < args.frames:
    ok, frame = cap.read()
    if not ok:
        print("카메라 읽기 실패, 중단")
        break
    result = track(frame)
    if result.boxes is not None and len(result.boxes) > 0:
        with_person += 1
    count += 1
    if count % 20 == 0:
        print(f"  {count}/{args.frames}  지금까지 {count / (time.time() - start):.1f} fps")

elapsed = time.time() - start
cap.release()
print("\n=== 결과 ===")
print(f"모델: {args.model} / 입력 크기: {args.imgsz}")
print(f"평균 FPS: {count / elapsed:.1f}  ({count}프레임, {elapsed:.1f}초)")
print(f"사람이 잡힌 프레임: {with_person}/{count}")