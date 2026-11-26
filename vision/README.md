# AI-Door Vision

## 1. 프로젝트 개요

AI-Door의 Vision 모듈은 카메라 영상을 이용해 사람의 움직임을 추적하고, 주변 물품을 탐지하는 역할을 담당합니다.

기본적으로 YOLO11n과 ByteTrack을 활용해 사람에게 고유 ID를 부여하고 위치 정보를 수집합니다. 추가로 학습한 객체 탐지 모델을 이용해 유모차, 휠체어, 캐리어, 자전거, 택배 상자를 인식하고, 사람과 물품의 바운딩박스 위치 관계를 분석합니다.

현재는 Windows 노트북 웹캠에서 기능을 테스트했으며, 추후 Raspberry Pi에서 NCNN 기반으로 실행하고 Intent 및 Integration 모듈과 연동할 예정입니다.

## 2. 구현 기능

### 사람 탐지 및 추적
- YOLO11n을 이용한 사람 탐지
- ByteTrack을 이용한 객체별 `track_id` 부여 및 유지
- `classes=[0]` 설정으로 사람 클래스만 추적
- 사람의 중심 좌표, 바운딩박스 크기, 탐지 신뢰도 수집
- 추적 결과를 CSV 파일로 저장

### 물품 탐지
별도로 학습한 YOLO11 모델을 이용해 다음 물품을 탐지합니다.

| 모델 | 탐지 대상 |
|---|---|
| `stroller_best.pt` | 휠체어, 캐리어, 유모차, 자전거 |
| `cardboard_best.pt` | 택배 상자 |

Stroller 모델에는 사람 클래스도 포함되어 있지만, 사람 추적은 별도의 YOLO11n 모델에서 처리합니다.

### 사람과 물품 연결
`object_association.py`에서 사람과 물품의 바운딩박스를 비교해 연결 가능성을 계산합니다.

- 바운딩박스 겹침 비율 및 중심점 거리 비교
- 탐지된 물품을 위치상 가장 적합한 사람 ID에 연결
- 최근 프레임에서 같은 물품 종류가 반복 연결되면 `confirmed` 표시
- 연결되지 않은 물품은 `unmatched_objects`로 구분

현재 연결 방식은 위치 관계를 기반으로 하므로, 실제로 사람이 물품을 들거나 밀고 있는지까지 판단하지는 않습니다.

## 3. 파일 구성

```text
vision/
├── webcam_tracking.py
├── multi_object_test.py
├── object_association.py
└── README.md

models/
├── stroller_best.pt
└── cardboard_best.pt
```

| 파일 | 설명 |
|---|---|
| `webcam_tracking.py` | 사람 탐지 및 ByteTrack 추적, CSV 저장 |
| `multi_object_test.py` | 사람 및 물품 동시 탐지, 연결 결과 테스트 |
| `object_association.py` | 사람과 물품의 위치 관계 계산 |
| `README.md` | Vision 모듈 설명 및 실행 방법 |

## 4. 실행 환경

현재 테스트 환경:

- Windows / Python 3.11
- Ultralytics YOLO11
- OpenCV
- 노트북 내장 웹캠

프로젝트 루트에서 패키지를 설치합니다.

```bash
python -m pip install -r requirements.txt
```

기본 사람 탐지 모델인 `yolo11n.pt`는 최초 실행 시 다운로드할 수 있습니다. 추가 학습한 물품 모델은 `models/` 폴더에 별도로 준비해야 합니다.

## 5. 실행 방법

### 5.1 사람 추적 테스트

```bash
python vision/webcam_tracking.py --camera 0 --delay 5 --duration 20
```

웹캠에서 사람을 탐지하고 ID를 추적하며, 결과를 CSV로 저장합니다.

### 5.2 사람 및 물품 탐지 테스트

```bash
python vision/multi_object_test.py \
  --camera 0 \
  --stroller-model models/stroller_best.pt \
  --cardboard-model models/cardboard_best.pt \
  --imgsz 480 \
  --conf 0.6 \
  --show \
  --save \
  --duration 20
```

`--object-every 3` 옵션을 추가하면 물품 탐지를 3프레임마다 한 번씩 실행합니다. 사람 추적은 매 프레임 수행합니다.

Windows Git Bash나 Linux에서는 위 명령어를 사용할 수 있습니다.

실행 중 `q`를 누르면 웹캠 창이 종료됩니다.

## 6. 데이터 저장

### 사람 추적 CSV

저장 위치:

```text
data/vision/
```

주요 기록 항목:

- `timestamp`, `elapsed_sec`, `frame_id`
- `track_id`, `class`
- `center_x`, `center_y`
- `width`, `height`
- `confidence`, `frame_w`, `frame_h`

### 물품 탐지 JSONL

저장 위치:

```text
data/vision/raw/_multi_object/
```

JSONL 파일은 프레임당 하나의 JSON 객체를 저장합니다.

주요 기록 항목:

- `timestamp`, `elapsed_sec`, `frame_id`
- `object_inference_frame`
- `people`
- `unmatched_objects`

`people`에는 추적 ID, 정규화된 위치 정보, 바운딩박스와 연결된 물품 목록이 포함됩니다.

물품 정보에는 종류, 신뢰도, 바운딩박스, 연결 점수 및 `confirmed` 값이 포함됩니다.

`confirmed=True`는 일정 프레임 동안 같은 물품 종류가 해당 사람에게 반복 연결되었다는 의미이며, 실제 소지 여부가 검증되었다는 뜻은 아닙니다.

## 7. 테스트 결과 및 한계

노트북 웹캠에서 사람 추적과 물품 탐지를 테스트했습니다.

- 사람 한 명을 대상으로 ByteTrack ID가 유지되는 것을 확인
- 아이패드에 표시한 유모차 사진을 Stroller 모델이 탐지하는 것을 확인
- 사람과 유모차 이미지가 함께 등장하는 상황에서 바운딩박스 연결 결과 확인
- 사람이 카메라에 가까워지면 물품으로 잘못 탐지되는 사례 발견
- Cardboard 모델에서 사람이나 화면 일부를 택배 상자로 인식하는 오탐 발생

현재 물품 탐지 결과는 실험 단계이며, 실제 Raspberry Pi 환경에서의 인식 정확도와 처리 속도는 추가 검증이 필요합니다.

특히 물품이 잘못 탐지되면 사람과 물품 연결 결과에도 영향을 줄 수 있으므로, Integration에서는 이를 보조 정보로 사용하는 것이 적절합니다.

## 8. 향후 개발 계획

1. Raspberry Pi에서 물품 탐지 모델의 NCNN 변환 및 실행 테스트
2. 물품 탐지 간격과 입력 이미지 크기 조정을 통한 FPS 개선
3. Cardboard 및 Stroller 모델의 오탐 감소
4. 사람·물품 탐지 결과를 실시간으로 전달하는 인터페이스 구현
5. Intent 및 Integration 모듈과 연동
6. ESP32 레이더 센서 및 서보모터 연동 후 실제 문 제어 테스트

최종적으로는 기존 사람 접근 의도 판단을 유지하면서, 유모차·휠체어·택배 상자 등 물품 동반이 추정되는 경우 문 열림 유지 시간을 조정하는 방향으로 활용할 계획입니다.

단, 실제 문 닫힘 여부는 통행 및 장애물 감지 결과를 함께 고려하도록 설계할 예정입니다.