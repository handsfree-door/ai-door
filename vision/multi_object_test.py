"""Independent Vision experiment: COCO person tracking + trained object detectors.

Does NOT control a door. Produces JSONL records for a future Intent integration.
Examples:
  python vision/multi_object_test.py --stroller-model models/stroller_best.pt \\
      --cardboard-model models/cardboard_best.pt --show --save
  python vision/multi_object_test.py --camera 1 --show --duration 20 \\
      --stroller-model models/stroller_best.pt --cardboard-model models/cardboard_best.pt
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from object_association import Detection, match_objects

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'data' / 'vision' / 'raw' / '_multi_object'
# Class map verified from supplied Stroller checkpoint: 0-human, 1-wheelchair,
# 2-suitcase, 3-stroller, 4-bicycle. COCO person detection is used separately.
STROLLER_LABELS = {1: 'wheelchair', 2: 'suitcase', 3: 'stroller', 4: 'bicycle'}


def parse_args():
    p = argparse.ArgumentParser(description='Vision-only multi-model tracking and proximity test')
    p.add_argument('--camera', default='0', help='Camera index (0/1) or video file path')
    p.add_argument('--person-model', default='yolo11n.pt', help='COCO person tracking weights')
    p.add_argument('--stroller-model', required=True, help='Stroller multi-class best.pt')
    p.add_argument('--cardboard-model', required=True, help='Cardboard best.pt')
    p.add_argument('--imgsz', type=int, default=480)
    p.add_argument('--object-every', type=int, default=1,
                   help='Run object models once every N frames (person tracking stays every frame)')
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--confirm-hits', type=int, default=2,
                   help='Frames associated in recent window before confirmed')
    p.add_argument('--window-frames', type=int, default=4)
    p.add_argument('--duration', type=float, default=None)
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--show', action='store_true', help='Visualize bounding boxes (omit on SSH)')
    p.add_argument('--save', action='store_true', help='Save JSONL predictions to raw/_multi_object/')
    args = p.parse_args()
    if args.object_every < 1 or args.window_frames < 1 or args.confirm_hits < 1 or args.confirm_hits > args.window_frames:
        p.error('Require object-every >= 1 and 1 <= confirm-hits <= window-frames')
    if not 0 < args.conf <= 1:
        p.error('conf must be in (0,1]')
    return args


def load_model(path):
    p = Path(path)
    if path != 'yolo11n.pt' and not p.exists():
        raise SystemExit(f'Model not found: {path}')
    return YOLO(path)


def boxes_to_detections(result, name, labels, min_conf):
    found = []
    if result.boxes is None:
        return found
    for b in result.boxes:
        idx = int(b.cls.item())
        if idx not in labels:
            continue
        conf = float(b.conf.item())
        if conf < min_conf:
            continue
        found.append(Detection(labels[idx], conf, tuple(map(float, b.xyxy[0].tolist())), name))
    return found


def get_people(result, width, height, conf):
    out = []
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return out
    for b in boxes:
        if int(b.cls.item()) != 0 or float(b.conf.item()) < conf:
            continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        tid = int(b.id.item())
        out.append({
            'track_id': tid, 'confidence': round(float(b.conf.item()), 4),
            'bbox_xyxy': [round(x, 2) for x in (x1, y1, x2, y2)],
            'x': round((x1+x2)/(2*width), 5),
            'h': round((y2-y1)/height, 5),
            'objects': [],
        })
    return out


def draw(frame, people, unmatched):
    for person in people:
        x1, y1, x2, y2 = map(int, person['bbox_xyxy'])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (40,220,40), 2)
        names = ','.join(f"{o['label']}{'*' if o['confirmed'] else '?'}" for o in person['objects'])
        caption = f"ID {person['track_id']} {names}"
        cv2.putText(frame, caption[:85], (x1,max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,220,40), 2)
        for obj in person['objects']:
            a,b,c,d = map(int, obj['bbox_xyxy'])
            cv2.rectangle(frame, (a,b), (c,d), (255,150,0), 2)
    for obj in unmatched:
        x1,y1,x2,y2 = map(int, obj['bbox_xyxy'])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,165,255), 2)
        cv2.putText(frame, f"{obj['label']} (unmatched)", (x1,max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,165,255), 1)


def main():
    args = parse_args()
    source = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f'Cannot open camera/video: {args.camera}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    person_model = load_model(args.person_model)
    stroller_model = load_model(args.stroller_model)
    cardboard_model = load_model(args.cardboard_model)
    print('Models loaded. Stroller classes:', stroller_model.names)
    print('Cardboard classes:', cardboard_model.names)
    # Checkpoint-verified class mapping. Do not reinterpret index 0 as a person for every model.
    if dict(stroller_model.names) != {0:'0-human',1:'1-wheelchair',2:'2-suitcase',3:'3-stroller',4:'4-bicycle'}:
        raise SystemExit('Unexpected stroller model classes; check weights/class mapping first.')
    if len(cardboard_model.names) != 1 or 'cardboard' not in str(cardboard_model.names[0]).lower():
        raise SystemExit('Unexpected cardboard model classes; check weights/class mapping first.')

    f = None
    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f'{datetime.now():%Y%m%d_%H%M%S}_multi_object.jsonl'
        f = path.open('w', encoding='utf-8')
        print('Saving:', path)
    history = defaultdict(lambda: deque(maxlen=args.window_frames))
    last_seen = {}
    start = time.monotonic()
    frame_id = 0
    latest_objects = []
    object_fresh = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            height, width = frame.shape[:2]
            person_result = person_model.track(frame, persist=True,
                tracker='bytetrack.yaml', classes=[0], imgsz=args.imgsz,
                conf=args.conf, verbose=False)[0]
            people = get_people(person_result, width, height, args.conf)

            # Only associate objects freshly inferred on this frame. Otherwise we
            # would pretend stale box locations were current as people move.
            object_fresh = (frame_id-1) % args.object_every == 0
            if object_fresh:
                s = stroller_model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
                c = cardboard_model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
                latest_objects = boxes_to_detections(s, 'stroller_model', STROLLER_LABELS, args.conf)
                latest_objects += boxes_to_detections(c, 'cardboard_model', {0:'cardboard'}, args.conf)
                associations, unmatched = match_objects(people, latest_objects)
            else:
                associations = {p['track_id']: [] for p in people}
                unmatched = []
            now = time.monotonic() - start  # timestamp AFTER inference, for downstream latency
            for person in people:
                tid = person['track_id']
                matched = associations.get(tid, [])
                # Temporal confirmation evaluated over actual object-inference frames.
                if object_fresh:
                    observed = {o['label'] for o in matched}
                    for label in (*STROLLER_LABELS.values(), 'cardboard'):
                        history[(tid, label)].append(label in observed)
                    last_seen[tid] = now
                for obj in matched:
                    hit_count = sum(history[(tid, obj['label'])])
                    obj['confirmed'] = hit_count >= args.confirm_hits
                    obj['evidence'] = 'spatial_proximity_only'
                person['objects'] = matched
            # Limit state growth if tracker IDs change frequently.
            if frame_id % 100 == 0:
                expired = {tid for tid,t in last_seen.items() if now-t > 10}
                for tid in expired:
                    last_seen.pop(tid, None)
                for key in list(history):
                    if key[0] in expired:
                        del history[key]
            record = {
                'schema_version': 1, 'timestamp': datetime.now().isoformat(timespec='milliseconds'),
                'elapsed_sec': round(now, 3), 'frame_id': frame_id,
                'frame_w': width, 'frame_h': height,
                'object_inference_frame': object_fresh,
                'people': people, 'unmatched_objects': unmatched,
            }
            if f:
                f.write(json.dumps(record, ensure_ascii=False)+'\n')
            if frame_id % 20 == 0:
                print(f'frame={frame_id} time={now:.1f}s people={len(people)} objects={len(latest_objects) if object_fresh else 0}')
            if args.show:
                draw(frame, people, unmatched)
                cv2.imshow('Vision multi-object test (q to quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if args.duration is not None and now >= args.duration:
                break
            if args.max_frames is not None and frame_id >= args.max_frames:
                break
    except KeyboardInterrupt:
        print('Stopped.')
    finally:
        cap.release()
        if f:
            f.close()
        if args.show:
            cv2.destroyAllWindows()
    print(f'Finished: {frame_id} frames in {time.monotonic()-start:.1f}s')


if __name__ == '__main__':
    main()
