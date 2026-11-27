# 사람 위치를 받아서 OPEN / HOLD / CLOSE / ALERT 명령을 정함
# main.py(CSV 재생)와 live.py(웹캠 실시간)가 똑같이 이걸 씀
#
# 사용법
#   ctrl = DoorController(door)                                   # door = FakeDoor() 또는 SerialDoor("COM3")
#   ctrl.step(now, [(track_id, t, x, h), ...])                    # 물품 정보 없이
#   ctrl.step(now, [(track_id, t, x, h, {"stroller"}), ...])      # 5번째에 확인된 물품 이름들

import sys
from collections import deque
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "intent"))  # intent 폴더 코드 불러오기

from rule_baseline import decide, WINDOW_SEC   # noqa: E402
from door_state import decide_while_open, REOPEN_BLOCK_SEC  # noqa: E402

HOLD_MS = 2000     # 사람이 보이는 동안 2초씩 열림 유지를 연장
REFRESH_SEC = 0.5  # HOLD 연장 명령을 보내는 간격
LOST_SEC = 1.0     # 문을 열어준 사람이 1초 동안 안 보이면 지나간 걸로 봄
TAILGATE_SEC = 2.0 # 앞사람 때문에 문이 열린 뒤 이 시간 안에 다음 사람이 붙으면 → 강한 경고
ALERT_MAX_GAP_SEC = 5.0  # 문이 열리고 이 시간보다 늦게 온 사람은 꼬리물기가 아니라 다음 방문자로 봄 → 경고 안 함

# 물품 동반 (유모차·휠체어·캐리어 등) → 천천히 지나가니까 문을 더 오래 열어 둠
ITEM_LABELS = {"stroller", "wheelchair", "suitcase", "bicycle", "cardboard"}  # 반영할 물품 (오탐 많은 건 여기서 빼기)
ITEM_NAMES = {"stroller": "유모차", "wheelchair": "휠체어", "suitcase": "캐리어",
              "bicycle": "자전거", "cardboard": "택배상자"}
ITEM_MEMORY_SEC = 3.0  # 마지막으로 물품이 확인된 뒤 이 시간 동안은 '물품 동반'으로 봄
ITEM_HOLD_MS = 5000    # 물품 동반이면 HOLD 시간 (기본 2000)
ITEM_LOST_SEC = 3.0    # 물품 동반이면 사람이 안 보이고 이만큼 지나야 닫음 (기본 1초)


def log(now, msg):
    print(f"[{now:6.2f}s] {msg}")


class DoorController:
    def __init__(self, door):
        self.door = door
        self.history = {}     # track_id → 최근 기록 (t, x, h)
        self.opened_for = {}  # 문을 열어준 track_id → 마지막으로 보인 시각
        self.crossed = set()  # 가로지른다고 판단한 track_id (로그 한 번만 찍으려고)
        self.staying = set()  # 문 앞에 서 있다고 판단한 track_id (로그 한 번만 찍으려고)
        self.blocked = {}     # 멀어져서 닫은 track_id → 이 시각 전까지는 다시 안 엶
        self.alerted = set()  # 이미 경고한 track_id (한 사람당 한 번만 경고)
        self.last_hold = -999.0
        self.open_since = None  # 지금 열림이 시작된 시각 (따라옴 간격 계산용)
        self.events = []      # (시각, 명령) 기록
        self.item_seen = {}   # track_id → (물품 이름들, 마지막으로 확인된 시각)
        self.item_logged = set()  # 물품 동반 로그를 이미 찍은 track_id

    def send(self, now, cmd):
        if not cmd.startswith("HOLD"):  # HOLD 연장은 자주 보내서 출력 생략
            log(now, f"→ 명령: {cmd}")
        self.door.send(cmd)
        self.events.append((now, cmd))

    def step(self, now, people):
        # people: 이번에 보인 사람들 [(track_id, 찍힌 시각, x, h[, 물품]), ...]  x, h는 0~1로 맞춘 값
        self.door.update(now)

        # 1) 기록 쌓기 (물품 정보가 있으면 같이)
        seen = set()
        for person in people:
            tid, t, x, h = person[:4]
            self.history.setdefault(tid, deque()).append((t, x, h))
            seen.add(tid)
            items = set(person[4]) & ITEM_LABELS if len(person) > 4 else set()
            if items:
                self.item_seen[tid] = (items, now)

        # 2) 사람마다 최근 1초 움직임으로 판단
        for tid in seen:
            self._judge(now, tid)

        # 3) 문을 열어준 사람이 아직 보이면 열림 유지 연장 (물품 동반이면 더 오래 기다림)
        active = [t for t, last in self.opened_for.items()
                  if now - last <= (ITEM_LOST_SEC if self.items_of(t, now) else LOST_SEC)]
        if active and now - self.last_hold >= REFRESH_SEC:
            with_item = any(self.items_of(t, now) for t in active)
            self.send(now, f"HOLD {ITEM_HOLD_MS if with_item else HOLD_MS}")
            self.last_hold = now

        # 4) 문을 열어준 사람이 모두 사라지면 닫기
        if self.opened_for and not active:
            log(now, "판단: 문을 연 사람이 지나감 → 문 닫기")
            self.send(now, "CLOSE")
            self.opened_for.clear()

    def items_of(self, tid, now):
        # 최근 ITEM_MEMORY_SEC초 안에 확인된 물품 (물품 모델을 매 프레임 안 돌려도 이어서 봄)
        items, last = self.item_seen.get(tid, (set(), -999.0))
        return items if now - last <= ITEM_MEMORY_SEC else set()

    def _log_item(self, now, tid):
        # 문을 열어준 사람이 물품 동반이면 한 번만 로그
        items = self.items_of(tid, now)
        if items and tid not in self.item_logged:
            names = ", ".join(ITEM_NAMES.get(i, i) for i in sorted(items))
            log(now, f"판단: ID {tid} {names} 동반 → 열림 유지 연장 (HOLD {ITEM_HOLD_MS}ms)")
            self.item_logged.add(tid)

    def _judge(self, now, tid):
        hist = self.history[tid]
        # 1초 전보다 오래된 기록은 버리되, 1초 전에 가장 가까운 한 줄은 남김 (FPS가 낮아도 비교할 수 있게)
        while len(hist) >= 2 and hist[1][0] <= now - WINDOW_SEC:
            hist.popleft()
        if hist[0][0] > now - WINDOW_SEC:
            return  # 아직 1초치 기록이 안 쌓임

        window = pd.DataFrame(list(hist), columns=["t", "x", "h"])

        # 문을 열어준 사람이면: 서 있는지 / 떠나는지 판단
        if tid in self.opened_for:
            motion = decide_while_open(window)
            if motion == "LEAVING":
                log(now, f"판단: ID {tid} 문에서 멀어지는 중 → 이 사람 몫의 열림 해제")
                del self.opened_for[tid]
                self.staying.discard(tid)
                self.blocked[tid] = now + REOPEN_BLOCK_SEC
                if not self.opened_for:  # 기다리는 다른 사람이 없으면 바로 닫기
                    self.send(now, "CLOSE")
                return
            if motion == "STAY" and tid not in self.staying:
                log(now, f"판단: ID {tid} 문 앞에 서 있음 → 열어 둠")
                self.staying.add(tid)
            elif motion != "STAY":
                self.staying.discard(tid)

        decision = decide(window)

        # 멀어지는 중이면 박스가 커져도 다시 열지 않음 (문 덜컹거림 방지)
        if decision == "OPEN" and decide_while_open(window) == "LEAVING":
            decision = "WAIT"
        if decision == "OPEN" and now < self.blocked.get(tid, -1.0):
            decision = "WAIT"  # 방금 멀어져서 닫은 사람이면 잠깐 다시 열지 않음

        if decision == "OPEN" and tid not in self.opened_for:
            others = [o for o, last in self.opened_for.items() if now - last <= LOST_SEC]
            if others:
                # 앞사람 때문에 이미 열린 문으로 다른 사람이 따라옴
                gap = now - self.open_since
                if gap <= ALERT_MAX_GAP_SEC and tid not in self.alerted:
                    level = 2 if gap < TAILGATE_SEC else 1
                    log(now, f"판단: ID {tid}가 ID {others[0]} 뒤를 따라옴 (간격 {gap:.1f}초) → 경고 {level}단계")
                    self.send(now, f"ALERT {level}")
                    self.alerted.add(tid)
                elif tid not in self.alerted:
                    log(now, f"판단: ID {tid} 열린 문으로 다가옴 (간격 {gap:.1f}초) → 다음 방문자로 보고 경고 안 함")
            else:
                log(now, f"판단: ID {tid} 다가오는 중 → 문 열기")
                self.open_since = now
            if self.door.target == 0:
                self.send(now, "OPEN")  # OPEN 바로 뒤에 HOLD를 붙여 보내면 보드가 열기를 시작 안 해서, 첫 HOLD는 0.5초 뒤 연장 때 보냄
            else:
                self.send(now, f"HOLD {ITEM_HOLD_MS if self.items_of(tid, now) else HOLD_MS}")
            self.last_hold = now
        elif decision == "CROSS" and tid not in self.crossed and tid not in self.opened_for:
            log(now, f"판단: ID {tid} 가로지르는 중 → 안 엶")
            self.crossed.add(tid)

        if decision == "OPEN" or tid in self.opened_for:
            self.opened_for[tid] = now  # 마지막으로 보인 시각 갱신
            self._log_item(now, tid)

    def summary(self):
        print("\n=== 요약 ===")
        if not self.events:
            print("문이 한 번도 열리지 않았어요.")
        for t, cmd in self.events:
            if not cmd.startswith("HOLD"):
                print(f"{t:6.2f}s  {cmd}")
        print("마지막 상태:", self.door.read_state())