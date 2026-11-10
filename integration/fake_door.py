# 진짜 ESP32 대신 쓰는 가짜 문
# 명령(OPEN / HOLD ms / CLOSE / ALERT n)을 받으면 문 각도를 바꾸고 상태를 출력함

OPEN_ANGLE = 90         # 완전히 열렸을 때 각도
SPEED_DEG_PER_SEC = 90  # 1초에 90도 움직임 → 1초면 다 열림


class FakeDoor:
    def __init__(self):
        self.now = 0.0
        self.angle = 0.0       # 현재 각도 (0 = 닫힘)
        self.target = 0.0      # 가려는 각도
        self.hold_until = None # 이 시각까지 열림 유지
        self.state = "CLOSED"

    def log(self, msg):
        print(f"[{self.now:6.2f}s] [문] {msg}")

    def update(self, now):
        # 시간이 흐른 만큼 문을 움직임
        step = SPEED_DEG_PER_SEC * (now - self.now)
        if self.angle < self.target:
            self.angle = min(self.target, self.angle + step)
        elif self.angle > self.target:
            self.angle = max(self.target, self.angle - step)
        self.now = now

        # HOLD 시간이 끝나면 스스로 닫음 (라즈베리파이가 멈춰도 문이 계속 열려 있지 않게)
        if self.hold_until is not None and now >= self.hold_until:
            self.log("HOLD 시간 끝 → 자동으로 닫기 시작")
            self.target = 0.0
            self.hold_until = None

        self._update_state()

    def send(self, cmd):
        parts = cmd.split()
        name = parts[0]
        if name != "HOLD":  # HOLD 연장은 자주 와서 출력 생략
            self.log(f"명령 받음: {cmd}")

        if name == "OPEN":
            self.target = OPEN_ANGLE
        elif name == "HOLD":
            self.hold_until = self.now + int(parts[1]) / 1000
        elif name == "CLOSE":
            self.target = 0.0
            self.hold_until = None
        elif name == "ALERT":
            self.log(f"경고 {parts[1]}단계")
        else:
            self.log("알 수 없는 명령")

        self._update_state()

    def _update_state(self):
        if self.angle == self.target:
            new_state = "OPEN" if self.angle > 0 else "CLOSED"
        else:
            new_state = "OPENING" if self.target > self.angle else "CLOSING"

        if new_state != self.state:
            self.log(f"상태 {self.state} → {new_state} (각도 {self.angle:.0f}도)")
            self.state = new_state

    def read_state(self):
        reed = 1 if self.angle == 0 else 0  # 닫혀 있으면 리드 스위치 1
        return f"STATE angle={self.angle:.0f} tof1=999 tof2=999 reed={reed}"