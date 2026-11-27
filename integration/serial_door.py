# 진짜 문(ESP32 또는 아두이노)에 USB 시리얼로 명령을 보내는 코드
# fake_door.py의 FakeDoor와 똑같이 쓸 수 있게 만듦 (send / update / read_state / target)
#
# 명령 형식은 docs/protocol.md 그대로: OPEN / HOLD ms / CLOSE / ALERT n  (115200, 줄바꿈 \n)

import time
import serial  # pip install pyserial

BAUD = 115200
OPEN_ANGLE = 90
HIDE_PREFIXES = ("[초음파]",)  # 보드가 계속 보내는 줄은 화면에 안 찍음 (OPEN/CLOSE 로그가 묻혀서)
SHOW_KEYWORDS = ("OPEN", "CLOSE", "HOLD", "미충족", "대기", "서보")  # 숨기는 줄이라도 이 단어가 있으면 보여줌

class SerialDoor:
    def __init__(self, port):
        # port 예시: 윈도우 "COM3", 라즈베리파이 "/dev/ttyACM0" 또는 "/dev/ttyUSB0"
        self.ser = serial.serial_for_url(port, BAUD, timeout=0)
        time.sleep(0 if port.startswith("loop") else 2)  # 보드는 연결하면 리셋되니까 잠깐 기다림
        self.ser.reset_input_buffer()
        self.buf = b""          # 아직 줄바꿈이 안 온 글자 모아두기 (줄이 잘려 보이지 않게)
        self.now = 0.0
        self.target = 0.0      # 우리가 보낸 명령 기준으로 문이 가려는 각도 (controller가 이 값을 봄)
        self.hold_until = None
        self.last_state = "(아직 상태 못 받음)"
        print(f"[문] {port} 연결됨")

    def log(self, msg):
        print(f"[{self.now:6.2f}s] [문] {msg}")

    def update(self, now):
        self.now = now
        # 보드는 HOLD 시간이 끝나면 스스로 닫음 → 여기서도 똑같이 닫힌 걸로 기억
        if self.hold_until is not None and now >= self.hold_until:
            self.target = 0.0
            self.hold_until = None
        # 보드가 보낸 글자를 모아서, 줄바꿈이 온 줄만 꺼내 읽기
        if self.ser.in_waiting:
            self.buf += self.ser.read(self.ser.in_waiting)
        while b"\n" in self.buf:
            raw, self.buf = self.buf.split(b"\n", 1)
            line = raw.decode(errors="ignore").strip()
            if not line:
                continue
            if line.startswith("STATE"):
                self.last_state = line
            elif not line.startswith(HIDE_PREFIXES) or any(k in line for k in SHOW_KEYWORDS):
                self.log(f"보드: {line}")

    def send(self, cmd):
        self.ser.write((cmd + "\n").encode())
        name = cmd.split()[0]
        if name != "HOLD":
            self.log(f"보냄: {cmd}")
        if name == "OPEN":
            self.target = OPEN_ANGLE
        elif name == "HOLD":
            self.hold_until = self.now + int(cmd.split()[1]) / 1000
        elif name == "CLOSE":
            self.target = 0.0
            self.hold_until = None

    def read_state(self):
        return self.last_state

    def close(self):
        # 끝날 때 포트를 닫아서 다음 실행 때 "포트를 열 수 없음" 에러가 안 나게
        if self.ser.is_open:
            self.ser.close()
            print("[문] 포트 닫음")