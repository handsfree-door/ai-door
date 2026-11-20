# 진짜 문(ESP32 또는 아두이노)에 USB 시리얼로 명령을 보내는 코드
# fake_door.py의 FakeDoor와 똑같이 쓸 수 있게 만듦 (send / update / read_state / target)

import time
import serial  # pip install pyserial

BAUD = 115200
OPEN_ANGLE = 90


class SerialDoor:
    def __init__(self, port):
        # port 예시: 윈도우 "COM3", 라즈베리파이 "/dev/ttyACM0" 또는 "/dev/ttyUSB0"
        self.ser = serial.serial_for_url(port, BAUD, timeout=0)
        time.sleep(2)  # 아두이노는 연결하면 리셋되니까 잠깐 기다림
        self.ser.reset_input_buffer()
        self.now = 0.0
        self.target = 0.0      # 우리가 보낸 명령 기준으로 문이 가려는 각도 (main.py가 이 값을 봄)
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
        # 보드가 보낸 줄 읽기 (STATE ... 같은 것)
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            if line.startswith("STATE"):
                self.last_state = line
            else:
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