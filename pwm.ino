#define MOTOR_PWM_PIN 9  // PWM 출력 핀 설정

void setup() {
  Serial.begin(115200);  // 시리얼 통신 시작
  pinMode(MOTOR_PWM_PIN, OUTPUT);  // PWM 핀 설정
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // Python에서 전송한 데이터를 읽기
    int speed = input.toInt();  // 문자열을 정수로 변환

    // 속도 값을 PWM 출력으로 사용 (0-255 범위로 가정)
    if (speed >= 0 && speed <= 255) {
      analogWrite(MOTOR_PWM_PIN, speed);  // PWM 신호 출력
    }
  }
}
