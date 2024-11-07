#define ENA1 6   // 모터 1의 PWM 핀
#define ENA2 11  // 모터 2의 PWM 핀

#define IN1 2   // 모터 1의 방향 제어 핀 1
#define IN2 3   // 모터 1의 방향 제어 핀 2
#define IN3 4   // 모터 2의 방향 제어 핀 1
#define IN4 5   // 모터 2의 방향 제어 핀 2

void setup() {
  Serial.begin(115200);
  // 핀 모드 설정
  pinMode(ENA1, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
}

void loop() {
  // 두 모터를 동시에 정방향 회전
  int speed = Serial.parseInt(); 
  Serial.print("받은 속도: ");
  Serial.println(speed); 
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA1, 150+speed);  // 모터 1 속도 제어 (0~255)
  delay(100);  // 2초 동안 두 모터가 역방향 회전


  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  analogWrite(ENA2, 150+speed);  // 모터 2 속도 제어 (0~255)

  delay(100);  // 2초 동안 두 모터가 정방향 회전


}
