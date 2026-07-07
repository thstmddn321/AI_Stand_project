#include <Servo.h>

const int N20_A = 5;
const int N20_B = 6;
const int STEP_PIN = 3;
const int DIR_PIN = 2;
const int SERVO_PIN = 9;
const int EN_PIN = 8;
Servo tiltServo;

int stepDirection = 0;  // 0=정지, 1=CW, -1=CCW

unsigned long servoStartTime = 0;
bool isServoMoving = false;

String inputString = "";
bool stringComplete = false;

void setup() {
  Serial.begin(115200);
  pinMode(N20_A, OUTPUT);
  pinMode(N20_B, OUTPUT);
  digitalWrite(N20_A, LOW);
  digitalWrite(N20_B, LOW);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW);
  tiltServo.attach(SERVO_PIN);
  tiltServo.write(90);
  inputString.reserve(200);
}

void loop() {
  if (stringComplete) {
    parseCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
  updateStepper();
  checkServoStop();
}

void moveN20(int direction) {
  if (direction > 0) {
    digitalWrite(N20_A, HIGH);
    digitalWrite(N20_B, LOW);
  } else if (direction < 0) {
    digitalWrite(N20_A, LOW);
    digitalWrite(N20_B, HIGH);
  } else {
    digitalWrite(N20_A, LOW);
    digitalWrite(N20_B, LOW);
  }
}

void updateStepper() {
  if (stepDirection != 0) {
    digitalWrite(DIR_PIN, stepDirection > 0 ? LOW : HIGH);
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(800);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(800);
  }
}

void startPulseServo(int direction) {
  if (direction > 0) tiltServo.write(120);
  else if (direction < 0) tiltServo.write(60);
  servoStartTime = millis();
  isServoMoving = true;
}

void checkServoStop() {
  if (isServoMoving && (millis() - servoStartTime >= 150)) {
    tiltServo.write(90);
    isServoMoving = false;
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '>') {
      stringComplete = true;
    } else if (inChar != '<') {
      inputString += inChar;
    }
  }
}

void parseCommand(String cmd) {
  int commaIndex = cmd.indexOf(',');
  if (commaIndex == -1) return;
  char type = cmd.charAt(0);
  int value = cmd.substring(commaIndex + 1).toInt();

  switch (type) {
    case 'H': moveN20(value); break;
    case 'R':
      if (value > 0) stepDirection = 1;
      else if (value < 0) stepDirection = -1;
      else stepDirection = 0;
      break;
    case 'T': startPulseServo(value); break;
    case 'S':
      digitalWrite(N20_A, LOW);
      digitalWrite(N20_B, LOW);
      stepDirection = 0;
      tiltServo.write(90);
      break;
  }
}
