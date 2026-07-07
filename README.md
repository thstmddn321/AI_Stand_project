# 스마트 노트북 거치대 (Smart Laptop Stand)

웹캠과 비전 AI를 활용하여 사용자의 자세와 제스처에 따라 자동으로 높낮이, 회전, 기울기를 조절해주는 스마트 거치대 프로젝트입니다.

## 주요 기능

- **자동 모드**: 얼굴 추적 및 거리/각도에 따른 자동 위치 조정
- **제스처 모드**: 손동작(엄지, 손바닥, 주먹 등)을 통한 수동 제어
- **한글 로그**: 웹캠 화면에 실시간 동작 상태 출력

## 설치 및 실행 방법

1. 아두이노 업로드: `arduino_stand/arduino_stand.ino`
2. 파이썬 라이브러리 설치: `pip install -r python_vision/requirements.txt`
3. 실행: `python python_vision/main.py`

## 하드웨어 구성

- N20 모터 (높이)
- NEMA17 스테핑 모터 (회전)
- MG90S 서보 모터 (기울기)
