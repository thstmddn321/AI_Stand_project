# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import serial
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os

# --- Configuration ---
SERIAL_PORT = "/dev/tty.usbmodem1401"
BAUD_RATE = 115200
USE_SERIAL = True

MODE_AUTO = "자동 조정 모드"
MODE_GESTURE = "제스처 제어 모드"
MODE_POSTURE = "자세 교정 모드"
MODE_IDLE = "대기 모드"

MODE_BUTTONS = [MODE_AUTO, MODE_GESTURE, MODE_POSTURE, MODE_IDLE]
BUTTON_X = 10
BUTTON_Y = 58
BUTTON_WIDTH = 190
BUTTON_HEIGHT = 42
BUTTON_GAP = 8
clicked_mode = None


# --- Model paths ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")
FACE_LANDMARK_MODEL = os.path.join(MODEL_DIR, "face_landmarker.task")
FACE_DETECT_MODEL = os.path.join(MODEL_DIR, "face_detector.tflite")

# --- Initialize MediaPipe Tasks ---
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.IMAGE,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

face_lm_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_LANDMARK_MODEL),
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE,
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_lm_options)

face_det_options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_DETECT_MODEL),
    min_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE,
)
face_detector = vision.FaceDetector.create_from_options(face_det_options)

current_mode = MODE_AUTO
target_height = 0
status_msg = "준비됨"

pending_mode = None
mode_intent_start_time = 0
MODE_SWITCH_DELAY = 1.0

last_face_x, last_face_y = 0.5, 0.5
posture_start_time, posture_phase = 0, 0
bad_posture_type = ""
bad_posture_start = 0
BAD_POSTURE_DURATION = 5.0

baseline_pitch = None
baseline_roll = None
baseline_face_sz = None
calibration_data = []

last_detected_gesture = None
gesture_hold_start = 0
GESTURE_HOLD_DELAY = 0.5
gesture_loss_start = 0
GESTURE_LOSS_TOLERANCE = 0.3

FONT_CANDIDATES = [
    "malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
]
font = None
for _fp in FONT_CANDIDATES:
    try:
        font = ImageFont.truetype(_fp, 20)
        break
    except: pass
if font is None:
    font = ImageFont.load_default()

def put_korean_text(img, text, pos, color=(255, 255, 255)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

ser = None
if USE_SERIAL:
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1, write_timeout=0.1)
        time.sleep(2)
    except: pass

def send_command(cmd_type, value):
    if USE_SERIAL and ser:
        try:
            msg = f"<{cmd_type},{int(round(value))}>"
            ser.write(msg.encode())
            if cmd_type in ("T", "R"):
                print(f"[CMD] {msg}")
        except:
            pass
    else:
        if cmd_type == "T":
            print(f"[CMD 실패 - 시리얼 없음] {cmd_type},{value}")


def switch_mode(new_mode):
    global current_mode, pending_mode, status_msg
    global posture_phase, posture_start_time, calibration_data
    global baseline_pitch, baseline_roll, baseline_face_sz, bad_posture_start

    if new_mode == current_mode:
        pending_mode = None
        return

    current_mode = new_mode
    pending_mode = None
    if current_mode == MODE_POSTURE:
        posture_phase = 1
        posture_start_time = time.time()
        calibration_data = []
        baseline_pitch = baseline_roll = baseline_face_sz = None
        bad_posture_start = 0
    elif current_mode == MODE_IDLE:
        send_command("S", 0)
    status_msg = f"{current_mode} 전환됨"


def on_mouse(event, x, y, flags, param):
    global clicked_mode
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for index, mode in enumerate(MODE_BUTTONS):
        top = BUTTON_Y + index * (BUTTON_HEIGHT + BUTTON_GAP)
        if BUTTON_X <= x <= BUTTON_X + BUTTON_WIDTH and top <= y <= top + BUTTON_HEIGHT:
            clicked_mode = mode
            return


def draw_mode_buttons(image):
    overlay = image.copy()
    panel_bottom = BUTTON_Y + len(MODE_BUTTONS) * (BUTTON_HEIGHT + BUTTON_GAP)
    cv2.rectangle(overlay, (0, 45), (BUTTON_X + BUTTON_WIDTH + 10, panel_bottom), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)

    for index, mode in enumerate(MODE_BUTTONS):
        top = BUTTON_Y + index * (BUTTON_HEIGHT + BUTTON_GAP)
        active = mode == current_mode
        fill = (40, 145, 40) if active else (55, 55, 55)
        border = (100, 255, 100) if active else (145, 145, 145)
        cv2.rectangle(image, (BUTTON_X, top),
                      (BUTTON_X + BUTTON_WIDTH, top + BUTTON_HEIGHT), fill, -1)
        cv2.rectangle(image, (BUTTON_X, top),
                      (BUTTON_X + BUTTON_WIDTH, top + BUTTON_HEIGHT), border, 2)
        image = put_korean_text(image, mode, (BUTTON_X + 10, top + 9), (255, 255, 255))
    return image

def get_hand_pose(hand_landmarks, handedness_list, hand_idx):
    lm = hand_landmarks

    label = "Right"
    if handedness_list and hand_idx < len(handedness_list):
        cats = handedness_list[hand_idx]
        if cats:
            label = cats[0].category_name

    pts = [(l.x, l.y) for l in lm]
    hand_size = np.linalg.norm(np.array([pts[0][0] - pts[9][0], pts[0][1] - pts[9][1]]))
    if hand_size < 0.001: hand_size = 0.1

    f_up = []
    thumb_vec = np.array([pts[4][0] - pts[2][0], pts[4][1] - pts[2][1]])
    thumb_dist = np.linalg.norm(thumb_vec)
    f_up.append(thumb_dist > hand_size * 0.6)

    for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        tip_wrist = np.linalg.norm(np.array([pts[tip][0] - pts[0][0], pts[tip][1] - pts[0][1]]))
        mcp_wrist = np.linalg.norm(np.array([pts[mcp][0] - pts[0][0], pts[mcp][1] - pts[0][1]]))
        f_up.append(tip_wrist > mcp_wrist * 1.2)

    v1 = np.array([pts[5][0] - pts[0][0], pts[5][1] - pts[0][1]])
    v2 = np.array([pts[17][0] - pts[0][0], pts[17][1] - pts[0][1]])
    cp = v1[0] * v2[1] - v1[1] * v2[0]
    if label == "Left": is_palm = cp < 0
    else: is_palm = cp > 0

    ptr_vec = np.array([pts[8][0] - pts[5][0], pts[8][1] - pts[5][1]])
    norm = np.linalg.norm(ptr_vec)
    if norm > 0: ptr_vec /= norm

    ptr_dir = "NONE"
    if abs(ptr_vec[0]) > 0.5:
        ptr_dir = "LEFT" if ptr_vec[0] < 0 else "RIGHT"
    elif abs(ptr_vec[1]) > 0.5:
        ptr_dir = "DOWN" if ptr_vec[1] > 0 else "UP"

    is_upright = pts[12][1] < pts[9][1] - (hand_size * 0.2)

    return f_up, is_palm, ptr_dir, hand_size, is_upright, thumb_vec

def get_face_posture_data(face_lm_result, img_h, img_w):
    if not face_lm_result.face_landmarks: return None, None
    lm = face_lm_result.face_landmarks[0]
    nose_tip = lm[1]
    chin = lm[152]
    forehead = lm[10]
    left_eye = lm[33]
    right_eye = lm[263]
    face_height = abs(chin.y - forehead.y)
    if face_height < 0.001: return 0, 0
    pitch_ratio = (chin.y - nose_tip.y) / face_height
    roll_angle = np.degrees(np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x))
    return pitch_ratio, abs(roll_angle)

def draw_hand_landmarks(image, hand_lm_result):
    if not hand_lm_result.hand_landmarks: return image
    h, w = image.shape[:2]
    for hand_lms in hand_lm_result.hand_landmarks:
        pts = [(int(l.x * w), int(l.y * h)) for l in hand_lms]
        connections = mp.solutions.hands.HAND_CONNECTIONS if hasattr(mp, 'solutions') else [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(0,17),(17,18),(18,19),(19,20)
        ]
        for c in connections:
            cv2.line(image, pts[c[0]], pts[c[1]], (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(image, pt, 4, (255, 0, 0), -1)
    return image


def select_camera():
    print("\n" + "="*30)
    print(" [카메라 선택 모드]")
    print("="*30)

    available_indices = []
    devnull = open(os.devnull, 'w')
    old_stderr = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                available_indices.append(i)
            cap.release()
    os.dup2(old_stderr, 2)
    os.close(old_stderr)
    devnull.close()

    if not available_indices:
        print("연결된 카메라를 찾을 수 없습니다. 기본값(0)으로 시도합니다.")
        return 0

    for idx in available_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f" [{idx}] 카메라 {idx}  ({w}x{h})")

    while True:
        try:
            choice = input(f"\n사용할 카메라 번호를 선택하세요 ({'/'.join(map(str, available_indices))}): ").strip()
            if not choice:
                choice = available_indices[0]
            else:
                choice = int(choice)
            if choice not in available_indices:
                print("목록에 있는 번호를 입력해주세요.")
                continue
            # 실제 프레임 읽기 테스트
            test_cap = cv2.VideoCapture(choice, cv2.CAP_AVFOUNDATION)
            time.sleep(1.0)
            ok = False
            for _ in range(10):
                ret, _ = test_cap.read()
                if ret:
                    ok = True
                    break
            test_cap.release()
            if ok:
                return choice
            else:
                print(f" 카메라 {choice} 프레임 읽기 실패. 다른 번호를 선택하세요.")
        except ValueError:
            print("숫자를 입력해주세요.")

camera_idx = select_camera()
cap = cv2.VideoCapture(camera_idx, cv2.CAP_AVFOUNDATION)
WINDOW_NAME = "Smart Laptop Stand"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)
loop_count = 0
try:
    while cap.isOpened():
        if clicked_mode is not None:
            requested_mode = clicked_mode
            clicked_mode = None
            switch_mode(requested_mode)

        success, image = cap.read()
        if not success: break
        image = cv2.flip(image, 1)
        h, w = image.shape[:2]
        small = cv2.resize(image, (640, 360))
        sh, sw = small.shape[:2]
        rgb_image = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        hand_result = hand_landmarker.detect(mp_image)
        face_det_result = face_detector.detect(mp_image)
        face_lm_result = face_landmarker.detect(mp_image)

        gesture_detected = False
        loop_count += 1

        image = draw_hand_landmarks(image, hand_result)

        if hand_result.hand_landmarks:
            gesture_detected = True
            for hand_idx, hand_lms in enumerate(hand_result.hand_landmarks):
                f_up, is_palm, ptr_dir, hand_size, upright, thumb_vec = get_hand_pose(
                    hand_lms, hand_result.handedness, hand_idx
                )

                num_up = sum(f_up)
                num_no_thumb = sum(f_up[1:])

                detected_intent = None
                if num_up == 0: detected_intent = MODE_IDLE
                elif not f_up[0]:
                    if num_no_thumb == 1: detected_intent = MODE_AUTO
                    elif num_no_thumb == 2: detected_intent = MODE_GESTURE
                    elif num_no_thumb == 3: detected_intent = MODE_POSTURE

                # 제스처 모드에서 검지만 좌우로 가리킬 때 자동 모드 전환 방지
                if current_mode == MODE_GESTURE and detected_intent == MODE_AUTO:
                    if f_up[1] and not (f_up[0] or f_up[2] or f_up[3] or f_up[4]) and ptr_dir in ("LEFT", "RIGHT"):
                        detected_intent = None

                if detected_intent and detected_intent != current_mode:
                    if pending_mode == detected_intent:
                        if time.time() - mode_intent_start_time > MODE_SWITCH_DELAY:
                            switch_mode(detected_intent)
                    else:
                        pending_mode = detected_intent
                        mode_intent_start_time = time.time()
                else:
                    pending_mode = None

                if current_mode == MODE_GESTURE:
                    detected_gesture = None

                    if f_up[1] and not (f_up[0] or f_up[2] or f_up[3] or f_up[4]):
                        if ptr_dir == "LEFT":
                            detected_gesture = "rotate_left"
                            status_msg = "회전: 좌회전 (유지 중...)"
                        elif ptr_dir == "RIGHT":
                            detected_gesture = "rotate_right"
                            status_msg = "회전: 우회전 (유지 중...)"
                    elif f_up[0] and num_no_thumb == 0:
                        if abs(thumb_vec[1]) > abs(thumb_vec[0]):
                            if thumb_vec[1] < 0:
                                detected_gesture = "height_up"
                                status_msg = "높이: 올리기 (유지 중...)"
                            else:
                                detected_gesture = "height_down"
                                status_msg = "높이: 내리기 (유지 중...)"
                    elif num_up >= 4 and not (f_up[1] and f_up[2] and not f_up[3] and not f_up[4]):
                        if is_palm:
                            detected_gesture = "tilt_back"
                            status_msg = "기울기: 뒤로"
                        else:
                            detected_gesture = "tilt_forward"
                            status_msg = "기울기: 앞으로"

                    INSTANT_GESTURES = {"tilt_back", "tilt_forward"}

                    if detected_gesture:
                        if detected_gesture in INSTANT_GESTURES:
                            if loop_count % 2 == 0:
                                if detected_gesture == "tilt_back":
                                    send_command("T", 1)
                                elif detected_gesture == "tilt_forward":
                                    send_command("T", -1)
                            last_detected_gesture = None
                            gesture_loss_start = 0
                        else:
                            now = time.time()
                            if detected_gesture == last_detected_gesture:
                                gesture_loss_start = 0
                                if now - gesture_hold_start >= GESTURE_HOLD_DELAY:
                                    if loop_count % 2 == 0:
                                        if detected_gesture == "rotate_left":
                                            status_msg = "회전: 좌회전"
                                            send_command("R", -1)
                                        elif detected_gesture == "rotate_right":
                                            status_msg = "회전: 우회전"
                                            send_command("R", 1)
                                        elif detected_gesture == "height_up":
                                            status_msg = "높이: 올리기"
                                            send_command("H", 1)
                                        elif detected_gesture == "height_down":
                                            status_msg = "높이: 내리기"
                                            send_command("H", -1)
                                else:
                                    status_msg = status_msg + " (유지 중...)"
                            else:
                                if last_detected_gesture is None:
                                    # 이전 제스처 없음 → 즉시 새 제스처 시작
                                    last_detected_gesture = detected_gesture
                                    gesture_hold_start = now
                                    gesture_loss_start = 0
                                else:
                                    # 다른 제스처에서 전환 → GESTURE_LOSS_TOLERANCE 동안 무시
                                    if gesture_loss_start == 0:
                                        gesture_loss_start = now
                                    if now - gesture_loss_start >= GESTURE_LOSS_TOLERANCE:
                                        last_detected_gesture = detected_gesture
                                        gesture_hold_start = now
                                        gesture_loss_start = 0
                    else:
                        now = time.time()
                        if last_detected_gesture is not None:
                            # 제스처 유실 → GESTURE_LOSS_TOLERANCE 동안 기다린 후 리셋
                            if gesture_loss_start == 0:
                                gesture_loss_start = now
                            if now - gesture_loss_start >= GESTURE_LOSS_TOLERANCE:
                                last_detected_gesture = None
                                gesture_loss_start = 0
                        else:
                            gesture_loss_start = 0

        if current_mode == MODE_IDLE:
            status_msg = "대기 모드: 정지 중"
        elif current_mode == MODE_POSTURE:
            elapsed = time.time() - posture_start_time
            pitch, roll = get_face_posture_data(face_lm_result, h, w)
            if posture_phase == 1:
                countdown = 3 - int(elapsed)
                status_msg = f"자세 교정: 바른 자세 유지하세요! ({max(0, countdown)})"
                if pitch is not None and face_det_result.detections:
                    det = face_det_result.detections[0]
                    bb = det.bounding_box
                    sz = (bb.width / sw) * (bb.height / sh)
                    calibration_data.append((pitch, roll, sz))
                if elapsed > 3.0:
                    if calibration_data:
                        baseline_pitch = float(np.mean([d[0] for d in calibration_data]))
                        baseline_roll = float(np.mean([d[1] for d in calibration_data]))
                        baseline_face_sz = float(np.mean([d[2] for d in calibration_data]))
                    else:
                        baseline_pitch, baseline_roll, baseline_face_sz = 0.5, 0.0, 0.05
                    calibration_data = []
                    posture_phase = 2
            elif posture_phase == 2:
                cur_face_sz = None
                if face_det_result.detections:
                    det = face_det_result.detections[0]
                    bb = det.bounding_box
                    cur_face_sz = (bb.width / sw) * (bb.height / sh)
                if pitch is not None:
                    bad = False
                    if abs(pitch - baseline_pitch) > 0.15:
                        bad = True
                        bad_posture_type = "목 각도"
                    elif roll > baseline_roll + 10:
                        bad = True
                        bad_posture_type = "기울어짐"
                    elif cur_face_sz is not None and cur_face_sz > baseline_face_sz * 1.2:
                        bad = True
                        bad_posture_type = "거북목"
                    if bad:
                        if bad_posture_start == 0:
                            bad_posture_start = time.time()
                        bad_elapsed = time.time() - bad_posture_start
                        remaining = int(BAD_POSTURE_DURATION - bad_elapsed) + 1
                        status_msg = f"자세 교정: 감시 중 ({bad_posture_type} 감지, {remaining}초)"
                        if bad_elapsed >= BAD_POSTURE_DURATION:
                            posture_phase = 3
                            posture_start_time = time.time()
                            bad_posture_start = 0
                    else:
                        bad_posture_start = 0
                        status_msg = "자세 교정: 감시 중"
            elif posture_phase == 3:
                status_msg = f"불안정한 자세 감지!! ({bad_posture_type})"
                send_command("H", 1)
                if elapsed > 1.5:
                    posture_phase = 4
                    posture_start_time = time.time()
            elif posture_phase == 4:
                status_msg = "3초 동안 바른 자세로 돌아오세요!"
                send_command("H", 0)
                if elapsed > 3.0:
                    posture_phase = 5
                    posture_start_time = time.time()
            elif posture_phase == 5:
                status_msg = "자세 복귀 중..."
                send_command("H", -1)
                if elapsed > 1.5:
                    send_command("H", 0)
                    posture_phase = 1
                    posture_start_time = time.time()

        elif current_mode == MODE_AUTO:
            if not gesture_detected and loop_count % 3 == 0:
                pitch, _ = get_face_posture_data(face_lm_result, h, w)
                if face_det_result.detections:
                    det = face_det_result.detections[0]
                    bb = det.bounding_box
                    # bounding_box는 small 이미지(640x360) 픽셀 단위 → 정규화
                    cx = (bb.origin_x + bb.width / 2) / sw
                    cy = (bb.origin_y + bb.height / 2) / sh
                    face_sz = (bb.width / sw) * (bb.height / sh)
                    last_face_x, last_face_y = cx, cy
                    if pitch is not None and pitch > 0.60:
                        status_msg = "상태: 천장 감지"
                        send_command("S", 0)
                    elif cy < 0.25:
                        send_command("H", 1)
                        status_msg = "상태: 위쪽 추적"
                    elif cy > 0.75:
                        send_command("H", -1)
                        status_msg = "상태: 아래쪽 추적"
                    elif cx < 0.25:
                        send_command("R", -1)
                        status_msg = "상태: 왼쪽 추적"
                    elif cx > 0.75:
                        send_command("R", 1)
                        status_msg = "상태: 오른쪽 추적"
                    elif face_sz > 0.12:
                        send_command("T", 1)
                        status_msg = "상태: 가까움"
                    else:
                        err_x = cx - 0.5
                        if abs(err_x) > 0.12:
                            send_command("R", 1 if err_x > 0 else -1)
                            status_msg = "상태: 중앙 정렬"
                        else:
                            status_msg = "상태: 정상"
                            send_command("H", 0)
                            send_command("S", 0)
                else:
                    if last_face_y < 0.35:
                        send_command("H", 1)
                        status_msg = "유실: 위"
                    elif last_face_y > 0.65:
                        send_command("H", -1)
                        status_msg = "유실: 아래"
                    elif last_face_x < 0.35:
                        send_command("R", -1)
                        status_msg = "유실: 왼"
                    elif last_face_x > 0.65:
                        send_command("R", 1)
                        status_msg = "유실: 오"
                    else:
                        status_msg = "상태: 얼굴 없음"
                        send_command("S", 0)

        if current_mode == MODE_GESTURE and not gesture_detected:
            status_msg = "제스처: 없음"
            send_command("S", 0)
            target_height = 0

        cv2.rectangle(image, (0, 0), (640, 45), (0, 0, 0), -1)
        if pending_mode:
            progress = int((time.time() - mode_intent_start_time) / MODE_SWITCH_DELAY * 640)
            cv2.rectangle(image, (0, 40), (progress, 45), (0, 255, 255), -1)

        image = put_korean_text(image, f"모드: {current_mode}", (10, 10), (0, 255, 0))
        image = put_korean_text(image, f"로그: {status_msg}", (220, 10), (255, 255, 255))
        image = draw_mode_buttons(image)

        cv2.imshow(WINDOW_NAME, image)
        if cv2.waitKey(1) & 0xFF == 27: break
finally:
    if cap: cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    face_landmarker.close()
    face_detector.close()
    if ser: ser.close()
