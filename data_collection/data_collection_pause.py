import cv2
import numpy as np
import os
import math
from collections import deque
from datetime import datetime
import mediapipe as mp

# === Settings === #
GESTURE_NAME = "pause"  # Change per gesture
SAVE_PATH = f"gesture_data/{GESTURE_NAME}"
os.makedirs(SAVE_PATH, exist_ok=True)

FRAME_WINDOW = 16
MOVEMENT_THRESHOLD = 5
RECORD_KEY = ord('r')
QUIT_KEY = ord('q')

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.8, min_tracking_confidence=0.7)

# Capture
cap = cv2.VideoCapture(0)
frame_history = deque(maxlen=FRAME_WINDOW)
recording = False
prev_landmarks = None

FINGERTIPS = [4, 8, 12, 16, 20]
# Create individual trail buffers for each fingertip
trail_buffers = {tip_id: deque(maxlen=30) for tip_id in FINGERTIPS}
# Unique colors for each fingertip
colors = {
    4: (255, 0, 0),     # Thumb - Blue
    8: (0, 255, 0),     # Index - Green
    12: (0, 255, 255),  # Middle - Yellow
    16: (255, 0, 255),  # Ring - Magenta
    20: (0, 128, 255)   # Pinky - Orange
}

def extract_features(landmarks, use_motion=False):
    features = []

    wrist = landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    # 1. Relative coordinates
    for lm in landmarks:
        features.extend([
            lm.x - wrist_x,
            lm.y - wrist_y,
            lm.z - wrist_z
        ])

    # 2. Fingertip distances (to wrist)
    for tip_id in FINGERTIPS:
        tip = landmarks[tip_id]
        dist = math.sqrt(
            (tip.x - wrist_x) ** 2 +
            (tip.y - wrist_y) ** 2 +
            (tip.z - wrist_z) ** 2
        )
        features.append(dist)

    # 3. Motion tracking
    if use_motion and prev_landmarks is not None:
        prev_wrist = prev_landmarks[0]
        dx = wrist_x - prev_wrist.x
        dy = wrist_y - prev_wrist.y
        angle = math.degrees(math.atan2(dy, dx))
        features.extend([dx, dy, angle])

    return np.array(features)

def is_significant_movement(frames):
    diffs = [np.linalg.norm(frames[i] - frames[i - 1]) for i in range(1, len(frames))]
    return np.mean(diffs) > MOVEMENT_THRESHOLD

print("Press 'r' to record a gesture, 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            features = extract_features(hand_landmarks.landmark, use_motion=True)
            frame_history.append(features)
            prev_landmarks = hand_landmarks.landmark

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Update trail buffers
            for tip_id in FINGERTIPS:
                lm = hand_landmarks.landmark[tip_id]
                cx, cy = int(lm.x * w), int(lm.y * h)
                trail_buffers[tip_id].append((cx, cy))

    # Draw fingertip trails
    for tip_id, trail in trail_buffers.items():
        color = colors[tip_id]
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], color, 2)

    # Show recording status
    if recording:
        cv2.putText(frame, f"Recording: {len(frame_history)} frames", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if len(frame_history) == FRAME_WINDOW:
            if is_significant_movement(list(frame_history)):
                filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".npy"
                np.save(os.path.join(SAVE_PATH, filename), np.array(frame_history))
                print(f"Saved: {filename}")
            else:
                print("Ignored: Not enough motion")
            frame_history.clear()
            recording = False

    cv2.imshow("Recording", frame)
    key = cv2.waitKey(1)

    if key == RECORD_KEY:
        recording = True
        frame_history.clear()
        print("Started recording...")

    elif key == QUIT_KEY:
        break

cap.release()
cv2.destroyAllWindows()
