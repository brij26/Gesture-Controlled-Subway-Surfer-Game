import cv2
import numpy as np
import math
import joblib
from collections import deque
import mediapipe as mp
import pyautogui  # NEW: to simulate key presses

from pynput import keyboard
import threading

# Define your key press handler
def on_press(key):
    try:
        if key == keyboard.Key.right:
            print("Right arrow is pressed")
        elif key == keyboard.Key.left:
            print("Left arrow is pressed")
        elif key == keyboard.Key.space:
            print("Spacebar pressed")
    except:
        pass

# Start the listener in a background thread
def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

# Add this line at the top of your main script (before your gesture code starts)
start_keyboard_listener()

# === Settings === #
MODEL_PATH = "saved_models\\random_forest.pkl"
FRAME_WINDOW = 16
MOVEMENT_THRESHOLD = 3

# Load model
model = joblib.load(MODEL_PATH)

# === Label Map === #
label_map = {
    0: "go_left",
    1: "go_right",
    2: "hoverboard",
    3: "jump",
    4: "pause",
    5: "relax",
    6: "roll"
}

gesture_to_key = {
    "go_left": "a",
    "go_right": "d",
    "jump": "w",
    "roll": "s",
    "pause": "p"
}

LABELS = list(label_map.values())

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.75, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)
frame_history = deque(maxlen=FRAME_WINDOW)
prev_landmarks = None
last_prediction = "Waiting..."
previous_action = None  # NEW: track last action to avoid repeats

FINGERTIPS = [4, 8, 12, 16, 20]

def extract_features(landmarks, prev_landmarks=None):
    features = []
    wrist = landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    for lm in landmarks:
        features.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

    for tip_id in FINGERTIPS:
        tip = landmarks[tip_id]
        dist = math.sqrt((tip.x - wrist_x) ** 2 +
                         (tip.y - wrist_y) ** 2 +
                         (tip.z - wrist_z) ** 2)
        features.append(dist)

    if prev_landmarks is not None:
        prev_wrist = prev_landmarks[0]
        dx = wrist_x - prev_wrist.x
        dy = wrist_y - prev_wrist.y
        angle = math.degrees(math.atan2(dy, dx))
        features.extend([dx, dy, angle])
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features)

def is_significant_movement(frames):
    diffs = [np.linalg.norm(frames[i] - frames[i - 1]) for i in range(1, len(frames))]
    return np.mean(diffs) > MOVEMENT_THRESHOLD

print("Perform a gesture... (Press 'q' to quit)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            features = extract_features(hand_landmarks.landmark, prev_landmarks)
            prev_landmarks = hand_landmarks.landmark
            frame_history.append(features)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            if len(frame_history) == FRAME_WINDOW:
                input_sequence = np.array(frame_history).flatten().reshape(1, -1)

                if is_significant_movement(list(frame_history)):
                    prediction_index = model.predict(input_sequence)[0]
                    last_prediction = label_map.get(prediction_index, f"unknown({prediction_index})")
                else:
                    last_prediction = "relax"

                # === Action Mapping === #
                if last_prediction != "relax" and last_prediction != previous_action:
                    if last_prediction == "hoverboard":
                        pyautogui.press("space")
                        pyautogui.press("space")
                        print("Pressed: space (x2) for hoverboard")
                    else:
                        key = gesture_to_key.get(last_prediction)
                        if key:
                            pyautogui.press(key)
                            print(f"Pressed: {key}")
                    previous_action = last_prediction

                if last_prediction == "relax":
                    previous_action = None  # reset to allow next detection

                frame_history.popleft()

    # Display the most recent prediction
    cv2.putText(frame, f"Gesture: {last_prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Gesture Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
