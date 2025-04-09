import cv2
import numpy as np
import joblib
from collections import deque
import mediapipe as mp
import time

# === Load Trained Model and Label Encoder ===
model = joblib.load("gesture_random_forest.pkl")
label_encoder = joblib.load("gesture_label_encoder.pkl")

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7
)

# === Configuration ===
FRAME_WINDOW = 16
MOVEMENT_THRESHOLD = 5  # To ignore 'relax' state
PREDICTION_COOLDOWN = 1  # seconds

frame_history = deque(maxlen=FRAME_WINDOW)
last_prediction_time = 0
prediction_text = ""

# === Finger Tip Indices ===
tip_ids = [4, 8, 12, 16, 20]

def get_fingertip_trail(landmarks, width, height):
    trail = []
    for tip_id in tip_ids:
        lm = landmarks[tip_id]
        x = int(lm.x * width)
        y = int(lm.y * height)
        z = lm.z * 100  # scale Z
        trail.extend([x, y, z])
    return np.array(trail)

def is_significant_movement(frames):
    diffs = [np.linalg.norm(frames[i] - frames[i - 1]) for i in range(1, len(frames))]
    return np.mean(diffs) > MOVEMENT_THRESHOLD

# === Webcam Loop ===
cap = cv2.VideoCapture(0)

print("Starting live gesture prediction...")
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
            trail = get_fingertip_trail(hand_landmarks.landmark, w, h)
            frame_history.append(trail)

            # Draw hand
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # === Predict if ready ===
    current_time = time.time()
    if len(frame_history) == FRAME_WINDOW:
        if is_significant_movement(list(frame_history)):
            if current_time - last_prediction_time > PREDICTION_COOLDOWN:
                input_data = np.array(frame_history).flatten().reshape(1, -1)
                prediction = model.predict(input_data)
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                prediction_text = predicted_label
                last_prediction_time = current_time
        else:
            prediction_text = "Relax"
        frame_history.clear()

    # === Show Prediction ===
    cv2.putText(frame, f"Gesture: {prediction_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Live Gesture Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
