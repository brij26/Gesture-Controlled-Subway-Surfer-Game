import cv2
import numpy as np
from vispy import app, scene
from vispy.scene import visuals
import mediapipe as mp
from threading import Thread

# === MediaPipe Setup === #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# === Vispy Setup === #
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black', size=(800, 600))
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=70, azimuth=0, elevation=90, distance=500)

# === Trail Visuals for Each Finger Tip === #
finger_tips = {
    "thumb": mp_hands.HandLandmark.THUMB_TIP,
    "index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
    "middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    "ring": mp_hands.HandLandmark.RING_FINGER_TIP,
    "pinky": mp_hands.HandLandmark.PINKY_TIP
}

colors = {
    "thumb": 'red',
    "index": 'cyan',
    "middle": 'green',
    "ring": 'yellow',
    "pinky": 'magenta'
}

trail_objects = {
    name: scene.visuals.Line(color=color, method='gl', width=4, parent=view.scene)
    for name, color in colors.items()
}

# === Shared Data === #
max_trail_length = 30
trail_buffers = {name: [] for name in finger_tips.keys()}
capture_running = True

import time  

def update_vispy():
    while capture_running:
        for name, buffer in trail_buffers.items():
            if buffer:
                trail_objects[name].set_data(pos=np.array(buffer, dtype=np.float32))
        canvas.update()
        time.sleep(0.03)  

def run_camera():
    global capture_running
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for name, landmark_enum in finger_tips.items():
                    landmark = hand_landmarks.landmark[landmark_enum]
                    x_pixel = landmark.x * 640
                    y_pixel = (1 - landmark.y) * 480
                    z_scaled = landmark.z * 100

                    pos = [x_pixel - 320, y_pixel - 240, -z_scaled]
                    trail_buffers[name].append(pos)

                    if len(trail_buffers[name]) > max_trail_length:
                        trail_buffers[name].pop(0)

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            capture_running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# Start both threads
camera_thread = Thread(target=run_camera)
vispy_thread = Thread(target=update_vispy)

camera_thread.start()
vispy_thread.start()

app.run()
