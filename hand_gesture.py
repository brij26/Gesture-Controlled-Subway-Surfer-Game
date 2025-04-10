import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
  while capture.isOpened():
      ret, frame = capture.read()
      frame = cv2.flip(frame, 1)
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      detected_image = hands.process(image)
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
      if detected_image.multi_hand_landmarks:
          for hand_lms in detected_image.multi_hand_landmarks:
              mp_drawing.draw_landmarks(image, hand_lms,
                                        mp_hands.HAND_CONNECTIONS,
                                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                            color=(255, 0, 255), thickness=4, circle_radius=2),
                                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                            color=(20, 180, 90), thickness=2, circle_radius=2)
                                        )
  
      cv2.imshow('Webcam', image)
  
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

capture.release()
cv2.destroyAllWindows()