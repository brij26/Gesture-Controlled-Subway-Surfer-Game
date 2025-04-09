# 🖐️ Hand Gesture Controlled Subway Surfers

Control the **Subway Surfers** game using your hand gestures in real time via webcam! This project uses **MediaPipe**, **scikit-learn**, and **OpenCV** to recognize gestures and simulate keyboard presses to control game actions like moving left/right, jumping, rolling, using hoverboard, and pausing the game.

---

## 🎮 Demo
- click to watch demo video(https://github.com/brij26/Gesture-Controlled-Subway-Surfer-Game/blob/main/output.mp4)



---

## 🚀 Features

- Real-time gesture detection using webcam
- Machine learning model for gesture classification
- Gesture-to-keyboard mapping using `pyautogui`
- Works with Subway Surfers (Web or PC version)
- Easy to train your own gestures and extend functionality

---

## ✋ Recognized Gestures

| Gesture       | Action         | Key Pressed |
|---------------|----------------|-------------|
| `go_left`     | Move Left      | ⬅️ `a` |
| `go_right`    | Move Right     | ➡️ `d` |
| `jump`        | Jump           | ⬆️ `w` |
| `roll`        | Roll           | ⬇️ `s` |
| `hoverboard`  | Use Hoverboard | `Space` (twice) |
| `pause`       | Pause Game     | `P`          |
| `relax`       | No Action      | —            |

---

## 🧠 How It Works

1. **Data Collection**: You record gestures using MediaPipe and store them as `.npy` or `.csv`.
2. **Model Training**: Use a classifier like Random Forest, LSTM, or Transformer to learn gesture patterns.
3. **Live Prediction**: The webcam captures your hand, predicts gesture, and maps it to a keyboard action.

---

## 🛠️ Tech Stack

- [Python](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://google.github.io/mediapipe/)
- [scikit-learn](https://scikit-learn.org/)
- [pyautogui](https://pyautogui.readthedocs.io/en/latest/)
- [NumPy](https://numpy.org/)
- [joblib](https://joblib.readthedocs.io/en/latest/)

---

## 📦 Installation

```bash
git https://github.com/brij26/Gesture-Controlled-Subway-Surfer-Game
cd gesture-controlled-subway-surfers

# Create a  environment using conda
conda create -p venv python==3.10 -y
conda activate ./venv      # On Windows

# if conda is not avialable than install it first and than try it again

# Install dependencies
pip install -r requirements.txt
