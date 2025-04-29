# import os
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import LabelEncoder
# import joblib

# # === CONFIG ===
# DATA_DIR = "trail_data"  # contains folders like move_left, jump, roll, etc.

# X = []
# y = []

# # === Load and Flatten Gesture Data ===
# for gesture_name in os.listdir(DATA_DIR):
#     gesture_path = os.path.join(DATA_DIR, gesture_name)
#     if not os.path.isdir(gesture_path):
#         continue

#     for file in os.listdir(gesture_path):
#         if file.endswith(".npy"):
#             path = os.path.join(gesture_path, file)
#             data = np.load(path)  # Shape: (16, 15)
#             X.append(data.flatten())  # Flatten to (240,) for each file
#             y.append(gesture_name)

# print(f"Loaded {len(X)} samples from {len(set(y))} classes.")

# # === Encode Labels ===
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # === Split and Train ===
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # === Evaluate ===
# y_pred = model.predict(X_test)
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# # === Save the model and label encoder ===
# joblib.dump(model, "gesture_random_forest.pkl")
# joblib.dump(label_encoder, "gesture_label_encoder.pkl")
# print("\nModel saved as 'gesture_random_forest.pkl'")

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# === Load data === #
def load_data(base_path='gesture_data'):
    X, y = [], []
    for label in os.listdir(base_path):
        folder = os.path.join(base_path, label)
        for file in os.listdir(folder):
            if file.endswith('.npy'):
                data = np.load(os.path.join(folder, file))
                X.append(data)
                y.append(label)
    return np.array(X), np.array(y)

X, y = load_data()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# === Split === #
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train.reshape(X_train.shape[0], -1), y_train.argmax(axis=1))

y_pred_rf = rf.predict(X_test.reshape(X_test.shape[0], -1))
print("Random Forest Report:\n", classification_report(y_test.argmax(axis=1), y_pred_rf))

os.makedirs("saved_models", exist_ok=True)

joblib.dump(rf, "saved_models/random_forest.pkl")
print("Random Forest model saved.")

joblib.dump(label_encoder, "saved_models/label_encoder.pkl")


