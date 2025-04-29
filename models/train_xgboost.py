import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# === Load data === #
def load_data(base_path='gesture_data'):
    X, y = [], []
    for label in os.listdir(base_path):
        folder = os.path.join(base_path, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith('.npy'):
                data = np.load(os.path.join(folder, file))
                X.append(data)
                y.append(label)
    return np.array(X), np.array(y)

X, y = load_data()

# === Load saved LabelEncoder === #
label_encoder = joblib.load("saved_models/label_encoder.pkl")
y_encoded = label_encoder.transform(y)  # use transform, NOT fit_transform
y_onehot = to_categorical(y_encoded)

# === Split === #
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# === Train XGBoost === #
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train.reshape(X_train.shape[0], -1), y_train.argmax(axis=1))

# === Evaluation === #
y_pred_xgb = xgb.predict(X_test.reshape(X_test.shape[0], -1))
print("XGBoost Report:\n", classification_report(y_test.argmax(axis=1), y_pred_xgb))

# === Save Model === #
xgb.save_model("saved_models/xgboost_model.json")
print("XGBoost model saved.")
