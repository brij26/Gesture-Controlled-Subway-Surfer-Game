import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

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
y_encoded = label_encoder.transform(y)
y_onehot = to_categorical(y_encoded)

# === Split === #
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# === LSTM Model === #
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === EarlyStopping === #
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# === Train === #
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# === Evaluate === #
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("LSTM Report:\n", classification_report(y_true, y_pred))

# === Save Model === #
model.save("saved_models/lstm_model.h5")
print("LSTM model saved.")
