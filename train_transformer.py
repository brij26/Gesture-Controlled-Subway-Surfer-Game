import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

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

# === Transformer Encoder Block === #
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Multi-head self-attention
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x_ff, x])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# === Build Transformer Model === #
input_shape = X.shape[1:]  # (timesteps, features)
inputs = Input(shape=input_shape)

x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)

x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
outputs = Dense(y_onehot.shape[1], activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# === EarlyStopping === #
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# === Train === #
model.fit(
    X_train, y_train,
    epochs=120,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# === Evaluate === #
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Transformer Report:\n", classification_report(y_true, y_pred))

# === Save Model === #
model.save("saved_models/transformer_model.h5")
print("Transformer model saved.")
