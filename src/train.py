import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Load dataset
df = pd.read_csv("data/k8s_large_dataset.csv")

# Feature scaling
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['cpu_usage', 'memory_usage', 'pod_status', 'network_io', 'disk_usage']])

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Prepare training data
seq_length = 10
X, y = [], []

for i in range(len(df_scaled) - seq_length):
    X.append(df_scaled[i:i + seq_length])
    y.append(df_scaled[i + seq_length])

X, y = np.array(X), np.array(y)

# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(y.shape[1])
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# Save model
model.save("models/k8s_failure_model.h5")

print("✅ Model training complete. Model saved in 'models/k8s_failure_model.h5'.")
