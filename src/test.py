import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = load_model("models/k8s_failure_model.h5", compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Load scaler
scaler = joblib.load("models/scaler.pkl")

# Load test data
test_data = pd.read_csv("data/k8s_large_dataset.csv")

# Preprocess test data
seq_length = 10
features = ['cpu_usage', 'memory_usage', 'pod_status', 'network_io', 'disk_usage']
test_scaled = scaler.transform(test_data[features])

X_test = [test_scaled[i:i + seq_length] for i in range(len(test_scaled) - seq_length)]
X_test = np.array(X_test)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions
predictions = scaler.inverse_transform(predictions)

# Reduce the number of points for better clarity
sample_size = 1500  # Adjust for better readability
indices = np.linspace(0, len(predictions) - 1, sample_size, dtype=int)

# Define colors & markers
colors = ['red', 'green', 'blue', 'purple', 'orange']
markers = ['x', 'o', 's', 'd', '^']
labels = ['CPU Usage', 'Memory Usage', 'Pod Status', 'Network IO', 'Disk Usage']

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)
fig.suptitle("Actual vs Predicted Resource Usage (Scatter Plots)", fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < len(features):  # Ensure only valid features are plotted
        ax.scatter(indices, predictions[indices, i], color=colors[i], marker=markers[i], label=f"Predicted {labels[i]}", alpha=0.7, s=20)
        ax.scatter(indices, test_data[features[i]][seq_length:].values[indices], color='black', marker='.', label=f"Actual {labels[i]}", alpha=0.3, s=10)
        
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel(labels[i], fontsize=12)
        ax.set_title(f"{labels[i]}: Actual vs Predicted", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True)

# Hide the last empty subplot (if there are only 5 graphs in a 2x3 layout)
fig.delaxes(axes[1, 2])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
plt.show()

print("✅ All scatter plots displayed in one window successfully.")
