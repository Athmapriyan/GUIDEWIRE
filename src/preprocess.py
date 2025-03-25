import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
df = pd.read_csv("data/k8s_large_dataset.csv")

# Feature scaling
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['cpu_usage', 'memory_usage', 'pod_status', 'network_io', 'disk_usage']])

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save processed data
pd.DataFrame(df_scaled, columns=['cpu_usage', 'memory_usage', 'pod_status', 'network_io', 'disk_usage']).to_csv("data/k8s_processed.csv", index=False)

print("✅ Data preprocessing complete. Scaled dataset saved.")
