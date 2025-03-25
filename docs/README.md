# Kubernetes Failure Prediction

#Made with by Team Name : Mutta Puffs

## 📌 Overview
This project focuses on predicting Kubernetes system failures using deep learning. It uses time-series data of CPU usage, memory usage, pod status, network I/O, and disk usage to train a predictive model.

## 📂 Repository Structure
```
📦 Kubernetes-Failure-Prediction
├── 📂 src/                 # Code for data collection, training, and evaluation
│   ├── preprocess.py  # Script to collect data (if applicable)
│   ├── train.py      # Model training script
│   ├── utils.py   # Model evaluation script
│   ├── test.py          # Script for making predictions
│
├── 📂 models/              # Trained models
│   ├── k8s_failure_model.h5  # Saved deep learning model
│   ├── scaler.pkl            # Preprocessing scaler
│
├── 📂 data/                # Dataset files
│   ├── k8s_large_dataset.csv # Large dataset (uploaded dataset)
│
├── 📂 docs/                # Documentation
│   ├── README.md           # Project documentation
│   ├── model_architecture.png # Model diagram
│
├── 📂 presentation/        # Slides and recorded demos
│   ├── slides.pptx         # Presentation slides
│
├── requirements.txt        # Dependencies
├── LICENSE                 # License file
└── .gitignore              # Git ignore file
```

## 📊 Dataset
### **1. Public Datasets (Recommended)**
- [Google Cluster Workload Traces](https://github.com/google/cluster-data)
- [Microsoft Azure VM Failure Dataset](https://github.com/Azure/AzurePublicDataset)
- [Alibaba Cloud Cluster Data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=1)

### **2. Provided Dataset**(RECOMMENDED)
A large dataset (`k8s_large_dataset.csv`) is included in the `/data` directory.
https://drive.google.com/drive/folders/18ggeRfdxg8IwaX3n8h-1kMRhaBHJ5X-I?usp=drive_link

## 🧑‍💻 Installation
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the training script:**
   ```bash
   python src/train_model.py
   ```
3. **Make predictions:**
   ```bash
   python src/predict.py
   ```

## 🔍 Model Details
- **Architecture:** LSTM-based Time Series Model
- **Input Features:**
  - CPU Usage
  - Memory Usage
  - Pod Status
  - Network I/O
  - Disk Usage
- **Output:** Predicted system health metrics

## 📈 Results
- Achieved **MAE of ~2.5%** on test data.
- Can help **prevent downtime** in Kubernetes environments.

## 🤝 Contributing
Feel free to submit issues and pull requests to improve this project!


