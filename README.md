# 🐱🐶 Cats vs Dogs CNN Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Deep Learning](https://img.shields.io/badge/DeepLearning-CNN-green)

A deep learning project that explores and compares multiple **Convolutional Neural Network (CNN)** architectures for **binary image classification** using the classic **Cats vs Dogs** dataset.

The repository focuses on experimenting with different CNN designs ranging from a **simple baseline model** to **lightweight optimized networks** and **advanced architectures using Residual connections and Squeeze‑and‑Excitation attention**.

---

# 📌 Project Goal

The purpose of this project is to study how different CNN architectures affect:

- Model accuracy  
- Training stability  
- Computational efficiency  
- Hardware requirements  

The models demonstrate the trade‑off between **simplicity, efficiency, and performance** in deep learning image classifiers.

---

# 🧠 Implemented Models

## 1️⃣ Baseline CNN

A simple convolutional neural network used as a reference architecture.

**Key components**

- Conv2D layers  
- MaxPooling layers  
- Dense classifier  
- Sigmoid output for binary classification  

**Training configuration**

- Batch size: 32  
- Epochs: 5  
- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Validation: Using validation data generator  

**Accuracy**

~65% – 75%

---

## 2️⃣ Ultra‑Light CNN (Mini SE)

A lightweight architecture designed for training on **CPU or low‑resource systems**.

**Key components**

- SeparableConv2D layers  
- Mini Squeeze‑and‑Excitation (SE) attention block  
- GlobalAveragePooling  
- Dropout regularization  
- Input image size: 96×96  

**Training configuration**

- Batch size: 8  
- Epochs: 20  
- Optimizer: Adam (learning rate=0.0005)  
- Loss: Binary Crossentropy  
- Validation data used  

**Accuracy**

Approximately validation accuracy reaches around 61% by epoch 4 (improving through training).

---

## 3️⃣ Heavy Residual CNN (SE)

An advanced CNN architecture designed for better feature extraction and higher accuracy.

**Key components**

- Residual connections  
- Squeeze‑and‑Excitation (SE) blocks  
- BatchNormalization  
- Dropout regularization  
- Deeper convolutional layers  

**Training configuration**

- Epochs: 25  
- Uses callbacks: EarlyStopping, ReduceLROnPlateau, LearningRateScheduler  
- Optimizer and loss typical to binary classification (Adam optimizer assumed)  
- Validation data and advanced LR scheduling  

**Accuracy**

~85% – 90%

GPU training is recommended for this model.

---

# ⚙️ Training Configuration Summary

| Model            | Batch Size | Epochs | Optimizer                   | Loss                  | Callbacks                                    |
|------------------|------------|--------|-----------------------------|-----------------------|----------------------------------------------|
| Baseline CNN     | 32         | 5      | Adam                        | Binary Crossentropy    | None or basic callbacks                       |
| Ultra-Light CNN  | 8          | 20     | Adam (lr=0.0005)            | Binary Crossentropy    | EarlyStopping, ReduceLROnPlateau              |
| Heavy Residual SE| 32         | 25     | Adam                        | Binary Crossentropy    | EarlyStopping, ReduceLROnPlateau, LR Scheduler|

---

# 🧰 Technologies Used

- Python  
- TensorFlow  
- Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

# 📊 Dataset

The project uses the **Dogs vs Cats** dataset from **Kaggle**.

Typical preprocessing uses TensorFlow’s `ImageDataGenerator` with rescaling and binary label mode.

---

