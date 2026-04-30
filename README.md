# 🐱🐶 Cats vs Dogs CNN Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Deep Learning](https://img.shields.io/badge/DeepLearning-CNN-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

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

**Accuracy**

~65% – 75%

---

## 2️⃣ Ultra‑Light CNN (Mini SE)

A lightweight architecture designed for training on **CPU or low‑resource systems**.

**Key components**

- SeparableConv2D layers  
- Mini Squeeze‑and‑Excitation attention block  
- GlobalAveragePooling  
- Dropout regularization  

**Accuracy**

~60% – 70%

---

## 3️⃣ Heavy Residual CNN (SE)

An advanced CNN architecture designed for better feature extraction and higher accuracy.

**Key components**

- Residual connections  
- Squeeze‑and‑Excitation (SE) blocks  
- BatchNormalization  
- Dropout regularization  
- Deeper convolutional layers  

**Accuracy**

~85% – 90%

GPU training is recommended for this model.


# ⚙️ Training Configuration

Typical training setup used in the models:

**Loss Function**

Binary Crossentropy

**Optimizer**

Adam

**Metric**

Accuracy

**Callbacks**

- EarlyStopping  
- ReduceLROnPlateau  

These techniques help stabilize training and reduce overfitting.

---

# 🧰 Technologies Used

- Python  
- TensorFlow  
- Keras  
- NumPy  
- Matplotlib  
- Scikit‑learn  

---
