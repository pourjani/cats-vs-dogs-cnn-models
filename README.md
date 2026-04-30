# 🐱🐶 Cats vs Dogs CNN Models

This repository contains multiple Convolutional Neural Network (CNN) architectures for binary image classification — identifying whether an image shows a **cat** or a **dog** using TensorFlow and Keras.

The project focuses on comparing different CNN designs from a simple baseline model to more advanced architectures.
## 🧠 Model Descriptions

### 1️⃣ Baseline CNN  
**File:** `cats_vs_dogs_cnn_baseline`

A simple CNN used as a reference model.

**Highlights**
- Basic Conv2D + MaxPooling layers  
- Dense output with Sigmoid activation  
- Easy to understand and modify  

**Performance**
- Accuracy: ~65–75%  
- Runs efficiently on CPU  

---

### 2️⃣ Ultra-Light CNN (Mini SE)  
**File:** `cats_dogs_ultralight_cnn_mini_se`

A lightweight CNN designed for efficient training on systems with limited hardware resources.

**Highlights**
- SeparableConv2D layers for faster computation  
- Mini Squeeze-and-Excitation (SE) attention block  
- GlobalAveragePooling  
- Dropout for regularization  

**Performance**
- Accuracy: ~60–70%  
- Optimized for CPU usage  

---

### 3️⃣ Heavy Residual CNN (SE)  
**File:** `cats_dogs_heavy_residual_se_cnn`

An advanced CNN architecture that integrates **Residual connections** and **Squeeze-and-Excitation (SE)** blocks for improved feature learning.

**Highlights**
- Residual blocks  
- SE attention modules  
- BatchNormalization  
- Dropout  

**Performance**
- Accuracy: ~85–90%  
- GPU recommended for faster training  

---

## ⚙️ Training Configuration

- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy  
- **Callbacks:** EarlyStopping, ReduceLROnPlateau  

---

## 🔬 Technologies Used

- Python  
- TensorFlow  
- Keras  
- NumPy  
- Matplotlib  

---
