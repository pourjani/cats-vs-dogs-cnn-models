# cats-vs-dogs-cnn-models
This project implements multiple Convolutional Neural Network (CNN) architectures for classifying images of cats and dogs using TensorFlow/Keras.
The goal of this project is to compare different CNN designs from a simple baseline model to more advanced architectures.

---

## Models Included

### 1. Baseline CNN
File: `cats_vs_dogs_cnn_baseline.ipynb`

A simple convolutional neural network used as a starting point for the classification task.

Features:
- Basic Conv2D + MaxPooling layers
- Simple architecture
- Good for understanding the fundamentals

Expected accuracy:
~65–75%

Runs well on:
CPU

---

### 2. Ultra-Light CNN (Mini SE)
File: `cats_dogs_ultralight_cnn_mini_se.ipynb`

A lightweight CNN designed to run efficiently on CPU with minimal resource usage.

Features:
- SeparableConv layers
- Lightweight architecture
- Small model size
- CPU friendly

Expected accuracy:
~60–70%

Runs well on:
Low-end laptops and CPUs

---

### 3. Heavy Residual SE CNN
File: `cats_dogs_heavy_residual_se_cnn.ipynb`

A more advanced deep CNN architecture that includes Residual connections and Squeeze-and-Excitation (SE) blocks.

Features:
- Residual connections
- SE attention blocks
- Deeper architecture
- Higher representational power

Expected accuracy:
~85–90%

Recommended hardware:
GPU (may run slowly on CPU)

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

---
