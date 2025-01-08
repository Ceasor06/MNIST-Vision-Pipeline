# **MNIST-CNN-Inference**

This project implements a **Convolutional Neural Network (CNN)** for **handwritten digit recognition** using the **MNIST dataset**. It covers the entire pipeline—from **training in Python** with **PyTorch** to **C++ inference** using **LibTorch**.

---

## **Project Overview**

### **Key Features:**
- **Python Training Script**: Builds and trains a CNN on the MNIST dataset.  
- **C++ Inference Code**: Loads the trained model for batch processing of images.  
- **TorchScript Export**: Converts the model to a format deployable in C++.  
- **GPU Acceleration**: Supports **MPS (Metal Performance Shaders)** for faster inference on macOS.  
- **Batch Processing**: Processes all images in a folder and logs results to a file.  

---

## **Setup Instructions**

### **1. Clone Repository**
```bash
git clone <repository_url>
cd MNIST-CNN-Inference
