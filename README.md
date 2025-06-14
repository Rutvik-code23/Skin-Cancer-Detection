
# 🧬 Skin Cancer Detection with CNN & Data Augmentation

A robust image classification pipeline using deep learning (CNNs) and advanced data augmentation to detect skin cancer with high accuracy.

---

### 🧠 Problem Statement

Skin cancer is a rapidly growing health concern globally. Early and accurate diagnosis is crucial but often limited by:

❌ Shortage of trained dermatologists  
📸 Visual similarities between benign and malignant lesions  
🔎 Difficulty in consistent and objective diagnosis  

Traditional screening methods are manual, slow, and prone to human error. Automated systems can support clinical decision-making and increase early detection rates.

---

### ⚠️ The Challenge

Build an image classification model that learns to differentiate between benign and malignant skin lesions using CNNs and augmented dermoscopic image data.

---

### 🚀 Objectives

| 🎯 Goal | 📝 Description |
|--------|----------------|
| 🧠 Image Classification | Train a CNN to classify images as benign or malignant |
| 🔄 Data Augmentation | Improve generalization through real-time transformations |
| 🧪 Model Evaluation | Use metrics like accuracy, loss |
| 📊 Visualization | Track training curves and sample predictions |

---

### 📊 Context

Dermoscopic images are challenging to classify due to noise, lighting, shape irregularities, and overlapping patterns. By applying real-time **data augmentation**, and training a **convolutional neural network**, we aim to create a model that performs reliably across various skin lesion types.

---

### 🛠️ Tech Stack

| Category | Tools / Frameworks |
|---------|---------------------|
| 🔍 DL | TensorFlow / Keras / CNN |
| 🖼️ Data Pipeline | ImageDataGenerator / tf.data |
| 📂 Data Format | Folder-structured image dataset (Train / Val) |
| 🧪 Experimentation | Google Colab / Jupyter Notebooks |

---

## 📁 Dataset Split

The dataset is divided into training and validation sets as follows:

| Set          | Number of Images | Number of Classes | Description                                      |
|--------------|------------------|-------------------|--------------------------------------------------|
| Training     | 32,966           | 2                 | Used to train the model and learn key features.  |
| Validation   | 14,132           | 2                 | Used to evaluate model performance and detect overfitting. |

---

### ⚙️ Features

- ✅ Binary classification using CNN
- ✅ Real-time image augmentation (flip, zoom, rotate, shear, shift)
- ✅ Evaluation on validation dataset
- ✅ Beginner-friendly TensorFlow implementation

---

### 📈 Results

- High training and validation accuracy
- Improved generalization due to augmentation
- Good separation of benign and malignant classes

---

### 🔮 Future Work

- Deploy using Streamlit or Flask for real-world testing
