# 🏭 Machine Learning for Industrial Inspection

This repository contains a collection of machine learning models and datasets developed for **industrial inspection and quality control**, focusing on pharmaceutical bottle defect detection and classification.

The project is developed **strictly for academic and research purposes**, providing a foundation for studying and improving automated defect detection techniques in manufacturing environments.

---

## 📂 Repository Contents

### 🔹 Classification Models
- Jupyter notebooks for training and evaluating classification models.
- Pretrained weights included for reproducibility.
- Dataset consists of labeled images of pharmaceutical bottles categorized as *Good* or *Defective*.

### 🔹 Detection Models
- Includes **YOLOv8** and **Faster R-CNN** implementations.
- Models trained and evaluated across multiple viewpoints:
  - **Side View**
  - **Top View**
  - **Bottom View**
  - **Combined View**
- Annotated datasets are provided for localized defect detection.

---

## 📦 Dataset

The dataset used for both classification and detection can be accessed below:

👉 **[Download Original Dataset](https://drive.google.com/drive/folders/1WxkK6sdmGOCMZgY8wXui_Eav5Qov-9-C?usp=drive_link)**



---

## ⚙️ Models and Weights

All trained model weights are stored within the project directories:
- `Classification model/weights/`
- `Detection model/Yolov8/weights/`
- `Detection model/Faster R-CNN/weights/`

---

## 🧠 Objectives

- Develop and evaluate **classification** and **object detection** models for defect inspection.  
- Assess model performance across multiple viewpoints for enhanced generalization.  
- Contribute to academic understanding of **machine learning in industrial inspection systems**.

---

## 📊 Results Overview

- **YOLOv8** achieved high mAP scores across all viewpoints, with *Side* and *Top* views performing best.  
- **Faster R-CNN** demonstrated consistent detection with strong precision.  
- The **classification model** achieved over **85% accuracy** on the validation dataset.

---

## 🧩 Tools and Libraries

- Python (NumPy, Pandas, OpenCV, Matplotlib)
- PyTorch / TensorFlow
- Ultralytics YOLOv8
- Scikit-learn
- Jupyter Notebook

---

## 🎓 Academic Purpose

This repository and its contents are intended **solely for academic, educational, and research use**.  
Redistribution, commercialization, or industrial deployment of this work without prior authorization is **not permitted**.

---

## 📧 Contact & Enquiries

For academic or research enquiries, please contact:  
**📩 talifhaninemaangani@gmail.com**

---

## 🧑‍💻 Author
**Talifhani Nemaangani**  
GitHub: [@Talifhani-cloud](https://github.com/Talifhani-cloud)

---

## 📜 License
This project is licensed for **academic and research purposes only**.  
Please cite or acknowledge this repository when referencing or using any part of this work.
