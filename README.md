
# 🎬 Real-Time Face Segmentation for Movie Cast Identification

## 📌 Project Overview
This project focuses on detecting and segmenting human faces in movie scenes using deep learning techniques. The goal is to enable a "Pause-and-Identify" feature in streaming platforms, where users can instantly view actor details.

---

## 🚀 Features
- Face segmentation using U-Net architecture
- MobileNetV2 encoder (transfer learning)
- Real-time inference using Streamlit
- Image upload and webcam support
- Performance dashboard with metrics
- Downloadable logs and results

---

## 🧠 Problem Statement
Streaming platforms need a system that can:
- Detect faces in complex movie scenes
- Segment faces accurately
- Identify actors in real-time

---

## 💼 Business Use Cases
- 🎥 Pause & Identify actors
- 🎯 Personalized recommendations
- 🛡 Content moderation
- 📢 Targeted advertising

---

## 🗂 Dataset
- Movie scene images
- Corresponding binary face masks
- Preprocessing:
  - Resize to 256x256
  - Data augmentation (flip, rotate, brightness)

---

## 🏗 Model Architecture
- U-Net architecture
- MobileNetV2 as encoder
- Skip connections for better localization
- Custom Dice Loss function

---

## ⚙️ Training Details
- Optimizer: Adam
- Loss: Dice Loss + Binary Crossentropy
- Metrics: Dice Coefficient, IoU
- Techniques:
  - Early Stopping
  - Model Checkpoint
  - Learning Rate Scheduling

---

## 📊 Evaluation Metrics
| Metric | Target |
|--------|--------|
| Dice Coefficient | > 0.92 |
| IoU | > 0.88 |
| F1 Score | > 0.90 |
| Inference Speed | < 100ms |

---

## 🔍 Results
- High accuracy in face segmentation
- Works well in crowded scenes
- Real-time performance achieved

---

## 🖥 Streamlit App
### Features:
- Upload images
- Real-time webcam detection
- Visualization of masks
- Metrics dashboard

### Run App:
```bash
streamlit run app.py
```

---

## 📦 Installation
```bash
git clone <repo_url>
cd project
pip install -r requirements.txt
```

---

## 📁 Project Structure
```
├── data/
├── models/
├── notebooks/
├── app.py
├── requirements.txt
├── README.md
```

---

## 📌 Future Improvements
- Add face recognition (actor identification)
- Improve inference speed
- Deploy on cloud (AWS / HuggingFace)

---

## 🧑‍💻 Tech Stack
- Python
- OpenCV
- TensorFlow / Keras
- Streamlit
- NumPy, Pandas, Matplotlib

---


## 🙌 Acknowledgements
- Open-source community
- TensorFlow & Streamlit contributors
