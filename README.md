# FacialStressDetector: real-time stress monitoring works immediately using the provided trained model.
 
Real-Time Facial Stress Detection using Deep Learning

FacialStressDetector is a real-time **facial stress monitoring system** built using deep learning and computer vision.  
It detects faces from a live camera feed and predicts whether a person appears **STRESSED** or **NOT STRESSED** based on facial visual cues.

The repository includes a **pre-trained model**, allowing instant real-time stress monitoring without requiring any dataset or training.

---

## Highlights

- ✅ Plug-and-play real-time stress detection (no training required)
- ✅ Face detection using **MTCNN**
- ✅ Stress classification using **MobileNetV2 (transfer learning)**
- ✅ Prediction smoothing for stable live output
- ✅ Supports both **binary** and **multiclass** training
- ✅ CPU-only execution (no GPU needed)
- ✅ Automatic dataset keyword mapping (stress / non-stress)

---

## System Overview

1. **Face Detection**
   - Faces are detected from live webcam feed using MTCNN.
   - Only high-confidence detections are processed.

2. **Stress Classification**
   - A CNN model based on MobileNetV2 predicts stress likelihood.
   - Multiclass emotion predictions are internally mapped to stress states.

3. **Temporal Smoothing**
   - Predictions are averaged over a sliding window to avoid flickering results.

4. **Live Visualization**
   - Bounding boxes and stress labels are displayed in real time.

---

## Quick Start (Real-Time Monitoring)

> ⚡ No dataset required — the model is already trained.

### 1️⃣ Install dependencies

pip install tensorflow opencv-python mtcnn numpy tqdm
python stress_detection.py

## 📊 Dataset (Training Only)

The model included in this repository is **already trained** and can be used
directly for real-time stress monitoring.

A dataset is **NOT required** to run live detection.

### Training Dataset Overview
- Multiclass facial emotion image dataset
- Class folders such as: `happy`, `sad`, `angry`, `neutral`, `anxious`, etc.
- Images are automatically mapped to **stressed / not_stressed**
  using keyword-based logic during training
- Only face regions are used for learning

### Notes
- The dataset is **not included** in this repository due to size and licensing
- Any standard facial emotion dataset can be used for retraining
- Training is optional and only required if model re-training is desired

