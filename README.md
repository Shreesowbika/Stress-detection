Stress Detection System using MobileNetV2
This repository contains a complete pipeline for Real-Time Stress Detection using deep learning and computer vision. The system uses MobileNetV2 for classification and MTCNN for robust face detection in live video streams.
Features
Automated Dataset Mapping: Automatically categorizes image folders into stressed or not_stressed based on folder names (e.g., "fear," "angry," "happy," "neutral").

Transfer Learning: Leverages the pre-trained MobileNetV2 architecture for high-performance feature extraction.

Dual Mode Architecture: Supports both Binary classification (Direct Stress/No Stress) and Multiclass classification with keyword inference.

Live Inference: Real-time camera processing with MTCNN face tracking and predictive smoothing.

CPU Optimized: Explicitly configured to run efficiently on CPU environments.
