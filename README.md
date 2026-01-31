# Deepfake Detection - Unified Ensemble Model

**A high-precision deepfake detection system powering a modern web application.**
This project utilizes a powerful **Ensemble Architecture** that combines three distinct deep learning models (CNN, LSTM, and ResNeXt) to achieve superior accuracy in distinguishing real images from AI-generated deepfakes.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Ensemble](https://img.shields.io/badge/Ensemble-3_Models-green.svg)

---

## 🧠 The Architecture: Triple-Model Ensemble

Unlike simple detection systems, this project leverages **three specialized models** working in unison:

### 1. 🔍 CNN (Convolutional Neural Network)
- **Role**: Standard feature extraction.
- **Strength**: Excellent at detecting local visual artifacts and pattern irregularities typical in deepfakes.

### 2. ⏱️ LSTM (Long Short-Term Memory)
- **Role**: Spatial-Sequence Analysis.
- **Architecture**: A hybrid model that extracts features using CNN layers and then processes them as a sequence using LSTM.
- **Strength**: Catching inconsistencies across the "spatial flow" of an image, treating image segments like a sequence to find structural breaks.

### 3. 🕸️ ResNeXt (Custom Lightweight)
- **Role**: Grouped Feature Processing.
- **Architecture**: A custom implementation of ResNeXt blocks data into multiple paths (cardinality).
- **Strength**: Captures complex, multi-scale features that standard CNNs might miss, effectively seeing "more details" without a massive computational cost.

### ✨ The Ensemble (The "Meta-Learner")
The **Ensemble Model** aggregates predictions from all three networks. By combining their unique perspectives—spatial patterns (CNN), sequential structure (LSTM), and multi-path features (ResNeXt)—it delivers a final confidence score that is more robust than any single model alone.

---

## 🚀 Features

- **🛡️ Multi-Model Security**: Harder to fool thanks to the 3-model voting system.
- **⚡ FastAPI Backend**: High-performance asynchronous API handling requests.
- **🎨 Modern Frontend**: Beautiful, dark-mode UI with drag-and-drop support.
- **📊 Detailed Analytics**: Returns confidence scores for "Real" vs "Fake" probabilities.
- **🏎️ GPU Acceleration**: Automatic GPU detection for fast inference.

---

## 🛠️ Project Structure

```
DeepFakeDetection/
├── app.py                    # FastAPI Backend (Inference Server)
├── requirements.txt          # Project Dependencies
├── frontend/                 # Modern Web Interface
│   ├── index.html           
│   ├── styles.css           
│   └── script.js            
└── src/                      # Model Source Code
    ├── model_ensemble.py     # 🧠 The implementation of the Ensemble logic
    ├── model_cnn.py          # CNN Architecture
    ├── model_lstm.py         # LSTM Hybird Architecture
    ├── model_resnext.py      # ResNeXt Architecture
    └── train_ensemble.py     # Script to train all models together
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the System
Start the backend server:
```bash
uvicorn app:app --reload
```
*Access the API at `http://localhost:8000`*

### 3. Launch Frontend
Open `frontend/index.html` in your browser.

---

## 🔧 Training the Ensemble

To train the full system from scratch, use the provided training scripts in the `src/` directory.

1. **Prepare Dataset**: Place your Real/Fake dataset in `data/Dataset/`.
2. **Train Models**:
   ```bash
   # Train the complete ensemble
   python src/train_ensemble.py
   ```
   *This script will train all individual models and evaluate the ensemble performance.*

---

## ⚠️ Note on Model Files
Due to GitHub file size limits, the trained `.h5` model files (which can imply huge sizes for an ensemble) are not included in this repository. 
- You interpret the code to understand the architecture.
- Use the training scripts to generate your own fresh models!

---

## 📜 License
MIT License - Open for any use.
