# DeepFake Detection — Production Ensemble System

[![CI](https://github.com/Sahil-aka/DeepFakeDetection/actions/workflows/ci.yml/badge.svg)](https://github.com/Sahil-aka/DeepFakeDetection/actions)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-yellow)](https://huggingface.co/spaces/Sahil-aka/DeepFakeDetection)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](Dockerfile)

A production-grade deepfake detection system using a hybrid **ResNeXt-50 (32x4d) + LSTM sequence model** that achieves **87.8% accuracy** on the FaceForensics++ benchmark. Supports single-image detection, video timeline analysis with suspicious segment flagging, and Grad-CAM spatial explainability heatmaps.

---

## 🌟 Key Features

- **📷 Single-Image Detection**: Instant authenticity verification with confidence scoring and probability breakdown.
- **🎥 Video Temporal Analysis**: Uniform and configurable FPS frame sampling with timeline analysis, flagging suspicious intervals (≥70% manipulation confidence).
- **🔥 Grad-CAM Explainability**: Generates gradient-weighted class activation mapping overlays to highlight manipulated pixel regions.
- **⚡ High-Performance FastAPI Backend**: Async processing, strict CORS configuration, input validation, and batch inference.
- **🐳 Containerized & CI/CD Ready**: Multi-stage Docker container, Docker Compose orchestration, and automated GitHub Actions test pipeline.
- **💻 Modern Interactive UI**: Dark-mode glassmorphism interface supporting drag-and-drop image and video uploads, real-time status checking, and interactive Grad-CAM heatmap toggles.

---

## 📊 Benchmark Results

Evaluated on **FaceForensics++** and sequence benchmark splits:

| Model Architecture | Input Format | Accuracy | AUC-ROC | F1 Score | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline CNN | Single Frame (224x224) | 78.4% | 0.832 | 0.776 | ~45ms |
| Spatial ResNeXt-50 (32x4d) | Single Frame (224x224) | 83.1% | 0.884 | 0.825 | ~65ms |
| Temporal LSTM Sequence | 20 Frames | 81.6% | 0.865 | 0.809 | ~110ms |
| **ResNeXt-50 + LSTM (This System)** | **20 Frames Sequence** | **87.8%** | **0.934** | **0.871** | **~85ms (GPU)** |

> **Dataset:** FaceForensics++ (10,000 Real / 10,000 Manipulated facial frames). Model trained and benchmarked across 20-frame sequences (`model_87_acc_20_frames_final_data.pt`).

---

## 🧠 System Architecture

```
                       ┌────────────────────────┐
                       │  Uploaded Media File   │
                       └───────────┬────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
           [ Image Pipeline ]            [ Video Pipeline ]
            RGB Resize 224x224            Sample at N fps
            ImageNet Normalize            Uniform Frame Slices
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │     ResNeXt-50 Backbone (Grouped Conv 32x4d)    │
          │    Extracts 2048-dim spatial feature vectors    │
          └────────────────────────┬────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          [ Feature Maps (7x7) ]        [ Pooled Vectors (2048) ]
                    │                             │
                    │                             ▼
                    │                 ┌───────────────────────┐
                    │                 │ LSTM Sequence Layer   │
                    │                 │ (Hidden size: 2048)   │
                    │                 └───────────┬───────────┘
                    │                             │
                    │                             ▼
                    │                 ┌───────────────────────┐
                    │                 │ Dropout (0.4) + FC    │
                    │                 │ Softmax Classification│
                    │                 └───────────┬───────────┘
                    │                             │
                    ▼                             ▼
       ┌────────────────────────┐    ┌───────────────────────────┐
       │   Grad-CAM Heatmap     │    │ Verdict: Real / Fake      │
       │ (manipulation overlay) │    │ Confidence & Timeline     │
       └────────────────────────┘    └───────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Sahil-aka/DeepFakeDetection.git
cd DeepFakeDetection

# Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify / Download Model Weights

```bash
python scripts/download_models.py
```

### 3. Start API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open `frontend/index.html` in your browser (or serve with any static web server).

---

## 🐳 Docker Deployment

Run the complete production stack (Backend on port 8000 + Frontend on port 8080) with a single command:

```bash
docker-compose up --build
```

Then visit:
- **Web Interface**: `http://localhost:8080`
- **FastAPI Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

---

## 📡 API Reference

| Method | Endpoint | Description | Request Body / Query |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health & GPU diagnostic | None |
| `POST` | `/predict` | Image deepfake detection | Multipart `file` (JPG/PNG/WEBP) |
| `POST` | `/predict/explain` | Image detection + Grad-CAM heatmap | Multipart `file` (JPG/PNG/WEBP) |
| `POST` | `/predict/video` | Video frame timeline & verdict | Multipart `file` + `?sample_fps=1` |
| `POST` | `/predict/batch` | Batch image analysis | Multipart `files` list |

### Example Response: `/predict/explain`

```json
{
  "success": true,
  "prediction": "Fake",
  "confidence": 95.8,
  "real_probability": 4.2,
  "fake_probability": 95.8,
  "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "heatmap_note": "Warm colors (red/yellow) indicate spatial regions flagged as manipulated.",
  "model_breakdown": {
    "resnext_spatial": { "real_prob": 4.2, "fake_prob": 95.8 },
    "lstm_temporal": { "real_prob": 4.2, "fake_prob": 95.8 },
    "ensemble": { "real_prob": 4.2, "fake_prob": 95.8 }
  }
}
```

---

## 🧪 Testing & Quality Assurance

Run the complete automated test suite:

```bash
pytest tests/ -v
```

Linting with Ruff:

```bash
ruff check .
```

---

## 💡 Technical Interview Highlights

### Q: Why combine ResNeXt with an LSTM instead of using a standard 2D CNN?
> **Answer:** Deepfakes exhibit two distinct classes of flaws:
> 1. **Spatial artifacts** (blending edges, warped skin texture, chromatic aberration around lips/eyes), captured effectively by the grouped convolutions (cardinality 32) of **ResNeXt-50**.
> 2. **Temporal inconsistencies** (unnatural eye blinks, frame jitter, flickering lighting across frames), captured by the recurrent memory cells of the **LSTM** layer across sequential frame vectors.

### Q: How is Grad-CAM implemented on this hybrid network?
> **Answer:** During backpropagation, we extract gradients from the target class logit with respect to the final convolutional feature maps (`layer4` of ResNeXt). We calculate global-average-pooled importance weights and compute a rectified linear weighted combination, generating a 2D spatial heatmap that indicates exactly which facial regions triggered the model's manipulation decision.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
