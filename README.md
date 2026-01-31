# Deepfake Detection - CNN Model

A web-based deepfake detection system using a CNN (Convolutional Neural Network) model with FastAPI backend and modern HTML/CSS/JavaScript frontend.

## Features

- 🤖 **CNN Deep Learning Model** - Trained model for detecting deepfake images
- 🚀 **FastAPI Backend** - Fast, modern API for image prediction
- 🎨 **Modern Web Interface** - Beautiful, responsive UI with drag-and-drop upload
- 📊 **Real-time Results** - Instant prediction with confidence scores
- 🎯 **High Accuracy** - CNN model trained on real and fake images

---

## Quick Start

> **⚠️ Important Note**: The trained model file (`deepfake_cnn_gpu.h5`) is not included in this repository due to GitHub's file size limits. You'll need to either:
> - Train your own model using the instructions below, OR
> - Download a pre-trained model (if available separately)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Backend

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

You can view the API documentation at `http://localhost:8000/docs`

### 3. Open the Frontend

Open `frontend/index.html` in your web browser, or serve it with a simple HTTP server:

```bash
cd frontend
python -m http.server 8080
```

Then navigate to `http://localhost:8080`

---

## Usage

1. **Upload an Image**: Drag and drop an image or click to browse
2. **Analyze**: Click the "Analyze Image" button
3. **View Results**: See if the image is Real or Fake with confidence score

---

## Project Structure

```
DeepFakeModelProject/
├── app.py                    # FastAPI Backend
├── requirements.txt          # Python Dependencies
├── deepfake_cnn_gpu.h5      # Trained CNN Model
├── frontend/
│   ├── index.html           # Web Interface
│   ├── styles.css           # Styling
│   └── script.js            # Frontend Logic
├── src/
│   ├── model_cnn.py         # CNN Architecture
│   ├── train_cnn.py         # Training Script
│   └── data_utils.py        # Data Loading
└── data/                    # Dataset (Not in git)
```

---

## API Endpoints

### `GET /`
Root endpoint - API status

### `GET /health`
Health check endpoint

### `POST /predict`
Predict if an image is real or fake

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file

**Response:**
```json
{
  "success": true,
  "prediction": "Real",
  "confidence": 95.5,
  "raw_score": 0.045,
  "details": {
    "is_fake": false,
    "fake_probability": 4.5,
    "real_probability": 95.5
  }
}
```

---

## Training Your Own Model

If you want to train the CNN model from scratch:

### 1. Get the Dataset

Download the dataset and organize it as:
```
data/
└── Dataset/
    ├── Train/
    │   ├── Real/
    │   └── Fake/
    ├── Validation/
    │   ├── Real/
    │   └── Fake/
    └── Test/
        ├── Real/
        └── Fake/
```

### 2. Run Training

```bash
python src/train_cnn.py
```

The trained model will be saved in `src/models/`

---

## GPU Configuration

The backend automatically detects and uses GPU if available. For GTX 1650 (3.5GB VRAM), memory is limited to prevent crashes.

To change the GPU memory limit, edit `app.py`:
```python
tf.config.LogicalDeviceConfiguration(memory_limit=3500)  # Change to your VRAM
```

---

## Technologies Used

- **Backend**: FastAPI, TensorFlow/Keras, Python
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Model**: Convolutional Neural Network (CNN)
- **Styling**: Custom CSS with glassmorphism and animations

---

## License

MIT License - See LICENSE file for details
