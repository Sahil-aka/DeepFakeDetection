
import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure GPU memory (if available)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Limit GPU memory to 3.5GB for GTX 1650
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3500)]
        )
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="CNN-based deepfake image detection system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for model
model = None

# Model path - check multiple locations
MODEL_PATHS = [
    "deepfake_cnn_gpu.h5",
    "src/models/basic_cnn_best.h5",
    "src/models/basic_cnn_final.h5"
]

def load_model():
    """Load the trained CNN model"""
    global model
    
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            try:
                model = tf.keras.models.load_model(model_path)
                print("Model loaded successfully!")
                return
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                continue
    
    raise RuntimeError("No trained model found! Please train the CNN model first.")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for CNN model
    - Resize to 224x224
    - Convert to RGB
    - Normalize to [0, 1]
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Deepfake Detection API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict if an uploaded image is real or fake
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON with prediction result and confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)[0][0]
        
        # Convert to percentage
        confidence = float(prediction) * 100
        
        # Determine label (sigmoid output: 0 = Fake, 1 = Real)
        # prediction > 0.5 means closer to 1 (Real)
        is_real = prediction > 0.5
        label = "Real" if is_real else "Fake"
        
        # Adjust confidence for display
        # If Real (p > 0.5), confidence is p * 100
        # If Fake (p <= 0.5), confidence is (1 - p) * 100
        display_confidence = confidence if is_real else (100 - confidence)
        
        return JSONResponse(content={
            "success": True,
            "prediction": label,
            "confidence": round(display_confidence, 2),
            "raw_score": round(float(prediction), 4),
            "details": {
                "is_fake": not is_real,
                "fake_probability": round(100 - confidence, 2),
                "real_probability": round(confidence, 2)
            }
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
