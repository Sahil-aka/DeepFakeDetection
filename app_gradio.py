"""
app_gradio.py — Hugging Face Spaces Demo
Public deployment interface for DeepFake Detection system.
Features:
- Image Analysis with authenticity verdict & confidence
- Grad-CAM heatmap visualization highlighting manipulated facial regions
- Video Deepfake Detection with frame sampling timeline
"""

import os

import cv2
import gradio as gr
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.model_resnext_lstm import DeepFakeResNeXtLSTM

# ─── Model Setup ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device for Gradio: {device}")

model_paths = [
    "models/model_87_acc_20_frames_final_data.pt",
    "model_87_acc_20_frames_final_data.pt",
    "../model_87_acc_20_frames_final_data.pt",
]

model = DeepFakeResNeXtLSTM()
loaded = False
for p in model_paths:
    if os.path.exists(p):
        try:
            state_dict = torch.load(p, map_location=device, weights_only=False)
            if isinstance(state_dict, dict) and "model" in state_dict and not any(k.startswith("model.") for k in state_dict):
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=True)
            loaded = True
            print(f"✅ Loaded model weights from {p}")
            break
        except Exception as e:
            print(f"⚠️ Could not load {p}: {e}")

model.to(device)
model.eval()

# ─── Inference Functions ─────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def detect_image(image):
    if image is None:
        return "No image provided", {}, None

    pil_img = Image.fromarray(image).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        fake_prob = float(probs[0].item())
        real_prob = float(probs[1].item())

    is_real = real_prob > 0.5
    verdict = "REAL (Authentic)" if is_real else "FAKE (Manipulated)"
    conf = (real_prob if is_real else fake_prob) * 100.0

    breakdown = {
        "Authentic Probability": f"{real_prob * 100:.1f}%",
        "Deepfake Probability": f"{fake_prob * 100:.1f}%",
        "Model Backbone": "ResNeXt-50 (32x4d)",
        "Sequence Layer": "LSTM (hidden size 2048)",
        "Benchmark Accuracy": "87.8% (FaceForensics++)"
    }

    # Generate Grad-CAM heatmap
    try:
        heatmap = model.generate_gradcam(tensor, target_class=0)
        img_np = np.array(pil_img.resize((224, 224)))
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    except Exception as e:
        print(f"Heatmap error: {e}")
        overlay = None

    result_text = f"## Verdict: {verdict}\n\n**Confidence:** {conf:.1f}%"
    return result_text, breakdown, overlay


def detect_video(video_path):
    if not video_path:
        return "No video provided", {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Failed to open video file", {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(fps))  # 1 frame per second

    timeline = []
    idx = 0
    while cap.isOpened() and len(timeline) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=-1)[0]
                fake_prob = float(probs[0].item())
                timeline.append({
                    "time_sec": round(idx / fps, 1),
                    "fake_prob": round(fake_prob * 100, 1),
                    "status": "Fake" if fake_prob > 0.5 else "Real"
                })
        idx += 1
    cap.release()

    if not timeline:
        return "No frames extracted", {}

    fake_count = sum(1 for f in timeline if f["status"] == "Fake")
    fake_pct = (fake_count / len(timeline)) * 100.0
    overall = "DEEPFAKE DETECTED" if fake_pct >= 50.0 else "LIKELY AUTHENTIC"

    summary = {
        "Overall Verdict": overall,
        "Flagged Frames": f"{fake_count} / {len(timeline)} ({fake_pct:.1f}%)",
        "Frames Analyzed": len(timeline),
        "Timeline": timeline
    }
    return f"## Verdict: {overall}\n\nFlagged frames: {fake_pct:.1f}%", summary


# ─── Gradio UI Construction ──────────────────────────────────────────────────
with gr.Blocks(title="DeepFake Detection — Production Ensemble") as demo:
    gr.Markdown("# 🛡️ DeepFake Detection System")
    gr.Markdown(
        "Production-grade deepfake detection using a hybrid **ResNeXt-50 (32x4d) + LSTM** architecture "
        "achieving **87.8% accuracy** on the FaceForensics++ benchmark. Includes Grad-CAM explainability heatmaps."
    )

    with gr.Tab("📷 Image Detection & Grad-CAM"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="numpy", label="Upload Face Image")
                btn_img = gr.Button("Analyze Image", variant="primary")
            with gr.Column():
                txt_verdict = gr.Markdown(label="Verdict")
                json_breakdown = gr.JSON(label="Confidence Breakdown")
                img_heatmap = gr.Image(label="Grad-CAM Manipulation Heatmap (Red/Yellow = Flagged Regions)")

        btn_img.click(
            fn=detect_image,
            inputs=[img_input],
            outputs=[txt_verdict, json_breakdown, img_heatmap]
        )

    with gr.Tab("🎥 Video Deepfake Detection"):
        with gr.Row():
            with gr.Column():
                vid_input = gr.Video(label="Upload Video (MP4, MOV, AVI)")
                btn_vid = gr.Button("Analyze Video", variant="primary")
            with gr.Column():
                txt_vid_verdict = gr.Markdown(label="Video Verdict")
                json_vid_timeline = gr.JSON(label="Timeline & Analysis")

        btn_vid.click(
            fn=detect_video,
            inputs=[vid_input],
            outputs=[txt_vid_verdict, json_vid_timeline]
        )

if __name__ == "__main__":
    demo.launch()
